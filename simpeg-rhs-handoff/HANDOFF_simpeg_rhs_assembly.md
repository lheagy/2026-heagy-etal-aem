# Handoff — TDEM source terms are rebuilt from scratch at every time step

**Status:** diagnosed, not fixed. Intended as a separate SimPEG PR.
**Found in:** SimPEG `95843cbef` (local checkout `~/git/simpeg/simpeg`), 0.24.1.dev33.
**Impact here:** at 50 soundings on the global mesh, RHS assembly is **456 s of a
530 s forward (86%)**. Extrapolated to the 100-sounding forward, ~912 s of its
1501 s. See `caber-synthetic/COMPUTE_COST.md` §5.

---

## TL;DR

`s_e` for a magnetic-dipole/loop source factorises exactly as

```
s_e(t) = [ Cᵀ (M_{f,1/µ} b) ] · waveform.eval(t)
         └──── constant vector ────┘   └─ scalar ─┘
```

The constant vector is recomputed **in full** on every call — including a
re-assembly of a sparse inner-product matrix the simulation already caches. For
a `StepOffWaveform` the scalar is **0** for every step after the first, so the
whole thing is rebuilt and then multiplied by zero, 79 times out of 80.

A ~20-line cache makes `s_e` **576× faster** with bit-identical results
(demonstrated in `simpeg_rhs_minimal_example.py`, section 4).

---

## Where it is

| what | file | line |
|---|---|---|
| `Simulation3DElectricField.getRHS` | `electromagnetics/time_domain/simulation.py` | 1547 |
| `BaseTDEMSimulation.getSourceTerm` | `electromagnetics/time_domain/simulation.py` | 444 |
| `MagDipole.s_e` ← the hot spot | `electromagnetics/time_domain/sources.py` | 1516 |
| `CircularLoop(MagDipole)` | `electromagnetics/time_domain/sources.py` | 1557 |

Three layers compound:

**1. `getRHS` evaluates the source term twice per step** (simulation.py:1547):

```python
s_m, s_e   = self.getSourceTerm(tInd)
_,   s_en1 = self.getSourceTerm(tInd - 1)
```

`getSourceTerm(tInd - 1)` is exactly what the previous step already computed as
its `tInd`. Nothing is carried forward — an immediate 2× on everything below.

**2. `getSourceTerm` loops every source** (simulation.py:444), calling
`src.eval(sim, t)` → `s_m` + `s_e`. So the cost scales linearly with source
count, which is why this dominates multi-source forwards but is invisible in a
single-sounding benchmark.

**3. `MagDipole.s_e` recomputes two time-independent quantities** (sources.py:1516):

```python
C = simulation.mesh.edge_curl
b = self._bSrc(simulation)                                   # geometry only
if simulation._formulation == "EB":
    MfMui = simulation.mesh.get_face_inner_product(1.0/self.mu)   # rebuilt!
    ...
    return C.T * (MfMui * b) * self.waveform.eval(time)
```

- `self._bSrc(simulation)` — the analytic loop/dipole vector potential evaluated
  on all mesh edges, then curled. Depends only on source geometry, `mu` and the
  mesh. **Takes no `time` argument.**
- `mesh.get_face_inner_product(1.0/self.mu)` — assembles a sparse matrix over
  the whole mesh. The simulation already exposes exactly this as
  `simulation.MfMui`, cached. Measured **~2600× cheaper** to read the cached
  property than to rebuild.

Only `self.waveform.eval(time)` varies with time, and it is a scalar.

---

## Measurements

From `simpeg_rhs_minimal_example.py` on a 13,824-cell TensorMesh (45,000 edges),
one loop source, 20 steps. *The box was running a 50-worker inversion at the
time, so absolute times are inflated — the ratios are the point.*

| | time | share |
|---|---:|---:|
| `src.s_e(sim, t)` | 46.94 ms | 100% |
| ├ `_bSrc(simulation)` | 34.35 ms | 73% |
| ├ `mesh.get_face_inner_product(1/mu)` | 3.08 ms | 7% |
| └ `sim.MfMui` (cached, for comparison) | 0.0012 ms | — |
| **after the prototype cache** | **0.081 ms** | **576× faster** |

Per forward, per source: **1.88 s → 0.003 s** on that small mesh.

On the real problem (global OcTree, 147,204 edges, 80 steps, `StepOffWaveform`):
`getRHS` costs **113 ms per source per step**, i.e. **~9.1 s per source per
forward**, and the returned vector is **exactly zero for steps 2–80** (verified
directly). At 50 sources that is 456 s of a 530 s forward.

---

## Proposed fix

Cache the constant vector on the source, keyed so it invalidates correctly, and
use the simulation's cached `MfMui`:

```python
def s_e(self, simulation, time):
    key = (id(simulation.mesh), self.mu, simulation._formulation)
    if getattr(self, "_s_e_const_key", None) != key:
        C = simulation.mesh.edge_curl
        b = self._bSrc(simulation)
        if simulation._formulation == "EB":
            self._s_e_const = C.T * (simulation.MfMui * b)   # not a rebuild
        else:
            self._s_e_const = C * (1.0 / self.mu * b)
        self._s_e_const_key = key
    if self.waveform.has_initial_fields and time < simulation.time_steps[1]:
        return self._s_e_const
    return self._s_e_const * self.waveform.eval(time)
```

Two further wins, in rough order of value-per-risk:

1. **Short-circuit a zero waveform.** If `waveform.eval(time) == 0`, return
   `Zero()` and let `getRHS` skip the `Cᵀ M s_m` work too. For step-off that
   removes 79/80 of the remaining cost outright.
2. **Carry `getSourceTerm(tInd-1)` forward** in `getRHS` instead of recomputing
   it. Halves the call count for every source type and every waveform.

---

## Correctness caveats to settle in the PR

- **`mu` dependence.** `MfMui` is built from `simulation.mu`. If a simulation
  ever inverts for or mutates `mu`, the cached vector must invalidate. The key
  above uses `self.mu` (the *source's* mu, used for the analytic field) — check
  whether `simulation.mu` can differ and vary, and key on it if so.
  Note `MagDipole.bInitial` (sources.py:1466) already branches on
  `np.all(simulation.mu == self.mu)`, so the two are not assumed equal.
- **Mesh identity.** `id(mesh)` is a weak key — fine within a run, wrong if a
  mesh is garbage-collected and the id reused. Prefer a `weakref` or a
  simulation-side cache.
- **Tiled / meta simulations.** The same source object is *not* shared across
  tiles in our setup (each tile gets its own `Survey([src])`), but a shared
  source across differing meshes is exactly what the cache key must catch. Worth
  a test.
- **Other source types.** `LineCurrent` (sources.py:1703) and `RawVec_Grounded`
  (2212) have their own `s_e`; check whether the same factorisation applies
  before generalising.
- **Waveforms with initial fields.** The `time < simulation.time_steps[1]`
  branch must be preserved exactly — it is what makes the first step carry the
  un-scaled source.

---

## Reproducing

```bash
python simpeg_rhs_minimal_example.py
```

Self-contained (simpeg + discretize + numpy). Prints, in order: the RHS being
identically zero after step 1; the cost breakdown of one `s_e`; the two
`getSourceTerm` calls per `getRHS`; and the prototype fix with an equality check
against the original.

## Before opening the PR

- Search SimPEG issues/PRs — this is an obvious enough hot spot that it may
  already be known or partly addressed upstream.
- Benchmark a `RampOffWaveform` / `VTEMWaveform` case too. Those have a
  non-zero waveform at most steps, so win (1) above does not apply and the
  caching win is the whole story — that is the more general case and the better
  justification for the PR.
