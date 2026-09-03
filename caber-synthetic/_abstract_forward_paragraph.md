# Abstract (revised draft)

Airborne electromagnetic (AEM) surveys collect dense, high-quality data, yet are
most often inverted under a 1D layered-earth assumption that breaks down over
geologically complex targets. Full 3D inversion can resolve these settings but is
computationally expensive. We present a tiled 3D inversion workflow, implemented
in the open-source SimPEG framework, in which each source is simulated on a small
local OcTree mesh (about one-tenth the size of a global mesh) and the independent
per-source simulations are distributed across a worker pool. A single-sounding
forward runs about 35× faster than the same simulation on the global mesh; across
the full survey, distributing the per-source solves over the worker pool yields a
roughly 90× speedup. Because each tile solve is small and self-contained, this
acceleration is achieved at peak memory comparable to a single global solve, so
many solves run concurrently on modest hardware. Sensitivities are propagated
through composable mappings, so the Jacobian is only ever formed on the small
local meshes, and the same code supports both voxel and low-dimensional
parametric models. On a synthetic dipping conductor beneath a conductive
overburden, a stitched 1D inversion smears and under-recovers the target and a
cold-started 3D inversion fails to converge, whereas a two-stage
hybrid-parametric inversion—warm-started from a recovered ellipsoid—converges
quickly and recovers a coherent dipping conductor together with the overburden.
The framework's flexibility also enables an efficient 3D forward check on 1D
inversion results, indicating where the 1D assumption holds and where 3D
inversion is warranted.

---

# Abstract (revised draft — lighter touch, exact numbers deferred to methods)

Airborne electromagnetic (AEM) surveys collect dense, high-quality data, yet are
most often inverted under a 1D layered-earth assumption that breaks down over
geologically complex targets. Full 3D inversion can resolve these settings but is
computationally expensive. We present a tiled 3D inversion workflow, implemented
in the open-source SimPEG framework, in which each source is simulated on a small
local OcTree mesh (about one-tenth the size of a global mesh) and the independent
per-source simulations are distributed across a worker pool. A single-sounding
forward runs about an order of magnitude faster than the same simulation on the
global mesh, and distributing the solves across the pool accelerates the full
survey by roughly two orders of magnitude, at peak memory comparable to a single
global solve. Sensitivities are propagated through composable mappings, so the
Jacobian is only ever formed on the small local meshes, and the same code
supports both voxel and low-dimensional parametric models. On a synthetic dipping
conductor beneath a conductive overburden, a stitched 1D inversion smears and
under-recovers the target and a cold-started 3D inversion fails to converge,
whereas a two-stage hybrid-parametric inversion—warm-started from a recovered
ellipsoid—converges quickly and recovers a coherent dipping conductor together
with the overburden. The framework's flexibility also enables an efficient 3D
forward check on 1D inversion results, indicating where the 1D assumption holds
and where 3D inversion is warranted.

### Notes on the abstract rewrite (not for the abstract)

- **Two variants above.** Both make the same one substantive change to your
  draft — the forward-cost sentence — and frame the benefit honestly as
  throughput at *comparable* peak memory rather than a memory saving.
  - *Variant 1 (exact numbers):* states the per-sounding 35× and full-survey
    ~90× explicitly. Use if the abstract is the only place these numbers appear,
    or if you want the figures up front.
  - *Variant 2 (lighter touch):* says "about an order of magnitude" per sounding
    and "roughly two orders of magnitude" for the full survey, deferring the
    exact 35×/90× to the methods paragraph. Use if methods already carries the
    precise figures, to avoid printing identical numbers twice.
- In either variant the memory clause is optional and can be cut for length
  without affecting the rest.
- Everything else is unchanged in substance and matches the workflow (parametric
  ellipsoid warm start, stitched-1D under-recovery, cold-start non-convergence,
  1D-on-3D forward check).

---

# Abstract — forward-modeling / tiling paragraph (draft)

The global OcTree mesh uses 20 m base cells over an 8 km domain, refined at the
receivers and over the target, for ~52,300 cells. Each per-source tile mesh
shares the same base cell size but is refined only beneath its source, giving
~5,500 cells per tile, about one-tenth of the global mesh. To solve the TDEM
problem, Maxwell's equations are discretized in space with a mimetic
finite-volume method on the OcTree mesh and backward-Euler for the time-stepping
(Haber, 2014). We use a direct solver (Pardiso) for the forward. Per sounding,
a single forward takes 2.7 s on a tile mesh versus 96 s on the global mesh,
roughly 35× faster. The independent per-source solves distribute across a worker
pool; running 48 workers in parallel, the full 100-sounding forward completes in
~16 s versus ~25 min on the global mesh (~90×), at comparable peak memory
(~13.5 vs ~11.3 GB). The tiled cost is dominated by holding all per-sounding
meshes and operators in the driver, which grows with survey size; building each
tile lazily within its worker would reduce this and is a natural next step for
larger surveys.

---

## Notes (not for the abstract)

- **35× is per-sounding, 90× is full-survey** — two different comparisons, both
  kept intentionally. Per sounding: 2.7 s vs 96 s (tile vs global solve). Full
  survey (100 soundings): ~16 s tiled across 48 workers vs ~1500 s global. The
  full-survey global is one shared factorization followed by 100 × 80 implicit
  backward-Euler back-substitutions, which is why it is ~25 min rather than
  100 × 96 s.
- **Comparable peak memory** (~13.5 GB tiled vs ~11.3 GB global): the tiling win
  at survey scale is wall-clock, not memory. Tiled peak is dominated by a ~9 GB
  floor from holding all 100 simulation objects + TileMap operators in the
  driver (scales with sounding count); global peak is dominated by storing all
  100 sources' time-domain fields at once. Numbers from `_ram_usage.md` /
  `_ram_tiled.json` / `_ram_global.json`.
- Worker count = 48 = `cpu_count()` on the benchmark node (default
  `n_processes=None`), single-threaded workers (OMP/MKL = 1).

---

# Abstract (revised draft — trimmed to ≤200 words, 198 words)

Airborne electromagnetic (AEM) surveys collect dense, high-quality data, yet are
most often inverted under a 1D layered-earth assumption that breaks down over
geologically complex targets. Full 3D inversion can resolve these settings but is
computationally expensive. We present a tiled 3D inversion workflow in the
open-source SimPEG framework, in which each source is simulated on a small local
OcTree mesh and per-source simulations run in parallel. A single-sounding
forward runs about an order of magnitude faster than on the global mesh, and
distributing the solves accelerates the full survey by roughly two orders of
magnitude, at comparable peak memory. Sensitivities propagate
through composable mappings, so the Jacobian is only ever computed on the small
local meshes, and the same code supports voxel and parametric models. On a
synthetic dipping conductor beneath a conductive overburden, a stitched 1D
inversion smears and under-recovers the target and a naive cold-started 3D
inversion (no sensitivity weighting) stalls well above target, whereas a
two-stage hybrid-parametric inversion, warm-started from a recovered ellipsoid,
converges quickly and recovers a coherent dipping conductor with the overburden. The framework also enables an efficient 3D forward
check on 1D results, indicating where 1D suffices and where 3D is warranted.

# Cold-started 3D — methods paragraph (draft)

Cold-started 3D. To illustrate a failure mode that can arise when targeting a
highly conductive, compact body, we run a basic, uninformed inversion without
sensitivity weighting, starting from a uniform halfspace. It does not converge:
after 19 iterations the data misfit stalls roughly 20× above its target, the
conductor never forms, and strong "ringing" sensitivity artifacts develop (Yang
et al., 2018; Figures 2c, 3b). The inversion fits the early-time channels but not
the later times over the target, which are governed by the conductor. Other
strategies, such as sensitivity weighting, could improve the recovery; here we
instead illustrate a warm-starting approach using a parametric inversion.

### Cold-start fact-check (verified against `_iters_coldstart_80m_superseded/`)

- No sensitivity weighting: confirmed — `_test_full_coldstart.py` directives are
  `BetaEstimate_ByEig`, `BetaSchedule`, `SaveOutputDictEveryIteration`,
  `TargetMisfit` only (no `UpdateSensitivityWeights`).
- Iterations: saved run = **19** (not 20); `maxIter` was capped at 25.
- Misfit: final φ_d = 20,792 = **20.8× target** (n_data = 1000). "Roughly 20×."
- Early-vs-late: over the 3 soundings above the target, median |rel. misfit| is
  **~4% at the earliest channels and ~96% at the latest** — confirms "fits
  early-time, misses late-time over the target."

### Trim notes (vs. the full-length variants above)

Cuts are redundancy only, no claims removed: "implemented in"→"in";
"distributed in parallel"→"run in parallel"; dropped "the same simulation" and
shortened the memory clause to "at comparable peak memory"; merged the
mappings + voxel/parametric sentences ("both … low-dimensional"→"voxel and
parametric"); joined the two results sentences with "whereas"; and
"where the 1D assumption holds and where 3D inversion is warranted"→"where 1D
suffices and where 3D is warranted". Next ~4 words available by cutting "In our
example,".

---

# Conclusions (draft)

Tiling substantially reduces the cost of the forward and sensitivity
computations that dominate each step of a 3D AEM inversion, making a targeted 3D
inversion tractable on modest hardware. It remains more expensive than a 1D
inversion, but for focused 3D problems the cost is manageable. A key strength of
the framework is its flexibility: because the parametric and voxel inversions are
built from the same composable mappings, a low-dimensional parametric inversion
can supply the prior that warm-starts the full 3D inversion. In the example shown
here, this warm start enabled recovery of a 3D model that fit both the early- and
late-time data, whereas a naive cold start from a halfspace—without sensitivity
weighting—did not converge. Sensitivity weighting or other strategies could also
improve the cold-start result; the warm start is simply one effective option that
the shared parametric/voxel framework makes natural.

The data fits also point to a broader use of the tiled code. The apparent success
of a 1D inversion in fitting its own data does not guarantee that the recovered
model is consistent with the 3D physics. A 3D forward simulation of the stitched
1D result is a simple, inexpensive check on where the 1D assumption holds; where
discrepancies arise, they indicate where a targeted 3D inversion of the kind
shown here is warranted.

The framework's flexibility also makes it straightforward to experiment with
alternative parameterizations and priors. The tiling strategy is general enough
to support other physics—for example airborne frequency-domain EM, and
potentially natural-source EM—though the different character of the sensitivities
in those problems will require further testing.

### Conclusions edit notes

- Opening now implies the inversion benefit: "the forward and sensitivity
  computations that **dominate each step of a 3D AEM inversion**" — connects the
  forward-cost reduction to the inversion without quoting per-iteration times or
  invoking CG-iteration behavior.
- "greatly reduces the computational cost" → "substantially reduces"; dropped the
  "days into minutes to hours" total-wall-clock claim.
- Cold start softened to a "naive … without sensitivity weighting" baseline, with
  an explicit note that sensitivity weighting or other strategies could also help.
