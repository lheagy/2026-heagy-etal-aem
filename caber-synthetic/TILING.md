# Tiled meta-simulations: MT vs. time-domain

Pseudocode for the tiled `MetaSimulation` pattern in SimPEG, in both the
magnetotelluric and the time-domain case, and why the two tile along different
axes.

---

## MT — tile by frequency

```python
# 1 | one fine "global" mesh holds the true model
global_mesh   = build_global_mesh(frequencies, stations)
sigma         = sphere_in_halfspace(global_mesh)
active_cells  = global_mesh.cell_centers[:, 2] < 0.0

# 2 | frequency-dependent meshes
sims, mappings = [], []
for freq in frequencies:
    local_mesh = build_local_mesh(freq, stations)

    # 3 | the mapping carries (and averages) the model
    mappings.append(HarmonicTileMap(
        model_mesh, active_cells, mesh
    ))

    # 4 | a standard SimPEG MT simulation on that mesh
    sims.append(Simulation3DPrimarySecondary(
        local_mesh,
        survey=mt_survey(freq, stations),
        sigmaMap=InjectActiveCells(mesh, ...),
        solver=Pardiso)
    )

# 5 | compose: behaves like ONE simulation
meta  = MetaSimulation(sims, mappings)
dpred = meta.dpred(sigma[active])   # Jvec / Jtvec too
```

---

## Time domain — tile by transmitter

```python
# 1 | one fine "global" mesh holds the true model
global_mesh   = build_global_mesh(sounding_locations)
sigma         = dipping_target_in_halfspace(global_mesh)
active_cells  = global_mesh.cell_centers[:, 2] < 0.0

# 2 | source-dependent meshes  (TDEM tiles by TRANSMITTER, not by time)
sims, mappings = [], []
for src in survey.source_list:
    local_mesh = build_local_mesh(src, refine_depth=300)

    # 3 | the mapping carries (and volume-averages) the model
    tile = TileMap(global_mesh, active_cells, local_mesh)
    mappings.append(tile)

    # 4 | a standard SimPEG TDEM simulation on that mesh
    #     all 80 steps share this mesh; 4 unique dt -> 4 factorizations
    sims.append(Simulation3DElectricField(
        local_mesh,
        survey=Survey([src]),
        time_steps=[(3e-6, 20), (1e-5, 20), (3e-5, 20), (1e-4, 20)],
        sigmaMap=ExpMap() * InjectActiveCells(local_mesh, tile.local_active),
        solver=Pardiso)
    )

# 5 | compose: behaves like ONE simulation
meta  = MultiprocessingMetaSimulation(sims, mappings)   # sources -> processes
dpred = meta.dpred(log(sigma[active_cells]))            # Jvec / Jtvec too
```

---

## Why the tiling axis differs

**MT tiles by frequency. TDEM tiles by source.** This is forced by the
discretisation, not a stylistic choice:

- Frequencies are **independent** linear systems. Each can have its own mesh,
  sized to its skin depth, and they solve in any order.
- Time steps are **not**. Backward Euler couples step *n* to step *n−1* through
  `Asubdiag * f[..., tInd]`, so every step must live on the same mesh as the one
  before it. **There is no tiling axis in time.**

So in TDEM the parallel axis left to you is the transmitter — and for AEM that is
a good one, because each loop only illuminates its own footprint. Measured on
the dipping-target synthetic:

| | global mesh | local (tiled) mesh |
|---|---:|---:|
| cells | 52,340 | 5,510 |
| active cells | 42,284 | 3,602 |
| single-sounding `dpred` | 96.0 s | 2.7 s (**35× faster**) |

## The loop over sources is not a loop over solves

Within a tile, every source in that tile's survey is solved **simultaneously** as
extra right-hand-side columns, and the factorization is **shared** across all 20
steps in a `dt` block. The cost structure is therefore

> **4 factorizations + 80 solve calls** — independent of the number of sources,

not `80 × n_src`. One factorization costs about as much as 110 solves. See
[COMPUTE_COST.md](COMPUTE_COST.md) for the full breakdown.

## Two things easy to get wrong

**The mapping slot composes.** It is not restricted to tiling — the parametric
inversion plugs in here by pre-composing the ellipsoid:

```python
mappings.append(tile * ParametricEllipsoid(global_mesh, active_cells))  # 11 params -> tile
```

giving the chain `m → ParametricEllipsoid → TileMap → InjectActiveCells →
ExpMap → σ` (see `_plot_mappings.py`).

**Pin each worker to one thread.** With `MultiprocessingMetaSimulation`, set

```python
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
```

*before* importing numpy, so the per-source direct solves do not oversubscribe
cores against the process-level parallelism. Omitting this is easy and costly.

---

Implementation: `_test_parametric.py` (`build_global_mesh`,
`build_local_meshes`, and the `sim_full` / `sim_parametric` construction at
lines 205–275).
