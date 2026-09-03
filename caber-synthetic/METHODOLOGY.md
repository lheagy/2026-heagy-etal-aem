# Methodology — dipping-target synthetic AEM inversion workflow

Detailed, reproducible description of the forward modelling, the three inversion
results (two-stage parametric→3D, cold-start 3D, stitched 1D), and the
forward-simulation cost comparison. All work uses **SimPEG** (time-domain EM)
and **discretize** (OcTree / `TreeMesh`). The example is a synthetic dipping
conductive target beneath a heterogeneous conductive overburden, loosely
inspired by a real dipping-conductor exploration setting.

---

## 1. Workflow overview

- **Goal:** recover a dipping conductive target beneath a laterally variable
  conductive overburden from airborne TDEM (AEM) data, and show why a full 3D
  inversion is required (vs. stitched-1D, and vs. a cold-started 3D inversion).
- **Three inversions compared, all on the same synthetic data:**
  1. **Two-stage** — a parametric ellipsoid inversion (Phase 1) builds a
     starting/reference model for a full 3D voxel inversion (Phase 2).
  2. **Cold-start 3D** — the same 3D inversion started from a uniform halfspace
     (no parametric prior). Baseline.
  3. **Stitched 1D** — independent 1D layered inversions per sounding. Baseline.
- **Forward-cost study:** wall time and peak RAM of a single-sounding forward
  (`dpred`) on a small per-source *tiled* mesh vs. the full *global* mesh.

---

## 2. Synthetic (true) model

Conductivity model on the global OcTree mesh (S/m); air is `1e-8` S/m, all cells
with centre `z < 0` are "active" (subsurface).

- **Background halfspace:** 1000 Ω·m → `σ_back = 1e-3` S/m.
- **Dipping target (conductor):**
  - `σ_target = 5` S/m (200 Ω·m).
  - Dip **45° from horizontal** (slope +1; tilts *up* toward +x). [In code:
    `target_dip = 135`, slope `= tan(-dip·π/180) = +1`.]
  - Tabular slab, ~100 m thick (perpendicular), vertical extent `z ∈ [-300, -120]`
    m, strike extent `y ∈ [-100, 100]` m, along-dip extent limited to
    `x ∈ [-300, 300]` m. The slab subcrops (shallowest) near its up-dip (+x) end.
- **Conductive overburden (U-shaped / basin):**
  - `σ_overburden = 1/150 ≈ 6.7e-3` S/m (150 Ω·m).
  - Map extent `x ∈ [-100, 600]` m, `y ∈ [-200, 200]` m.
  - **Parabolic thickness** profile in x, zero at the footprint edges, maximum
    60 m at the centre (x = 250 m):
    `t(x) = 60·(1 − ((x − 250)/350)²)` for `x ∈ [-100, 600]`, else 0.
  - The basin blankets the target's shallow up-dip end while leaving clean
    halfspace on the far west — this is the lateral heterogeneity that defeats a
    1D assumption and that the parametric model (halfspace + ellipsoid) cannot
    represent.

---

## 3. Survey geometry

Airborne loop–loop TDEM, central-loop configuration, flown on parallel lines.

- **Transmitter:** circular loop, radius 10 m, `StepOffWaveform`, vertical
  (z-oriented) magnetic dipole moment; flight height 30 m.
- **Receiver:** co-located `PointMagneticFluxTimeDerivative` (dB/dt), z-component.
- **Time channels:** 20, log-spaced `2e-5` to `2e-3` s.
- **Line spacing:** 100 m (5 lines, `y = -200, -100, 0, 100, 200` m).
- **Along-line station spacing (this is the one survey difference between
  methods):**
  - **3D inversions (two-stage, cold-start):** 80 m → 10 stations/line ×
    5 lines = **50 soundings** (1000 data).
  - **Stitched 1D:** 40 m → 20 stations/line × 5 lines = **100 soundings**
    (2000 data). Finer along-line sampling is the natural deliverable for a
    stitched-1D product; the 80 m stations are a subset of the 40 m stations.

---

## 4. Forward simulation

- **Physics / discretisation:** SimPEG `time_domain.Simulation3DElectricField`
  (implicit backward-Euler E-field formulation) on an OcTree `TreeMesh`.
- **Model mapping:** `ExpMap` (invert for log-conductivity) composed with
  `InjectActiveCells` (active = subsurface; air fixed at `log(1e-8)`).
- **Time stepping:** `[(3e-6, 20), (1e-5, 20), (3e-5, 20), (1e-4, 20)]` = **80
  steps** (4 unique step sizes ⇒ 4 system factorizations reused across steps).
- **Linear solver:** MKL `Pardiso` (SimPEG default direct solver).
- **Global mesh:** OcTree, 20 m base cells, 8 km cubic domain (CCC origin),
  `diagonal_balance=True`. Refined to the finest level (i) at all receiver
  locations and (ii) over the target bounding box. **≈ 52,340 cells (≈ 42,284
  active).**
- **Tiled (per-source local) meshes:** one `TreeMesh` per transmitter, same base
  `h`/origin as the global mesh, refined to the finest level along a vertical
  column beneath the source (40 points from −300 m to the source height,
  `padding_cells_by_level=[4,2,2]`). **≈ 5,510 cells (≈ 3,602 active)** — about
  1/10 the global mesh.
  - The `[4,2,2]` refinement (4 fine cells laterally before coarsening) was
    important: a narrower fine-cell footprint left the dipping target poorly
    resolved laterally and produced ring/checkerboard artifacts in the 3D
    inversion.
- **Tiled meta-simulation:** `MultiprocessingMetaSimulation(sims, mappings)`.
  Each source's local simulation is wrapped with a `TileMap` (global active model
  → local active model). Sources are distributed across worker processes; **each
  worker runs single-threaded** (`OMP_NUM_THREADS = MKL_NUM_THREADS = 1`) so the
  per-source forwards do not oversubscribe cores against the process-level
  parallelism.

---

## 5. Synthetic data and uncertainties

- **Data are clean** — the forward response of the true model; **no random noise
  is added.** Uncertainties are *assigned* and used only to weight the misfit.
- **3D inversions:** relative error **5%** + noise floor `1e-12` (T/s).
- **Parametric Phase 1:** relative error **10%**, and only the **last 5 time
  channels** (channels 16–20, ≳ 8×10⁻⁴ s) — the late-time window after the
  overburden response has decayed, so the parametric target inversion is not
  biased by the (unmodelled) overburden.
- **Stitched 1D:** relative error **10%** + floor `1e-12` (see §8 for why 10%).
- **Misfit / target:** `L2DataMisfit` with χ² target = number of data
  (`TargetMisfit`).

---

## 6. Inversion 1 — Two-stage (parametric ellipsoid → full 3D)

### 6a. Phase 1 — parametric ellipsoid inversion

- **Model:** a single conductive ellipsoid in a halfspace, **11 parameters**:
  `[p₀ (log σ_back), log rx, log ry, log rz (semi-axes), φx, φy, φz (rotation),
  x₀, y₀, z₀ (centre), p_in (log σ_interior)]`.
  - Ellipsoid boundary `(x−x₀)ᵀ M (x−x₀) = 1`, with
    `M = Rᵀ diag(1/rx², 1/ry², 1/rz²) R`, `R = Rx·Ry·Rz`; interior/exterior
    conductivities blended across the boundary with a fixed sigmoid
    `boundary_sharpness = 10`. (Custom `ParametricEllipsoid` map → mesh
    log-conductivity.)
- **Data:** last 5 channels, 10% relative error (overburden-free window).
- **Simulation:** same tiled meta-simulation as the forward (each source's local
  mesh), mapping `TileMap · ParametricEllipsoid`.
- **Starting model:** disk-like ellipsoid, semi-axes (100, 100, 50) m, centred
  (0, 0, −200) m, no rotation, interior 5 S/m, background `1e-3` S/m.
- **Regularisation:** `Smallness` on the 11-parameter vector (light).
- **Bounds (`ProjectedGNCG`):** `log rx,ry,rz ∈ [log 5, log 1000]` m; angles
  `∈ [−π, π]`; `z₀ ≤ 0`; `p_in ∈ [log 1e-3, log 1e3]`; `p₀ ∈ [log 1e-6, log 1e-1]`.
- **Optimizer:** `ProjectedGNCG`, `maxIter = 10`, `cg_maxiter = 15`.
- **Result:** converges in **4 iterations** to φ_d ≈ 120 (target 250). Recovered
  ellipsoid: **dip φy ≈ 38.8°**, semi-axes ≈ (141, 121, 56) m, centre ≈
  (0, 0, −200) m, interior ≈ 3.6 S/m. A conservative but correctly-located and
  -oriented body — a good warm start.

### 6b. Phase 2 — full 3D voxel inversion

- **Mesh / simulation:** global OcTree, tiled meta-simulation (50 sources, all 20
  channels), `ExpMap`·`InjectActiveCells`. 1000 data, 5% error.
- **Starting & reference model `m₀`:** the Phase-1 ellipsoid mapped onto the
  global mesh (`ParametricEllipsoid · mopt_parametric`). The smallness
  *reference model* is also set to `m₀` (so smallness anchors the parametric
  structure rather than pulling toward a flat halfspace).
- **Regularisation:** `WeightedLeastSquares`, **`α_s = 0.01`**, `α_x = α_y =
  α_z = 1.0`. Smallness is deliberately down-weighted so spatial smoothness
  dominates — this lets the (unmodelled) overburden form in a data-driven way and
  stay laterally connected, instead of draining to the reference between stations
  ("donut" artifact) at larger `α_s`.
- **Trade-off (β) schedule:** `BetaEstimate_ByEig(beta0_ratio = 1e5)` then
  `BetaSchedule(coolingFactor = 2, coolingRate = 2)`. The large `beta0_ratio`
  starts the inversion **over-regularised**: the parametric `m₀` already nearly
  fits the data, so a weakly-regularised first Gauss–Newton step overshoots the
  target (or drives σ to the bound) in one iteration; starting strong and cooling
  toward the target is stable.
- **Optimizer:** `ProjectedGNCG`, `maxIter = 40`, `cg_maxiter = 40`, log-σ bounds
  `[1e-6, 1e2]` S/m, `tolF = tolX = 1e-10` (so termination comes from
  `TargetMisfit`/`maxIter`, not the optimizer's small-step test).
- **Result:** converges in **9 iterations** to φ_d ≈ 727 (target 1000). Recovers
  a coherent, clearly **dipping** conductor at the correct depth and location,
  σ_max ≈ **31 S/m**, no ring/checkerboard artifacts; the overburden appears as a
  near-surface conductive band.

---

## 7. Inversion 2 — Cold-start 3D (baseline)

- **Identical to Phase 2 in every respect** (mesh, tiled simulation, data,
  `α_s = 0.01`, `beta0_ratio = 1e5`, `coolingFactor = 2`, `cg_maxiter = 40`,
  bounds, `TargetMisfit`) **except the starting and reference model are a uniform
  halfspace** `m₀ = m_ref = log(σ_back)` — i.e. no parametric prior.
- `maxIter = 25` (the run is a non-convergence demonstration).
- **Result:** **fails to converge.** After 19 Gauss–Newton iterations it stalls
  at φ_d ≈ 20,800 — **~21× above the target** of 1000 — and the conductor never
  forms (σ_max ≈ 0.9 S/m). The data misfit plateaus while the model norm grows;
  the predicted late-time decay falls well below the observed data over the
  target. This isolates the value of the parametric warm start: same inversion,
  same data, only the starting model differs.

---

## 8. Inversion 3 — Stitched 1D (baseline)

Independent 1D layered inversions, one per sounding, stitched into a section.

- **Per-sounding forward:** SimPEG `time_domain.Simulation1DLayered` (semi-
  analytic), same loop (radius 10 m, `StepOff`) and 20 channels.
- **1D mesh:** 25 m layers over a 400 m core (16 layers) + 12 padding layers
  growing at 1.3× → **29 layers** (`TensorMesh`).
- **Data / uncertainty:** all 20 channels, **relative error 10%** + floor `1e-12`.
  - *Why 10% (vs 5% for the 3D):* the observed data are generated on the 3D
    OcTree mesh, but the 1D forward is semi-analytic. For an identical halfspace
    the two forwards disagree by ~20% (late time) up to ~2× (earliest channel),
    because the 10 m loop is under-resolved on the 20 m base cell. Inverting the
    3D-mesh data with an exact 1D forward at 5% forces the 1D model to invent
    spurious (deep) structure to fit that discrepancy; 10% absorbs it and a
    "background" sounding correctly returns ≈ halfspace.
- **Starting & reference model:** uniform **true background**
  `m₀ = m_ref = log(σ_back)` (so depths the data cannot constrain park at the
  true background rather than at an arbitrary start).
- **Regularisation:** `WeightedLeastSquares`, `α_s = 1/25 = 0.04`, `α_x = 1.0`,
  plus `UpdateSensitivityWeights` (depth/sensitivity weighting) and
  `UpdatePreconditioner`.
- **Fixed-β scheme (key for stitching):** β is **fixed at 10** for *all*
  iterations and *all* soundings — **no `BetaEstimate`, no `BetaSchedule`
  cooling.** A per-sounding-estimated or cooled β would stop each sounding at a
  different effective regularisation, producing a jumpy section; a single fixed β
  gives consistent regularisation and a smooth section.
- **Stopping:** `TargetMisfit(chifact = 1.1)` — stop as soon as
  χ² ≤ 1.1·N_data, catching the descent near the target rather than overshooting
  on the next Gauss–Newton step. `InexactGaussNewton`, **`maxIter = 5`**.
  - *Why `maxIter = 5`:* soundings directly over the target cannot reach the
    target misfit with a layered model and would otherwise run to `maxIter`,
    piling on laterally-inconsistent (choppy) structure; capping at 5 stops them
    before that plateau and yields a smoother, more coherent target smear.
- **Execution:** the 100 soundings are inverted independently and in parallel
  (`joblib`), then stitched (each recovered column placed at its station x along
  the central line).
- **Result:** clean background and a recovered conductive zone, but the target is
  **smeared and strongly under-recovered** (σ_max ≈ 0.12 S/m vs. true 5 S/m) and
  the overburden's amplitude/depth are not resolved (TDEM conductance–thickness
  equivalence). The 1D assumption cannot reconstruct the coherent dipping 3D body
  — motivating the 3D inversion.

---

## 9. Forward-simulation cost comparison (tiled vs global)

Quantifies the cost advantage of the per-source tiled meshes vs. a single global
mesh, for one sounding.

- **What is measured:** wall time and peak RAM of a **single `dpred` call** for
  one central sounding (y = 0, x ≈ 0), comparing:
  1. **tiled** — the source on its local mesh (≈ 5,510 cells), and
  2. **global** — the *same* source on the full global mesh (≈ 52,340 cells),
     same survey, same 80-step time discretisation, same Pardiso solver.
- **Protocol (important for the abstract):**
  - **Only `dpred` is timed.** All mesh / simulation / model construction is done
    first and excluded.
  - Reported time = **median of 3 consecutive cold `dpred` calls** (model reset
    between calls; the 3 runs agree to < 4%, so no field-caching artifact).
  - **Peak RAM** = process resident set size (RSS) sampled at 5 ms during
    `dpred`; we report both the **dpred increment** (peak − post-setup baseline,
    i.e. the forward solve's own footprint) and the total peak RSS.
  - **Single-threaded** (`OMP_NUM_THREADS = MKL_NUM_THREADS = 1`) for a clean,
    reproducible comparison — and because in production the tiling already
    parallelises across sources (one single-threaded worker per source).
- **Results (single thread):**

  | quantity | tiled (local mesh) | global mesh | ratio |
  |---|---:|---:|---:|
  | mesh cells | 5,510 | 52,340 | 9.5× |
  | `dpred` wall time (median) | **2.7 s** | **96 s** | **35× slower** |
  | `dpred` RAM increment | **61 MB** | **1,110 MB** | **18×** |
  | peak process RSS | 663 MB | 1,699 MB | 2.6× |

  The tiled per-source forward is **~35× faster** and uses **~18× less** memory
  for the solve. (The ~600 MB common baseline in peak RSS is the global mesh +
  model held in memory in both cases.)
- **Machine:** Intel Xeon E5-2670 v3 @ 2.30 GHz (2 sockets × 12 cores, 48
  threads), 126 GB RAM, Linux 5.15.

---

## 10. Software, versions, and reproducibility

- **Software:** SimPEG `0.24.1.dev33+g42a5db944`, discretize `0.11.3`,
  pymatsolver `0.3.1` (MKL Pardiso), NumPy `2.1.3`, SciPy `1.15.1`, Python
  `3.11`.
- **Scripts (this study):**
  - `_test_parametric.py` — true model, survey, meshes, Phase-1 parametric
    inversion; `parametric_ellipsoid.py` — the `ParametricEllipsoid` map.
  - `_test_full.py` — Phase-2 full 3D inversion (reads the Phase-1 model).
  - `_test_full_coldstart.py` — cold-start 3D baseline.
  - `_test_1d_single.py` / `_test_1d_stitched.py` — 1D single-sounding test and
    stitched run.
  - `_timing_test.py` + `_build_timing_report.py` → `_forward_timing.md`.
  - `_build_figures_notebook.py` → `dipping_target_figures.ipynb` (all figures).
- **Per-iteration output:** every 3D inversion saves
  `SaveOutputDictEveryIteration` (`iter, beta, phi_d, phi_m, f, m, dpred`) so
  convergence curves and per-iteration models/predicted data are recoverable.
- **Headline results:** two-stage recovers the dipping conductor (σ_max ≈ 31 S/m,
  φ_d → target in 9 iters); cold-start 3D stalls ~21× above target (conductor
  never forms); stitched 1D smears and under-recovers (σ_max ≈ 0.12 S/m). The
  tiled forward is ~35× faster / ~18× lighter than the global mesh per sounding.
