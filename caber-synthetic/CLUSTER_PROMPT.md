# Cluster session prompt — Caber-synthetic AEM inversion

> Paste the section below into a fresh Claude Code session on the cluster.
> The TL;DR up top is enough to start; the appendices give context if needed.

---

## TL;DR

We're developing a **two-stage AEM inversion workflow** in
`caber-synthetic/`: (1) parametric ellipsoid inversion produces a
starting model for (2) a full-mesh inversion. Phase 1 is in OK shape;
phase 2 keeps producing **ring artifacts** and the **σ × volume
degeneracy** (compact high-σ blobs in place of extended moderate-σ
targets).

I want to test whether **more data** and/or **a finer mesh** breaks the
degeneracy. The local machine ran out of compute headroom — this is
why we're on the cluster.

**Concrete deliverables I'd like first:**

1. Decide what "more data" should look like (more receivers? denser
   y-line spacing? earlier time channels? larger transmitter footprint?
   moving sources?). Recommend one to try first and why.
2. Refine the mesh: finer base cells (10 m instead of 20 m), more local
   refinement around sources. Re-run the parametric (phase 1) to verify
   the workflow still produces a reasonable starting model.
3. Re-run the full-mesh inversion (phase 2). The acceptance test is
   that the recovered model preserves m0's geometry (no checkerboarding
   / ring) AND reaches the target misfit without σ hitting the upper
   bound.

Use the existing `_test_parametric.py` and `_test_full.py` as the
starting point. They're self-contained scripts that mirror the
notebook setup, so you can iterate without running the notebook
end-to-end. Plotting helpers are `_plot_data.py`,
`_plot_parametric.py`, `_plot_full.py`, `_plot_intermediate.py`.

**Be efficient.** Each forward sim is a few minutes; iterate
deliberately. Before each run, predict what you expect to happen, then
compare the actual log to your prediction — that's faster than
"change one thing, run, look, repeat" 20 times.

---

## Repo layout

```
caber-synthetic/
├── caber-synthetic.ipynb          # original notebook (do not break)
├── parametric_ellipsoid.py        # ParametricEllipsoid mapping (refactored)
├── _test_parametric.py            # phase-1 standalone (parametric inversion)
├── _test_full.py                  # phase-2 standalone (full-mesh inversion)
├── _plot_data.py                  # plot dB/dt, anomaly, true model
├── _plot_parametric.py            # plot parametric recovery vs truth
├── _plot_full.py                  # plot full-mesh recovery vs truth
├── _plot_intermediate.py          # plot intermediate inversion iters
└── .gitignore                     # excludes run artifacts
```

The `_test_*` scripts use `MultiprocessingMetaSimulation`, so on a
cluster with more cores you should get a real speedup. The standalone
scripts cache `_dobs_cache.npy` so you only generate the synthetic
data once per (mesh + truth-model) configuration.

---

## Current settings (in `_test_parametric.py` at HEAD)

**Truth model:**
- `sigma_back = 1e-3 S/m` (1000 Ω·m halfspace, no overburden yet)
- `sigma_target = 5 S/m` (200 Ω·m) — we lowered this from 10 to ease
  the σ×volume degeneracy
- `target_dip = 135` (= 45° from horizontal, slope = +1)
- `target_thickness = 100 m` (slab is ~100 m thick in x at center depth)
- `target_z = [-300, -120]` (vertical extent)
- `target_y = [-100, 100]` (strike extent)

**Survey:**
- 30 stations (rx_x: 10 in [-380, 340] every 80 m, rx_y: 3 at -200/0/200)
- 20 time channels logspace(2e-5, 2e-3) s
- StepOffWaveform, CircularLoop (radius 10 m)
- Tx height 30 m

**Mesh:**
- Global TreeMesh, 20 m base cells, 8 km domain
- Refined at receivers and over the target bounding box
- Local meshes per source for `MultiprocessingMetaSimulation`

**Parametric inversion (phase 1):**
- Starting model: disk-like ellipsoid `(rx, ry, rz) = (100, 100, 50) m`,
  centered (0, 0, -200), no rotation, σ_target = 5 S/m, σ_back = 1e-3 S/m
- Layout `[p_0, log_rx, log_ry, log_rz, phi_x, phi_y, phi_z, x_0, y_0, z_0, p_in]`
- `boundary_sharpness = 10` (fixed, not inverted)
- Inversion: ProjectedGNCG, maxIter=10, cg_maxiter=15
- last 5 time channels (`times_invert = slice(15, None)`)
- relative error: 0.10
- Bounds: log(5) ≤ log_r ≤ log(1000); log(1e-3) ≤ p_interior ≤ log(1e3)

**Full-mesh inversion (phase 2):**
- m0 = parametric recovery (`global_ellipsoid * mopt_parametric`)
- reference_model = m0 (NOT background — using background as ref pulled
  m0 structure away)
- regularization: WeightedLeastSquares, alpha_s = 0.1, smoothness = 1
- `reference_model_in_smooth=False` (user prefers not adding this prior yet)
- BetaEstimate_ByEig(beta0_ratio=10), BetaSchedule(coolingFactor=2,
  coolingRate=2)
- ProjectedGNCG, maxIter=15, cg_maxiter=40, log-cond bounds [1e-6, 100] S/m
- SaveOutputDictEveryIteration(saveOnDisk=True)

---

## What's already known

**Phase 1 (parametric) works OK** for the σ=5 / 45° dip case. Recovered
~(127, 103, 71 m) horizontal disk at right depth, σ ≈ 1.3 S/m. Misses
the dip but gives a clean, conservative starting model.

**Phase 2 (full-mesh) fails** for this problem in two characteristic ways:
1. If `alpha_s` is too small, the smoothness term lets the model
   "wiggle" and produces a **checkerboard / ring artifact** in xy.
2. If `alpha_s` is too large (with reference = background), the
   smallness term **pulls the model toward the flat background**,
   wiping the m0 structure.

We tried lots of (β₀_ratio, α_s, reference_model) combinations. Even
giving the inversion an **ideal hand-crafted m0** that exactly
matched the true dipping slab geometry still produced the ring
artifact. So the issue isn't the starting model — the **data alone
doesn't constrain the spatial distribution well enough**.

**Hypothesis for the cluster session:** the 30-station survey × 20
channels is too sparse to break the σ×volume degeneracy. The 20 m
base cell size is fine — what's likely too small is the **fine-cell
footprint** around each source on the *local* meshes. Each local
mesh is refined with `padding_cells_by_level=[2, 2, 2]`, which means
only 2 cells of fine refinement before the cells start coarsening.
For a dipping target that extends laterally, the source needs to
"see" the target through a wider band of fine cells.

## What NOT to do (already explored, didn't help)

- `alpha_s ∈ {1e-6, 1e-4, 1e-3, 0.1}` with reference=background
- `beta0_ratio ∈ {1, 10, 100}`
- `coolingFactor ∈ {2}`, `coolingRate ∈ {1, 2}`
- ProjectedGNCG vs InexactGaussNewton
- Various ellipsoid starting models (vertical pipe, disk, x-prolate)
- Boundary sharpness 5 vs 10 vs 15 in parametric
- Bumping CG max iter from 15 to 40

Increasing CG iters did help convergence of the linear subproblem
(residual went from 11% → 3%), so 40 is a good default. The issue
above CG is the geology/data resolution.

## Things to try (in rough priority order)

1. **Wider fine-cell footprint on the local meshes.** Change
   `padding_cells_by_level=[2, 2, 2]` to `[4, 2, 2]` in
   `build_local_meshes` (and try also in `build_global_mesh`). This
   doubles the lateral extent of the finest-cell zone around each
   source without going to a finer base cell — keeps cell count
   manageable but gives each source a broader view of the target.
   **Try this first**; it's cheap and addresses the most likely cause.
2. **More receivers in y** (denser y-line spacing). Currently only 3
   y-lines (−200/0/+200) so any feature elongated in y is undersampled.
3. **More time channels** (40 instead of 20). More direct measurement
   of the diffusion profile.
4. **Moving sources** (more "station footprint" coverage, e.g. shift
   sources in y too instead of co-located with receivers).
5. **`reference_model_in_smooth=True`** (we deferred this — it's a
   strong prior, but worth testing once we know other things aren't
   the bottleneck).

**Don't** drop the base cell size below 20 m — it isn't the
bottleneck and triples the cell count for marginal benefit.

For each: predict whether it will (a) break the σ×volume degeneracy
or (b) just smooth out the artifacts you already see. If (b), try
the next thing.

## Acceptance test

A passing recovery should show, in `_plot_full.py`:
- xz slice: a conductor that **clearly dips** (not vertical pipe, not
  horizontal blanket)
- xy slice: a single coherent blob at the right depth — no
  checkerboard, no donut, no halo extending to the receiver locations
- σ_recovered_max ≤ ~10 S/m (well below the 100 S/m upper bound)
- φ_d hits target (~600) without overshoot

If you can hit those three conditions, take screenshots of
`_plot_full.png` and the convergence log, then re-run with the same
settings but the **original σ_target = 10** case (rho_target = 0.1)
to see if the workflow now stretches to the harder case.

## Notebook

The notebook `caber-synthetic.ipynb` was updated with the same
settings the scripts use (parametric m0, β₀ ratio, etc.). You can
update it to match whatever the cluster run lands on, but don't
break the existing cell structure.

---

## Appendix A — model convention (so you don't get bitten)

`parametric_ellipsoid.py` boundary: `(x - x_0)^T M (x - x_0) = 1` with
`M = R^T diag(1/r_x², 1/r_y², 1/r_z²) R`, `R = Rx @ Ry @ Rz`. So
`r_x, r_y, r_z` are the **semi-axes** (not full axes). Parameters are
`log_r`, not `r`. Body-x in world frame = `R^T e_x = first row of R`.

Dip convention in the notebook: `slope = tan(-dip * π/180)`. So
`dip=135` → slope +1 → 45° from horizontal in xz plane, tilting up at
+x.

## Appendix B — coding/workflow preferences

- Iterate by editing `_test_parametric.py` / `_test_full.py` directly,
  not by re-templating new scripts every time.
- When changing parameters, **change one thing at a time** and write
  down what you predict. Don't change three things simultaneously.
- Be terse in user-facing summaries — short tables and bullets over
  long prose. Show the convergence numbers (φ_d / target / β / σ
  range / bound hits) every time.
- Plots are cheap; if anything looks weird, `_plot_*.py` first, debate
  second.
- Don't tighten the σ upper bound silently — flag it before doing it.
- Don't introduce `reference_model_in_smooth=True` silently — flag it
  before doing it.

## Appendix C — file caching

- `_dobs_cache.npy` is invalidated by any change to truth-model
  geometry or σ values. Delete it whenever you change `sigma_target`,
  `target_dip`, `target_thickness`, the mesh, or the survey.
- `_mopt_test.npy` is the latest parametric recovery — overwritten
  every parametric run.
- `_mrec_full.npy` is the latest full-mesh recovery — overwritten
  every full-mesh run.
- `InversionModel_*.npz` is one file per iteration. Delete before each
  new run so you don't mix old/new iters.
