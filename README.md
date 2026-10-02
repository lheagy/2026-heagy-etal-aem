# 3D hybrid-parametric inversion of AEM data accelerated by tiling

> Heagy, L. J., Capriotti, J., Kang, S., Fournier, D., Weis, J., Kuttai, J.,
> Cowan, D., and Soler, S. R. (2026). *3D hybrid-parametric inversion of AEM
> data accelerated by tiling.* AEM 2026.

**Abstract** 
Airborne electromagnetic (AEM) surveys collect dense and, high-quality data, yet are most often inverted under a 1D layered-earth assumption that breaks down over geologically complex targets. Full 3D inversion can resolve these settings but is computationally expensive. We present a tiled 3D inversion workflow, implemented in the open-source SimPEG framework, in which each source is simulated on a small local OcTree mesh, and the per-source simulations are distributed in parallel. In our example, a single-sounding forward runs about an order of magnitude faster than on the global mesh, and distributing the solves accelerates the full survey by roughly two orders of magnitude, at comparable peak memory. Sensitivities are propagated through composable mappings, so the Jacobian is only ever computed on the small local meshes. The same code supports both voxel and parametric models. On a synthetic dipping conductor beneath a conductive overburden, a stitched 1D inversion smears and under-recovers the target and a naive cold-started 3D inversion stalls above the target misfit, whereas a two-stage hybrid-parametric inversion, warm-started from a recovered ellipsoid, converges quickly and recovers a coherent dipping conductor with the overburden. The framework also enables an efficient 3D forward check on 1D results, indicating where 1D suffices and where 3D is warranted.


## Layout

- `abstract/`: the abstract (`heagy-et-al-2026-aem.{pdf,docx}`) and the talk (`Heagy-et-al-AEM.{pdf,pptx}`).
- `dipping-conductor/`: the synthetic dipping conductor beneath a conductive overburden.
  - `dipping_target_setup.py`: mesh, true model, survey and tile meshes, plus the Phase-1 parametric inversion when run as a script. Everything else imports it.
  - `parametric_ellipsoid.py`: the `ParametricEllipsoid` mapping.
  - `dipping_target_figures.ipynb`: builds every figure from the cached results.
  - `fields_3_soundings.ipynb`: field snapshots and the field movie.
  - `METHODOLOGY.md`: the inversion setup in detail.
  - `COMPUTE_COST.md`, `_iteration_timing.md`, `TILING.md`: compute-cost benchmarks and notes on tiling.

## Figures

| Published as | File (in `dipping-conductor/`) | Made by |
|---|---|---|
| Abstract Fig 1 | `fig1_meshes.png` | `dipping_target_figures.ipynb` |
| Abstract Fig 2 | `fig2_model_comparison.png` | `dipping_target_figures.ipynb` |
| Abstract Fig 3 | `fig3_datafit_1d3d.png` | `dipping_target_figures.ipynb` |
| Talk (mappings slide) | `fig_mappings.png` | `_plot_mappings.py` |
| Talk (slide 4 movie) | `tdem_fields_src1.mp4` | `fields_3_soundings.ipynb` |

Re-running the notebook reproduces Figs 1 and 2 pixel-for-pixel. It reproduces
Fig 3's data, but the published PNG was restyled afterwards (figure size, fonts,
legend placement), so the regenerated version looks slightly different.
The notebook also writes `figS1`–`figS4` (data fit, convergence, forward cost,
overburden plan view). These supplementary figures appear in neither the
abstract nor the talk.

The compute-cost numbers quoted in the abstract and talk come from
`_timing_{global,tiled}.json`, `_ram_{global,tiled}.json`,
`_timing_breakdown_*.json` (summarised in `COMPUTE_COST.md`) and
`_iteration_timing.md`.

## The cached results are the record

The production inversions (Phase-2 hybrid-parametric 3D, cold-start 3D and
stitched 1D) were run on a cluster with scripts that were not preserved. Their
outputs in `dipping-conductor/` are the only record of those runs:

- `_mrec_ovbU5.npy`: recovered hybrid-parametric 3D model
- `_models_1d_40m.npy`: stitched-1D models
- `_dobs_40m.npy`, `_dobs_80m.npy`: observed data (40 m and 80 m surveys)
- `_dpred_1d_on_3d.npy`: 3D forward of the stitched-1D model (`_test_1d_forward3d.py`)
- `_mopt_test.npy`: ellipsoid parameters from a local Phase-1 run, used by `_plot_mappings.py`
- `_iters_phase1_ovbU5/`, `_iters_ovbU_5line/`, `_iters_coldstart_80m/`:
  per-iteration output (`InversionModel_*.npz`) for the parametric,
  hybrid-parametric and cold-start runs

## Environment

Python 3.11, SimPEG 0.24.1.dev33+g42a5db944, discretize 0.11.3,
pymatsolver 0.3.1 (MKL Pardiso), NumPy 2.1.3, SciPy 1.15.1.
Run the scripts and notebooks from inside `dipping-conductor/`.
