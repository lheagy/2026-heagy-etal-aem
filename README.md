# 3D hybrid-parametric inversion of AEM data accelerated by tiling

Code, cached results, and figures for

> Heagy, L. J., Capriotti, J., Kang, S., Fournier, D., Weis, J., Kuttai, J.,
> Cowan, D., and Soler, S. R. (2026). *3D hybrid-parametric inversion of AEM
> data accelerated by tiling.* AEM 2026.

The extended abstract and the slides are in [`abstract/`](abstract/).

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
