"""Baseline B: stitched 1D inversions.

Per-station Simulation1DLayered inversions using the validated single-sounding
recipe from _test_1d_single.make_inversion:
- 25 m layers over a 400 m core, 12 padding layers at 1.3x
- all 20 time channels
- FIXED beta = 10 (no cooling), TargetMisfit(chifact=1.1)
- 10% relative error + 1e-12 floor (the 3D-mesh dobs carries a ~20%-to-2x
  forward discrepancy vs the exact 1D forward; 5% would convert that into
  spurious deep structure, so we loosen to 10%)
- m0 = reference = true background (sigma_back)

Reads _dobs_cache.npy (must match the current survey in _test_parametric).
Writes _models_1d.npy (n_stations x n_layers, log-sigma) and
_plot_1d_stitched.png (stitched section at y=0 vs truth).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import discretize
from simpeg import (
    maps, Data, data_misfit, inverse_problem,
    regularization, optimization, directives, inversion,
)
from simpeg.electromagnetics import time_domain as tdem

from _test_parametric import (
    build_global_mesh, build_true_model,
    rx_locs, rx_times, rx_x, rx_y, sigma_back,
)
from _test_1d_single import make_inversion, THICK, N_LAYERS

# validated single-sounding recipe (see _test_1d_single)
thicknesses1d = THICK
n_layers = N_LAYERS


def run_inv(ind, dobs_sounding):
    # maxIter=5: soundings over the target can't reach the target misfit with
    # a layered model and would otherwise run to maxIter, piling on laterally-
    # inconsistent structure (choppy target zone). Capping at 5 stops them
    # before that plateau, giving a smoother, more coherent target smear.
    inv, sim, dmis = make_inversion(rx_locs[ind], dobs_sounding, beta=10.0, maxIter=5)
    m0 = np.log(sigma_back) * np.ones(N_LAYERS)
    mopt = inv.run(m0)
    return ind, mopt, dmis(mopt)  # dmis = chi^2 = sum((r/std)^2)


def main():
    n_t = len(rx_times)
    n_stations = rx_locs.shape[0]
    dobs = np.load("_dobs_cache.npy")
    assert dobs.size == n_stations * n_t, (
        f"dobs cache ({dobs.size}) does not match survey "
        f"({n_stations} x {n_t})"
    )
    d_per_station = dobs.reshape(n_stations, n_t)

    # sanity check on one sounding first
    print("test sounding 0 ...", flush=True)
    _, mtest, phi_d0 = run_inv(0, d_per_station[0])
    print(f"  station 0: phi_d={phi_d0:.1f} (target {n_t}), "
          f"sigma range [{np.exp(mtest).min():.3g}, {np.exp(mtest).max():.3g}]",
          flush=True)

    from joblib import Parallel, delayed
    # n_jobs kept modest: a heavy 3D job may be sharing the box. 1D-layered
    # forwards are cheap, so this still finishes in minutes.
    print(f"running {n_stations} 1D inversions ...", flush=True)
    results = Parallel(n_jobs=12)(
        delayed(run_inv)(ind, d_per_station[ind]) for ind in range(n_stations)
    )
    models = np.zeros((n_stations, n_layers))
    phi_ds = np.zeros(n_stations)
    for ind, mopt, phi_d in results:
        models[ind] = mopt
        phi_ds[ind] = phi_d
    np.save("_models_1d.npy", models)
    print(f"saved -> _models_1d.npy; phi_d: median {np.median(phi_ds):.1f}, "
          f"max {phi_ds.max():.1f} (target {n_t})", flush=True)

    # ---- stitch: section at y=0 ----
    z_edges = -np.r_[0.0, np.cumsum(np.r_[thicknesses1d, thicknesses1d[-1]])]
    iy_c = len(rx_y) // 2
    nx = len(rx_x)
    # station index = iy * nx + ix  (ndgrid: x cycles fastest)
    sec = np.exp(models[iy_c * nx:(iy_c + 1) * nx])  # (nx, n_layers)

    dx = np.diff(rx_x).mean()
    x_edges = np.r_[rx_x - dx / 2, rx_x[-1] + dx / 2]

    global_mesh = build_global_mesh()
    sigma_true = build_true_model(global_mesh)

    # true model spans 1e-3..10 (target = 5 S/m); the 1D recovery is much
    # weaker and dips below 1e-3, so give it its own 1e-4..1 scale.
    norm_true = LogNorm(vmin=1e-3, vmax=10)
    norm_1d = LogNorm(vmin=1e-4, vmax=1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    out = global_mesh.plot_slice(
        sigma_true, normal="y", pcolor_opts={"norm": norm_true}, ax=axes[0],
    )
    axes[0].set_title("true model: xz at y=0")
    plt.colorbar(out[0], ax=axes[0], label="σ (S/m)")

    pc = axes[1].pcolormesh(x_edges, z_edges, sec.T, norm=norm_1d)
    axes[1].set_title("stitched 1D inversions: section at y=0")
    plt.colorbar(pc, ax=axes[1], label="σ (S/m)")
    for ax in axes:
        ax.plot(rx_locs[:, 0], np.zeros(n_stations), "kv", ms=4)
        ax.set_xlim([-600, 600])
        ax.set_ylim([-450, 50])
        ax.set_aspect(1)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
    plt.tight_layout()
    plt.savefig("_plot_1d_stitched.png", dpi=110)
    print("saved -> _plot_1d_stitched.png", flush=True)


if __name__ == "__main__":
    main()
