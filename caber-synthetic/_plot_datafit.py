"""Observed vs predicted data for a parametric model (default _mopt_test.npy).

Observed = black dots, predicted = blue lines, -dB/dt vs along-line x, one
panel per inversion time channel, for the central (y=0) line. Shows whether
the parametric m0 fits the late-channel data the dip is meant to be
constrained by.

    python _plot_datafit.py [model.npy]
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import numpy as np
import matplotlib.pyplot as plt

from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver
from simpeg.meta import MultiprocessingMetaSimulation

from parametric_ellipsoid import ParametricEllipsoid
from _test_parametric import (
    build_global_mesh, build_local_meshes,
    rx_locs, rx_times, rx_x, rx_y, TIME_STEPS,
)

TIMES_INVERT = slice(15, None)  # last 5 channels, as in _test_parametric


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "_mopt_test.npy"
    mopt = np.load(model_path)
    dip_angle = np.degrees(mopt[5])
    print(f"model {model_path}: phi_y (dip) = {dip_angle:.1f} deg", flush=True)

    Solver = get_default_solver()
    global_mesh = build_global_mesh()
    active_cells = global_mesh.cell_centers[:, 2] < 0
    active_cells_map = maps.InjectActiveCells(
        global_mesh, active_cells, value_inactive=np.log(1e-8),
    )

    times_inv = rx_times[TIMES_INVERT]
    full_src, inv_src = [], []
    for i in range(rx_locs.shape[0]):
        loc = rx_locs[i, :]
        wf = tdem.sources.StepOffWaveform()
        kw = dict(location=loc, orientation="z", radius=10, waveform=wf)
        rx_f = tdem.receivers.PointMagneticFluxTimeDerivative(loc, rx_times, orientation="z")
        rx_i = tdem.receivers.PointMagneticFluxTimeDerivative(loc, times_inv, orientation="z")
        full_src.append(tdem.sources.CircularLoop(receiver_list=[rx_f], **kw))
        inv_src.append(tdem.sources.CircularLoop(receiver_list=[rx_i], **kw))
    survey_full = tdem.Survey(full_src)
    survey_inv = tdem.Survey(inv_src)

    mesh_list = build_local_meshes(global_mesh, survey_full)
    global_ellipsoid = ParametricEllipsoid(
        global_mesh, active_cells=active_cells, boundary_sharpness=10.0,
    )
    mappings, sims = [], []
    for ii, local_mesh in enumerate(mesh_list):
        tile_map = maps.TileMap(global_mesh, active_cells, local_mesh)
        local_actmap = maps.InjectActiveCells(
            local_mesh, active_cells=tile_map.local_active,
            value_inactive=np.log(1e-8),
        )
        mappings.append(tile_map * global_ellipsoid)
        sims.append(tdem.simulation.Simulation3DElectricField(
            mesh=local_mesh, survey=tdem.Survey([survey_inv.source_list[ii]]),
            time_steps=TIME_STEPS, solver=Solver,
            sigmaMap=maps.ExpMap(local_mesh) * local_actmap,
        ))
    sim = MultiprocessingMetaSimulation(sims, mappings)

    print("predicting data from parametric model ...", flush=True)
    dpred = sim.dpred(mopt)
    sim.join()

    nx, ny = len(rx_x), len(rx_y)
    n_it = len(times_inv)
    dobs = np.load("_dobs_cache.npy").reshape(ny, nx, len(rx_times))[:, :, TIMES_INVERT]
    dpred = dpred.reshape(ny, nx, n_it)

    rel_err = 0.05
    misfit = (dpred - dobs) / (rel_err * np.abs(dobs) + 1e-12)
    chi2_n = np.mean(misfit ** 2)
    print(f"overall chi^2/N (inv channels) = {chi2_n:.2f}", flush=True)

    np.savez("_datafit_arrays.npz", dobs=dobs, dpred=dpred,
             times_inv=times_inv, rx_x=rx_x, rx_y=rx_y, dip=dip_angle)

    iy_c = ny // 2  # y = 0
    fig, axes = plt.subplots(1, n_it, figsize=(2.9 * n_it, 4), sharey=True)
    if n_it == 1:
        axes = [axes]
    for it, ax in enumerate(axes):
        ax.plot(rx_x, -dobs[iy_c, :, it], "ko", ms=4, label="observed")
        ax.plot(rx_x, -dpred[iy_c, :, it], "b-", lw=1.5, label="predicted")
        ax.set_yscale("log")
        ax.set_xlabel("x (m)")
        ax.set_title(f"t = {times_inv[it]*1e3:.2f} ms", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        if it == 0:
            ax.set_ylabel("-dB/dt (T/s)")
            ax.legend(fontsize=8)
    fig.suptitle(
        f"central line (y=0): observed vs predicted   "
        f"[dip={dip_angle:.1f} deg, chi2/N={chi2_n:.2f} @5% err]", y=1.0,
    )
    plt.tight_layout()
    out = "_plot_datafit.png"
    plt.savefig(out, dpi=95)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
