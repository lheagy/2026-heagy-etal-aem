"""Regenerate dobs and plot:
1. -dB/dt vs x per y-line, all time channels (full model and layered bg)
2. anomaly = (full - bg_overburden) / |bg_overburden|: target visibility
   over the layered (halfspace + overburden) background
3. central-station sounding: overburden response decay vs target emergence
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from simpeg import maps, Data
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver
from simpeg.meta import MultiprocessingMetaSimulation

from _test_parametric import (
    build_global_mesh,
    build_local_meshes,
    build_true_model,
    sigma_back, sigma_target, sigma_air, sigma_overburden,
    rx_locs, rx_times, rx_x, rx_y,
    TIME_STEPS,
)
from simpeg import maps as _maps  # alias to avoid shadowing in closure


def main():
    Solver = get_default_solver()

    global_mesh = build_global_mesh()
    active_cells = global_mesh.cell_centers[:, 2] < 0
    print(f"global mesh: {global_mesh.n_cells} cells", flush=True)

    active_cells_map = maps.InjectActiveCells(
        global_mesh, active_cells, value_inactive=np.log(1e-8),
    )

    source_list = []
    for i in range(rx_locs.shape[0]):
        loc = rx_locs[i, :]
        rx = tdem.receivers.PointMagneticFluxTimeDerivative(loc, rx_times, orientation="z")
        wf = tdem.sources.StepOffWaveform()
        src = tdem.sources.CircularLoop(
            receiver_list=[rx], location=loc, orientation="z", radius=10, waveform=wf,
        )
        source_list.append(src)
    survey = tdem.Survey(source_list)
    n_data = len(source_list) * len(rx_times)
    print(f"survey: {len(source_list)} sources x {len(rx_times)} times = {n_data} data", flush=True)

    mesh_list = build_local_meshes(global_mesh, survey)
    time_steps = TIME_STEPS

    mappings, sims = [], []
    for ii, local_mesh in enumerate(mesh_list):
        tile_map = maps.TileMap(global_mesh, active_cells, local_mesh)
        local_actmap = maps.InjectActiveCells(
            local_mesh, active_cells=tile_map.local_active,
            value_inactive=np.log(1e-8),
        )
        mappings.append(tile_map)
        sims.append(tdem.simulation.Simulation3DElectricField(
            mesh=local_mesh,
            survey=tdem.Survey([survey.source_list[ii]]),
            time_steps=time_steps,
            solver=Solver,
            sigmaMap=maps.ExpMap() * local_actmap,
        ))
    sim = MultiprocessingMetaSimulation(sims, mappings)

    # --- three models: full (target + overburden), layered bg, halfspace ---
    sigma_full = build_true_model(global_mesh)
    sigma_ovb = build_true_model(global_mesh, include_target=False)
    sigma_bg_only = build_true_model(
        global_mesh, include_target=False, include_overburden=False,
    )

    import time
    dpreds = {}
    for name, sig in [("full", sigma_full), ("bg+ovb", sigma_ovb),
                      ("halfspace", sigma_bg_only)]:
        print(f"computing dpred for {name} model...", flush=True)
        t0 = time.time()
        dpreds[name] = sim.dpred(np.log(sig[active_cells]))
        print(f"  done in {time.time() - t0:.1f} s", flush=True)
    dobs_full = dpreds["full"]
    np.save("_dobs_cache.npy", dobs_full)
    print("  cached full-model dobs -> _dobs_cache.npy", flush=True)

    n_t = len(rx_times)
    d_full = dobs_full.reshape(len(rx_y), len(rx_x), n_t)
    d_ovb  = dpreds["bg+ovb"].reshape(len(rx_y), len(rx_x), n_t)
    d_bg   = dpreds["halfspace"].reshape(len(rx_y), len(rx_x), n_t)
    # target visibility over the LAYERED background
    anomaly = (d_full - d_ovb) / np.abs(d_ovb)
    # overburden signature relative to the halfspace
    ovb_anomaly = (d_ovb - d_bg) / np.abs(d_bg)

    # --- plot ---
    ny = len(rx_y)
    fig, axes = plt.subplots(ny, 3, figsize=(15, 3 * ny))
    if ny == 1:
        axes = axes[np.newaxis, :]
    cmap_times = plt.cm.viridis(np.linspace(0, 1, n_t))

    for j, y_val in enumerate(rx_y):
        # col 0: total dB/dt with target
        ax = axes[j, 0]
        for ti in range(n_t):
            ax.plot(rx_x, -d_full[j, :, ti], color=cmap_times[ti], lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("-dB/dt (T/s)")
        ax.set_title(f"y={y_val:.0f} m: target (all 20 channels)")
        ax.grid(True, alpha=0.3)

        # col 1: layered background (halfspace dashed for reference)
        ax = axes[j, 1]
        for ti in range(n_t):
            ax.plot(rx_x, -d_ovb[j, :, ti], color=cmap_times[ti], lw=1)
            ax.plot(rx_x, -d_bg[j, :, ti], color=cmap_times[ti], lw=0.6, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("x (m)")
        ax.set_title(f"y={y_val:.0f} m: bg+overburden (halfspace dashed)")
        ax.grid(True, alpha=0.3)

        # col 2: target anomaly relative to the layered background
        ax = axes[j, 2]
        for ti in range(n_t):
            ax.plot(rx_x, anomaly[j, :, ti], color=cmap_times[ti], lw=1)
        ax.axhline(0.05, color="r", ls="--", lw=0.8, label="5% noise")
        ax.axhline(-0.05, color="r", ls="--", lw=0.8)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("(full - bg_ovb) / |bg_ovb|")
        ax.set_title(f"y={y_val:.0f} m: target anomaly")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = "_plot_data.png"
    plt.savefig(out, dpi=110)
    print(f"saved -> {out}")

    # --- central-station sounding: overburden decay vs target emergence ---
    jc = len(rx_y) // 2
    ic = int(np.argmin(np.abs(rx_x)))
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4.5))
    ax3[0].loglog(rx_times, -d_bg[jc, ic, :], "k--", label="halfspace")
    ax3[0].loglog(rx_times, -d_ovb[jc, ic, :], "C0", label="+ overburden")
    ax3[0].loglog(rx_times, -d_full[jc, ic, :], "C3", label="+ target")
    ax3[0].set_xlabel("time (s)")
    ax3[0].set_ylabel("-dB/dt (T/s)")
    ax3[0].set_title(f"sounding at (x,y)=({rx_x[ic]:.0f},{rx_y[jc]:.0f})")
    ax3[0].legend()
    ax3[0].grid(True, alpha=0.3, which="both")

    ax3[1].semilogx(rx_times, np.abs(ovb_anomaly[jc, ic, :]), "C0",
                    label="|overburden vs halfspace|")
    ax3[1].semilogx(rx_times, np.abs(anomaly[jc, ic, :]), "C3",
                    label="|target vs bg+overburden|")
    ax3[1].axhline(0.05, color="r", ls="--", lw=0.8, label="5% noise")
    ax3[1].set_xlabel("time (s)")
    ax3[1].set_ylabel("relative anomaly")
    ax3[1].set_title("overburden decay vs target emergence")
    ax3[1].legend(fontsize=8)
    ax3[1].grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("_plot_data_sounding.png", dpi=110)
    print("saved -> _plot_data_sounding.png")

    # summarize anomaly visibility
    print(f"\nanomaly statistics (target σ = {sigma_target} S/m, "
          f"overburden σ = {sigma_overburden} S/m):")
    print(f"  target  anomaly (vs layered bg): max {anomaly.max():+.2f}, "
          f"min {anomaly.min():+.2f}, frac |.|>5%: "
          f"{(np.abs(anomaly) > 0.05).mean():.2%}")
    print(f"  ovb     anomaly (vs halfspace):  max {ovb_anomaly.max():+.2f}, "
          f"min {ovb_anomaly.min():+.2f}")
    with np.printoptions(precision=2, suppress=False):
        print(f"  central-station |ovb anomaly| per channel:\n"
              f"    {np.abs(ovb_anomaly[jc, ic, :])}")
        print(f"  central-station |target anomaly| per channel:\n"
              f"    {np.abs(anomaly[jc, ic, :])}")
    sim.join()

    # --- true model slices ---
    from _test_parametric import target_z
    norm = LogNorm(vmin=sigma_back / 5, vmax=max(sigma_target * 5, sigma_back * 50))
    z_center = float(np.mean(target_z))
    z_min = global_mesh.origin[2]
    dz_min = global_mesh.h[2][0]
    ind_z = int(round((z_center - z_min) / dz_min))

    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 4))
    out_xz = global_mesh.plot_slice(
        sigma_full, normal="y", pcolor_opts={"norm": norm}, ax=ax2[0],
    )
    ax2[0].plot(rx_locs[:, 0], rx_locs[:, 2], "wo", ms=3, mec="k", mew=0.5)
    ax2[0].set_xlim(800 * np.r_[-1, 1])
    ax2[0].set_ylim(np.r_[-400, 50])
    ax2[0].set_aspect(1)
    ax2[0].set_title(f"true model: xz slice at y=0   "
                     f"(σ_target = {sigma_target} S/m, dip 45°)")
    plt.colorbar(out_xz[0], ax=ax2[0], label="σ (S/m)")

    out_xy = global_mesh.plot_slice(
        sigma_full, normal="z", ind=ind_z,
        pcolor_opts={"norm": norm}, ax=ax2[1],
    )
    ax2[1].plot(rx_locs[:, 0], rx_locs[:, 1], "wo", ms=3, mec="k", mew=0.5)
    ax2[1].set_xlim(500 * np.r_[-1, 1])
    ax2[1].set_ylim(400 * np.r_[-1, 1])
    ax2[1].set_aspect(1)
    ax2[1].set_title(f"true model: xy slice at z={z_min + ind_z * dz_min:.0f} m")
    plt.colorbar(out_xy[0], ax=ax2[1], label="σ (S/m)")

    plt.tight_layout()
    out2 = "_plot_true_model.png"
    plt.savefig(out2, dpi=110)
    print(f"saved -> {out2}")


if __name__ == "__main__":
    main()
