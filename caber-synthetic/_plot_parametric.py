"""Plot the recovered parametric ellipsoid (from _mopt_test.npy).

Shows xz and xy slices through the recovered conductivity model with the
true-target outline overlaid for context.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import discretize
from simpeg import maps

from parametric_ellipsoid import ParametricEllipsoid
from _test_parametric import (
    build_global_mesh,
    build_true_model,
    sigma_back,
    sigma_target,
    target_z,
    rx_locs,
)


def main():
    global_mesh = build_global_mesh()
    active_cells = global_mesh.cell_centers[:, 2] < 0
    active_cells_map = maps.InjectActiveCells(
        global_mesh, active_cells, value_inactive=np.log(1e-8),
    )

    mopt = np.load("_mopt_test.npy")
    LABELS = ["p_0","log_rx","log_ry","log_rz","phi_x","phi_y","phi_z","x_0","y_0","z_0","p_in"]
    print("recovered parametric model:")
    for lab, v in zip(LABELS, mopt):
        print(f"  {lab:>7} = {v:+10.4g}")
    rx_, ry_, rz_ = np.exp(mopt[1:4])
    print(f"\n  semi-axes (m):  rx={rx_:.1f}  ry={ry_:.1f}  rz={rz_:.1f}")
    print(f"  center (m):     ({mopt[7]:.1f}, {mopt[8]:.1f}, {mopt[9]:.1f})")
    print(f"  angles (deg):   ({np.degrees(mopt[4]):.1f}, {np.degrees(mopt[5]):.1f}, {np.degrees(mopt[6]):.1f})")
    print(f"  sigma_back:     {np.exp(mopt[0]):.4g} S/m")
    print(f"  sigma_target:   {np.exp(mopt[10]):.4g} S/m")

    ellipsoid = ParametricEllipsoid(
        global_mesh, active_cells=active_cells, boundary_sharpness=10.0,
    )

    # Recovered sigma on the global mesh (log-cond -> sigma)
    sigma_param = maps.ExpMap() * active_cells_map * ellipsoid * mopt
    # True sigma for comparison
    sigma_true = build_true_model(global_mesh)

    # color scale
    norm = LogNorm(vmin=sigma_back / 5, vmax=sigma_target * 5)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # ----- xz slice (normal y) at y=0 -----
    for ax, model, title in zip(axes[0], [sigma_true, sigma_param],
                                ["true model", "parametric recovery"]):
        out = global_mesh.plot_slice(
            model, normal="y", pcolor_opts={"norm": norm}, ax=ax,
        )
        ax.plot(rx_locs[:, 0], rx_locs[:, 2], "wo", ms=3, mec="k", mew=0.5)
        ax.set_xlim(800 * np.r_[-1, 1])
        ax.set_ylim(np.r_[-400, 50])
        ax.set_aspect(1)
        ax.set_title(f"{title}: xz slice at y=0")
        plt.colorbar(out[0], ax=ax, label="σ (S/m)")

    # ----- xy slice (normal z) near target depth -----
    z_center = float(np.mean(target_z))  # ~-210
    # TreeMesh plot_slice needs ind = (z - z_min) / dz_min on the finest grid.
    z_min = global_mesh.origin[2]
    dz_min = global_mesh.h[2][0]
    ind_z = int(round((z_center - z_min) / dz_min))
    z_ind_val = z_min + ind_z * dz_min
    print(f"\nplotting xy slice at z = {z_ind_val:.1f}  (ind={ind_z})")

    for ax, model, title in zip(axes[1], [sigma_true, sigma_param],
                                ["true model", "parametric recovery"]):
        out = global_mesh.plot_slice(
            model, normal="z", ind=ind_z, pcolor_opts={"norm": norm}, ax=ax,
        )
        ax.plot(rx_locs[:, 0], rx_locs[:, 1], "wo", ms=3, mec="k", mew=0.5)
        ax.set_xlim(500 * np.r_[-1, 1])
        ax.set_ylim(400 * np.r_[-1, 1])
        ax.set_aspect(1)
        ax.set_title(f"{title}: xy slice at z={z_ind_val:.0f} m")
        plt.colorbar(out[0], ax=ax, label="σ (S/m)")

    plt.tight_layout()
    out = "_plot_parametric.png"
    plt.savefig(out, dpi=130)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
