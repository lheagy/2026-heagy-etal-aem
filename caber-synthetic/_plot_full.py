"""Plot the recovered full-mesh inversion result (_mrec_full.npy) vs the true
model and the parametric m0."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
MOPT_PARAMETRIC = np.load("_mopt_test.npy")


def main():
    global_mesh = build_global_mesh()
    active_cells = global_mesh.cell_centers[:, 2] < 0
    active_cells_map = maps.InjectActiveCells(
        global_mesh, active_cells, value_inactive=np.log(1e-8),
    )

    sigma_true = build_true_model(global_mesh)

    ellipsoid = ParametricEllipsoid(
        global_mesh, active_cells=active_cells, boundary_sharpness=10.0,
    )
    sigma_param = maps.ExpMap() * active_cells_map * ellipsoid * MOPT_PARAMETRIC

    import os
    if os.path.exists("_mrec_full.npy"):
        mrec = np.load("_mrec_full.npy")
        sigma_rec = maps.ExpMap() * active_cells_map * mrec
        print(f"mrec range: [{mrec.min():.3f}, {mrec.max():.3f}] log(S/m)")
        print(f"   -> sigma: [{np.exp(mrec.min()):.3e}, {np.exp(mrec.max()):.3e}] S/m")
    else:
        sigma_rec = sigma_param.copy()
        print("(no _mrec_full.npy yet; plotting m0 in the recovery slot)")

    norm = LogNorm(vmin=sigma_back / 5, vmax=sigma_target * 5)

    # 3 columns: true, m0 (parametric), mrec (full)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # ---- xz slices ----
    for ax, model, title in zip(
        axes[0],
        [sigma_true, sigma_param, sigma_rec],
        ["true model", "parametric m0", "full recovery"],
    ):
        out = global_mesh.plot_slice(
            model, normal="y", pcolor_opts={"norm": norm}, ax=ax,
        )
        ax.plot(rx_locs[:, 0], rx_locs[:, 2], "wo", ms=3, mec="k", mew=0.5)
        ax.set_xlim(800 * np.r_[-1, 1])
        ax.set_ylim(np.r_[-400, 50])
        ax.set_aspect(1)
        ax.set_title(f"{title}: xz slice at y=0")
        plt.colorbar(out[0], ax=ax, label="σ (S/m)")

    # ---- xy slices at z ~ -200 ----
    z_center = float(np.mean(target_z))
    z_min = global_mesh.origin[2]
    dz_min = global_mesh.h[2][0]
    ind_z = int(round((z_center - z_min) / dz_min))
    z_val = z_min + ind_z * dz_min

    for ax, model, title in zip(
        axes[1],
        [sigma_true, sigma_param, sigma_rec],
        ["true model", "parametric m0", "full recovery"],
    ):
        out = global_mesh.plot_slice(
            model, normal="z", ind=ind_z, pcolor_opts={"norm": norm}, ax=ax,
        )
        ax.plot(rx_locs[:, 0], rx_locs[:, 1], "wo", ms=3, mec="k", mew=0.5)
        ax.set_xlim(500 * np.r_[-1, 1])
        ax.set_ylim(400 * np.r_[-1, 1])
        ax.set_aspect(1)
        ax.set_title(f"{title}: xy at z={z_val:.0f} m")
        plt.colorbar(out[0], ax=ax, label="σ (S/m)")

    plt.tight_layout()
    out_path = "_plot_full.png"
    plt.savefig(out_path, dpi=110)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
