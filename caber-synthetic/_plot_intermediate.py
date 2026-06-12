"""Plot intermediate inversion iterations from saved npz files."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from simpeg import maps

from parametric_ellipsoid import ParametricEllipsoid
from _test_parametric import (
    build_global_mesh, build_true_model, sigma_back, sigma_target,
    target_z, rx_locs,
)
from _test_full import MOPT_PARAMETRIC


def load_iter(path):
    out = np.load(path, allow_pickle=True)["arr_0"].item()
    return out


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
    sigma_m0 = maps.ExpMap() * active_cells_map * ellipsoid * MOPT_PARAMETRIC

    files = sorted(glob.glob("InversionModel_*.npz"))
    print(f"found {len(files)} intermediate files")
    if not files:
        return

    # pick iters to show: m0, mid, and last
    iters_to_show = [files[0], files[len(files)//2], files[-1]]
    print("plotting:", [os.path.basename(f) for f in iters_to_show])
    sigmas = [maps.ExpMap() * active_cells_map * load_iter(f)["m"] for f in iters_to_show]
    titles = [
        f"iter {int(os.path.basename(f).split('_')[-1].replace('.npz', ''))}, "
        f"phi_d={load_iter(f).get('phi_d', float('nan')):.0f}"
        for f in iters_to_show
    ]

    norm = LogNorm(vmin=sigma_back / 5, vmax=sigma_target * 5)

    # 2 rows (xz, xy), 5 cols (true, m0, iter1, iter mid, iter last)
    z_center = float(np.mean(target_z))
    z_min = global_mesh.origin[2]
    dz_min = global_mesh.h[2][0]
    ind_z = int(round((z_center - z_min) / dz_min))

    n_cols = 2 + len(sigmas)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    all_titles = ["true", "m0"] + titles
    all_sigmas = [sigma_true, sigma_m0] + sigmas

    for ax, model, title in zip(axes[0], all_sigmas, all_titles):
        out = global_mesh.plot_slice(
            model, normal="y", pcolor_opts={"norm": norm}, ax=ax,
        )
        ax.plot(rx_locs[:, 0], rx_locs[:, 2], "wo", ms=3, mec="k", mew=0.5)
        ax.set_xlim(800 * np.r_[-1, 1])
        ax.set_ylim(np.r_[-400, 50])
        ax.set_aspect(1)
        ax.set_title(f"{title}\nxz @ y=0", fontsize=10)
        plt.colorbar(out[0], ax=ax, label="σ (S/m)")

    for ax, model, title in zip(axes[1], all_sigmas, all_titles):
        out = global_mesh.plot_slice(
            model, normal="z", ind=ind_z, pcolor_opts={"norm": norm}, ax=ax,
        )
        ax.plot(rx_locs[:, 0], rx_locs[:, 1], "wo", ms=3, mec="k", mew=0.5)
        ax.set_xlim(500 * np.r_[-1, 1])
        ax.set_ylim(400 * np.r_[-1, 1])
        ax.set_aspect(1)
        ax.set_title(f"{title}\nxy @ z=-200", fontsize=10)
        plt.colorbar(out[0], ax=ax, label="σ (S/m)")

    plt.tight_layout()
    out_path = "_plot_intermediate.png"
    plt.savefig(out_path, dpi=110)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
