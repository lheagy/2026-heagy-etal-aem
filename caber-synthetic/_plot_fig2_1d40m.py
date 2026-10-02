"""Copy of Figure 2 with the 40 m stations drawn on the stitched-1D panel.

The stitched-1D model in fig2_model_comparison.png is already the 40 m result
(`_models_1d_40m.npy`, 20 stations x 5 lines, 40 m cells). Only the *markers*
were wrong: the top row plotted `rx_locs_80` on all four panels, so the 1D
section carried 10 dots above a 20-station model. The bottom row already got
this right. This script fixes the top row and writes a separate file.

Writes fig2b_model_comparison_1d40m.png (does not touch fig2_model_comparison.png).
Run: python _plot_fig2_1d40m.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import discretize
from simpeg import maps

from _test_parametric import build_global_mesh, build_true_model, rx_y, target_z

CMAP = plt.get_cmap("Spectral_r").copy()
CMAP.set_bad("white")
NORM2 = LogNorm(vmin=1e-3, vmax=10)
RX = dict(marker="o", linestyle="none", ms=3.5, mfc="k", mec="w", mew=0.5)


def load_iters(folder):
    recs = [np.load(f, allow_pickle=True)["arr_0"].item()
            for f in sorted(glob.glob(os.path.join(folder, "*.npz")))]
    recs.sort(key=lambda o: o["iter"])
    return recs


def main():
    mesh = build_global_mesh()
    active = mesh.cell_centers[:, 2] < 0
    AIR = ~active
    actmap = maps.InjectActiveCells(mesh, active, value_inactive=np.log(1e-8))
    sigma_true = build_true_model(mesh)

    rx_x_40 = (np.linspace(-500, 500, 26))[3:-3]      # 1D survey: 20/line
    rx_x_80 = rx_x_40[::2]                            # 3D survey: 10/line
    rz = 30.0
    rx_locs_80 = discretize.utils.ndgrid([rx_x_80, rx_y, np.r_[rz]])
    rx_locs_40 = discretize.utils.ndgrid([rx_x_40, rx_y, np.r_[rz]])

    def sigma_full(m_active):
        s = np.asarray(maps.ExpMap() * actmap * m_active, dtype=float)
        s[AIR] = np.nan
        return s

    sigma_true_plot = sigma_true.copy(); sigma_true_plot[AIR] = np.nan
    Z_SLICE = float(np.mean(target_z))
    IZ = int(round((Z_SLICE - mesh.origin[2]) / mesh.h[2][0]))

    mrec_two = np.load("_mrec_ovbU5.npy")
    mrec_cold = load_iters("_iters_coldstart_80m_superseded")[-1]["m"]
    models_1d = np.load("_models_1d_40m.npy")         # (100, 29) = 20 x 5 stations

    _CS, _CORE, _NPAD, _PF = 25.0, 400.0, 12, 1.3
    THICK = discretize.utils.unpack_widths(
        [(_CS, int(np.ceil(_CORE / _CS))), (_CS, _NPAD, _PF)])
    z1d_top = np.r_[0.0, -np.cumsum(THICK)]
    z1d_cen = np.r_[(z1d_top[:-1] + z1d_top[1:]) / 2, z1d_top[-1] - THICK[-1] / 2]

    nx40, ny = len(rx_x_40), len(rx_y)
    iy0 = ny // 2
    sec_1d = np.exp(models_1d[iy0 * nx40:(iy0 + 1) * nx40])
    z1d_edges = np.r_[0.0, -np.cumsum(np.r_[THICK, THICK[-1]])]
    dx = np.diff(rx_x_40).mean(); dy = np.diff(rx_y).mean()
    x1d_edges = np.r_[rx_x_40 - dx / 2, rx_x_40[-1] + dx / 2]
    y1d_edges = np.r_[rx_y - dy / 2, rx_y[-1] + dy / 2]
    il = int(np.argmin(np.abs(z1d_cen - Z_SLICE)))
    slice_1d = np.exp(models_1d[:, il]).reshape(ny, nx40)

    print(f"1D model: {models_1d.shape[0]} soundings = {nx40} x {ny}, "
          f"along-line spacing {dx:.0f} m")
    print(f"3D survey: {rx_locs_80.shape[0]} soundings, "
          f"spacing {np.diff(rx_x_80).mean():.0f} m")

    def msh_xz(sig, ax):
        return mesh.plot_slice(sig, normal="y", ax=ax,
                               pcolor_opts={"norm": NORM2, "cmap": CMAP})[0]

    def msh_xy(sig, ax):
        return mesh.plot_slice(sig, normal="z", ind=IZ, ax=ax,
                               pcolor_opts={"norm": NORM2, "cmap": CMAP})[0]

    fig, ax = plt.subplots(2, 4, figsize=(19, 7.5))
    cols = ["(a) true", "(b) stitched 1D", "(c) cold-start 3D",
            "(d) two-stage 3D"]

    # --- top row: xz ---
    im = msh_xz(sigma_true_plot, ax[0, 0])
    ax[0, 1].pcolormesh(x1d_edges, z1d_edges, sec_1d.T, norm=NORM2, cmap=CMAP)
    msh_xz(sigma_full(mrec_cold), ax[0, 2])
    msh_xz(sigma_full(mrec_two), ax[0, 3])
    for j, c in enumerate(cols):
        # THE FIX: the 1D column is a 40 m survey, the rest are 80 m
        locs = rx_locs_40 if j == 1 else rx_locs_80
        on_line = np.abs(locs[:, 1]) < 1e-6          # only the y = 0 stations
        ax[0, j].plot(locs[on_line, 0], locs[on_line, 2], **RX)
        ax[0, j].set_xlim([-600, 600]); ax[0, j].set_ylim([-450, 50])
        ax[0, j].set_aspect(1)
        spacing = 40 if j == 1 else 80
        ax[0, j].set_title(f"{c}\nxz at y=0   ({spacing} m stations)")
        ax[0, j].set_xlabel("x (m)")
    ax[0, 0].set_ylabel("z (m)")
    for j in range(1, 4):          # plot_slice sets its own "z" ylabel
        ax[0, j].set_ylabel("")

    # --- bottom row: xy ---
    msh_xy(sigma_true_plot, ax[1, 0])
    ax[1, 1].pcolormesh(x1d_edges, y1d_edges, slice_1d, norm=NORM2, cmap=CMAP)
    msh_xy(sigma_full(mrec_cold), ax[1, 2])
    msh_xy(sigma_full(mrec_two), ax[1, 3])
    for j in range(4):
        locs = rx_locs_40 if j == 1 else rx_locs_80
        ax[1, j].plot(locs[:, 0], locs[:, 1], **RX)
        ax[1, j].set_xlim([-500, 500]); ax[1, j].set_ylim([-400, 400])
        ax[1, j].set_aspect(1)
        n = locs.shape[0]
        ax[1, j].set_title(f"xy at z={Z_SLICE:.0f} m   ({n} soundings)")
        ax[1, j].set_xlabel("x (m)")
    ax[1, 0].set_ylabel("y (m)")
    for j in range(1, 4):
        ax[1, j].set_ylabel("")

    fig.colorbar(im, ax=ax, shrink=0.6, label="σ (S/m)")
    fig.savefig("fig2b_model_comparison_1d40m.png", dpi=200,
                bbox_inches="tight")
    print("wrote fig2b_model_comparison_1d40m.png")


if __name__ == "__main__":
    main()
