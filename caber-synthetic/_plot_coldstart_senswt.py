"""Figures for the sensitivity-weighted cold-start inversion.

Writes:
  fig_coldstart_senswt.png      model comparison (true / cold / cold+senswt / two-stage)
  fig_senswt_diagnostics.png    sensitivity weights + convergence curves

Run: python _plot_coldstart_senswt.py
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

CMAP = plt.get_cmap("Spectral_r").copy(); CMAP.set_bad("white")
NORM = LogNorm(vmin=1e-3, vmax=10)
RX = dict(marker="o", linestyle="none", ms=3.5, mfc="k", mec="w", mew=0.5)
XLIM = np.r_[-600.0, 600.0]
ZLIM = np.r_[-600.0, 80.0]        # deeper than fig2: the artifact sits at -370 m


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

    rx_x_80 = (np.linspace(-500, 500, 26))[3:-3][::2]
    rx_locs_80 = discretize.utils.ndgrid([rx_x_80, rx_y, np.r_[30.0]])
    on_line = np.abs(rx_locs_80[:, 1]) < 1e-6

    def sig_full(m_active):
        s = np.asarray(maps.ExpMap() * actmap * m_active, dtype=float)
        s[AIR] = np.nan
        return s

    sigma_true_plot = sigma_true.copy(); sigma_true_plot[AIR] = np.nan
    Z_SLICE = float(np.mean(target_z))
    IZ = int(round((Z_SLICE - mesh.origin[2]) / mesh.h[2][0]))

    it_cold = load_iters("_iters_coldstart_80m_superseded")
    it_sw = load_iters("_iters_coldstart_senswt")
    mrec_cold = it_cold[-1]["m"]
    mrec_sw = np.load("_mrec_coldstart_senswt.npy")
    mrec_two = np.load("_mrec_ovbU5.npy")
    wr = np.load("_wr_coldstart_senswt.npy")

    cols = [("(a) true", sigma_true_plot),
            ("(b) cold-start\n(no weighting)", sig_full(mrec_cold)),
            ("(c) cold-start\n+ sensitivity weighting", sig_full(mrec_sw)),
            ("(d) two-stage\n(parametric warm start)", sig_full(mrec_two))]

    # ---------------------------------------------------- model comparison
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.2))
    for j, (title, sig) in enumerate(cols):
        im = mesh.plot_slice(sig, normal="y", ax=ax[0, j],
                             pcolor_opts={"norm": NORM, "cmap": CMAP})[0]
        ax[0, j].plot(rx_locs_80[on_line, 0], rx_locs_80[on_line, 2], **RX)
        ax[0, j].set_xlim(XLIM); ax[0, j].set_ylim(ZLIM); ax[0, j].set_aspect(1)
        ax[0, j].set_title(f"{title}\nxz at y=0", fontsize=10)
        ax[0, j].set_xlabel("x (m)")
        mesh.plot_slice(sig, normal="z", ind=IZ, ax=ax[1, j],
                        pcolor_opts={"norm": NORM, "cmap": CMAP})
        ax[1, j].plot(rx_locs_80[:, 0], rx_locs_80[:, 1], **RX)
        ax[1, j].set_xlim([-500, 500]); ax[1, j].set_ylim([-400, 400])
        ax[1, j].set_aspect(1)
        ax[1, j].set_title(f"xy at z = {Z_SLICE:.0f} m", fontsize=10)
        ax[1, j].set_xlabel("x (m)")
        if j:
            ax[0, j].set_ylabel(""); ax[1, j].set_ylabel("")
    ax[0, 0].set_ylabel("z (m)"); ax[1, 0].set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="σ (S/m)")
    fig.savefig("fig_coldstart_senswt.png", dpi=200, bbox_inches="tight")
    print("wrote fig_coldstart_senswt.png")

    # ---------------------------------------------------- diagnostics
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 4.2),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    wr_full = np.full(mesh.n_cells, np.nan); wr_full[active] = wr
    im2 = mesh.plot_slice(wr_full, normal="y", ax=ax2[0],
                          pcolor_opts={"norm": LogNorm(3e-3, 1.0),
                                       "cmap": "viridis"})[0]
    ax2[0].plot(rx_locs_80[on_line, 0], rx_locs_80[on_line, 2], **RX)
    ax2[0].set_xlim(XLIM); ax2[0].set_ylim(ZLIM); ax2[0].set_aspect(1)
    ax2[0].set_xlabel("x (m)"); ax2[0].set_ylabel("z (m)")
    ax2[0].set_title("sensitivity weights $w_r$ (y=0)", fontsize=10)
    plt.colorbar(im2, ax=ax2[0], shrink=0.85, label="$w_r$")

    for recs, lab, c in ((it_cold, "cold-start (no weighting)", "C3"),
                         (it_sw, "cold-start + sensitivity weighting", "C0")):
        it = [o["iter"] for o in recs]; pd = [o["phi_d"] for o in recs]
        ax2[1].semilogy(it, pd, "-o", ms=3.5, color=c, label=lab)
    ax2[1].axhline(1000, color="k", ls="--", lw=1.2, label="target $\\phi_d$ = 1000")
    ax2[1].set_xlabel("iteration"); ax2[1].set_ylabel("$\\phi_d$")
    ax2[1].set_title("convergence", fontsize=10)
    ax2[1].legend(fontsize=8, frameon=False); ax2[1].grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig("fig_senswt_diagnostics.png", dpi=200, bbox_inches="tight")
    print("wrote fig_senswt_diagnostics.png")


if __name__ == "__main__":
    main()
