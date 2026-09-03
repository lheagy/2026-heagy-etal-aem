"""Figure: the SimPEG mapping chain for the parametric (tiled) simulation.

This is the literal chain `sim_parametric` runs, in application order
(_test_parametric.py:267,273 -- `param_mappings = tile_map * global_ellipsoid`
and `sigmaMap = ExpMap * local_actmap`):

  m (11 params)
     --ParametricEllipsoid-->  (a) log-sigma on the GLOBAL ACTIVE cells
     --TileMap-------------->  (b) volume-averaged onto the LOCAL ACTIVE cells
     --InjectActiveCells---->  (c) air added, on the FULL local mesh

`ExpMap` is applied in every panel, since all three plot conductivity.

Run: python _plot_mappings.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem

from parametric_ellipsoid import ParametricEllipsoid
from _test_parametric import (
    build_global_mesh, build_local_meshes, rx_times,
)

AIR_SIGMA = 1e-8
XLIM = np.r_[-600.0, 600.0]
ZLIM = np.r_[-450.0, 80.0]

# 80 m survey stations on the y = 0 line; panel (c) shows the tile for the one
# at x = 100 m -- one station further from the ellipsoid centre than the tile in
# fig1_meshes, so more of the body falls in the tile's coarse cells.
RX_X = (np.linspace(-500, 500, 26))[3:-3][::2]
TX_HEIGHT = 30.0
SRC_LOC = np.r_[100.0, 0.0, TX_HEIGHT]

CMAP = plt.get_cmap("Spectral_r").copy()
CMAP.set_bad("white")                  # NaN (= no model value at all) -> white
CMAP.set_under("0.72")                 # air (1e-8, below vmin) -> grey
# same scale as fig1_meshes. Air sits five decades below vmin, so it lands on
# the "under" colour and the colorbar is drawn with extend="min".
NORM = LogNorm(vmin=1e-3, vmax=10.0)
GRID = dict(color="k", linewidth=0.3)

RX = dict(marker="o", linestyle="none", ms=4.5, mfc="k", mec="k")
RX_FOCUS = dict(marker="s", linestyle="none", ms=10, mfc="k", mec="k")


def main():
    # ---------------------------------------------------------------- setup
    mesh = build_global_mesh()
    active = mesh.cell_centers[:, 2] < 0
    n_active = int(active.sum())

    m = np.load("_mopt_test.npy")                     # 11 recovered parameters
    ellipsoid = ParametricEllipsoid(mesh, active_cells=active,
                                    boundary_sharpness=10.0)

    src = tdem.sources.CircularLoop(
        receiver_list=[tdem.receivers.PointMagneticFluxTimeDerivative(
            SRC_LOC, rx_times, orientation="z")],
        location=SRC_LOC, orientation="z", radius=10,
        waveform=tdem.sources.StepOffWaveform(),
    )
    local_mesh = build_local_meshes(mesh, tdem.Survey([src]))[0]
    tile = maps.TileMap(mesh, active, local_mesh)
    local_active = tile.local_active
    n_local_active = int(local_active.sum())
    local_actmap = maps.InjectActiveCells(
        local_mesh, active_cells=local_active,
        value_inactive=np.log(AIR_SIGMA))

    # (a) ParametricEllipsoid: 11 params -> log-sigma on the GLOBAL active cells
    m_active = ellipsoid * m
    sigma_a = np.full(mesh.n_cells, np.nan)           # air: no value yet
    sigma_a[active] = np.exp(m_active)

    # (b) TileMap: global active -> LOCAL active cells (volume averaged).
    #     Still no air: the vector only covers the local active cells.
    m_local = tile * m_active
    sigma_b = np.full(local_mesh.n_cells, np.nan)
    sigma_b[local_active] = np.exp(m_local)

    # (c) InjectActiveCells: local active -> the full local mesh, air = 1e-8
    sigma_c = np.asarray(maps.ExpMap() * local_actmap * m_local, dtype=float)

    print(f"m: {m.size} parameters")
    print(f"(a) ParametricEllipsoid -> {m_active.size} global active cells")
    print(f"(b) TileMap             -> {m_local.size} local active cells")
    print(f"(c) InjectActiveCells   -> {sigma_c.size} local mesh cells")
    print(f"    sigma: background {np.exp(m[0]):.2e}, "
          f"interior {np.exp(m[-1]):.2f} S/m")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 3.5))

    panels = [
        (axes[0], mesh, sigma_a,
         r"(a)  $\mathbf{ParametricEllipsoid}\;\cdot\;\mathbf{m}$",
         f"11 parameters  $\\rightarrow$  {n_active:,} global active cells"),
        (axes[1], local_mesh, sigma_b,
         r"(b)  $\mathbf{TileMap}\;\cdot\;(\cdots)$",
         f"{n_active:,}  $\\rightarrow$  {n_local_active:,} local active cells"),
        (axes[2], local_mesh, sigma_c,
         r"(c)  $\mathbf{InjectActiveCells}\;\cdot\;(\cdots)$",
         f"{n_local_active:,}  $\\rightarrow$  {local_mesh.n_cells:,} "
         f"local mesh cells   (air added)"),
    ]

    others = RX_X[RX_X != SRC_LOC[0]]
    for ax, msh, sig, title, sub in panels:
        msh.plot_slice(sig, normal="y", ax=ax, grid=True, grid_opts=GRID,
                       pcolor_opts={"norm": NORM, "cmap": CMAP})
        ax.plot(others, np.full(others.size, TX_HEIGHT), zorder=5,
                clip_on=False, **RX)
        ax.plot(SRC_LOC[0], SRC_LOC[2], zorder=6, clip_on=False, **RX_FOCUS)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlim(XLIM); ax.set_ylim(ZLIM); ax.set_aspect(1)
        ax.set_xlabel("x (m)")
        ax.set_title(f"{title}\n{sub}", fontsize=10.5, pad=8)
    axes[0].set_ylabel("z (m)")
    for a in axes[1:]:
        a.set_ylabel("")

    fig.subplots_adjust(left=0.042, right=0.895, top=0.78, bottom=0.15,
                        wspace=0.20)

    cax = fig.add_axes([0.912, 0.17, 0.010, 0.56])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax,
                      extend="min", label="$\\sigma$ (S/m)")
    cb.ax.text(0.5, -0.085, "air", transform=cb.ax.transAxes, ha="center",
               va="top", fontsize=8.5, color="0.45")

    fig.suptitle(
        "SimPEG mappings, parametric simulation:   $\\mathbf{m}$  "
        "$\\rightarrow$  ParametricEllipsoid  $\\rightarrow$  TileMap  "
        "$\\rightarrow$  InjectActiveCells  $\\rightarrow$  ExpMap  "
        "$\\rightarrow$  $\\sigma$", fontsize=12, y=0.985)

    fig.savefig("fig_mappings.png", dpi=220)
    print("wrote fig_mappings.png")


if __name__ == "__main__":
    main()
