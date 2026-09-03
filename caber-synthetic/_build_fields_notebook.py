"""Build fields_3_soundings.ipynb -- runs a 3-sounding TDEM forward on the
GLOBAL OcTree mesh, caches the fields on /t40array, and makes cross-section
figures + a movie of the currents and dB/dt.
Run: python _build_fields_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))


md("""# Currents and dB/dt for three soundings — global-mesh forward

Runs a single `Simulation3DElectricField` on the **global OcTree mesh** (the same
mesh `_test_parametric.build_global_mesh()` builds for the paper's forward and
inversions) with **three transmitters**, at `x = -200, 0, 200 m`, all on the
`y = 0` line at the 30 m flight height. The true model is the dipping conductor
under the U-shaped conductive overburden.

Because all three loops sit on `y = 0`, the `y`-normal cross-section through them
is the natural section: the loop's induced currents are **azimuthal about the
vertical axis through each transmitter**, so in this plane the current is purely
**out-of-plane** (`j_y`) and can be shown as a signed image, while **dB/dt lies
in the plane** and is shown as a vector field (streamlines over amplitude).

The fields are expensive (~3 min) so they are computed once and cached to
`/t40array/lheagy/2026-heagy-etal-aem/fields-3-soundings/`. Re-running the
notebook reloads the cache.

**Plotting controls — including the times — are all in the "Plot controls" cell
below.**""")

# ------------------------------------------------------------------ setup
md("## Setup")

code('''%matplotlib inline
import os, json, time

# threads for the Pardiso factorizations (this is a single, un-tiled simulation,
# so we want the solver multi-threaded -- unlike the tiled inversion scripts)
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import discretize
from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver

FIELDS_DIR = "/t40array/lheagy/2026-heagy-etal-aem/fields-3-soundings"
MOVIE_DIR = "."          # where the .mp4 / .gif land
os.makedirs(FIELDS_DIR, exist_ok=True)

# The conda ffmpeg in this env is broken (it links libx264.so.138, the env ships
# .164), so point matplotlib at the self-contained binary that ships with
# imageio-ffmpeg. `pip install imageio-ffmpeg` if this is missing.
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    HAVE_FFMPEG = True
except Exception as _err:
    HAVE_FFMPEG = False
    print(f"no usable ffmpeg ({_err}) -- GIF only. pip install imageio-ffmpeg")

plt.rcParams["figure.dpi"] = 110
SIGMA_CMAP = plt.get_cmap("Spectral_r").copy(); SIGMA_CMAP.set_bad("white")
print("fields cache:", FIELDS_DIR)''')

# --------------------------------------------------------------- geometry
md("""## Geometry, model, and the global mesh

`build_global_mesh()` refines at **all 130 stations of the full survey** (not just
our three) — this is deliberate: it is the exact mesh the paper's forward
modelling and 3D inversions use, so these field pictures are the physics those
runs actually solve.""")

code('''from _test_parametric import (
    build_global_mesh, build_true_model, rx_times, TIME_STEPS,
    sigma_back, sigma_target, sigma_overburden,
)

# ---- the three sounding locations (change here) ----------------------------
SRC_X = np.r_[-200.0, 0.0, 200.0]
SRC_Y = 0.0
TX_HEIGHT = 30.0

src_locs = np.c_[SRC_X, np.full(SRC_X.size, SRC_Y), np.full(SRC_X.size, TX_HEIGHT)]
n_src = src_locs.shape[0]
SRC_LABELS = [f"x = {x:+.0f} m" for x in SRC_X]

mesh = build_global_mesh()
active = mesh.cell_centers[:, 2] < 0
actmap = maps.InjectActiveCells(mesh, active, value_inactive=np.log(1e-8))
sigma_true = build_true_model(mesh)
m_true = np.log(sigma_true[active])

print(f"global mesh: {mesh.n_cells} cells ({int(active.sum())} active)")
print(f"sigma: background {sigma_back:.1e}, overburden {sigma_overburden:.2e}, "
      f"target {sigma_target:.1f} S/m")
for lab, loc in zip(SRC_LABELS, src_locs):
    print(f"  source {lab}: {loc}")''')

code('''source_list = [
    tdem.sources.CircularLoop(
        receiver_list=[tdem.receivers.PointMagneticFluxTimeDerivative(
            loc, rx_times, orientation="z")],
        location=loc, orientation="z", radius=10,
        waveform=tdem.sources.StepOffWaveform(),
    )
    for loc in src_locs
]
survey = tdem.Survey(source_list)

sim = tdem.Simulation3DElectricField(
    mesh=mesh,
    survey=survey,
    time_steps=TIME_STEPS,          # [(3e-6,20),(1e-5,20),(3e-5,20),(1e-4,20)]
    solver=get_default_solver(),
    sigmaMap=maps.ExpMap() * actmap,
)
times = sim.times                    # 81 values, 0 -> 2.86e-3 s
n_t = len(times)
print(f"{n_t} stored times, 0 -> {times[-1]:.3e} s "
      f"({len(TIME_STEPS)} unique step sizes -> {len(TIME_STEPS)} factorizations)")''')

# ------------------------------------------------------------------ fields
md("""## Run the forward and cache the fields on `/t40array`

`sim.fields()` solves all three sources simultaneously at each time step (the
factorization is shared), so three sources cost little more than one.

What gets cached:

| file | shape | what |
|---|---|---|
| `jy_cc.npy` | `(n_src, n_cells, n_t)` | out-of-plane current density `j_y`, cell-centred |
| `dbdt_cc.npy` | `(n_src, 3*n_cells, n_t)` | `dB/dt` averaged to cell centres, `[vx; vy; vz]` |
| `dpred.npy` | `(n_src, 20)` | the dB/dt decay at the 20 survey channels |
| `esolution.npy` | `(n_edges, n_src, n_t)` | the raw solved E-field (lets you recompute anything later) |

Set `RECOMPUTE = True` to force a re-run.""")

code('''RECOMPUTE = False
SAVE_ESOLUTION = True     # ~290 MB; the raw primitive, keep it unless space is tight

P = {k: os.path.join(FIELDS_DIR, f"{k}.npy")
     for k in ("jy_cc", "dbdt_cc", "dpred", "esolution", "times", "src_locs", "sigma_true")}
META = os.path.join(FIELDS_DIR, "meta.json")
_need = ("jy_cc", "dbdt_cc", "dpred", "times")

if RECOMPUTE or not all(os.path.exists(P[k]) for k in _need):
    t0 = time.time()
    f = sim.fields(m_true)
    print(f"sim.fields: {time.time() - t0:.1f} s", flush=True)

    ey = slice(mesh.n_edges_x, mesh.n_edges_x + mesh.n_edges_y)   # the y-edges
    Ay = mesh.average_edge_y_to_cell
    Af = mesh.average_face_to_cell_vector

    j_all = f[:, "j", :]                                          # (n_edges, n_src, n_t)
    jy_cc = np.stack([Ay @ j_all[ey, i, :] for i in range(n_src)])
    del j_all

    db_all = f[:, "dbdt", :]                                      # (n_faces, n_src, n_t)
    dbdt_cc = np.stack([Af @ db_all[:, i, :] for i in range(n_src)])
    del db_all

    dpred = sim.dpred(m_true, f=f).reshape(n_src, len(rx_times))

    np.save(P["jy_cc"], jy_cc)
    np.save(P["dbdt_cc"], dbdt_cc)
    np.save(P["dpred"], dpred)
    np.save(P["times"], times)
    np.save(P["src_locs"], src_locs)
    np.save(P["sigma_true"], sigma_true)
    if SAVE_ESOLUTION:
        np.save(P["esolution"], f[:, "eSolution", :])
    with open(META, "w") as fh:
        json.dump({"n_cells": int(mesh.n_cells), "n_edges": int(mesh.n_edges),
                   "n_src": int(n_src), "n_t": int(n_t),
                   "src_locs": src_locs.tolist(),
                   "time_steps": [[float(dt), int(n)] for dt, n in TIME_STEPS]}, fh, indent=2)
    del f
    print("cached ->", FIELDS_DIR)
else:
    meta = json.load(open(META))
    assert meta["n_cells"] == mesh.n_cells, "cache was built on a different mesh"
    assert np.allclose(meta["src_locs"], src_locs), "cache was built with different sources"
    jy_cc = np.load(P["jy_cc"])
    dbdt_cc = np.load(P["dbdt_cc"])
    dpred = np.load(P["dpred"])
    times = np.load(P["times"]); n_t = len(times)
    print("loaded cache from", FIELDS_DIR)

print(f"jy_cc {jy_cc.shape}   dbdt_cc {dbdt_cc.shape}   dpred {dpred.shape}")''')

# ---------------------------------------------------------- plot controls
md("""## Plot controls

**This is the cell to edit.** `PLOT_TIMES_MS` picks the times (in
**milliseconds**) shown in the static figures; `ti_at()` snaps any time to the
nearest stored time step. The stepping is coarse late in time, so ask for a time
and check what you actually got (printed below).""")

code('''# ---- times to plot (milliseconds) ------------------------------------------
PLOT_TIMES_MS = [0.02, 0.06, 0.15, 0.4, 1.0, 2.0]

# ---- cross-section geometry -------------------------------------------------
SLICE_Y = 10.0             # y of the section (cell centres straddle y=0, so use +/-10)
XLIM = np.r_[-650.0, 650.0]
ZLIM = np.r_[-450.0, 80.0]
SECTION_DX = 10.0          # octree is resampled onto this regular grid for plotting

# ---- colour / scaling -------------------------------------------------------
NORMALIZE = True           # scale each panel by its own peak (recommended: the
                           # fields decay ~7 decades over the time range, so
                           # absolute scaling makes late times invisible)
J_CMAP = "RdBu_r"          # signed, for out-of-plane current
J_LINTHRESH = 1e-3         # symlog linear region, as a fraction of the peak
DBDT_CMAP = "viridis"
DBDT_DECADES = 4           # colour range below the peak

# ---- the sounding used for the time-strip figure and the movie --------------
FOCUS_SRC = 1              # 0 -> x=-200, 1 -> x=0, 2 -> x=+200


def ti_at(t_ms):
    """Index of the stored time step nearest `t_ms` milliseconds."""
    return int(np.argmin(np.abs(times - t_ms * 1e-3)))


PLOT_TI = [ti_at(t) for t in PLOT_TIMES_MS]
for t_req, ti in zip(PLOT_TIMES_MS, PLOT_TI):
    print(f"  requested {t_req:6.3f} ms  ->  index {ti:3d}, actual {times[ti]*1e3:6.4f} ms")''')

# ------------------------------------------------------------ section mesh
md("""## Section mesh

The octree fields are sampled onto a regular 2D grid on the `y = SLICE_Y` plane
(nearest containing cell — no interpolation, so the values are the true cell
values). This gives a consistent grid for the image, the streamlines and the
model contours, and makes the animation fast (each frame is fancy-indexing plus
one `pcolormesh`).""")

code('''nx = int(np.diff(XLIM)[0] // SECTION_DX)
nz = int(np.diff(ZLIM)[0] // SECTION_DX)
mesh2d = discretize.TensorMesh(
    [np.full(nx, SECTION_DX), np.full(nz, SECTION_DX)], origin=[XLIM[0], ZLIM[0]]
)
pts3d = np.c_[mesh2d.cell_centers[:, 0],
              np.full(mesh2d.n_cells, SLICE_Y),
              mesh2d.cell_centers[:, 1]]
IND2D = mesh.get_containing_cells(pts3d)     # octree cell -> section pixel

SIG2D = sigma_true[IND2D]
AIR2D = pts3d[:, 2] > 0
CX, CZ = mesh2d.cell_centers_x, mesh2d.cell_centers_y
SIG_GRID = SIG2D.reshape(nz, nx)

# geometric mid-points between the three conductivities -> model outlines
CONTOUR_LEVELS = np.log10([
    np.sqrt(sigma_back * sigma_overburden),
    np.sqrt(sigma_overburden * sigma_target),
])
print(f"section: {nx} x {nz} pixels at {SECTION_DX:.0f} m, y = {SLICE_Y:.0f} m")''')

# ------------------------------------------------------------------ helpers
md("## Plotting helpers")

code('''def _frame(ax, title=""):
    ax.set_aspect(1)
    ax.set_xlim(XLIM); ax.set_ylim(ZLIM)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title(title, fontsize=10)


def outline_model(ax, color="k", lw=0.9):
    """Outline the target and overburden on top of a field panel."""
    ax.contour(CX, CZ, np.log10(SIG_GRID), levels=CONTOUR_LEVELS,
               colors=color, linewidths=lw)
    ax.axhline(0.0, color=color, lw=lw, ls="-")


def mark_sources(ax, focus=None, color="k"):
    ax.plot(src_locs[:, 0], src_locs[:, 2], marker="v", ls="none",
            ms=7, mfc=color, mec="w", mew=0.7, zorder=5, clip_on=False)
    if focus is not None:
        ax.plot(src_locs[focus, 0], src_locs[focus, 2], marker="v", ls="none",
                ms=10, mfc="C1", mec="k", mew=0.8, zorder=6, clip_on=False)


def plot_model(ax=None, colorbar=True, cax=None, sources=True, focus=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 3.2))
    s = SIG2D.copy(); s[AIR2D] = np.nan
    out = mesh2d.plot_image(
        s, ax=ax, pcolor_opts={"norm": LogNorm(1e-3, 10), "cmap": SIGMA_CMAP}
    )
    _frame(ax, "conductivity")
    if sources:
        mark_sources(ax, focus=focus)
    if colorbar:
        cb = plt.colorbar(out[0], ax=None if cax is not None else ax, cax=cax)
        cb.set_label("$\\\\sigma$ (S/m)")
    return ax


def plot_currents(src_ind, ti, ax=None, colorbar=True, cax=None,
                  normalize=None, vmax=None, annotate=True, outline=True):
    """Out-of-plane current density j_y on the y = SLICE_Y section."""
    normalize = NORMALIZE if normalize is None else normalize
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 3.2))

    v = jy_cc[src_ind, IND2D, ti]
    peak = float(np.abs(v).max())
    if normalize:
        v = v / max(peak, 1e-300)
        vm = 1.0
    else:
        vm = peak if vmax is None else vmax
    norm = SymLogNorm(linthresh=J_LINTHRESH * vm, vmin=-vm, vmax=vm, base=10)

    out = mesh2d.plot_image(v, ax=ax, pcolor_opts={"norm": norm, "cmap": J_CMAP})
    if outline:
        outline_model(ax)
    mark_sources(ax, focus=src_ind)
    _frame(ax, f"currents  $j_y$  —  t = {times[ti]*1e3:.3f} ms")
    if annotate:
        ax.text(0.015, 0.06, f"peak {peak:.2e} A/m$^2$", transform=ax.transAxes,
                fontsize=8, va="bottom", ha="left",
                bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.5))
    if colorbar:
        cb = plt.colorbar(out[0], ax=None if cax is not None else ax, cax=cax)
        cb.set_label("$j_y$ / peak" if normalize else "$j_y$ (A/m$^2$)")
    return ax


def plot_dbdt(src_ind, ti, ax=None, colorbar=True, cax=None,
              normalize=None, vmax=None, annotate=True, outline=True,
              density=1.1):
    """-dB/dt as an in-plane vector field (streamlines over amplitude)."""
    normalize = NORMALIZE if normalize is None else normalize
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 3.2))

    d = -dbdt_cc[src_ind, :, ti].reshape(mesh.n_cells, 3, order="F")[IND2D]
    peak = float(np.linalg.norm(d[:, [0, 2]], axis=1).max())
    if normalize:
        d = d / max(peak, 1e-300)
        vhi = 1.0
    else:
        vhi = peak if vmax is None else vmax
    vlo = vhi * 10.0 ** (-DBDT_DECADES)

    out = mesh2d.plot_image(
        np.r_[d[:, 0], d[:, 2]], "CCv", view="vec", ax=ax,
        pcolor_opts={"norm": LogNorm(vmin=vlo, vmax=vhi), "cmap": DBDT_CMAP},
        stream_threshold=vlo,
        stream_opts={"color": "w", "density": density, "linewidth": 0.7,
                     "arrowsize": 0.9},
    )
    if outline:
        outline_model(ax, color="k", lw=0.9)
    mark_sources(ax, focus=src_ind, color="w")
    _frame(ax, f"$-\\\\partial \\\\mathbf{{B}}/\\\\partial t$  —  t = {times[ti]*1e3:.3f} ms")
    if annotate:
        ax.text(0.015, 0.06, f"peak {peak:.2e} T/s", transform=ax.transAxes,
                fontsize=8, va="bottom", ha="left", color="k",
                bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.5))
    if colorbar:
        cb = plt.colorbar(out[0], ax=None if cax is not None else ax, cax=cax)
        cb.set_label("$|-d\\\\mathbf{B}/dt|$ / peak" if normalize
                     else "$|-d\\\\mathbf{B}/dt|$ (T/s)")
    return ax


def plot_decay(ax=None, ti=None, focus=None, sources=None, color=None):
    """The 20-channel dB/dt decays.

    `sources` limits which soundings are drawn (default: all three).
    `color` forces one colour for every curve -- pass "k" to show a single
    sounding's observed data in black.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 3.2))
    inds = range(n_src) if sources is None else list(sources)
    for i in inds:
        is_focus = focus is not None and i == focus
        if color is not None:
            c, lw, alpha = color, 2.0, 1.0
        elif is_focus:
            # this movie's own sounding: black and heavy, the others for context
            c, lw, alpha = "k", 2.2, 1.0
        else:
            c, lw, alpha = f"C{i}", 1.2, 0.75
        ax.loglog(rx_times * 1e3, -dpred[i], "-o", ms=4, lw=lw, alpha=alpha,
                  color=c, zorder=3 if is_focus else 2,
                  label=("observed" if color is not None else SRC_LABELS[i]))
    if ti is not None:
        ax.axvline(times[ti] * 1e3, color="0.3", lw=1.2, ls="--")
    ax.set_xlabel("time (ms)"); ax.set_ylabel(r"$-\\partial B_z/\\partial t$ (T/s)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3, which="both")
    return ax''')

# ------------------------------------------------------------------ Fig 1
md("""## Figure — the model and where the soundings sit""")

code('''fig, ax = plt.subplots(1, 2, figsize=(13, 3.4), width_ratios=[2.3, 1])
plot_model(ax=ax[0])
plot_decay(ax=ax[1])
ax[1].set_title("observed decays", fontsize=10)
fig.tight_layout()
fig.savefig("fig_fields_model.png", dpi=200, bbox_inches="tight")''')

# ------------------------------------------------------------------ Fig 2
md("""## Figure — currents and dB/dt at one time, for each sounding

Rows are the three transmitters; left column is the out-of-plane current, right
column is `-dB/dt` as an in-plane vector field. Change `SNAPSHOT_MS`.

The three panels share **one absolute colour scale** (not per-panel
normalization) — comparing soundings is the whole point here, and the amplitude
difference between the sounding over the target and the one over clean halfspace
is part of the story.""")

code('''SNAPSHOT_MS = 0.6
ti = ti_at(SNAPSHOT_MS)

# one absolute scale shared by all three soundings
J_SHARED = float(np.abs(jy_cc[:, IND2D, ti]).max())
B_SHARED = float(max(
    np.linalg.norm(dbdt_cc[i, :, ti].reshape(mesh.n_cells, 3, order="F")[IND2D][:, [0, 2]],
                   axis=1).max()
    for i in range(n_src)
))

fig, ax = plt.subplots(n_src, 2, figsize=(14, 3.0 * n_src), sharex=True, sharey=True)
for i in range(n_src):
    plot_currents(i, ti, ax=ax[i, 0], normalize=False, vmax=J_SHARED, colorbar=False)
    plot_dbdt(i, ti, ax=ax[i, 1], normalize=False, vmax=B_SHARED, colorbar=False)
    ax[i, 0].set_ylabel(f"{SRC_LABELS[i]}\\nz (m)")
    ax[i, 1].set_ylabel("")
for a in ax[:-1, :].flatten():
    a.set_xlabel("")

# one colorbar per column, spanning the rows
fig.colorbar(
    ScalarMappable(cmap=J_CMAP, norm=SymLogNorm(
        linthresh=J_LINTHRESH * J_SHARED, vmin=-J_SHARED, vmax=J_SHARED, base=10)),
    ax=ax[:, 0], shrink=0.75, pad=0.015,
).set_label("$j_y$ (A/m$^2$)")
fig.colorbar(
    ScalarMappable(cmap=DBDT_CMAP, norm=LogNorm(
        vmin=B_SHARED * 10.0 ** (-DBDT_DECADES), vmax=B_SHARED)),
    ax=ax[:, 1], shrink=0.75, pad=0.015,
).set_label("$|-d\\\\mathbf{B}/dt|$ (T/s)")

fig.suptitle(f"t = {times[ti]*1e3:.3f} ms   (shared colour scale)", y=0.995)
fig.savefig(f"fig_fields_per_sounding_{times[ti]*1e3:.3f}ms.png", dpi=200,
            bbox_inches="tight")''')

# ------------------------------------------------------------------ Fig 3
md("""## Figure — time evolution for one sounding

The smoke ring leaving the transmitter, being held up by the conductive
overburden, and then hanging in the dipping target. Set `FOCUS_SRC` and
`PLOT_TIMES_MS` in the controls cell.""")

code('''n_col = len(PLOT_TI)
# panels are aspect-1, so size the figure from the section's own aspect ratio
PANEL_W = 3.4
PANEL_H = PANEL_W * np.diff(ZLIM)[0] / np.diff(XLIM)[0]
fig, ax = plt.subplots(2, n_col, figsize=(PANEL_W * n_col, 2 * PANEL_H + 1.4),
                       sharex=True, sharey=True)
for k, ti in enumerate(PLOT_TI):
    plot_currents(FOCUS_SRC, ti, ax=ax[0, k], colorbar=False, annotate=False)
    plot_dbdt(FOCUS_SRC, ti, ax=ax[1, k], colorbar=False, annotate=False, density=0.8)
    ax[0, k].set_title(f"{times[ti]*1e3:.3f} ms", fontsize=10)
    ax[1, k].set_title("")
    if k > 0:
        ax[0, k].set_ylabel(""); ax[1, k].set_ylabel("")
    ax[0, k].set_xlabel("")
ax[0, 0].set_ylabel("$j_y$\\nz (m)")
ax[1, 0].set_ylabel(r"$-\\partial \\mathbf{B}/\\partial t$" + "\\nz (m)")

fig.colorbar(
    ScalarMappable(cmap=J_CMAP,
                   norm=SymLogNorm(linthresh=J_LINTHRESH, vmin=-1, vmax=1, base=10)),
    ax=ax[0, :], shrink=0.85, pad=0.01,
).set_label("$j_y$ / panel peak")
fig.colorbar(
    ScalarMappable(cmap=DBDT_CMAP,
                   norm=LogNorm(vmin=10.0 ** (-DBDT_DECADES), vmax=1.0)),
    ax=ax[1, :], shrink=0.85, pad=0.01,
).set_label("$|-d\\\\mathbf{B}/dt|$ / panel peak")

fig.suptitle(f"sounding {SRC_LABELS[FOCUS_SRC]}  (colour normalized per panel)",
             y=0.99)
fig.savefig(f"fig_fields_time_strip_src{FOCUS_SRC}.png", dpi=200, bbox_inches="tight")''')

# ------------------------------------------------------------------ movie
md("""## Movie

Same 2×2 layout as the
[ground-and-borehole talk](https://www.appliedgeophysics.org/articles/2025-heagy-roundup/ground-and-borehole-talk)
notebook: model, decay curves with a moving time marker, currents, and dB/dt.

Colour scales are **fixed** across the movie (each frame normalized by its own
peak, which is printed on the frame), so the colorbars are built once from
standalone `ScalarMappable`s and stay valid as the axes are cleared and redrawn.

Each movie is written **twice** — `.mp4` for slides (far smaller, seeks and
loops properly in Keynote/PowerPoint) and `.gif` for inline display anywhere.

> **ffmpeg note.** The `ffmpeg` in the `py311` conda env is broken: it links
> `libx264.so.138` but the env ships `libx264.so.164`, so `ani.save(...)` fails
> with a `CalledProcessError`. Worse,
> `animation.writers.is_available("ffmpeg")` still returns `True`, so it cannot
> be detected up front. The setup cell sidesteps this by pointing
> `rcParams["animation.ffmpeg_path"]` at the self-contained binary shipped with
> **imageio-ffmpeg** (`pip install imageio-ffmpeg`), which is what actually
> writes the mp4s here.""")

code('''MOVIE_SRC = FOCUS_SRC
MOVIE_MAX_MS = 2.0        # last frame
MOVIE_FPS = 8
MOVIE_DPI = 90            # 14x7 in -> 1260x630 px; bump for a bigger render
MOVIE_STRIDE = 1          # 2 -> every other time step (half the frames)

frame_ti = [i for i in range(1, n_t) if times[i] <= MOVIE_MAX_MS * 1e-3][::MOVIE_STRIDE]
print(f"{len(frame_ti)} frames, {times[frame_ti[0]]*1e3:.4f} -> "
      f"{times[frame_ti[-1]]*1e3:.4f} ms")


def save_animation(ani, basename, fps=MOVIE_FPS, dpi=MOVIE_DPI, title=""):
    """Write both an .mp4 and a .gif. Returns the gif path (for inline display).

    mp4 is the one to put in a slide deck: ~30x smaller than the gif at the
    same size, and it seeks/loops properly in Keynote/PowerPoint. The gif is
    kept because it renders inline anywhere with no codec.
    """
    written = []
    mp4 = os.path.join(MOVIE_DIR, basename + ".mp4")
    if HAVE_FFMPEG:
        try:
            ani.save(mp4, writer="ffmpeg", fps=fps, dpi=dpi,
                     metadata={"title": title, "artist": "Lindsey Heagy"})
            print(f"  wrote {mp4} ({os.path.getsize(mp4)/1e6:.1f} MB)")
            written.append(mp4)
        except Exception as err:
            print(f"  mp4 failed ({type(err).__name__}: {err})")
    gif = os.path.join(MOVIE_DIR, basename + ".gif")
    ani.save(gif, writer="pillow", fps=fps, dpi=dpi)
    print(f"  wrote {gif} ({os.path.getsize(gif)/1e6:.1f} MB)")
    written.append(gif)
    return gif''')

code('''# Decay panel: True -> only the sounding this movie is about, in black.
# False -> all three soundings, with this movie's own one black and heavy and
# the other two in colour behind it.
DECAY_SINGLE_SOURCE = False


def make_movie_figure(src_ind, decay_single=None):
    """Build the 2x2 frame and return (fig, draw_frame). The colour scales are
    fixed, so the colorbars are made once from standalone ScalarMappables and
    survive the per-frame `ax.clear()`."""
    decay_single = DECAY_SINGLE_SOURCE if decay_single is None else decay_single
    # both cross-sections go in the (wide) left column so they render at the
    # same size; the model thumbnail and the decay curves share the right column
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0],
                          height_ratios=[1, 1], hspace=0.28, wspace=0.42)
    ax_j = fig.add_subplot(gs[0, 0])
    ax_model = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_decay = fig.add_subplot(gs[1, 1])

    cax_model = make_axes_locatable(ax_model).append_axes("right", size="3%", pad=0.08)
    cax_j = make_axes_locatable(ax_j).append_axes("right", size="2%", pad=0.08)
    cax_b = make_axes_locatable(ax_b).append_axes("right", size="2%", pad=0.08)

    norm_model = LogNorm(1e-3, 10)
    norm_j = SymLogNorm(linthresh=J_LINTHRESH, vmin=-1, vmax=1, base=10)
    norm_b = LogNorm(vmin=10.0 ** (-DBDT_DECADES), vmax=1.0)

    fig.colorbar(ScalarMappable(norm=norm_model, cmap=SIGMA_CMAP),
                 cax=cax_model).set_label("$\\\\sigma$ (S/m)")
    fig.colorbar(ScalarMappable(norm=norm_j, cmap=J_CMAP),
                 cax=cax_j).set_label("$j_y$ / peak")
    fig.colorbar(ScalarMappable(norm=norm_b, cmap=DBDT_CMAP),
                 cax=cax_b).set_label("$|-d\\\\mathbf{B}/dt|$ / peak")

    def draw_frame(ti):
        for a in (ax_model, ax_decay, ax_j, ax_b):
            a.clear()
        plot_currents(src_ind, ti, ax=ax_j, colorbar=False, normalize=True)
        plot_dbdt(src_ind, ti, ax=ax_b, colorbar=False, normalize=True, density=1.0)
        plot_model(ax=ax_model, colorbar=False, focus=src_ind)
        if decay_single:
            plot_decay(ax=ax_decay, ti=ti, focus=src_ind,
                       sources=[src_ind], color="k")
            ax_decay.set_title(f"observed data — {SRC_LABELS[src_ind]}",
                               fontsize=10)
        else:
            plot_decay(ax=ax_decay, ti=ti, focus=src_ind)
            ax_decay.set_title("observed decays", fontsize=10)
        ax_j.set_xlabel("")
        fig.suptitle(f"sounding {SRC_LABELS[src_ind]}    "
                     f"t = {times[ti]*1e3:7.4f} ms", y=0.965)
        return (ax_j, ax_model, ax_b, ax_decay)

    return fig, draw_frame


# preview a single frame before committing to the full render
fig, draw_frame = make_movie_figure(MOVIE_SRC)
draw_frame(frame_ti[len(frame_ti) // 2])
fig''')

code('''fig, draw_frame = make_movie_figure(MOVIE_SRC)
ani = animation.FuncAnimation(fig, draw_frame, frames=frame_ti, blit=False)
out = save_animation(
    ani, f"tdem_fields_src{MOVIE_SRC}",
    title=f"TDEM currents and dB/dt, sounding {SRC_LABELS[MOVIE_SRC]}",
)
plt.close(fig)''')

md("Play it inline (works for either mp4 or gif):")

code('''from IPython.display import Video, Image, display

display(Image(out) if out.endswith(".gif") else Video(out))''')

md("""### One movie per sounding

Same movie for all three transmitters — the point of the comparison is how
differently the smoke ring behaves at `x = -200` (clean halfspace), `x = 0`
(over the target) and `x = +200` (under the thick overburden).""")

code('''for s in range(n_src):
    fig_s, draw_s = make_movie_figure(s)
    ani_s = animation.FuncAnimation(fig_s, draw_s, frames=frame_ti, blit=False)
    save_animation(ani_s, f"tdem_fields_src{s}",
                   title=f"TDEM currents and dB/dt, sounding {SRC_LABELS[s]}")
    plt.close(fig_s)''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "py311", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
with open("fields_3_soundings.ipynb", "w") as fh:
    nbf.write(nb, fh)
print(f"wrote fields_3_soundings.ipynb ({len(cells)} cells)")
