"""Export the global OcTree mesh + models to VTK for ParaView.

Writes (into this directory unless --out-dir is given):
  caber_global_mesh.vtu        the global TreeMesh with the true model and every
                               recovered model that lives on it, as cell arrays
  caber_local_mesh_center.vtu  local (tiled) mesh for a source over the target,
  caber_local_mesh_corner.vtu  and for the corner station of the 80 m survey --
                               each carrying the TileMap volume-averaged model,
                               i.e. what the tiled forward actually solves on
  caber_soundings.vtp          the 40 m survey stations as a point cloud
  caber_field_soundings.vtp    the 3 soundings used by fields_3_soundings.ipynb
  caber_local_tx.vtp           the two transmitters the local meshes belong to

Run: python _export_vtk.py [--out-dir DIR]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob

import numpy as np
import pyvista as pv
from simpeg import maps

from simpeg.electromagnetics import time_domain as tdem

from _test_parametric import (
    build_global_mesh, build_local_meshes, build_true_model, rx_times, rx_y,
)


AIR_LOG_SIGMA = np.log(1e-8)


def _last_iteration(folder):
    """Model vector from the final SaveOutputDictEveryIteration record."""
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not files:
        return None
    recs = [np.load(f, allow_pickle=True)["arr_0"].item() for f in files]
    recs.sort(key=lambda o: o["iter"])
    return np.asarray(recs[-1]["m"], dtype=float)


def collect_models(mesh, active):
    """{name: sigma on the full mesh}. Air cells are left at their real value
    here; they are blanked to NaN when the arrays are attached."""
    n_active = int(active.sum())
    actmap = maps.InjectActiveCells(mesh, active, value_inactive=AIR_LOG_SIGMA)
    to_sigma = maps.ExpMap() * actmap          # active log-sigma -> full-mesh sigma

    models = {"true": build_true_model(mesh)}

    # voxel models stored as log-conductivity on active cells
    # (_iters_ovbU_5line's last iteration is identical to _mrec_ovbU5.npy,
    # so it is not exported separately)
    candidates = {
        "two_stage": ("_mrec_ovbU5.npy", np.load),
        "cold_start": ("_iters_coldstart_80m_superseded", _last_iteration),
    }
    for name, (path, loader) in candidates.items():
        if not os.path.exists(path):
            print(f"  skip {name}: {path} not found")
            continue
        m = loader(path)
        if m is None or m.shape != (n_active,):
            print(f"  skip {name}: shape {None if m is None else m.shape} "
                  f"!= ({n_active},)")
            continue
        models[name] = np.asarray(to_sigma * m, dtype=float)

    # stitched 1D, already interpolated onto the full mesh as conductivity
    if os.path.exists("_sigma_1d_on_3d.npy"):
        s = np.load("_sigma_1d_on_3d.npy")
        if s.shape == (mesh.n_cells,):
            models["stitched_1d"] = np.asarray(s, dtype=float)
        else:
            print(f"  skip stitched_1d: shape {s.shape} != ({mesh.n_cells},)")

    return models


def _source_at(loc):
    rx = tdem.receivers.PointMagneticFluxTimeDerivative(loc, rx_times, orientation="z")
    return tdem.sources.CircularLoop(
        receiver_list=[rx], location=loc, orientation="z", radius=10,
        waveform=tdem.sources.StepOffWaveform(),
    )


def export_local_mesh(global_mesh, active, sigma_true, loc, label, out_dir):
    """Write one per-source local (tiled) mesh carrying the volume-averaged model.

    `build_local_meshes` keys the refinement off `src.location` alone, so the
    mesh built here is identical to the one that source gets in the full tiled
    run. The model is pushed through `TileMap`, so what lands in the file is the
    volume-averaged model the tiled forward actually sees -- not the global model
    sampled onto a finer grid.
    """
    local_mesh = build_local_meshes(global_mesh, tdem.Survey([_source_at(loc)]))[0]
    tile = maps.TileMap(global_mesh, active, local_mesh)
    local_actmap = maps.InjectActiveCells(
        local_mesh, active_cells=tile.local_active, value_inactive=AIR_LOG_SIGMA,
    )
    m_local = tile * np.log(sigma_true[active])           # volume-averaged log-sigma
    sigma = np.asarray(maps.ExpMap() * local_actmap * m_local, dtype=float)

    local_air = local_mesh.cell_centers[:, 2] >= 0
    sigma[local_air] = np.nan

    cell_data = {
        "active": (~local_air).astype(np.uint8),
        "sigma_true_tiled": sigma,
        "log10_sigma_true_tiled": np.log10(sigma),
        "rho_true_tiled": 1.0 / sigma,
    }
    grid = pv.wrap(local_mesh.to_vtk(models=cell_data))
    path = os.path.join(out_dir, f"caber_local_mesh_{label}.vtu")
    grid.save(path)

    finest = int((local_mesh._cell_levels_by_indexes() == local_mesh.max_level).sum())
    print(f"\nwrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    print(f"  tx at ({loc[0]:.0f}, {loc[1]:.0f}, {loc[2]:.0f})")
    print(f"  {local_mesh.n_cells} cells ({int(tile.local_active.sum())} active), "
          f"{finest} at the finest level "
          f"-- {global_mesh.n_cells / local_mesh.n_cells:.1f}x smaller than global")
    print(f"  sigma range {np.nanmin(sigma):.4g} .. {np.nanmax(sigma):.4g} S/m")
    return local_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mesh = build_global_mesh()
    active = mesh.cell_centers[:, 2] < 0
    print(f"global mesh: {mesh.n_cells} cells ({int(active.sum())} active)")

    models = collect_models(mesh, active)

    # cell arrays: sigma, log10(sigma) and resistivity, with air blanked to NaN
    # so ParaView's colour range is set by the ground. `active` is there to
    # Threshold the air away geometrically.
    cell_data = {"active": active.astype(np.uint8)}
    for name, sigma in models.items():
        s = np.array(sigma, dtype=float)
        s[~active] = np.nan
        cell_data[f"sigma_{name}"] = s
        cell_data[f"log10_sigma_{name}"] = np.log10(s)
        cell_data[f"rho_{name}"] = 1.0 / s

    grid = pv.wrap(mesh.to_vtk(models=cell_data))
    mesh_path = os.path.join(args.out_dir, "caber_global_mesh.vtu")
    grid.save(mesh_path)          # XML VTU, binary + compressed

    print(f"\nwrote {mesh_path} ({os.path.getsize(mesh_path)/1e6:.1f} MB)")
    print(f"  {grid.n_cells} cells, {grid.n_points} points")
    print(f"  bounds x {grid.bounds[0]:.0f}..{grid.bounds[1]:.0f}  "
          f"y {grid.bounds[2]:.0f}..{grid.bounds[3]:.0f}  "
          f"z {grid.bounds[4]:.0f}..{grid.bounds[5]:.0f}")
    print("  cell arrays:")
    for k in grid.cell_data:
        a = np.asarray(grid.cell_data[k], dtype=float)
        print(f"    {k:28s} min {np.nanmin(a):11.4g}  max {np.nanmax(a):11.4g}")

    # ---- survey stations -----------------------------------------------------
    rx_x_40 = (np.linspace(-500, 500, 26))[3:-3]      # stitched-1D survey
    rx_x_80 = rx_x_40[::2]                            # 3D-inversion survey
    locs = np.array([[x, y, 30.0] for y in rx_y for x in rx_x_40])

    # ---- two local (tiled) meshes -------------------------------------------
    # corner: first station of the 80 m survey. centre: the station over the
    # target used for Figure 1 in dipping_target_figures.ipynb.
    loc_corner = np.r_[rx_x_80.min(), rx_y.min(), 30.0]
    loc_center = np.r_[20.0, 0.0, 30.0]
    for loc, label in ((loc_center, "center"), (loc_corner, "corner")):
        export_local_mesh(mesh, active, models["true"], loc, label, args.out_dir)

    tx = pv.PolyData(np.vstack([loc_center, loc_corner]))
    tx["is_center"] = np.array([1, 0], dtype=np.uint8)
    tx_path = os.path.join(args.out_dir, "caber_local_tx.vtp")
    tx.save(tx_path)
    print(f"\nwrote {tx_path} (2 transmitters: centre and corner)")

    pts = pv.PolyData(locs)
    pts["in_80m_survey"] = np.isin(locs[:, 0], rx_x_80).astype(np.uint8)
    pts["line_y"] = locs[:, 1]

    pts_path = os.path.join(args.out_dir, "caber_soundings.vtp")
    pts.save(pts_path)
    print(f"\nwrote {pts_path} ({os.path.getsize(pts_path)/1e3:.0f} kB)")
    print(f"  {pts.n_points} stations at 40 m spacing on {len(rx_y)} lines; "
          f"{int(pts['in_80m_survey'].sum())} are in the 80 m survey")

    # The three soundings used by fields_3_soundings.ipynb. These are NOT survey
    # stations -- the 40 m grid runs -380, -340, ... 380, so it never lands on
    # -200 / 0 / +200 -- hence a separate file.
    field = pv.PolyData(np.array([[-200.0, 0.0, 30.0],
                                  [0.0, 0.0, 30.0],
                                  [200.0, 0.0, 30.0]]))
    field["src_index"] = np.arange(3, dtype=np.int32)
    field_path = os.path.join(args.out_dir, "caber_field_soundings.vtp")
    field.save(field_path)
    print(f"wrote {field_path} "
          f"({field.n_points} field-movie sounding locations)")


if __name__ == "__main__":
    main()
