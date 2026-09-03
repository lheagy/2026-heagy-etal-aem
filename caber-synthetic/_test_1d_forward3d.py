"""Forward-model the stitched-1D recovered model in 3D.

Maps the per-sounding 1D layered models (_models_1d_40m.npy) onto the global
OcTree mesh -- each cell within the survey footprint takes the conductivity of
the nearest sounding's 1D column at that depth; everything else is filled with
the true background. Then runs the tiled (MultiprocessingMetaSimulation)
forward to get the 3D predicted data, and compares it to the observed data.

This tests self-consistency of the stitched 1D: it fits each sounding with a 1D
forward, but does the assembled model reproduce the actual 3D response?

Writes _sigma_1d_on_3d.npy (full-mesh conductivity) and _dpred_1d_on_3d.npy.
Run with the 40 m survey config in _test_parametric.py (the 1D survey).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import discretize
from scipy.spatial import cKDTree

from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver
from simpeg.meta import MultiprocessingMetaSimulation

from _test_parametric import (
    build_global_mesh, build_local_meshes,
    rx_locs, rx_times, rx_x, rx_y, sigma_back, sigma_air, TIME_STEPS,
)

# 1D layer geometry (matches _test_1d_single)
_CS, _CORE, _NPAD, _PF = 25.0, 400.0, 12, 1.3
THICK = discretize.utils.unpack_widths(
    [(_CS, int(np.ceil(_CORE / _CS))), (_CS, _NPAD, _PF)])
LAYER_TOPS = np.r_[0.0, np.cumsum(THICK)]   # positive-down layer boundaries


def map_1d_to_3d(mesh, models_1d):
    """Per-cell conductivity from the nearest 1D sounding column; background
    outside the survey footprint."""
    active = mesh.cell_centers[:, 2] < 0
    cc = mesh.cell_centers
    sigma = np.full(mesh.n_cells, sigma_air)
    sigma[active] = sigma_back                          # default subsurface
    # footprint = soundings' bounding box (+ half a station/line spacing)
    fp = (active
          & (cc[:, 0] >= rx_x.min() - 20) & (cc[:, 0] <= rx_x.max() + 20)
          & (cc[:, 1] >= rx_y.min() - 50) & (cc[:, 1] <= rx_y.max() + 50))
    ic = np.where(fp)[0]
    tree = cKDTree(rx_locs[:, :2])                      # sounding (x, y)
    _, s_near = tree.query(cc[ic, :2])
    depth = -cc[ic, 2]
    lay = np.clip(np.searchsorted(LAYER_TOPS, depth, side="right") - 1,
                  0, models_1d.shape[1] - 1)
    sigma[ic] = np.exp(models_1d[s_near, lay])
    return sigma, active


def main():
    Solver = get_default_solver()
    mesh = build_global_mesh()
    models_1d = np.load("_models_1d_40m.npy")
    print(f"mesh {mesh.n_cells} cells; 1D model {models_1d.shape} on "
          f"{rx_locs.shape[0]} soundings", flush=True)

    sigma, active = map_1d_to_3d(mesh, models_1d)
    np.save("_sigma_1d_on_3d.npy", sigma)
    print(f"1D-on-3D sigma range (subsurface): "
          f"[{sigma[active].min():.2e}, {sigma[active].max():.2e}] S/m", flush=True)

    # ---- tiled forward ----
    source_list = []
    for i in range(rx_locs.shape[0]):
        loc = rx_locs[i, :]
        rx = tdem.receivers.PointMagneticFluxTimeDerivative(loc, rx_times, orientation="z")
        src = tdem.sources.CircularLoop(
            receiver_list=[rx], location=loc, orientation="z", radius=10,
            waveform=tdem.sources.StepOffWaveform())
        source_list.append(src)
    survey = tdem.Survey(source_list)

    mesh_list = build_local_meshes(mesh, survey)
    mappings, sims = [], []
    for ii, lm in enumerate(mesh_list):
        tile = maps.TileMap(mesh, active, lm)
        lactmap = maps.InjectActiveCells(lm, active_cells=tile.local_active,
                                         value_inactive=np.log(1e-8))
        mappings.append(tile)
        sims.append(tdem.simulation.Simulation3DElectricField(
            mesh=lm, survey=tdem.Survey([survey.source_list[ii]]),
            time_steps=TIME_STEPS, solver=Solver,
            sigmaMap=maps.ExpMap() * lactmap))
    sim = MultiprocessingMetaSimulation(sims, mappings)

    import time as _t
    t0 = _t.time()
    dpred = sim.dpred(np.log(sigma[active]))
    sim.join()
    np.save("_dpred_1d_on_3d.npy", dpred)
    print(f"3D forward of the 1D model: {dpred.size} data in {_t.time()-t0:.1f}s "
          f"-> _dpred_1d_on_3d.npy", flush=True)

    # quick comparison to observed
    dobs = np.load("_dobs_40m.npy")
    rel = (dpred - dobs) / (0.10 * np.abs(dobs) + 1e-12)
    print(f"vs observed (40 m): chi^2/N @10% = {np.mean(rel**2):.1f}  "
          f"(median |rel misfit| = {np.median(np.abs((dpred-dobs)/np.abs(dobs))):.2f})",
          flush=True)


if __name__ == "__main__":
    main()
