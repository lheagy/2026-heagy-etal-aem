"""Focused test of the parametric ellipsoid inversion.

Replicates the notebook setup through the parametric inversion only,
so we can iterate quickly on starting model / sharpness / error / iter
choices without paying for the full-mesh second stage.

Run with the py311 env active:
    python _test_parametric.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import numpy as np

import discretize
from simpeg import (
    maps,
    Data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
)
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver
from simpeg.meta import MultiprocessingMetaSimulation

from parametric_ellipsoid import ParametricEllipsoid


# ---------------------------------------------------------------- physics
rho_back = 1000.0
sigma_back = 1.0 / rho_back
rho_target = 0.2
sigma_target = 1.0 / rho_target  # = 5 S/m
sigma_air = 1e-8

target_dip = 135  # 45 deg from horizontal (slope = -tan(135 deg) = +1)
target_z = np.r_[-300, -120]


# ---------------------------------------------------------------- survey
tx_height = 30
rx_x = (np.linspace(-500, 500, 26))[3:-3][::2]
rx_y = (np.linspace(-400, 400, 5))[1:-1]
rx_z = tx_height
rx_locs = discretize.utils.ndgrid([rx_x, rx_y, rx_z])
rx_times = np.logspace(np.log10(2e-5), np.log10(2e-3), 20)


# ---------------------------------------------------------------- meshes
def build_global_mesh():
    base_cell_width = 20
    domain_extent = 8000
    n_base_cells = 2 ** int(
        np.ceil(np.log(domain_extent / base_cell_width) / np.log(2.0))
    )
    h = [(base_cell_width, n_base_cells)]
    mesh = discretize.TreeMesh([h, h, h], origin="CCC", diagonal_balance=True)
    mesh.refine_points(
        rx_locs, level=-1, padding_cells_by_level=[2, 2, 2],
        finalize=False, diagonal_balance=True,
    )
    bounding_points = np.array([
        [-500, rx_y.min(), target_z.min() - base_cell_width * 4],
        [ 500, rx_y.max(), 0],
    ])
    mesh.refine_bounding_box(
        bounding_points, level=-1,
        diagonal_balance=True, finalize=False,
        padding_cells_by_level=[2, 2, 2],
    )
    mesh.finalize()
    return mesh


def build_local_meshes(global_mesh, survey, refine_depth=300):
    mesh_list = []
    for src in survey.source_list:
        m_local = discretize.TreeMesh(
            global_mesh.h, origin=global_mesh.origin, diagonal_balance=True,
        )
        refine_points = discretize.utils.ndgrid(
            np.r_[src.location[0]],
            np.r_[src.location[1]],
            np.linspace(-refine_depth, src.location[2], 40),
        )
        m_local.refine_points(
            refine_points, level=-1,
            padding_cells_by_level=[2, 2, 2],
            finalize=True, diagonal_balance=True,
        )
        mesh_list.append(m_local)
    return mesh_list


# ---------------------------------------------------------------- true model
def dipping_target_indices(
    mesh, target_x_center, target_z_center, dip, target_thickness,
    target_xlim=None, target_ylim=None, target_zlim=None,
):
    slope = np.tan(-dip * np.pi / 180)
    tz = target_z_center + target_thickness / 2 * np.r_[-1, 1]
    z_bot = (mesh.cell_centers[:, 0] - target_x_center) * slope + tz.min()
    z_top = (mesh.cell_centers[:, 0] - target_x_center) * slope + tz.max()
    ind = (mesh.cell_centers[:, 2] >= z_bot) & (mesh.cell_centers[:, 2] <= z_top)
    if target_xlim is not None:
        ind &= (mesh.cell_centers[:, 0] >= target_xlim.min()) & (mesh.cell_centers[:, 0] <= target_xlim.max())
    if target_ylim is not None:
        ind &= (mesh.cell_centers[:, 1] >= target_ylim.min()) & (mesh.cell_centers[:, 1] <= target_ylim.max())
    if target_zlim is not None:
        ind &= (mesh.cell_centers[:, 2] >= target_zlim.min()) & (mesh.cell_centers[:, 2] <= target_zlim.max())
    return ind


def build_true_model(global_mesh):
    sigma = np.ones(global_mesh.n_cells) * sigma_air
    sigma[global_mesh.cell_centers[:, 2] < 0] = sigma_back
    target_x = np.r_[-300, 300]
    target_y = np.r_[-100, 100]
    ind = dipping_target_indices(
        global_mesh,
        target_x_center=0,
        target_z_center=np.mean(target_z),
        target_thickness=100,
        dip=target_dip,
        target_xlim=target_x,
        target_ylim=target_y,
        target_zlim=target_z,
    )
    sigma[ind] = sigma_target
    return sigma


# ---------------------------------------------------------------- main
def main():
    Solver = get_default_solver()

    global_mesh = build_global_mesh()
    print(f"global mesh: {global_mesh.n_cells} cells", flush=True)

    # full + parametric surveys (full time channels for forward, last 5 for inv)
    times_invert = slice(15, None)
    source_list = []
    source_list_invert = []
    for i in range(rx_locs.shape[0]):
        loc = rx_locs[i, :]
        rx_full = tdem.receivers.PointMagneticFluxTimeDerivative(loc, rx_times, orientation="z")
        rx_inv  = tdem.receivers.PointMagneticFluxTimeDerivative(loc, rx_times[times_invert], orientation="z")
        wf = tdem.sources.StepOffWaveform()
        src_kwargs = dict(location=loc, orientation="z", radius=10, waveform=wf)
        source_list.append(tdem.sources.CircularLoop(receiver_list=[rx_full], **src_kwargs))
        source_list_invert.append(tdem.sources.CircularLoop(receiver_list=[rx_inv], **src_kwargs))
    survey = tdem.Survey(source_list)
    survey_invert = tdem.Survey(source_list_invert)
    print(f"survey: {len(source_list)} sources x {len(rx_times)} times = {len(source_list)*len(rx_times)} data", flush=True)

    active_cells_map = maps.InjectActiveCells(
        global_mesh,
        global_mesh.cell_centers[:, 2] < 0,
        value_inactive=np.log(1e-8),
    )

    mesh_list = build_local_meshes(global_mesh, survey)

    time_steps = [(1e-5, 20), (3e-5, 20), (1e-4, 20)]

    # ---- full sim (for dobs generation) ----
    mappings, sims = [], []
    for ii, local_mesh in enumerate(mesh_list):
        tile_map = maps.TileMap(global_mesh, active_cells_map.active_cells, local_mesh)
        local_actmap = maps.InjectActiveCells(
            local_mesh,
            active_cells=tile_map.local_active,
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
    sim_full = MultiprocessingMetaSimulation(sims, mappings)

    # ---- generate dobs (or load if cache matches) ----
    expected_size = len(source_list) * len(rx_times)
    cache_path = "_dobs_cache.npy"
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.size == expected_size:
            dobs = cached
            print(f"loaded dobs from {cache_path}", flush=True)
        else:
            dobs = None
    else:
        dobs = None
    if dobs is None:
        sigma_true = build_true_model(global_mesh)
        model_true = np.log(sigma_true[active_cells_map.active_cells])
        print("generating dobs ...", flush=True)
        import time as _time
        t0 = _time.time()
        dobs = sim_full.dpred(model_true)
        print(f"  dobs ready: {_time.time()-t0:.1f} s, shape={dobs.shape}", flush=True)
        np.save(cache_path, dobs)

    # ---- inversion data (last 10 channels only) ----
    n_t = len(rx_times)
    dobs_invert = dobs.reshape(len(rx_y), len(rx_x), n_t)[:, :, times_invert].flatten()
    relative_error = 0.05
    noise_floor = 1e-12

    # ---- parametric sim ----
    param_mappings, param_sims = [], []
    global_ellipsoid = ParametricEllipsoid(
        global_mesh,
        active_cells=active_cells_map.active_cells,
        boundary_sharpness=10.0,
    )
    for ii, local_mesh in enumerate(mesh_list):
        tile_map = maps.TileMap(global_mesh, active_cells_map.active_cells, local_mesh)
        local_actmap = maps.InjectActiveCells(
            local_mesh,
            active_cells=tile_map.local_active,
            value_inactive=np.log(1e-8),
        )
        param_mappings.append(tile_map * global_ellipsoid)
        param_sims.append(tdem.simulation.Simulation3DElectricField(
            mesh=local_mesh,
            survey=tdem.Survey([survey_invert.source_list[ii]]),
            time_steps=time_steps,
            solver=Solver,
            sigmaMap=maps.ExpMap(local_mesh) * local_actmap,
        ))
    sim_parametric = MultiprocessingMetaSimulation(param_sims, param_mappings)

    # ---- inversion ----
    rel_err_parametric = 0.10
    data_invert = Data(
        survey_invert,
        dobs=dobs_invert,
        standard_deviation=np.abs(dobs_invert) * rel_err_parametric + noise_floor,
    )

    # p_0, log_rx, log_ry, log_rz, phi_x, phi_y, phi_z, x_0, y_0, z_0, p_interior
    # Disk-like starting model (wide x wide x thin) to match a sheet-like
    # dipping target. (50, 50, 150) vertical-pipe start sent the inversion
    # into a deep, near-vertical local minimum for this thinner / lower-sigma
    # case.
    parametric_model = np.r_[
        np.log(sigma_back),
        np.log(100), np.log(100), np.log(50),
        0, 0, 0,
        0, 0, -200,
        np.log(sigma_target),
    ]
    LABELS = ["p_0","log_rx","log_ry","log_rz","phi_x","phi_y","phi_z","x_0","y_0","z_0","p_in"]

    dmis = data_misfit.L2DataMisfit(simulation=sim_parametric, data=data_invert)
    regmesh = discretize.TensorMesh([len(parametric_model)])
    reg = regularization.Smallness(regmesh)

    lower = np.r_[
        np.log(1e-6),
        np.log(5), np.log(5), np.log(5),
        -np.pi, -np.pi, -np.pi,
        -np.inf, -np.inf, -np.inf,
        np.log(1e-3),
    ]
    upper = np.r_[
        np.log(1e-1),
        np.log(1000), np.log(1000), np.log(1000),
        np.pi, np.pi, np.pi,
        np.inf, np.inf, 0.0,
        np.log(1e3),
    ]
    opt = optimization.ProjectedGNCG(
        maxIter=10, lower=lower, upper=upper, cg_maxiter=15,
    )
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=0)
    save_iteration = directives.SaveOutputDictEveryIteration(saveOnDisk=True)
    target_misfit = directives.TargetMisfit()
    inv = inversion.BaseInversion(inv_prob, [save_iteration, target_misfit])

    print("\nstarting model:")
    for lab, v in zip(LABELS, parametric_model):
        print(f"  {lab:>7} = {v:+10.4g}")

    m0 = parametric_model.copy()
    mopt = inv.run(m0)

    print("\nrecovered model:")
    for lab, v, lo, up in zip(LABELS, mopt, lower, upper):
        flag = ""
        if np.isfinite(lo) and abs(v - lo) < 1e-4 * (abs(lo) + 1e-9):
            flag = "  <- at lower bound"
        if np.isfinite(up) and abs(v - up) < 1e-4 * (abs(up) + 1e-9):
            flag = "  <- at upper bound"
        print(f"  {lab:>7} = {v:+10.4g}{flag}")

    rx_, ry_, rz_ = np.exp(mopt[1:4])
    print(f"\n  semi-axes (m):  rx={rx_:.1f}  ry={ry_:.1f}  rz={rz_:.1f}")
    print(f"  center (m):     ({mopt[7]:.1f}, {mopt[8]:.1f}, {mopt[9]:.1f})")
    print(f"  angles (deg):   ({np.degrees(mopt[4]):.1f}, {np.degrees(mopt[5]):.1f}, {np.degrees(mopt[6]):.1f})")
    print(f"  sigma_back:     {np.exp(mopt[0]):.4g} S/m")
    print(f"  sigma_target:   {np.exp(mopt[10]):.4g} S/m")

    np.save("_mopt_test.npy", mopt)
    print("\nsaved -> _mopt_test.npy")


if __name__ == "__main__":
    main()
