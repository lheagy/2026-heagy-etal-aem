"""Caber synthetic TDEM inversion example.

Converted from caber-synthetic.ipynb.

This workflow uses simpeg's MultiprocessingMetaSimulation. Because multiprocessing
workers re-import this module (the "spawn" start method on macOS/Windows), the
execution code lives inside `if __name__ == "__main__":`. Module-level code
(imports and the helper function definitions) runs in every worker process; the
guarded block runs only in the main process.

Notebook plotting cells are commented out.
"""

import os

# Keep each worker single-threaded; must be set before numpy / simpeg import.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import time
import pickle

from concurrent.futures import ProcessPoolExecutor, as_completed

import discretize
# from simpeg import dask
from simpeg import (
    maps,
    Data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver

from simpeg.meta import MultiprocessingMetaSimulation, MetaSimulation

from parametric_ellipsoid import ParametricEllipsoid

Solver = get_default_solver()


def diffusion_distance(sigma, t):
    return 1260*np.sqrt(t/sigma)


def dipping_target_indices(
    mesh, target_x_center, target_z_center, dip, target_thickness, target_xlim=None, target_ylim=None, target_zlim=None
):
    """
    add a dipping target to the model. For now assumes the target dips in the x-direction
    """
    slope = np.tan(-dip*np.pi/180)
    target_z = target_z_center + target_thickness / 2 * np.r_[-1, 1]

    z_bottom = (mesh.cell_centers[:, 0] - target_x_center) * slope + target_z.min()
    z_top = (mesh.cell_centers[:, 0] - target_x_center) * slope + target_z.max()

    indices = (
        (mesh.cell_centers[:, 2] >= z_bottom) &
        (mesh.cell_centers[:, 2] <= z_top)
    )

    if target_xlim is not None:
        indices = indices & (
            (mesh.cell_centers[:, 0] >= target_xlim.min()) &
            (mesh.cell_centers[:, 0] <= target_xlim.max())
        )
    if target_ylim is not None:
        indices = indices & (
            (mesh.cell_centers[:, 1] >= target_ylim.min()) &
            (mesh.cell_centers[:, 1] <= target_ylim.max())
        )
    if target_zlim is not None:
        indices = indices & (
            (mesh.cell_centers[:, 2] >= target_zlim.min()) &
            (mesh.cell_centers[:, 2] <= target_zlim.max())
        )
    return indices


if __name__ == "__main__":
    # --- [cell 2] ---
    rho_back = 1000
    sigma_back = 1./rho_back

    rho_target = 0.1
    sigma_target = 1./rho_target

    rho_overburden = 50
    sigma_overburden = 1./rho_overburden

    sigma_air = 1e-8

    target_dips = np.r_[115] #, 45]
    target_z = np.r_[-300, -150]

    # --- [cell 3] ---
    # rx_locs = np.loadtxt(f"{directory}/rx_locs.txt")
    # rx_locs[:, 1] =
    tx_height = 30

    rx_x = (np.linspace(-500, 500, 26))

    rx_y = (np.linspace(-400, 400, 5))
    # rx_y = np.r_[-80, -40, 0, 40, 80]

    rx_z = tx_height

    rx_locs = discretize.utils.ndgrid([rx_x, rx_y, rx_z])
    rx_x

    # rx_times = np.loadtxt(f"{directory}/rx_times.txt")
    rx_times = np.logspace(np.log10(2e-5), np.log10(2e-3), 20)

    # --- [cell 4] ---
    print("rx_x =", rx_x)

    # --- [cell 5] ---
    print("rx_y =", rx_y)

    # --- [cell 7] ---
    print("diffusion distance, target:", diffusion_distance(sigma_target, 8e-3))

    # --- [cell 8] ---
    print("diffusion distance, background:", diffusion_distance(sigma_back, 8e-3))

    # --- [cell 9] ---
    base_cell_width = 20
    domain_extent = 8000

    n_base_cells = 2 ** int(
        np.ceil(np.log(domain_extent / base_cell_width) / np.log(2.0))
    )  # needs to be powers of 2 for the tree mesh

    h = [(base_cell_width, n_base_cells)]
    mesh = discretize.TreeMesh([h, h, h], origin="CCC", diagonal_balance=True)

    # refine near transmitters and receivers
    mesh.refine_points(
        rx_locs, level=-1, padding_cells_by_level=[2, 2, 2],
        finalize=False, diagonal_balance=True
    )

    # Refine core region of the mesh

    bounding_points = np.array([
        [-500, rx_y.min(), target_z.min() - base_cell_width * 4],
        [500, rx_y.max(), 0],
    ])
    mesh.refine_bounding_box(
        bounding_points, level=-1,
        diagonal_balance=True, finalize=False, padding_cells_by_level=[4, 2, 2]
    )

    mesh.finalize()

    # --- [cell 10] ---
    print(mesh)

    # --- [cell 12] ---
    models = {}

    target_x = np.r_[-300, 300]
    target_y = np.r_[-100, 100]
    target_thickness = 200

    overburden_x = np.r_[-640, 640]
    overburden_y = np.r_[-520, 520]
    overburden_thickness = 40

    # background model
    background = np.ones(mesh.n_cells) * sigma_air
    background[mesh.cell_centers[:, 2] < 0] = sigma_back
    models["background"] = background

    for dip in target_dips:
        conductivity_model = background.copy()
        indices = dipping_target_indices(
            mesh, target_x_center=0, target_z_center=np.mean(target_z),
            target_thickness=target_thickness, dip=dip,
            target_xlim=target_x,
            target_ylim=target_y,
            target_zlim=target_z
        )
        conductivity_model[indices] = sigma_target

        # add overburden
        overburden_indices = (
            (mesh.cell_centers[:, 0] >= overburden_x[0]) &
            (mesh.cell_centers[:, 0] < overburden_x[1]) &
            (mesh.cell_centers[:, 1] >= overburden_y[0]) &
            (mesh.cell_centers[:, 1] < overburden_y[1]) &
            (mesh.cell_centers[:, 2] >= -overburden_thickness) &
            (mesh.cell_centers[:, 2] < 0)
        )
        conductivity_model[overburden_indices] = sigma_overburden
        models[f"target_{dip}"] = conductivity_model

    # --- [cell 13] plot (commented out) ---
    # fig, ax = plt.subplots(1, 1, figsize=(8, 2))

    # plt.colorbar(
    #     mesh.plot_slice(
    #         conductivity_model,
    #         # grid=True,
    #         normal="y",
    #         pcolor_opts={"norm":LogNorm(1e-6, 1e0)},
    #         ax=ax)[0],
    #     ax=ax
    # )

    # ax.set_xlim(800*np.r_[-1, 1])
    # ax.set_ylim(np.r_[-400, 50])

    # ax.plot(rx_locs[:, 0], rx_locs[:, 2], "ro")
    # ax.set_aspect(1)

    # --- [cell 14] plot (commented out) ---
    # fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # # mesh_local = mesh_list[-1]

    # plt.colorbar(
    #     mesh.plot_slice(
    #         conductivity_model,
    #         # grid=True,
    #         normal="z",
    #         pcolor_opts={"norm":LogNorm(1e-3, 1e0)},
    #         ax=ax,
    #         ind=245
    #     )[0],
    #     ax=ax
    # )

    # xlim = 800*np.r_[-1, 1]
    # ax.set_xlim(xlim)
    # ax.set_ylim(xlim)

    # ax.plot(rx_locs[:, 0], rx_locs[:, 1], "ro")
    # ax.set_aspect(1)

    # --- [cell 15] ---
    source_list = []

    for i in range(rx_locs.shape[0]):
        rx = tdem.receivers.PointMagneticFluxTimeDerivative(rx_locs[i, :], rx_times, orientation="z")
        src = tdem.sources.CircularLoop(
            receiver_list=[rx], location=rx_locs[i, :], orientation="z", radius=10,
            waveform=tdem.sources.StepOffWaveform()
        )
        source_list.append(src)

    survey = tdem.Survey(source_list)

    # --- [cell 16] ---
    # set up local meshes
    global_mesh = mesh
    active_cells_map = maps.InjectActiveCells(global_mesh, global_mesh.cell_centers[:, 2]<0, value_inactive=np.log(1e-8))

    refine_depth = 300
    # def get_local_mesh(src):
    mesh_list = []
    x1 = np.mean(rx_locs[:, 0]) - (np.sum(global_mesh.h[0]) / 2)
    x2 = np.mean(rx_locs[:, 1]) - (np.sum(global_mesh.h[1]) / 2)
    x3 = np.mean(rx_locs[:, 2]) - (np.sum(global_mesh.h[2]) / 2)

    for src in survey.source_list:
    # src = survey.source_list[0]
        mesh_local = discretize.TreeMesh(global_mesh.h, origin=global_mesh.origin, diagonal_balance=True)
        refine_points = discretize.utils.ndgrid(
            np.r_[src.location[0]],
            np.r_[src.location[1]],
            np.linspace(-refine_depth, src.location[2], 40)
        )
        mesh_local.refine_points(
            refine_points,
            level=-1,
            padding_cells_by_level=[4, 4, 2],
            finalize=True,
            diagonal_balance=True
        )
        mesh_list.append(mesh_local)

        mesh_local.x0

    # --- [cell 17] ---
    print(mesh_local)

    # --- [cell 18] plot (commented out) ---
    # fig, ax = plt.subplots(1, 1)
    # plt.colorbar(
    #     mesh.plot_slice(
    #         models["target_115"], ax=ax, pcolor_opts={"norm":LogNorm()},
    #         normal="Z", ind=250
    #     )[0],
    #     ax=ax
    # )

    # ax.set_xlim(500*np.r_[-1, 1])
    # ax.set_ylim(500*np.r_[-1, 1])

    # ax.plot(rx_locs[:, 0], rx_locs[:, 1], "wo", ms=4)
    # ax.set_aspect(1)

    # --- [cell 19] ---
    nsteps = 20
    time_steps = [
        # (1e-6, nsteps),
        # (3e-6, nsteps),
        (1e-5, nsteps),
        (3e-5, nsteps),
        (1e-4, 20),
    ]


    mappings = []
    sims = []

    for ii, local_mesh in enumerate(mesh_list):

        tile_map = maps.TileMap(global_mesh, active_cells_map.active_cells, local_mesh)
        mappings.append(tile_map)

        local_actmap = maps.InjectActiveCells(
            local_mesh,
            active_cells=tile_map.local_active,
            value_inactive=np.log(1e-8)
        )

        local_survey = tdem.Survey([survey.source_list[ii]])
        sims.append(tdem.simulation.Simulation3DElectricField(
                mesh=local_mesh,
                survey=local_survey,
                time_steps=time_steps,
                solver=Solver,
                sigmaMap=maps.ExpMap() * local_actmap
            )
        )


    sim = MultiprocessingMetaSimulation(sims, mappings)

    # --- [cell 20] ---
    model = np.log(conductivity_model[active_cells_map.active_cells])

    # --- [cell 21] ---
    sims[0].dpred(mappings[0] * model)

    # --- [cell 22] ---
    print("Computing observed data (forward simulation)...")
    dobs = sim.dpred(model)
    np.save("dobs-caber.npy", dobs)

    # --- [cell 23] ---
    dobs_obj = Data(survey=survey, dobs=dobs)

    # --- [cell 24] ---
    n_times_invert = len(rx_times)
    dobs_reshaped = dobs.reshape(len(rx_y), len(rx_x), n_times_invert)

    # --- [cell 25] plot (commented out) ---
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # ax.semilogy(rx_x, -dobs_reshaped[2, :, :], "-ok", ms=4);

    # --- [cell 26] ---
    relative_error = 0.05
    noise_floor = 1e-12

    # --- [cell 27] ---
    print("n times (full):", len(rx_times))

    # --- [cell 28] ---
    times_invert = slice(15, None)
    print("inversion times:", rx_times[times_invert])

    # --- [cell 29] ---
    source_list_invert = []

    for i in range(rx_locs.shape[0]):
        rx = tdem.receivers.PointMagneticFluxTimeDerivative(rx_locs[i, :], rx_times[times_invert], orientation="z")
        src = tdem.sources.CircularLoop(
            receiver_list=[rx], location=rx_locs[i, :], orientation="z", radius=10,
            waveform=tdem.sources.StepOffWaveform()
        )
        source_list_invert.append(src)

    survey_invert = tdem.Survey(source_list_invert)

    # --- [cell 30] ---
    dobs_invert = dobs.reshape(len(rx_y), len(rx_x), n_times_invert)[:, :, times_invert].flatten()
    data_invert = Data(
        survey_invert, dobs=dobs_invert,
        standard_deviation=np.abs(dobs_invert)*relative_error + noise_floor
    )

    # --- [cell 31] ---
    param_mappings = []
    param_sims = []

    global_ellipsoid = ParametricEllipsoid(
        global_mesh,
        active_cells=active_cells_map.active_cells,
        # slopeFact=10,
    )

    for ii, local_mesh in enumerate(mesh_list):

        tile_map = maps.TileMap(global_mesh, active_cells_map.active_cells, local_mesh)

        local_actmap = maps.InjectActiveCells(
            local_mesh,
            active_cells=tile_map.local_active,
            value_inactive=np.log(1e-8)
        )
        param_mappings.append(tile_map * global_ellipsoid)

        local_survey = tdem.Survey([survey_invert.source_list[ii]])
        param_sims.append(tdem.simulation.Simulation3DElectricField(
                mesh=local_mesh,
                survey=local_survey,
                time_steps=time_steps,
                solver=Solver,
                sigmaMap=maps.ExpMap(local_mesh) * local_actmap
            )
        )

    sim_parametric = MultiprocessingMetaSimulation(param_sims, param_mappings)

    # --- [cell 32] ---
    # p_0, rx, ry, rz, phix, phiy, phiz, x_0, y_0, z_0, p_1, a
    parametric_model = np.r_[
        np.log(sigma_back), 100, 100, 300, 0, 0, 0, 0, 0, -200, np.log(sigma_target), 2
    ]

    # --- [cell 33] plot (commented out) ---
    # fig, ax = plt.subplots(1, 1)
    # plt.colorbar(
    #     mesh.plot_slice(
    #         maps.ExpMap()*active_cells_map*global_ellipsoid * parametric_model, normal="y", ax=ax,
    #         pcolor_opts={"norm":LogNorm()}
    #         )[0],
    #     ax=ax
    # )

    # ax.set_xlim(800*np.r_[-1, 1])
    # ax.set_ylim(np.r_[-400, 50])
    # ax.set_aspect(1)

    # --- [cell 34] plot (commented out) ---
    # ind = 38
    # local_mesh = mesh_list[ind]
    # local_sim = param_sims[ind]
    # local_mapping = param_mappings[ind]

    # fig, ax = plt.subplots(1, 1)
    # plt.colorbar(local_mesh.plot_slice(
    #     local_sim.sigmaMap * local_mapping * parametric_model, normal="y", pcolor_opts={"norm":LogNorm()},
    #     ax=ax,
    #     # grid=True,
    #     grid_opts={"color":"k", "lw":0.5},
    # )[0], ax=ax)

    # ax.set_xlim(800*np.r_[-1, 1])
    # ax.set_ylim(np.r_[-400, 50])
    # ax.set_aspect(1)

    # --- [cell 35] ---
    print("Computing parametric predicted data (forward simulation)...")
    dpred_parametric = sim_parametric.dpred(parametric_model)

    # --- [cell 36] ---
    n_times_invert = len(rx_times[times_invert])
    dpred_parametric_reshaped = dpred_parametric.reshape(len(rx_y), len(rx_x), n_times_invert)

    # --- [cell 37] plot (commented out) ---
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # ind = 2
    # ax.semilogy(rx_x, -dobs_reshaped[ind, :, times_invert], "-ok", ms=4);
    # ax.semilogy(rx_x, -dpred_parametric_reshaped[ind, :, :], "-oC0", ms=4);

    # --- [cell 38] ---
    rel_err_parametric = 0.1
    data_invert.standard_deviation = np.abs(dobs_invert)*rel_err_parametric + noise_floor
    dmis = data_misfit.L2DataMisfit(simulation=sim_parametric, data=data_invert)
    regmesh = discretize.TensorMesh([len(parametric_model)])
    reg = regularization.Smallness(
        regmesh
    )

    # --- [cell 39] ---
    # opt = optimization.InexactGaussNewton(maxIter=5, cg_maxiter=30)
    # p_0, rx, ry, rz, phix, phiy, phiz, x_0, y_0, z_0, p_1, a

    opt = optimization.ProjectedGNCG(
        maxIter=10,
        upper=np.hstack([11*[np.inf], [4]]),
        cg_maxiter=30,
    )
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=0)

    # --- [cell 40] ---
    # Defining a starting value for the trade-off parameter (beta) between the data
    # misfit and the regularization.
    save_iteration = directives.SaveOutputDictEveryIteration(
        saveOnDisk=True,
    )
    target_misfit = directives.TargetMisfit()

    # --- [cell 41] ---
    # The directives are defined as a list.
    directives_list = [
        save_iteration,
        target_misfit,
    ]

    # Here we combine the inverse problem and the set of directives
    inv = inversion.BaseInversion(inv_prob, directives_list)

    # --- [cell 42] ---
    m0 = parametric_model.copy()

    # --- [cell 43] ---
    print("Starting parametric inversion...")
    mopt_parametric = inv.run(m0)
    np.save("mopt_parametric", mopt_parametric)

    # --- [cell 45] ---
    print("recovered parametric model:", mopt_parametric)

    # --- [cell 46] ---
    dpred_reshaped = inv_prob.dpred.reshape(len(rx_y), len(rx_x), n_times_invert)

    # --- [cell 47] plot (commented out) ---
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # ind = 3
    # ax.semilogy(rx_x, -dobs_reshaped[ind, :, :], "-ok", ms=4);
    # ax.semilogy(rx_x, -dpred_reshaped[ind, :, :], "-oC0", ms=4);

    # --- [cell 48] plot (commented out) ---
    # fig, ax = plt.subplots(1, 1)
    # plt.colorbar(
    #     mesh.plot_slice(
    #         maps.ExpMap()*active_cells_map*global_ellipsoid * mopt_parametric, normal="y", ax=ax,
    #         pcolor_opts={"norm":LogNorm()}
    #         )[0],
    #     ax=ax
    # )

    # ax.set_xlim(800*np.r_[-1, 1])
    # ax.set_ylim(np.r_[-400, 50])
    # ax.set_aspect(1)

    # --- [cell 49] ---
    data_invert_full = Data(
        survey, dobs=dobs,
        standard_deviation=np.abs(dobs)*relative_error + noise_floor
    )

    # --- [cell 50] ---
    # set up an inversion now using the parametric as a reference model
    dmis_full = data_misfit.L2DataMisfit(simulation=sim, data=data_invert_full)
    reg = regularization.WeightedLeastSquares(
        global_mesh,
        active_cells = active_cells_map.active_cells,
        reference_model = global_ellipsoid * mopt_parametric
    )

    # --- [cell 51] ---
    opt = optimization.InexactGaussNewton(maxIter=20, cg_maxiter=30)
    inv_prob = inverse_problem.BaseInvProblem(dmis_full, reg, opt)

    # --- [cell 52] ---
    # Defining a starting value for the trade-off parameter (beta) between the data
    # misfit and the regularization.
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=100, n_pw_iter=2)
    cool_beta = directives.BetaSchedule(coolingFactor=2, coolingRate=4)
    # Options for outputting recovered models and predicted data for each beta.
    save_iteration = directives.SaveOutputDictEveryIteration(
        saveOnDisk=True,
    )
    target_misfit = directives.TargetMisfit()

    # --- [cell 53] ---
    # The directives are defined as a list.
    directives_list = [
        starting_beta,
        cool_beta,
        save_iteration,
        target_misfit,
    ]

    # Here we combine the inverse problem and the set of directives
    inv = inversion.BaseInversion(inv_prob, directives_list)

    # --- [cell 54] ---
    m0 = np.ones(active_cells_map.active_cells.sum()) * np.log(sigma_back)

    # --- [cell 55] ---
    print("Starting full (voxel-based) inversion...")
    mrec = inv.run(m0)
    np.save("mrec_full", mrec)

    # --- [cell 56] ---
    # out = np.load("InversionModel_2026-06-10-11-24_04.npz", allow_pickle=True)["arr_0"].item()

    # --- [cell 57] plot (commented out) ---
    # fig, ax = plt.subplots(1, 1)
    # plt.colorbar(
    #     mesh.plot_slice(
    #         maps.ExpMap()*active_cells_map*mrec, normal="y", ax=ax,
    #         pcolor_opts={"norm":LogNorm()}
    #         )[0],
    #     ax=ax
    # )

    # ax.set_xlim(800*np.r_[-1, 1])
    # ax.set_ylim(np.r_[-400, 50])
    # ax.set_aspect(1)
