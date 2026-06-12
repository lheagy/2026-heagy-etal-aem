"""Run the MT-Mill TDEM model design / inversion headless.

Stripped from model-design-mt-mill.ipynb: all plotting/REPL-display code removed.
Runs the forward problem and inversion, then saves the recovered model to mrec.npy.

Intended to be left running detached, e.g.:

    screen -S aem
    python model-design-mt-mill.py
    # Ctrl-a, d  to detach;  screen -r aem  to reattach
"""

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

from simpeg.meta import MultiprocessingMetaSimulation  # DaskMetaSimulation


Solver = get_default_solver()


# ---------------------------------------------------------------------------
# model + survey parameters
# ---------------------------------------------------------------------------
sigma_overburden = 0.01
depth_overburden = 60

sigma_alteration = 0.05
thickness_alteration = 150
depth_alteration = 300

sigma_stock = 0.001
width_stock = 400
depth_stock = depth_alteration

sigma_host = 0.002

sigma_air = 1e-8

rx_times = np.logspace(np.log10(2e-5), np.log10(2e-3), 20)

tx_height = np.r_[30]

rx_x = (np.linspace(-700, 700, 29))[::2]
rx_y = rx_x[::2]
rx_z = tx_height
rx_locs = discretize.utils.ndgrid([rx_x, rx_y, rx_z])

n_times_invert = 20

relative_error = 0.05
noise_floor = 1e-12

refine_depth = 300


def build_survey():
    source_list = []
    for i in range(rx_locs.shape[0]):
        rx = tdem.receivers.PointMagneticFluxTimeDerivative(
            rx_locs[i, :], rx_times, orientation="z"
        )
        src = tdem.sources.CircularLoop(
            receiver_list=[rx],
            location=rx_locs[i, :],
            orientation="z",
            radius=10,
            waveform=tdem.sources.StepOffWaveform(),
        )
        source_list.append(src)
    return tdem.Survey(source_list)


def build_model(global_mesh):
    model = np.ones(global_mesh.n_cells) * sigma_air
    model[global_mesh.cell_centers[:, 2] < 0] = sigma_host

    # alteration
    extent_alteration = width_stock / 2 + thickness_alteration
    inds_alteration = (
        (global_mesh.cell_centers[:, 0] > -extent_alteration)
        & (global_mesh.cell_centers[:, 0] < extent_alteration)
        & (global_mesh.cell_centers[:, 1] > -extent_alteration)
        & (global_mesh.cell_centers[:, 1] < extent_alteration)
        & (global_mesh.cell_centers[:, 2] > -depth_alteration)
        & (global_mesh.cell_centers[:, 2] < 0)
    )
    model[inds_alteration] = sigma_alteration

    # stock
    extent_stock = width_stock / 2
    inds_stock = (
        (global_mesh.cell_centers[:, 0] > -extent_stock)
        & (global_mesh.cell_centers[:, 0] < extent_stock)
        & (global_mesh.cell_centers[:, 1] > -extent_stock)
        & (global_mesh.cell_centers[:, 1] < extent_stock)
        & (global_mesh.cell_centers[:, 2] > -depth_stock)
        & (global_mesh.cell_centers[:, 2] < 0)
    )
    model[inds_stock] = sigma_stock

    # overburden
    extent_overburden = 1500  # mesh dependent
    inds_overburden = (
        (global_mesh.cell_centers[:, 0] > -extent_overburden)
        & (global_mesh.cell_centers[:, 0] < extent_overburden)
        & (global_mesh.cell_centers[:, 1] > -extent_overburden)
        & (global_mesh.cell_centers[:, 1] < extent_overburden)
        & (global_mesh.cell_centers[:, 2] > -depth_overburden)
        & (global_mesh.cell_centers[:, 2] < 0)
    )
    model[inds_overburden] = sigma_overburden

    return model


def build_local_meshes(global_mesh, survey):
    mesh_list = []
    for src in survey.source_list:
        mesh_local = discretize.TreeMesh(
            global_mesh.h, origin=global_mesh.origin, diagonal_balance=True
        )
        refine_points = discretize.utils.ndgrid(
            np.r_[src.location[0]],
            np.r_[src.location[1]],
            np.linspace(-refine_depth, src.location[2], 40),
        )
        mesh_local.refine_points(
            refine_points,
            level=-1,
            padding_cells_by_level=[2, 2, 2, 2],
            finalize=True,
            diagonal_balance=True,
        )
        mesh_list.append(mesh_local)
    return mesh_list


def main():
    print(f"Using solver: {Solver}")

    survey = build_survey()

    global_mesh = discretize.TreeMesh.read_UBC("td-octree-mt-mill/octree_mesh.txt")

    model = build_model(global_mesh)

    # set up local meshes
    active_cells_map = maps.InjectActiveCells(
        global_mesh, global_mesh.cell_centers[:, 2] < 0, value_inactive=np.log(1e-8)
    )

    mesh_list = build_local_meshes(global_mesh, survey)

    # set up tiled simulation
    nsteps = 20
    time_steps = [
        (1e-6, nsteps),
        (3e-6, nsteps),
        (1e-5, nsteps),
        (3e-5, nsteps),
        (1e-4, 20),
    ]

    mappings = []
    sims = []
    for ii, local_mesh in enumerate(mesh_list):
        tile_map = maps.TileMap(
            global_mesh, active_cells_map.active_cells, local_mesh
        )
        mappings.append(tile_map)

        local_actmap = maps.InjectActiveCells(
            local_mesh,
            active_cells=tile_map.local_active,
            value_inactive=np.log(1e-8),
        )

        local_survey = tdem.Survey([survey.source_list[ii]])
        sims.append(
            tdem.simulation.Simulation3DElectricField(
                mesh=local_mesh,
                survey=local_survey,
                time_steps=time_steps,
                solver=Solver,
                sigmaMap=maps.ExpMap() * local_actmap,
            )
        )

    sim = MultiprocessingMetaSimulation(sims, mappings)

    true_model = np.log(model)[active_cells_map.active_cells]

    # forward simulation -> synthetic data
    print("Running forward simulation...")
    dobs = sim.dpred(true_model)

    data_invert = Data(
        survey,
        dobs=dobs,
        standard_deviation=np.abs(dobs) * relative_error + noise_floor,
    )

    # set up inversion
    dmis = data_misfit.L2DataMisfit(simulation=sim, data=data_invert)
    reg = regularization.WeightedLeastSquares(
        global_mesh,
        active_cells=active_cells_map.active_cells,
    )

    opt = optimization.InexactGaussNewton(maxIter=20, maxIterCG=50)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10, n_pw_iter=1)
    cool_beta = directives.BetaSchedule(coolingFactor=2, coolingRate=4)
    save_iteration = directives.SaveOutputDictEveryIteration(saveOnDisk=True)
    target_misfit = directives.TargetMisfit()

    directives_list = [
        starting_beta,
        cool_beta,
        save_iteration,
        target_misfit,
    ]

    inv = inversion.BaseInversion(inv_prob, directives_list)

    m0 = np.log(sigma_host) * np.ones(np.sum(active_cells_map.active_cells))

    print("Running inversion...")
    mrec = inv.run(m0)

    # save the recovered model
    np.save("mrec.npy", mrec)
    print("Saved recovered model to mrec.npy")


if __name__ == "__main__":
    main()
