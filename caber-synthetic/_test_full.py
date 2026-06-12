"""Full-mesh second-stage inversion.

m0 = parametric recovery (iter 3, hardcoded from _run3.log)
reference = background
beta0_ratio = 1 (data dominates ~30:1 at m0)
ProjectedGNCG, maxIter=15, log-cond bounds [1e-6, 100] S/m.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

import discretize
from simpeg import (
    maps, Data, data_misfit, inverse_problem,
    regularization, optimization, directives, inversion,
)
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver
from simpeg.meta import MultiprocessingMetaSimulation

from parametric_ellipsoid import ParametricEllipsoid
from _test_parametric import (
    build_global_mesh,
    build_local_meshes,
    sigma_back,
    rx_locs,
    rx_times,
)


# "Ideal" parametric m0 -- ellipsoid matching the true dipping slab geometry.
# phi_y = +65 deg rotates the long axis (body x) into the dip direction
# (slope +2.144 -> dip 65 deg from horizontal in xz).
MOPT_PARAMETRIC = np.array([
    np.log(1e-3),        # p_0 (sigma_back)
    np.log(100),         # log_rx (along-dip, slab spine length ~99 m)
    np.log(100),         # log_ry (along strike, 100 m semi-axis)
    np.log(50),          # log_rz (slab-perpendicular, ~50 m semi-axis)
    0.0,                 # phi_x
    np.radians(65.0),    # phi_y
    0.0,                 # phi_z
    0.0,                 # x_0
    0.0,                 # y_0
    -210.0,              # z_0
    np.log(10.0),        # p_interior (sigma_target)
])


def main():
    Solver = get_default_solver()

    global_mesh = build_global_mesh()
    active_cells = global_mesh.cell_centers[:, 2] < 0
    n_active = int(active_cells.sum())
    print(f"global mesh: {global_mesh.n_cells} cells ({n_active} active)", flush=True)

    active_cells_map = maps.InjectActiveCells(
        global_mesh, active_cells, value_inactive=np.log(1e-8),
    )

    # full survey (all 20 channels)
    source_list = []
    for i in range(rx_locs.shape[0]):
        loc = rx_locs[i, :]
        rx = tdem.receivers.PointMagneticFluxTimeDerivative(loc, rx_times, orientation="z")
        src = tdem.sources.CircularLoop(
            receiver_list=[rx], location=loc, orientation="z", radius=10,
            waveform=tdem.sources.StepOffWaveform(),
        )
        source_list.append(src)
    survey = tdem.Survey(source_list)
    n_data = len(source_list) * len(rx_times)
    print(f"survey: {len(source_list)} sources x {len(rx_times)} times = {n_data} data", flush=True)

    mesh_list = build_local_meshes(global_mesh, survey)
    time_steps = [(1e-5, 20), (3e-5, 20), (1e-4, 20)]

    mappings, sims = [], []
    for ii, local_mesh in enumerate(mesh_list):
        tile_map = maps.TileMap(global_mesh, active_cells, local_mesh)
        local_actmap = maps.InjectActiveCells(
            local_mesh, active_cells=tile_map.local_active,
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
    sim = MultiprocessingMetaSimulation(sims, mappings)

    dobs = np.load("_dobs_cache.npy")
    assert dobs.size == n_data, f"dobs cache size mismatch: {dobs.size} != {n_data}"
    print(f"loaded dobs: shape={dobs.shape}", flush=True)

    relative_error = 0.05
    noise_floor = 1e-12
    data_obj = Data(
        survey, dobs=dobs,
        standard_deviation=np.abs(dobs) * relative_error + noise_floor,
    )

    # m0 from parametric recovery
    global_ellipsoid = ParametricEllipsoid(
        global_mesh, active_cells=active_cells, boundary_sharpness=10.0,
    )
    m0 = global_ellipsoid * MOPT_PARAMETRIC
    print(f"m0:  log-cond range [{m0.min():.3f}, {m0.max():.3f}] "
          f"-> sigma [{np.exp(m0.min()):.3e}, {np.exp(m0.max()):.3e}] S/m", flush=True)

    # reference = m0 (only affects smallness term -- smoothness still
    # penalizes |grad m| not |grad (m - m_ref)|, so sharp boundaries in m0
    # are still penalized by smoothness; smallness is what anchors structure)
    m_ref = m0.copy()

    dmis = data_misfit.L2DataMisfit(simulation=sim, data=data_obj)
    # alpha_s small, smoothness alphas at 1. This downweights "smallness"
    # (pull toward reference) and instead penalizes spatial roughness,
    # discouraging compact / cell-sized blobs in favor of extended targets.
    reg = regularization.WeightedLeastSquares(
        global_mesh, active_cells=active_cells, reference_model=m_ref,
        alpha_s=0.1, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
    )

    lower = np.full(n_active, np.log(1e-6))
    upper = np.full(n_active, np.log(1e2))
    opt = optimization.ProjectedGNCG(
        maxIter=15, lower=lower, upper=upper, cg_maxiter=40,
    )
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10, n_pw_iter=2)
    cool_beta = directives.BetaSchedule(coolingFactor=2, coolingRate=2)
    save_iteration = directives.SaveOutputDictEveryIteration(saveOnDisk=True)
    target_misfit = directives.TargetMisfit()

    inv = inversion.BaseInversion(
        inv_prob,
        [starting_beta, cool_beta, save_iteration, target_misfit],
    )

    print(f"\ntarget misfit (= N_data): {n_data}", flush=True)
    print("starting inversion ...", flush=True)
    mrec = inv.run(m0)

    np.save("_mrec_full.npy", mrec)
    print(f"\nsaved -> _mrec_full.npy", flush=True)
    print(f"recovered sigma range: [{np.exp(mrec).min():.3e}, "
          f"{np.exp(mrec).max():.3e}] S/m", flush=True)


if __name__ == "__main__":
    main()
