"""Baseline A: cold-started full-mesh inversion.

Identical to _test_full.py (same regularization, beta schedule, optimizer,
bounds) except m0 = reference = uniform halfspace (sigma_back) instead of
the phase-1 parametric recovery. Paper baseline: what does the 3D inversion
do without the parametric warm start?

Writes _mrec_coldstart.npy.
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

from _test_parametric import (
    build_global_mesh,
    build_local_meshes,
    sigma_back,
    rx_locs,
    rx_times,
    TIME_STEPS,
)


def main():
    Solver = get_default_solver()

    global_mesh = build_global_mesh()
    active_cells = global_mesh.cell_centers[:, 2] < 0
    n_active = int(active_cells.sum())
    print(f"global mesh: {global_mesh.n_cells} cells ({n_active} active)", flush=True)

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
            time_steps=TIME_STEPS,
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

    # cold start: uniform halfspace, no parametric information
    m0 = np.full(n_active, np.log(sigma_back))
    m_ref = m0.copy()
    print(f"m0: uniform halfspace log({sigma_back})", flush=True)

    dmis = data_misfit.L2DataMisfit(simulation=sim, data=data_obj)
    reg = regularization.WeightedLeastSquares(
        global_mesh, active_cells=active_cells, reference_model=m_ref,
        alpha_s=0.01, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
    )

    lower = np.full(n_active, np.log(1e-6))
    upper = np.full(n_active, np.log(1e2))
    # maxIter capped at 25: the cold start stalls far above target (it is a
    # baseline meant to demonstrate non-convergence), and at 100 sources a
    # full 40 iters would not finish before the deadline. 25 iters is more
    # than enough to show it fails to recover the target.
    opt = optimization.ProjectedGNCG(
        maxIter=25, lower=lower, upper=upper, cg_maxiter=40,
        tolF=1e-10, tolX=1e-10,
    )
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e5, n_pw_iter=2)
    cool_beta = directives.BetaSchedule(coolingFactor=2, coolingRate=2)
    save_iteration = directives.SaveOutputDictEveryIteration(saveOnDisk=True)
    target_misfit = directives.TargetMisfit()

    inv = inversion.BaseInversion(
        inv_prob,
        [starting_beta, cool_beta, save_iteration, target_misfit],
    )

    print(f"\ntarget misfit (= N_data): {n_data}", flush=True)
    print("starting cold-start inversion ...", flush=True)
    mrec = inv.run(m0)

    np.save("_mrec_coldstart.npy", mrec)
    print(f"\nsaved -> _mrec_coldstart.npy", flush=True)
    print(f"recovered sigma range: [{np.exp(mrec).min():.3e}, "
          f"{np.exp(mrec).max():.3e}] S/m", flush=True)
    sim.join()


if __name__ == "__main__":
    main()
