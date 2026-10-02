"""Validate the Hutchinson JtJ-diagonal estimator against the exact answer.

Single sounding => J is 20 x n_active, so every row can be built with 20 Jtvec
calls and diag(J^T J) formed exactly. Compare the Hutchinson estimate at
K = 8/16/32/64 against it, on the quantity that actually matters (the weights
`wr`, not the raw diagonal).

Run: python _validate_senswt.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time

import numpy as np

from simpeg import maps, Data
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver
from simpeg.meta import MetaSimulation

from _sensitivity_weights import (
    estimate_jtj_diag, exact_jtj_diag, sensitivity_weights,
)
from _test_parametric import (
    build_global_mesh, build_local_meshes, build_true_model,
    sigma_back, rx_times, TIME_STEPS,
)


def main():
    Solver = get_default_solver()
    global_mesh = build_global_mesh()
    active = global_mesh.cell_centers[:, 2] < 0
    n_active = int(active.sum())
    vol = global_mesh.cell_volumes[active]

    loc = np.r_[20.0, 0.0, 30.0]            # one sounding over the target
    src = tdem.sources.CircularLoop(
        receiver_list=[tdem.receivers.PointMagneticFluxTimeDerivative(
            loc, rx_times, orientation="z")],
        location=loc, orientation="z", radius=10,
        waveform=tdem.sources.StepOffWaveform(),
    )
    survey = tdem.Survey([src])

    local_mesh = build_local_meshes(global_mesh, survey)[0]
    tile = maps.TileMap(global_mesh, active, local_mesh)
    local_actmap = maps.InjectActiveCells(
        local_mesh, active_cells=tile.local_active, value_inactive=np.log(1e-8))
    sub = tdem.simulation.Simulation3DElectricField(
        mesh=local_mesh, survey=survey, time_steps=TIME_STEPS, solver=Solver,
        sigmaMap=maps.ExpMap() * local_actmap,
    )
    # serial MetaSimulation: same code path as the real run, no worker pool
    sim = MetaSimulation([sub], [tile])

    m0 = np.full(n_active, np.log(sigma_back))
    dobs = sim.dpred(m0)
    std = np.abs(dobs) * 0.05 + 1e-12
    W = 1.0 / std
    print(f"mesh {global_mesh.n_cells} cells ({n_active} active), "
          f"local {local_mesh.n_cells}; n_data = {dobs.size}\n", flush=True)

    t0 = time.perf_counter()
    jtj_exact = exact_jtj_diag(sim, m0, W, verbose=False)
    t_exact = time.perf_counter() - t0
    print(f"exact diag(JtJ): {t_exact:.1f} s for {dobs.size} rows "
          f"({t_exact/dobs.size:.1f} s/Jtvec)", flush=True)
    wr_exact = sensitivity_weights(jtj_exact, vol)

    print(f"\n{'K':>4}  {'time':>7}  {'pearson r':>10}  {'med |dwr|/wr':>13}  "
          f"{'p90 |dwr|/wr':>13}")
    print("-" * 56)
    for K in (8, 16, 32, 64):
        t0 = time.perf_counter()
        jtj_est = estimate_jtj_diag(sim, m0, W, n_probe=K, seed=0, verbose=False)
        dt = time.perf_counter() - t0
        wr_est = sensitivity_weights(jtj_est, vol)
        r = np.corrcoef(np.log10(wr_exact), np.log10(wr_est))[0, 1]
        rel = np.abs(wr_est - wr_exact) / wr_exact
        print(f"{K:>4}  {dt:>6.1f}s  {r:>10.4f}  {np.median(rel):>13.3f}  "
              f"{np.percentile(rel, 90):>13.3f}", flush=True)

    # sanity on the exact weights themselves
    cc = global_mesh.cell_centers[active]
    print(f"\nwr_exact: min {wr_exact.min():.3e}, max {wr_exact.max():.3e}")
    zc = cc[:, 2]
    for zlo, zhi in [(-50, 0), (-150, -100), (-300, -250), (-450, -400)]:
        sel = (zc >= zlo) & (zc < zhi)
        if sel.any():
            print(f"  z in [{zlo:5d}, {zhi:5d}): median wr "
                  f"{np.median(wr_exact[sel]):.4f}")
    np.save("_wr_exact_1src.npy", wr_exact)
    print("\nsaved _wr_exact_1src.npy")


if __name__ == "__main__":
    main()
