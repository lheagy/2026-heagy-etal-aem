"""Baseline A': cold-started full-mesh inversion WITH sensitivity weighting.

Identical to _test_full_coldstart.py in every respect (mesh, tiled simulation,
data, uncertainties, beta schedule, optimizer, bounds, m0 = m_ref = halfspace)
except the regularization carries sensitivity weights. The question: does the
cold start still fail once it is regularized the way a standard 3D inversion
would be, or was the "donut" in Figure 2c an artifact of the unweighted
smallness term?

`UpdateSensitivityWeights` cannot be used: it needs `sim.getJtJdiag`, which
`Simulation3DElectricField` does not implement. The weights come instead from a
Hutchinson estimate of diag((WJ)^T(WJ)) built from `Jtvec` -- see
_sensitivity_weights.py, validated against an exact single-source Jacobian in
_validate_senswt.py (Pearson r ~0.99 on log10 wr).

SURVEY: the 80 m / 50-sounding survey, matching the published two-stage and
cold-start runs (both have 1000 data; the cold start stalls at phi_d = 20792).
NOTE `_test_parametric.rx_locs` is the 40 m / 100-sounding survey used by the
stitched-1D baseline, and `_dobs_cache.npy` is its 2000-point data -- the
existing 3D scripts have drifted onto those. This script builds the 80 m survey
explicitly so it is comparable to Figure 2c.

Run:
    python _test_coldstart_senswt.py --jtj-only     # estimate weights, stop
    python _test_coldstart_senswt.py                # full inversion
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import time

import numpy as np

import discretize
from simpeg import (
    maps, Data, data_misfit, inverse_problem,
    regularization, optimization, directives, inversion,
)
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver
from simpeg.meta import MultiprocessingMetaSimulation

from _sensitivity_weights import estimate_jtj_diag, sensitivity_weights
from _test_parametric import (
    build_global_mesh, build_local_meshes, sigma_back, rx_y, rx_times,
    TIME_STEPS,
)

JTJ_CACHE = "_jtj_coldstart_senswt.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jtj-only", action="store_true",
                    help="estimate the JtJ diagonal, save it, and stop")
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--clip", type=float, default=3e-3,
                    help="weight floor as a fraction of the maximum")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--recompute-jtj", action="store_true")
    # beta0_ratio=1e5 (used by the unweighted runs) was tuned against the
    # UNWEIGHTED reg.deriv2. Sensitivity weighting changes that operator's
    # spectrum, and BetaEstimate_ByEig works from the ratio of largest
    # eigenvalues, so the same ratio gave beta_1 = 47.9 here versus 0.144 for
    # the unweighted cold start -- a 333x stronger damping term that froze the
    # Gauss-Newton step (phi_d flat at ~91,800 for 4 iterations, |dm| ~ 0.01;
    # archived in _iters_coldstart_senswt_beta1e5_stalled/). 1e5/333 = 300
    # starts this run at the same regularization strength as the unweighted
    # cold start, so the two differ only in the SHAPE of the regularization.
    ap.add_argument("--beta0-ratio", type=float, default=300.0)
    # alpha_s: 0.01 (inherited from the warm-start runs) turned out to
    # contribute 80% of phi_m at the recovered model, because smallness acts on
    # (m - m_ref) -- several log-units -- while smoothness acts on gradients.
    # Worse, the sensitivity weights multiply the smallness term too, so the
    # target band was pinned to the background halfspace 6x more firmly than
    # the deep zone (anchor alpha_s*w_r*vol: 3.09 vs 0.51). That is why the
    # target region sat at background for all 30 iterations of the first run.
    ap.add_argument("--alpha-s", type=float, default=1e-4)
    ap.add_argument("--wr-exponent", type=float, default=1.0,
                    help="soften the weights; 0.5 = 4th root overall")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, e.g. '_as1e-4'")
    args = ap.parse_args()

    Solver = get_default_solver()

    global_mesh = build_global_mesh()
    active_cells = global_mesh.cell_centers[:, 2] < 0
    n_active = int(active_cells.sum())
    vol = global_mesh.cell_volumes[active_cells]
    print(f"global mesh: {global_mesh.n_cells} cells ({n_active} active)",
          flush=True)

    # --- 80 m survey, 50 soundings (matches the published 3D runs) ----------
    rx_x_80 = (np.linspace(-500, 500, 26))[3:-3][::2]
    rx_locs = discretize.utils.ndgrid([rx_x_80, rx_y, np.r_[30.0]])
    source_list = [
        tdem.sources.CircularLoop(
            receiver_list=[tdem.receivers.PointMagneticFluxTimeDerivative(
                loc, rx_times, orientation="z")],
            location=loc, orientation="z", radius=10,
            waveform=tdem.sources.StepOffWaveform(),
        )
        for loc in rx_locs
    ]
    survey = tdem.Survey(source_list)
    n_data = len(source_list) * len(rx_times)
    print(f"survey: {len(source_list)} sources x {len(rx_times)} times "
          f"= {n_data} data (80 m spacing)", flush=True)

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

    dobs = np.load("_dobs_80m.npy")
    assert dobs.size == n_data, f"dobs size {dobs.size} != {n_data}"

    relative_error = 0.05
    noise_floor = 1e-12
    std = np.abs(dobs) * relative_error + noise_floor
    data_obj = Data(survey, dobs=dobs, standard_deviation=std)

    m0 = np.full(n_active, np.log(sigma_back))
    m_ref = m0.copy()

    # --- sensitivity weights ------------------------------------------------
    if os.path.exists(JTJ_CACHE) and not args.recompute_jtj:
        jtj = np.load(JTJ_CACHE)
        assert jtj.size == n_active, f"cached jtj {jtj.size} != {n_active}"
        print(f"loaded {JTJ_CACHE}", flush=True)
    else:
        print(f"estimating diag(JtJ) at m0 with {args.n_probe} probes ...",
              flush=True)
        t0 = time.perf_counter()
        jtj = estimate_jtj_diag(sim, m0, 1.0 / std, n_probe=args.n_probe,
                                seed=0)
        print(f"  done in {time.perf_counter()-t0:.0f} s", flush=True)
        np.save(JTJ_CACHE, jtj)

    wr_raw = np.sqrt(np.maximum(jtj, 0.0) / vol**2)
    wr_raw /= wr_raw.max()
    print(f"\nunclipped wr: min {wr_raw.min():.3e}, "
          f"median {np.median(wr_raw):.3e}, max {wr_raw.max():.3e}")
    for q in (1, 5, 25, 50, 75, 95):
        print(f"  p{q:<2d} {np.percentile(wr_raw, q):.3e}")
    wr = sensitivity_weights(jtj, vol, clip=args.clip,
                             exponent=args.wr_exponent)
    print(f"clip={args.clip:g} -> {100*np.mean(wr <= 1.001*args.clip):.1f}% "
          f"of cells at the floor", flush=True)
    np.save(f"_wr_coldstart_senswt{args.tag}.npy", wr)

    if args.jtj_only:
        print("\n--jtj-only: stopping before the inversion.")
        sim.join()
        return

    # --- inversion (everything below identical to _test_full_coldstart) -----
    dmis = data_misfit.L2DataMisfit(simulation=sim, data=data_obj)
    reg = regularization.WeightedLeastSquares(
        global_mesh, active_cells=active_cells, reference_model=m_ref,
        alpha_s=args.alpha_s, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        weights={"sensitivity": wr},
    )
    print(f"alpha_s = {args.alpha_s:g}  (alpha_x/y/z = 1)", flush=True)

    lower = np.full(n_active, np.log(1e-6))
    upper = np.full(n_active, np.log(1e2))
    opt = optimization.ProjectedGNCG(
        maxIter=args.max_iter, lower=lower, upper=upper, cg_maxiter=40,
        tolF=1e-10, tolX=1e-10,
    )
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    inv = inversion.BaseInversion(inv_prob, [
        directives.BetaEstimate_ByEig(beta0_ratio=args.beta0_ratio, n_pw_iter=2),
        directives.BetaSchedule(coolingFactor=2, coolingRate=2),
        directives.SaveOutputDictEveryIteration(saveOnDisk=True),
        directives.TargetMisfit(),
    ])

    print(f"\ntarget misfit (= N_data): {n_data}", flush=True)
    print("starting cold-start + sensitivity-weighted inversion ...", flush=True)
    t0 = time.perf_counter()
    mrec = inv.run(m0)
    print(f"\ntotal: {(time.perf_counter()-t0)/60:.1f} min", flush=True)

    out = f"_mrec_coldstart_senswt{args.tag}.npy"
    np.save(out, mrec)
    print(f"saved -> {out}", flush=True)
    print(f"recovered sigma: [{np.exp(mrec).min():.3e}, "
          f"{np.exp(mrec).max():.3e}] S/m  (true target 5 S/m)", flush=True)
    sim.join()


if __name__ == "__main__":
    main()
