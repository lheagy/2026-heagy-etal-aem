"""Per-time-step cost breakdown for one global-mesh TDEM forward.

Instruments `Simulation3DElectricField.fields()` to record, for every one of the
80 time steps: matrix assembly, factorization (only when dt changes), and the
solve. Writes JSON; `_build_timing_breakdown_report.py` turns it into a table.

The solver (pymatsolver Pardiso) is constructed with `factor=False` and
factorizes lazily inside the first solve, so the wrapper calls the underlying
`MKLPardisoSolver._factor()` explicitly to separate factorization from solve.
Verified this changes nothing numerically (residual stays at ~1e-14) and that
the isolated cost matches what the lazy first solve would otherwise absorb.

Run: python _timing_breakdown.py --n-src 1 --n-threads 1
"""
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--n-src", type=int, default=1)
ap.add_argument("--n-threads", type=int, default=1)
ap.add_argument("--repeat", type=int, default=3,
                help="timed runs after one warm-up (default 3)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

# must be set before numpy / MKL import
os.environ["OMP_NUM_THREADS"] = str(args.n_threads)
os.environ["MKL_NUM_THREADS"] = str(args.n_threads)

import json
import time

import numpy as np
from pydiso.mkl_solver import set_mkl_pardiso_threads, get_mkl_pardiso_max_threads

from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver

from _test_parametric import (
    build_global_mesh, build_true_model, rx_times, rx_y, TIME_STEPS,
)

BASE_SOLVER = get_default_solver()

REC = {"factor": [], "solve": [], "assemble_A": [], "assemble_sub": [],
       "assemble_rhs": []}


class TimedSolver(BASE_SOLVER):
    """Records analyze+factorize on construction and every subsequent solve.

    Subclasses (rather than wraps) the Pardiso solver because SimPEG requires
    `sim.solver` to be a pymatsolver BaseSolver subclass. `Base.__mul__`
    delegates to `__matmul__`, so overriding `__matmul__` catches the
    `Ainv * rhs` call in the time-stepping loop.
    """

    def __init__(self, A, **kwargs):
        t0 = time.perf_counter()
        super().__init__(A, **kwargs)
        self.solver._factor()             # force numeric factorization now
        REC["factor"].append(time.perf_counter() - t0)

    def __matmul__(self, rhs):
        t0 = time.perf_counter()
        out = super().__matmul__(rhs)
        REC["solve"].append(time.perf_counter() - t0)
        return out


class TimedSim(tdem.Simulation3DElectricField):
    def getAdiag(self, tInd):
        t0 = time.perf_counter()
        out = super().getAdiag(tInd)
        REC["assemble_A"].append(time.perf_counter() - t0)
        return out

    def getAsubdiag(self, tInd):
        t0 = time.perf_counter()
        out = super().getAsubdiag(tInd)
        REC["assemble_sub"].append(time.perf_counter() - t0)
        return out

    def getRHS(self, tInd):
        t0 = time.perf_counter()
        out = super().getRHS(tInd)
        REC["assemble_rhs"].append(time.perf_counter() - t0)
        return out


def main():
    set_mkl_pardiso_threads(args.n_threads)

    mesh = build_global_mesh()
    active = mesh.cell_centers[:, 2] < 0
    actmap = maps.InjectActiveCells(mesh, active, value_inactive=np.log(1e-8))
    sigma_true = build_true_model(mesh)
    m_true = np.log(sigma_true[active])

    # sources: the 80 m survey, truncated to --n-src (1 = the station over the
    # target, matching _forward_timing.md)
    rx_x_80 = (np.linspace(-500, 500, 26))[3:-3][::2]
    all_locs = np.array([[x, y, 30.0] for y in rx_y for x in rx_x_80])
    if args.n_src == 1:
        locs = np.array([[20.0, 0.0, 30.0]])
    else:
        locs = all_locs[: args.n_src]

    source_list = [
        tdem.sources.CircularLoop(
            receiver_list=[tdem.receivers.PointMagneticFluxTimeDerivative(
                loc, rx_times, orientation="z")],
            location=loc, orientation="z", radius=10,
            waveform=tdem.sources.StepOffWaveform(),
        )
        for loc in locs
    ]

    sim = TimedSim(
        mesh=mesh, survey=tdem.Survey(source_list), time_steps=TIME_STEPS,
        solver=TimedSolver, sigmaMap=maps.ExpMap() * actmap,
    )

    print(f"global mesh {mesh.n_cells} cells ({int(active.sum())} active), "
          f"{mesh.n_edges} edges", flush=True)
    print(f"{len(source_list)} source(s), {len(sim.time_steps)} time steps, "
          f"pardiso threads = {get_mkl_pardiso_max_threads()}", flush=True)

    # Warm-up: the first fields() call also builds discretize's mesh operators
    # (edge curl, inner-product matrices), which are then cached on the mesh.
    # _forward_timing.md reports the median of repeated calls, i.e. warm; time
    # the warm-up separately so the one-time cost is visible rather than hidden.
    t0 = time.perf_counter()
    f = sim.fields(m_true)
    t_coldstart = time.perf_counter() - t0
    print(f"cold first call (includes building mesh operators): "
          f"{t_coldstart:.2f} s", flush=True)

    runs = []
    for i in range(args.repeat):
        for k in REC:
            REC[k].clear()
        sim.model = None
        t0 = time.perf_counter()
        f = sim.fields(m_true)
        t_fields = time.perf_counter() - t0
        runs.append({"t_fields": t_fields,
                     **{f"t_{k}": list(v) for k, v in REC.items()}})
        print(f"  run {i+1}/{args.repeat}: {t_fields:.2f} s "
              f"(factor {sum(REC['factor']):.2f}, solve {sum(REC['solve']):.2f})",
              flush=True)

    # report the median run
    med = sorted(runs, key=lambda r: r["t_fields"])[len(runs) // 2]
    t_fields = med["t_fields"]
    for k in REC:
        REC[k] = med[f"t_{k}"]

    t0 = time.perf_counter()
    dpred = sim.dpred(m_true, f=f)
    t_dpred_interp = time.perf_counter() - t0

    dt_per_step = [float(dt) for dt in sim.time_steps]
    out = {
        "t_coldstart": t_coldstart,
        "runs_t_fields": [r["t_fields"] for r in runs],
        "repeat": args.repeat,
        "n_src": len(source_list),
        "n_threads": args.n_threads,
        "pardiso_threads": int(get_mkl_pardiso_max_threads()),
        "n_cells": int(mesh.n_cells),
        "n_active": int(active.sum()),
        "n_edges": int(mesh.n_edges),
        "time_steps": [[float(dt), int(n)] for dt, n in TIME_STEPS],
        "dt_per_step": dt_per_step,
        "t_fields_total": t_fields,
        "t_receiver_interp": t_dpred_interp,
        "n_data": int(dpred.size),
        **{f"t_{k}": v for k, v in REC.items()},
    }
    name = args.out or f"_timing_breakdown_src{len(source_list)}_thr{args.n_threads}.json"
    with open(name, "w") as fh:
        json.dump(out, fh, indent=2)

    tf, ts = sum(REC["factor"]), sum(REC["solve"])
    ta = sum(REC["assemble_A"]) + sum(REC["assemble_sub"]) + sum(REC["assemble_rhs"])
    print(f"\nfields()            {t_fields:8.2f} s")
    print(f"  factorizations    {tf:8.2f} s  ({len(REC['factor'])}x, "
          f"{100*tf/t_fields:.1f}%)")
    print(f"  solves            {ts:8.2f} s  ({len(REC['solve'])}x, "
          f"{100*ts/t_fields:.1f}%)")
    print(f"  matrix assembly   {ta:8.2f} s  ({100*ta/t_fields:.1f}%)")
    print(f"  other/overhead    {t_fields-tf-ts-ta:8.2f} s")
    print(f"receiver interp     {t_dpred_interp:8.2f} s")
    print(f"TOTAL forward       {t_fields+t_dpred_interp:8.2f} s  "
          f"-> {dpred.size} data")
    print("wrote", name)


if __name__ == "__main__":
    main()
