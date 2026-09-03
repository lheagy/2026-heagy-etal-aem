"""Forward-simulation timing/RAM for a single sounding:
  python _timing_test.py tiled     # one source on its local (tiled) mesh
  python _timing_test.py global    # one source on the full global mesh

Times ONLY the dpred call (all mesh/sim/model construction happens first and is
not timed). Peak RAM is sampled (process RSS) during the dpred call. Emits one
JSON line on stdout. Single-threaded (OMP/MKL=1) for a reproducible comparison.
"""
import os
import sys
# optional 2nd arg = thread count (default 1). Must be set before numpy/MKL load.
_threads = sys.argv[2] if len(sys.argv) > 2 else "1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _threads

import time
import json
import threading
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver

from _test_parametric import (
    build_global_mesh, build_local_meshes, build_true_model,
    rx_locs, rx_times, TIME_STEPS,
)


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


class RSSSampler(threading.Thread):
    """Poll process RSS in the background; record the peak."""
    def __init__(self, interval=0.005):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0.0
        self._run = True

    def run(self):
        while self._run:
            self.peak = max(self.peak, rss_mb())
            time.sleep(self.interval)

    def stop(self):
        self._run = False
        self.join()


def main():
    approach = sys.argv[1]
    Solver = get_default_solver()

    # ---- setup (NOT timed) ----
    gm = build_global_mesh()
    active = gm.cell_centers[:, 2] < 0
    sigma_true = build_true_model(gm)
    m_global = np.log(sigma_true[active])      # global-active log-sigma

    # single central sounding (y = 0, x nearest 0)
    xs = np.unique(rx_locs[:, 0])
    xc = xs[np.argmin(np.abs(xs))]
    s_ind = int(np.where((rx_locs[:, 0] == xc) & (rx_locs[:, 1] == 0.0))[0][0])
    loc = rx_locs[s_ind]
    rx = tdem.receivers.PointMagneticFluxTimeDerivative(loc, rx_times, orientation="z")
    src = tdem.sources.CircularLoop(
        receiver_list=[rx], location=loc, orientation="z", radius=10,
        waveform=tdem.sources.StepOffWaveform(),
    )
    survey = tdem.Survey([src])

    if approach == "tiled":
        mesh = build_local_meshes(gm, survey)[0]
        tile = maps.TileMap(gm, active, mesh)
        actmap = maps.InjectActiveCells(
            mesh, active_cells=tile.local_active, value_inactive=np.log(1e-8))
        m = tile * m_global
        n_active = int(tile.local_active.sum())
    elif approach == "global":
        mesh = gm
        actmap = maps.InjectActiveCells(gm, active, value_inactive=np.log(1e-8))
        m = m_global
        n_active = int(active.sum())
    else:
        raise SystemExit("approach must be 'tiled' or 'global'")

    sim = tdem.simulation.Simulation3DElectricField(
        mesh=mesh, survey=survey, time_steps=TIME_STEPS,
        solver=Solver, sigmaMap=maps.ExpMap() * actmap,
    )

    # _test_parametric pins MKL=1 at import; force the requested count at
    # runtime with threadpool_limits (overrides the env) and confirm it.
    from threadpoolctl import threadpool_limits, threadpool_info

    base_rss = rss_mb()
    sampler = RSSSampler()
    sampler.start()
    times = []
    mkl_threads = None
    with threadpool_limits(limits=int(_threads)):
        mkls = [p for p in threadpool_info() if p.get("internal_api") == "mkl"]
        mkl_threads = mkls[0]["num_threads"] if mkls else int(_threads)
        for _ in range(3):
            sim.model = None             # invalidate any cached fields
            t0 = time.perf_counter()
            d = sim.dpred(m)
            times.append(time.perf_counter() - t0)
    sampler.stop()

    print(json.dumps({
        "approach": approach,
        "threads_requested": int(_threads),
        "threads_mkl_active": mkl_threads,
        "mesh_cells": int(mesh.n_cells),
        "mesh_active": n_active,
        "n_time_steps": int(sum(n for _, n in TIME_STEPS)),
        "n_data": int(len(d)),
        "dpred_times_s": [round(t, 3) for t in times],
        "dpred_time_median_s": round(float(np.median(times)), 3),
        "rss_baseline_mb": round(base_rss, 1),
        "rss_peak_mb": round(sampler.peak, 1),
        "rss_dpred_increment_mb": round(sampler.peak - base_rss, 1),
    }))


if __name__ == "__main__":
    main()
