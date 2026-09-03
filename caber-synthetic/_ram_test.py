"""Measure total RAM of the tiled forward vs. the global forward.

The tiled approach trades a small *per-solve* footprint for holding many local
meshes at once plus one factorization per concurrent worker. This script
measures the honest aggregate: peak PSS (proportional set size) summed across
the parent process and ALL forked pool workers, so copy-on-write sharing is
counted once, not once per worker.

Usage:
  python _ram_test.py tiled   [n_processes]   # 100 local meshes, tiled dpred
  python _ram_test.py global                   # one global-mesh dpred

Run with the 40 m survey config in _test_parametric.py (100 soundings).
Writes _ram_tiled.json / _ram_global.json.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import time
import threading

import numpy as np
import psutil

from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver
from simpeg.meta import MultiprocessingMetaSimulation

from _test_parametric import (
    build_global_mesh, build_local_meshes,
    rx_locs, rx_times, sigma_back, TIME_STEPS,
)

MB = 1024.0 ** 2


def tree_pss(proc):
    """Sum PSS (MB) over a process and all its descendants. PSS divides each
    shared page by the number of processes sharing it, so forked workers that
    share read-only pages with the parent are not double-counted."""
    procs = [proc] + proc.children(recursive=True)
    total = 0.0
    for p in procs:
        try:
            total += p.memory_full_info().pss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / MB


class PeakSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0.0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            self.peak = max(self.peak, tree_pss(self.proc))
            self._stop_evt.wait(self.interval)

    def sample_now(self):
        v = tree_pss(self.proc)
        self.peak = max(self.peak, v)
        return v

    def stop(self):
        self._stop_evt.set()
        self.join()


def build_survey():
    source_list = []
    for i in range(rx_locs.shape[0]):
        loc = rx_locs[i, :]
        rx = tdem.receivers.PointMagneticFluxTimeDerivative(
            loc, rx_times, orientation="z")
        src = tdem.sources.CircularLoop(
            receiver_list=[rx], location=loc, orientation="z", radius=10,
            waveform=tdem.sources.StepOffWaveform())
        source_list.append(src)
    return tdem.Survey(source_list)


def run_tiled(n_processes):
    Solver = get_default_solver()
    sampler = PeakSampler()
    base = sampler.sample_now()

    mesh = build_global_mesh()
    active = mesh.cell_centers[:, 2] < 0
    survey = build_survey()

    mesh_list = build_local_meshes(mesh, survey)
    after_meshes = sampler.sample_now()
    local_cells = [lm.n_cells for lm in mesh_list]

    mappings, sims = [], []
    for ii, lm in enumerate(mesh_list):
        tile = maps.TileMap(mesh, active, lm)
        lactmap = maps.InjectActiveCells(
            lm, active_cells=tile.local_active, value_inactive=np.log(1e-8))
        mappings.append(tile)
        sims.append(tdem.simulation.Simulation3DElectricField(
            mesh=lm, survey=tdem.Survey([survey.source_list[ii]]),
            time_steps=TIME_STEPS, solver=Solver,
            sigmaMap=maps.ExpMap() * lactmap))
    sim = MultiprocessingMetaSimulation(sims, mappings, n_processes=n_processes)
    after_build = sampler.sample_now()
    n_proc = len(getattr(sim, "_sim_processes", []) or [])
    if not n_proc:
        n_proc = n_processes

    m = np.log(np.full(int(active.sum()), sigma_back))
    sampler.start()
    t0 = time.time()
    dpred = sim.dpred(m)            # first solve: factorizations alive in workers
    sim.dpred(m)                    # second pass to be sure peak is captured
    dt = time.time() - t0
    sim.join()
    sampler.stop()

    out = {
        "mode": "tiled",
        "n_sources": int(rx_locs.shape[0]),
        "n_processes": int(n_proc) if n_proc else None,
        "global_cells": int(mesh.n_cells),
        "local_cells_total": int(sum(local_cells)),
        "local_cells_mean": float(np.mean(local_cells)),
        "local_cells_max": int(max(local_cells)),
        "baseline_mb": round(base, 1),
        "after_meshes_mb": round(after_meshes, 1),
        "meshes_only_mb": round(after_meshes - base, 1),
        "after_build_mb": round(after_build, 1),
        "peak_mb": round(sampler.peak, 1),
        "peak_over_baseline_mb": round(sampler.peak - base, 1),
        "dpred_time_s": round(dt, 1),
        "n_data": int(dpred.size),
    }
    json.dump(out, open("_ram_tiled.json", "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


def run_global():
    Solver = get_default_solver()
    sampler = PeakSampler()
    base = sampler.sample_now()

    mesh = build_global_mesh()
    active = mesh.cell_centers[:, 2] < 0
    survey = build_survey()
    after_mesh = sampler.sample_now()

    actmap = maps.InjectActiveCells(mesh, active_cells=active,
                                    value_inactive=np.log(1e-8))
    sim = tdem.simulation.Simulation3DElectricField(
        mesh=mesh, survey=survey, time_steps=TIME_STEPS, solver=Solver,
        sigmaMap=maps.ExpMap() * actmap)

    m = np.log(np.full(int(active.sum()), sigma_back))
    sampler.start()
    t0 = time.time()
    dpred = sim.dpred(m)
    dt = time.time() - t0
    sampler.stop()

    out = {
        "mode": "global",
        "n_sources": int(rx_locs.shape[0]),
        "global_cells": int(mesh.n_cells),
        "baseline_mb": round(base, 1),
        "after_mesh_mb": round(after_mesh, 1),
        "peak_mb": round(sampler.peak, 1),
        "peak_over_baseline_mb": round(sampler.peak - base, 1),
        "dpred_time_s": round(dt, 1),
        "n_data": int(dpred.size),
    }
    json.dump(out, open("_ram_global.json", "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiled"
    if mode == "tiled":
        npx = int(sys.argv[2]) if len(sys.argv) > 2 else None
        run_tiled(npx)
    else:
        run_global()
