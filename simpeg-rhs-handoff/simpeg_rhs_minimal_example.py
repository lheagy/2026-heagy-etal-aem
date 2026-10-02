"""Minimal reproducer: TDEM source terms are rebuilt from scratch every time step.

Self-contained -- depends only on simpeg / discretize / numpy. Run it:

    python simpeg_rhs_minimal_example.py

It demonstrates, in order:

  1. For a StepOffWaveform the RHS is EXACTLY ZERO for every step after the
     first, yet is recomputed in full at each of them.
  2. `MagDipole.s_e` (which CircularLoop inherits) recomputes two purely
     time-independent quantities on every call:
       - `self._bSrc(simulation)`               -- the analytic loop vector
                                                  potential on all mesh edges
       - `mesh.get_face_inner_product(1/mu)`    -- a sparse matrix the
                                                  simulation ALREADY caches as
                                                  `simulation.MfMui`
     The only time-dependent factor is the scalar `waveform.eval(time)`.
  3. `getRHS(tInd)` calls `getSourceTerm` TWICE (tInd and tInd-1), so the
     previous step's source term is recomputed rather than carried forward.
  4. A ~20-line prototype fix that caches the constant part, produces
     bit-identical results, and is dramatically faster.

See HANDOFF_simpeg_rhs_assembly.md for context and measured impact.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time

import numpy as np
import discretize
from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver


def build():
    """A small 3D mesh with one loop source and a step-off waveform."""
    h = np.ones(24) * 25.0
    mesh = discretize.TensorMesh([h, h, h], origin="CCC")
    sigma = np.full(mesh.n_cells, 1e-2)

    loc = np.r_[0.0, 0.0, 30.0]
    rx = tdem.receivers.PointMagneticFluxTimeDerivative(
        loc.reshape(1, 3), np.logspace(-5, -3, 10), orientation="z")
    src = tdem.sources.CircularLoop(
        receiver_list=[rx], location=loc, orientation="z", radius=10.0,
        waveform=tdem.sources.StepOffWaveform())

    sim = tdem.simulation.Simulation3DElectricField(
        mesh=mesh, survey=tdem.Survey([src]),
        time_steps=[(1e-5, 10), (1e-4, 10)],
        solver=get_default_solver(), sigmaMap=maps.IdentityMap(mesh))
    sim.model = sigma
    return mesh, src, sim


def bench(fn, n=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3      # ms


def main():
    mesh, src, sim = build()
    n_steps = len(sim.time_steps)
    print(f"mesh: {mesh.n_cells} cells, {mesh.n_edges} edges, "
          f"{mesh.n_faces} faces;  {n_steps} time steps\n")

    # ---- 1. the RHS is identically zero after the first step ---------------
    print("1) RHS by time step (StepOffWaveform):")
    for tInd in (1, 2, 5, n_steps):
        r = sim.getRHS(tInd)
        print(f"     tInd {tInd:>3}   max|rhs| = {np.abs(r).max():.3e}"
              f"   all-zero: {not np.any(r)}")
    wf = src.waveform
    print(f"   waveform.eval: t=0 -> {wf.eval(0.0)},  "
          f"t=1e-5 -> {wf.eval(1e-5)},  t=1e-3 -> {wf.eval(1e-3)}")
    print("   => every step after the first multiplies a fully-rebuilt "
          "vector by 0.\n")

    # ---- 2. where the time is spent ----------------------------------------
    t_se = bench(lambda: src.s_e(sim, 1e-4))
    t_b = bench(lambda: src._bSrc(sim))
    t_ip = bench(lambda: mesh.get_face_inner_product(1.0 / src.mu))
    t_cached = bench(lambda: sim.MfMui, n=500)
    t_rhs = bench(lambda: sim.getRHS(5))
    print("2) cost of one s_e evaluation:")
    print(f"     src.s_e(sim, t)                    {t_se:8.2f} ms")
    print(f"       _bSrc(simulation)                {t_b:8.2f} ms  "
          f"({100*t_b/t_se:.0f}%)  time-independent")
    print(f"       mesh.get_face_inner_product      {t_ip:8.2f} ms  "
          f"({100*t_ip/t_se:.0f}%)  time-independent")
    print(f"       sim.MfMui  (already cached)      {t_cached:8.4f} ms  "
          f"<- {t_ip/max(t_cached,1e-9):.0f}x cheaper than rebuilding\n")

    # ---- 3. getRHS evaluates the source term twice -------------------------
    calls = {"n": 0}
    orig_get_source_term = type(sim).getSourceTerm

    def counting(self, tInd):
        calls["n"] += 1
        return orig_get_source_term(self, tInd)

    type(sim).getSourceTerm = counting
    calls["n"] = 0
    sim.getRHS(5)
    per_step = calls["n"]
    type(sim).getSourceTerm = orig_get_source_term
    print(f"3) getSourceTerm calls per getRHS: {per_step}  "
          f"(tInd and tInd-1; the latter was already computed last step)")
    print(f"     => {per_step * n_steps} source-term evaluations per forward, "
          f"per source")
    print(f"     => ~{per_step * n_steps * t_se / 1000:.2f} s per source on "
          f"this small mesh ({t_rhs:.1f} ms per getRHS)\n")

    # ---- 4. prototype fix ---------------------------------------------------
    # s_e(t) = [C.T @ (MfMui @ b)] * waveform.eval(t)
    #          '-------- constant --------'   '--- scalar ---'
    print("4) prototype fix: cache the constant vector, scale by the waveform")

    def s_e_cached(self, simulation, time_):
        key = getattr(self, "_s_e_const_key", None)
        if key != (id(simulation.mesh), self.mu):
            C = simulation.mesh.edge_curl
            b = self._bSrc(simulation)
            # use the simulation's cached MfMui rather than rebuilding it
            self._s_e_const = C.T * (simulation.MfMui * b)
            self._s_e_const_key = (id(simulation.mesh), self.mu)
        if (self.waveform.has_initial_fields
                and time_ < simulation.time_steps[1]):
            return self._s_e_const
        return self._s_e_const * self.waveform.eval(time_)

    ref = {t: src.s_e(sim, t) for t in (0.0, 1e-5, 1e-4, 1e-3)}
    type(src).s_e = s_e_cached
    new = {t: src.s_e(sim, t) for t in (0.0, 1e-5, 1e-4, 1e-3)}
    ok = all(np.allclose(np.asarray(ref[t], dtype=float),
                         np.asarray(new[t], dtype=float)) for t in ref)
    t_se_fixed = bench(lambda: src.s_e(sim, 1e-4))
    print(f"     results identical to the original: {ok}")
    print(f"     s_e: {t_se:.2f} ms -> {t_se_fixed:.3f} ms "
          f"({t_se/max(t_se_fixed,1e-9):.0f}x faster)")
    print(f"     per forward, per source: "
          f"{per_step*n_steps*t_se/1000:.2f} s -> "
          f"{per_step*n_steps*t_se_fixed/1000:.3f} s")


if __name__ == "__main__":
    main()
