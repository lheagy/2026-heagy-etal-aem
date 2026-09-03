# Per-iteration wall-clock time — 3D inversions

Representative per-iteration timings for the three inversions, taken from the
saved `SaveOutputDictEveryIteration` file timestamps of the production runs
(ovbU5 geometry: 80 m survey, 50 sources tiled across the multiprocessing pool).

**Caveat:** unlike the single-sounding forward comparison (`_forward_timing.md`),
these are *not* controlled benchmarks — they come from the iteration timestamps
of the production runs, so treat them as representative / order-of-magnitude.

## Summary

| inversion | per-iteration time | total run |
|---|---|---|
| Phase-1 parametric | **~3 min/iter** (3–3.5) | ~13 min (4 iters) |
| Phase-2 two-stage 3D | **~2 min early → ~8 min late** (~4 min typical) | ~35 min (9 iters) |
| Cold-start 3D | **~8–9 min/iter** (sustained) | ~2 h (19 iters, not converged) |

## Why the per-iteration cost varies — CG iterations

The per-iteration cost of the full 3D inversions (`ProjectedGNCG`) is dominated
by the conjugate-gradient (CG) iterations of the Gauss–Newton subproblem; each
CG step is ≈ a forward + adjoint solve. Both 3D runs use `cg_maxiter = 40`.

- **Phase-2 (warm-started)** needs few CG iterations early because the
  parametric `m₀` is already close to a data-fitting model. The log shows CG
  iterations per Gauss–Newton step of **8, 7, 9, 9, 12, 21, 28, 35, 40** —
  ramping up only as β cools toward convergence. The per-iteration times track
  this: ~2 min/iter early (CG ≈ 8) up to ~8 min/iter at the end (CG = 40).
  - measured per-iter (s): 101, 126, 126, 163, 272, 356, 441, 504

- **Cold-start** starts from a uniform halfspace, far from any data-fitting
  model, so the CG subproblem is ill-conditioned from the outset and hits the
  cap of 40 almost immediately, staying there for the whole run. Each iteration
  is ~8.5 min — matching Phase-2's *final* (CG = 40) iteration cost. So the cold
  start is more expensive both because it takes more iterations to stall *and*
  because each iteration is ~2× costlier.
  - measured per-iter (s): 100, 113, 151, 224, 259, 346, 380, 506, 504, 506,
    512, 521, 516, 532, 514, 523, 525, 524 (plateau at ~8.5 min)

- **Phase-1 parametric** inverts only 11 parameters (last-5 channels), ~3 min/
  iter, converging in 4 iterations.
  - measured per-iter (s): 162, 209, 209

## Takeaway for the abstract

The parametric warm start reduces the full-3D cost in two ways: it cuts the
number of Gauss–Newton iterations to convergence (9 vs. the cold start never
converging in 19), and it keeps the CG count — and hence the per-iteration wall
time — low until the final β-cooling steps (~2–4 min/iter for most of the run,
vs. ~8–9 min/iter sustained for the cold start).

## Machine

Intel Xeon E5-2670 v3 @ 2.30 GHz (2 × 12 cores, 48 threads), 126 GB RAM, Linux
5.15; SimPEG 0.24.1.dev33, MKL Pardiso solver; tiled forward
(`MultiprocessingMetaSimulation`), single-threaded workers.
