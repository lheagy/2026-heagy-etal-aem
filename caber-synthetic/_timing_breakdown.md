# Forward-simulation cost breakdown: global mesh

Where the time actually goes in **one** `Simulation3DElectricField` forward on the global OcTree mesh. Instrumented in `_timing_breakdown.py`, which subclasses the Pardiso solver to time each factorization and each solve, and the simulation to time each matrix assembly.

- **Mesh:** 52,340 cells (42,284 active), **147,204 edges** -- the E-field system is 147,204 x 147,204.
- **Time stepping:** (3e-06 s x 20), (1e-05 s x 20), (3e-05 s x 20), (0.0001 s x 20) = **80 steps**, **4 unique step sizes**.
- A factorization is reused while `dt` is constant, so there are **4 factorizations and 80 solves**.
- Reported times are the **median of 3 warm runs** (a preceding warm-up call builds discretize's mesh operators, which are then cached on the mesh).

## 1. Single sounding, single thread

The reference case, matching `_forward_timing.md` (95.3 s here vs 96.0 s there).

| stage | calls | total (s) | per call | share |
|---|---:|---:|---:|---:|
| factorization | 4 | 71.80 | 18.0 s | 75.7% |
| solve | 80 | 13.08 | 164 ms | 13.8% |
| RHS assembly (`getRHS`) | 80 | 9.02 | 113 ms | 9.5% |
| system matrix (`getAdiag`) | 4 | 0.21 | 52 ms | 0.2% |
| sub-diagonal (`getAsubdiag`) | 80 | 0.04 | 0.5 ms | 0.0% |
| other / field storage | - | 0.73 | - | 0.8% |
| **`fields()` total** | | **94.88** | | |
| receiver interpolation | | 0.43 | | |
| **forward total** | | **95.31** | | |

Cold first call, including one-time mesh-operator construction: **117.4 s** (+23 s over a warm call).

## 2. Per time-step block

One factorization per block, reused by every step in it. Single sounding, single thread.

| block | dt | steps | t range (s) | factorization (s) | solve/step (s) | block solves (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3e-06 | 20 | 0.00e+00 - 6.00e-05 | 17.95 | 0.164 | 3.27 |
| 2 | 1e-05 | 20 | 6.00e-05 - 2.60e-04 | 18.06 | 0.164 | 3.27 |
| 3 | 3e-05 | 20 | 2.60e-04 - 8.60e-04 | 17.85 | 0.163 | 3.26 |
| 4 | 0.0001 | 20 | 8.60e-04 - 2.86e-03 | 17.94 | 0.164 | 3.28 |

Every factorization costs about the same (17.9-18.1 s) -- the sparsity pattern is identical across blocks, only the `dt` scaling of the mass term changes. **One factorization (18.0 s) costs about as much as 110 solves (164 ms each).**

## 3. Threading

| threads | sources | `fields()` (s) | factorization (s) | solves (s) | assembly (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 94.9 | 71.8 | 13.1 | 9.3 |
| 24 | 1 | 28.2 | 12.0 | 5.1 | 10.3 |
| 24 | 10 | 125.0 | 12.0 | 19.0 | 88.9 |

24 threads speeds the whole forward up **3.4x** (95 -> 28 s). The gain is almost entirely in MKL: factorization 6.0x, solves 2.6x. **Assembly does not thread at all** (scipy sparse work on one core: 9.3 s -> 10.3 s), so it goes from 10% to 37% of the runtime and becomes the thing to optimise.

## 4. The RHS assembly dominates multi-source forwards

`getRHS` is called once per step and costs **113 ms per source per step**, because the source term is re-discretised onto the mesh every step. With a `StepOffWaveform` the transmitter is off for `t > 0`: the returned vector is **exactly zero for steps 2-80** (verified), so 79 of the 80 evaluations are wasted work.

At 24 threads:

| sources | RHS assembly (s) | share of `fields()` |
|---:|---:|---:|
| 1 | 10.1 | 36% |
| 10 | 88.7 | 71% |

It scales linearly with source count while the factorization is shared, so it takes over: at 10 sources it is **71% of the forward**, against 10% for the factorizations. Extrapolating (8.9 s/source) to the 100-source global forward in `_ram_usage.md` gives ~887 s of RHS assembly out of its 1501 s measured total.

> This is a SimPEG-side inefficiency, not a property of the problem. Caching the source term across steps of a step-off waveform would remove most of it.

## Machine / software

- **CPU:** Intel(R) Xeon(R) CPU E5-2670 v3 @ 2.30GHz -- 2 sockets x 12 cores, 48 logical
- **Linear solver:** pymatsolver Pardiso (MKL), `factor=False` then explicit `_factor()` to separate factorization from solve
- Python 3.11.11, SimPEG 0.24.1.dev33+g42a5db944, discretize 0.11.3, pymatsolver 0.3.1, numpy 2.1.3, scipy 1.15.1
