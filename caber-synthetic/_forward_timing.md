# Forward-simulation timing: tiled local mesh vs global mesh

Single TDEM sounding (one CircularLoop source, 20 time channels). **Only the `dpred` call is timed** — all mesh / simulation / model construction is excluded. Peak RAM is the process RSS sampled (5 ms) during `dpred`; the increment is peak minus the post-setup baseline. Reported time is the median of 3 consecutive cold `dpred` calls (model reset between calls).

## Results

| quantity | tiled (local mesh) | global mesh | ratio |
|---|---:|---:|---:|
| mesh cells | 5,510 | 52,340 | 9.5× |
| active cells | 3,602 | 42,284 | 11.7× |
| dpred wall time (median, s) | 2.71 | 96.04 | **35.5× slower** |
| dpred all runs (s) | [2.799, 2.709, 2.707] | [97.3, 96.04, 95.696] | |
| peak RSS (MB) | 663 | 1699 | **2.6×** |
| dpred RAM increment (MB) | 61 | 1110 | 18.2× |

The tiled local-mesh forward is **35× faster** (2.7 s vs 96 s). The forward solve itself uses **18× less memory** (61 vs 1110 MB of dpred increment). Peak process RSS is 2.6× lower (663 vs 1699 MB) — the ~600 MB baseline common to both is the global mesh + model held in memory in either case. The local mesh has only 5,510 cells vs 52,340.

## Machine / software

- **CPU:** Intel(R) Xeon(R) CPU E5-2670 v3 @ 2.30GHz
- **Sockets / cores-per-socket / logical CPUs:** 2 / 12 / 48
- **CPU max MHz:** 3100.0000
- **Total RAM:** 126 GB
- **OS / kernel:** Linux 5.15.0-171-generic
- **Threading:** OMP_NUM_THREADS = MKL_NUM_THREADS = 1 (single-threaded)
- **Linear solver:** Pardiso (SimPEG default)
- **Time stepping:** 80 steps (`[(3e-6,20),(1e-5,20),(3e-5,20),(1e-4,20)]`)

### Versions
- Python 3.11.11
- SimPEG 0.24.1.dev33+g42a5db944
- discretize 0.11.3
- pymatsolver 0.3.1
- numpy 2.1.3
- scipy 1.15.1
