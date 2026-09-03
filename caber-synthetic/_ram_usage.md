# Aggregate RAM and full-survey forward cost — tiled vs. global

The single-sounding benchmark (`_forward_timing.md`) shows the *per-solve* story:
a tile-mesh solve is ~35× faster and ~18× lighter than the same solve on the
global mesh (2.7 s / 61 MB vs. 96 s / 1.1 GB). But the per-solve footprint is not
the whole picture for a real survey: the tiled run holds *all* the local meshes
and one factorization per concurrent worker at once, while a single global
simulation holds the time-domain fields for *all* sources at once. This note
measures the aggregate for the full 100-sounding survey (40 m line spacing).

Memory is reported as **peak PSS** (proportional set size) summed across the
parent process and all forked pool workers, so copy-on-write pages shared
between workers are counted once, not once per worker. Measured with
`_ram_test.py` on the 48-core node (see Machine below).

## Headline

| | tiled (100 meshes, 48 workers) | global (1 mesh, 100 sources) |
|---|---|---|
| full-survey forward time | **~16 s** | **~1500 s (25 min)** |
| peak RAM (whole process tree) | **~13.5 GB** | **~11.3 GB** |

At full-survey scale the two approaches use **comparable** peak RAM — the tiled
run is actually a touch higher — but the tiled forward is **~90× faster**. So
the win from tiling here is overwhelmingly *time*, not memory. This matches the
per-solve picture (each tile solve is tiny) once you account for the fact that
the tiled run keeps many of those solves alive at once.

## Where the tiled memory goes (decomposition)

Building up the tiled simulation, peak PSS over the process tree:

| stage | PSS | added |
|---|---|---|
| baseline (imports, global mesh) | 0.39 GB | — |
| + 100 local meshes (bare) | 1.7 GB | **1.3 GB** (~13 MB/mesh) |
| + 100 simulations + TileMaps + mesh operators | 9.0 GB | **7.3 GB** (~73 MB/source) |
| + 48 workers factorizing during `dpred` | 13.5 GB | **4.5 GB** (~93 MB/worker) |

Two things stand out, and they are exactly the diminishing-return concern:

1. **Holding the simulation objects is the dominant fixed cost, ~9 GB**, and it
   is paid in the *parent* before any solve. It is dominated not by the bare
   meshes (1.3 GB) but by the per-source SimPEG simulation objects and their
   cached mesh operators plus the `TileMap` volume-averaging matrices
   (global→local, one sparse 52k×5.4k operator per source). This floor scales
   **linearly with the number of soundings** and is independent of how many
   workers you run — so past a few hundred soundings, just *holding* the tiled
   model becomes the bottleneck.
2. The worker factorizations add ~4.5 GB on top, scaling with the number of
   **concurrent workers** (each holds its tile's factorization + fields, ~93 MB).
   Fewer workers → lower peak but longer wall-clock; this is the time/memory
   knob.

## Why the global run is ~11 GB and ~25 minutes

A single global `Simulation3DElectricField` with 100 sources stores the
time-domain E-field for **every source at every time step** simultaneously
(~100 sources × 80 steps × ~160k edges), which is the ~11 GB — not the factor-
ization (the factorization alone is ~1.1 GB, per the single-sounding bench). The
25-minute time is 100 sources × 80 implicit backward-Euler steps of back-
substitution against the shared factorization. The tiled workers, by contrast,
compute each tile's fields on a tiny mesh and **discard them after extracting the
datum**, so they never hold more than (n_workers) small field sets at once.

## Takeaway

- The tiling advantage at survey scale is **~90× wall-clock**, with peak RAM
  *comparable* to the global solve (~13.5 vs ~11.3 GB here) — not a memory
  reduction. The honest framing is "small, independent solves distributed across
  a worker pool," not "uses less memory overall."
- Both approaches' aggregate RAM grows with the number of soundings (tiled: more
  meshes/operators to hold; global: more source-fields to store), so neither is
  free as the survey grows. For the tiled run specifically, the ~9 GB floor from
  holding all the simulation objects is the part that scales with sounding count
  and would become the limiting factor on very large surveys. (An implementation
  that built each tile's simulation lazily inside its worker, rather than all in
  the parent up front, would cut that floor — a worthwhile future optimization.)

## Numbers (machine-specific)

- tiled: peak 13,483 MB; meshes-only 1,338 MB; after-build 9,018 MB;
  full-survey forward 16.2 s (100 sources, 48 workers, local meshes ~5,438 cells
  each, 543,832 cells total). → `_ram_tiled.json`
- global: peak 11,278 MB; full-survey forward 1,500.8 s (52,340-cell mesh, 100
  sources). → `_ram_global.json`

## Machine

Intel Xeon E5-2670 v3 @ 2.30 GHz (2 × 12 cores, 48 threads), 126 GB RAM, Linux
5.15; SimPEG 0.24.1.dev33, MKL Pardiso solver, single-threaded workers
(OMP/MKL = 1). PSS sampled at 5 Hz across the process tree via psutil.
