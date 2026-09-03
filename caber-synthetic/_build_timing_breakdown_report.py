"""Turn _timing_breakdown_*.json into a markdown report.

Run: python _build_timing_breakdown_report.py
"""
import glob
import json
import platform
import subprocess

import numpy as np


def load(path):
    d = json.load(open(path))
    d["_path"] = path
    d["t_fields"] = d["t_fields_total"]     # median warm run
    return d


def dt_blocks(d):
    """Group the 80 steps into the 4 constant-dt blocks."""
    solve = np.array(d["t_solve"])
    dts = np.array(d["dt_per_step"])
    edges = [0] + list(np.nonzero(np.diff(dts))[0] + 1) + [len(dts)]
    out = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        out.append({
            "dt": float(dts[a]),
            "steps": b - a,
            "t_factor": d["t_factor"][k],
            "solves": solve[a:b],
            "t_end": float(np.cumsum(dts)[b - 1]),
        })
    return out


def cpu_name():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def main():
    runs = {}
    for p in sorted(glob.glob("_timing_breakdown_src*_thr*.json")):
        d = load(p)
        runs[(d["n_src"], d["n_threads"])] = d
    if not runs:
        raise SystemExit("no _timing_breakdown_*.json found -- run _timing_breakdown.py")

    ref = runs[(1, 1)]
    L = []
    L.append("# Forward-simulation cost breakdown: global mesh\n")
    L.append("Where the time actually goes in **one** `Simulation3DElectricField` "
             "forward on the global OcTree mesh. Instrumented in "
             "`_timing_breakdown.py`, which subclasses the Pardiso solver to time "
             "each factorization and each solve, and the simulation to time each "
             "matrix assembly.\n")
    L.append(f"- **Mesh:** {ref['n_cells']:,} cells ({ref['n_active']:,} active), "
             f"**{ref['n_edges']:,} edges** -- the E-field system is "
             f"{ref['n_edges']:,} x {ref['n_edges']:,}.")
    L.append(f"- **Time stepping:** "
             f"{', '.join(f'({dt:g} s x {n})' for dt, n in ref['time_steps'])} = "
             f"**{sum(n for _, n in ref['time_steps'])} steps**, "
             f"**{len(ref['time_steps'])} unique step sizes**.")
    L.append("- A factorization is reused while `dt` is constant, so there are "
             "**4 factorizations and 80 solves**.")
    L.append("- Reported times are the **median of 3 warm runs** (a preceding "
             "warm-up call builds discretize's mesh operators, which are then "
             "cached on the mesh).\n")

    # ---------------------------------------------------------------- headline
    L.append("## 1. Single sounding, single thread\n")
    L.append("The reference case, matching `_forward_timing.md` "
             f"(95.3 s here vs 96.0 s there).\n")
    L.append("| stage | calls | total (s) | per call | share |")
    L.append("|---|---:|---:|---:|---:|")
    tot = ref["t_fields"]
    rows = [
        ("factorization", ref["t_factor"]),
        ("solve", ref["t_solve"]),
        ("RHS assembly (`getRHS`)", ref["t_assemble_rhs"]),
        ("system matrix (`getAdiag`)", ref["t_assemble_A"]),
        ("sub-diagonal (`getAsubdiag`)", ref["t_assemble_sub"]),
    ]
    acc = 0.0
    for name, arr in rows:
        a = np.array(arr)
        acc += a.sum()
        m = a.mean()
        per = f"{m:.1f} s" if m >= 1 else (
            f"{m*1e3:.0f} ms" if m * 1e3 >= 1 else f"{m*1e3:.1f} ms")
        L.append(f"| {name} | {len(a)} | {a.sum():.2f} | {per} | "
                 f"{100*a.sum()/tot:.1f}% |")
    L.append(f"| other / field storage | - | {tot-acc:.2f} | - | "
             f"{100*(tot-acc)/tot:.1f}% |")
    L.append(f"| **`fields()` total** | | **{tot:.2f}** | | |")
    L.append(f"| receiver interpolation | | {ref['t_receiver_interp']:.2f} | | |")
    L.append(f"| **forward total** | | "
             f"**{tot + ref['t_receiver_interp']:.2f}** | | |\n")
    L.append(f"Cold first call, including one-time mesh-operator construction: "
             f"**{ref['t_coldstart']:.1f} s** "
             f"(+{ref['t_coldstart']-tot:.0f} s over a warm call).\n")

    # ---------------------------------------------------------------- per dt
    L.append("## 2. Per time-step block\n")
    L.append("One factorization per block, reused by every step in it. "
             "Single sounding, single thread.\n")
    L.append("| block | dt | steps | t range (s) | factorization (s) | "
             "solve/step (s) | block solves (s) |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|")
    t_start = 0.0
    for i, b in enumerate(dt_blocks(ref), 1):
        L.append(f"| {i} | {b['dt']:g} | {b['steps']} | "
                 f"{t_start:.2e} - {b['t_end']:.2e} | {b['t_factor']:.2f} | "
                 f"{b['solves'].mean():.3f} | {b['solves'].sum():.2f} |")
        t_start = b["t_end"]
    L.append("")
    facs = np.array(ref["t_factor"])
    sol = np.array(ref["t_solve"])
    L.append(f"Every factorization costs about the same "
             f"({facs.min():.1f}-{facs.max():.1f} s) -- the sparsity pattern is "
             f"identical across blocks, only the `dt` scaling of the mass term "
             f"changes. **One factorization ({facs.mean():.1f} s) costs about as "
             f"much as {facs.mean()/sol.mean():.0f} solves "
             f"({sol.mean()*1e3:.0f} ms each).**\n")

    # ---------------------------------------------------------------- threads
    L.append("## 3. Threading\n")
    L.append("| threads | sources | `fields()` (s) | factorization (s) | "
             "solves (s) | assembly (s) |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    for (ns, nt), d in sorted(runs.items()):
        asm = (np.sum(d["t_assemble_A"]) + np.sum(d["t_assemble_sub"])
               + np.sum(d["t_assemble_rhs"]))
        L.append(f"| {nt} | {ns} | {d['t_fields']:.1f} | "
                 f"{np.sum(d['t_factor']):.1f} | {np.sum(d['t_solve']):.1f} | "
                 f"{asm:.1f} |")
    L.append("")
    if (1, 24) in runs:
        a, b = runs[(1, 1)], runs[(1, 24)]
        asm_a = (np.sum(a["t_assemble_A"]) + np.sum(a["t_assemble_sub"])
                 + np.sum(a["t_assemble_rhs"]))
        asm_b = (np.sum(b["t_assemble_A"]) + np.sum(b["t_assemble_sub"])
                 + np.sum(b["t_assemble_rhs"]))
        L.append(f"24 threads speeds the whole forward up "
                 f"**{a['t_fields']/b['t_fields']:.1f}x** "
                 f"({a['t_fields']:.0f} -> {b['t_fields']:.0f} s). The gain is "
                 f"almost entirely in MKL: factorization "
                 f"{np.sum(a['t_factor'])/np.sum(b['t_factor']):.1f}x, solves "
                 f"{np.sum(a['t_solve'])/np.sum(b['t_solve']):.1f}x. "
                 f"**Assembly does not thread at all** "
                 f"(scipy sparse work on one core: {asm_a:.1f} s -> "
                 f"{asm_b:.1f} s), so it goes from "
                 f"{100*asm_a/a['t_fields']:.0f}% to "
                 f"{100*asm_b/b['t_fields']:.0f}% of the "
                 f"runtime and becomes the thing to optimise.\n")

    # ---------------------------------------------------------------- rhs
    L.append("## 4. The RHS assembly dominates multi-source forwards\n")
    r1 = np.array(ref["t_assemble_rhs"])
    L.append(f"`getRHS` is called once per step and costs "
             f"**{r1.mean()*1e3:.0f} ms per source per step**, because the "
             f"source term is re-discretised onto the mesh every step. With a "
             f"`StepOffWaveform` the transmitter is off for `t > 0`: the "
             f"returned vector is **exactly zero for steps 2-80** (verified), so "
             f"79 of the 80 evaluations are wasted work.\n")
    L.append("At 24 threads:\n")
    L.append("| sources | RHS assembly (s) | share of `fields()` |")
    L.append("|---:|---:|---:|")
    for (ns, nt), d in sorted(runs.items()):
        if nt != 24:
            continue
        r = np.sum(d["t_assemble_rhs"])
        L.append(f"| {ns} | {r:.1f} | {100*r/d['t_fields']:.0f}% |")
    L.append("")
    if (10, 24) in runs:
        d10 = runs[(10, 24)]
        per_src = np.sum(d10["t_assemble_rhs"]) / d10["n_src"]
        L.append(f"It scales linearly with source count while the factorization "
                 f"is shared, so it takes over: at 10 sources it is "
                 f"**{100*np.sum(d10['t_assemble_rhs'])/d10['t_fields']:.0f}% of "
                 f"the forward**, against {100*np.sum(d10['t_factor'])/d10['t_fields']:.0f}% "
                 f"for the factorizations. Extrapolating "
                 f"({per_src:.1f} s/source) to the 100-source global forward in "
                 f"`_ram_usage.md` gives ~{per_src*100:.0f} s of RHS assembly out "
                 f"of its 1501 s measured total.\n")
        L.append("> This is a SimPEG-side inefficiency, not a property of the "
                 "problem. Caching the source term across steps of a step-off "
                 "waveform would remove most of it.\n")

    # ---------------------------------------------------------------- machine
    L.append("## Machine / software\n")
    L.append(f"- **CPU:** {cpu_name()} -- 2 sockets x 12 cores, 48 logical")
    L.append(f"- **Linear solver:** pymatsolver Pardiso (MKL), "
             f"`factor=False` then explicit `_factor()` to separate "
             f"factorization from solve")
    import scipy
    import simpeg
    import discretize
    import pymatsolver
    L.append(f"- Python {platform.python_version()}, SimPEG {simpeg.__version__}, "
             f"discretize {discretize.__version__}, "
             f"pymatsolver {pymatsolver.__version__}, "
             f"numpy {np.__version__}, scipy {scipy.__version__}")

    text = "\n".join(L) + "\n"
    with open("_timing_breakdown.md", "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
