"""Assemble _forward_timing.md from the two timing JSON results + machine specs."""
import json
import platform
import subprocess
import numpy as np
import scipy
import simpeg
import discretize
import pymatsolver
from simpeg.utils.solver_utils import get_default_solver


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _lscpu():
    out = {}
    try:
        for line in subprocess.check_output(["lscpu"], text=True).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    except Exception:
        pass
    return out


def _total_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024.0 ** 2)
    except Exception:
        return float("nan")


tiled = json.load(open("_timing_tiled.json"))
glob = json.load(open("_timing_global.json"))
lscpu = _lscpu()
solver = get_default_solver()
solver_name = getattr(solver, "__name__", str(solver))

speedup = glob["dpred_time_median_s"] / tiled["dpred_time_median_s"]
mem_ratio = glob["rss_peak_mb"] / tiled["rss_peak_mb"]

lines = []
lines.append("# Forward-simulation timing: tiled local mesh vs global mesh\n")
lines.append("Single TDEM sounding (one CircularLoop source, 20 time channels). "
             "**Only the `dpred` call is timed** — all mesh / simulation / model "
             "construction is excluded. Peak RAM is the process RSS sampled "
             "(5 ms) during `dpred`; the increment is peak minus the post-setup "
             "baseline. Reported time is the median of 3 consecutive cold `dpred` "
             "calls (model reset between calls).\n")

lines.append("## Results\n")
lines.append("| quantity | tiled (local mesh) | global mesh | ratio |")
lines.append("|---|---:|---:|---:|")
lines.append(f"| mesh cells | {tiled['mesh_cells']:,} | {glob['mesh_cells']:,} | "
             f"{glob['mesh_cells']/tiled['mesh_cells']:.1f}× |")
lines.append(f"| active cells | {tiled['mesh_active']:,} | {glob['mesh_active']:,} | "
             f"{glob['mesh_active']/tiled['mesh_active']:.1f}× |")
lines.append(f"| dpred wall time (median, s) | {tiled['dpred_time_median_s']:.2f} | "
             f"{glob['dpred_time_median_s']:.2f} | **{speedup:.1f}× slower** |")
lines.append(f"| dpred all runs (s) | {tiled['dpred_times_s']} | {glob['dpred_times_s']} | |")
lines.append(f"| peak RSS (MB) | {tiled['rss_peak_mb']:.0f} | {glob['rss_peak_mb']:.0f} | "
             f"**{mem_ratio:.1f}×** |")
lines.append(f"| dpred RAM increment (MB) | {tiled['rss_dpred_increment_mb']:.0f} | "
             f"{glob['rss_dpred_increment_mb']:.0f} | "
             f"{glob['rss_dpred_increment_mb']/max(tiled['rss_dpred_increment_mb'],1e-9):.1f}× |")
lines.append("")
incr_ratio = glob["rss_dpred_increment_mb"] / max(tiled["rss_dpred_increment_mb"], 1e-9)
lines.append(
    f"The tiled local-mesh forward is **{speedup:.0f}× faster** "
    f"({tiled['dpred_time_median_s']:.1f} s vs {glob['dpred_time_median_s']:.0f} s). "
    f"The forward solve itself uses **{incr_ratio:.0f}× less memory** "
    f"({tiled['rss_dpred_increment_mb']:.0f} vs {glob['rss_dpred_increment_mb']:.0f} MB "
    f"of dpred increment). Peak process RSS is {mem_ratio:.1f}× lower "
    f"({tiled['rss_peak_mb']:.0f} vs {glob['rss_peak_mb']:.0f} MB) — the ~600 MB "
    f"baseline common to both is the global mesh + model held in memory in either "
    f"case. The local mesh has only {tiled['mesh_cells']:,} cells vs "
    f"{glob['mesh_cells']:,}.\n")

lines.append("## Machine / software\n")
lines.append(f"- **CPU:** {_cpu_model()}")
lines.append(f"- **Sockets / cores-per-socket / logical CPUs:** "
             f"{lscpu.get('Socket(s)','?')} / {lscpu.get('Core(s) per socket','?')} / "
             f"{lscpu.get('CPU(s)','?')}")
if "CPU max MHz" in lscpu:
    lines.append(f"- **CPU max MHz:** {lscpu['CPU max MHz']}")
lines.append(f"- **Total RAM:** {_total_ram_gb():.0f} GB")
lines.append(f"- **OS / kernel:** {platform.system()} {platform.release()}")
lines.append(f"- **Threading:** OMP_NUM_THREADS = MKL_NUM_THREADS = 1 (single-threaded)")
lines.append(f"- **Linear solver:** {solver_name} (SimPEG default)")
lines.append(f"- **Time stepping:** {tiled['n_time_steps']} steps "
             f"(`[(3e-6,20),(1e-5,20),(3e-5,20),(1e-4,20)]`)")
lines.append("")
lines.append("### Versions")
lines.append(f"- Python {platform.python_version()}")
lines.append(f"- SimPEG {simpeg.__version__}")
lines.append(f"- discretize {discretize.__version__}")
lines.append(f"- pymatsolver {pymatsolver.__version__}")
lines.append(f"- numpy {np.__version__}")
lines.append(f"- scipy {scipy.__version__}")
lines.append("")

with open("_forward_timing.md", "w") as f:
    f.write("\n".join(lines))
print("wrote _forward_timing.md")
print("\n".join(lines))
