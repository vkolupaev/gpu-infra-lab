#!/usr/bin/env python3
"""plot_gpu_monitor.py - plot the nvidia-smi log of one run group.

Reads <run_dir>/gpu-monitor.csv (nvidia-smi, 1 sample/s) and <run_dir>/c<N>.json
(vllm bench serve results). Benchmark phases come from the results: end = "date",
start = end - "duration". Writes two PNGs: the whole run and a zoom on the phase
with the highest concurrency. Three panels share the time axis: power draw, memory
utilization, SM clock. Seconds with the SW power cap bit (0x4) in
clocks_event_reasons are marked.

    python3 scripts/plot_gpu_monitor.py results/raw/<group_id>
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as md
import matplotlib.pyplot as plt

from plotstyle import BLUE, INK, INK2, ORANGE, PHASE, SURF, style_axis

SW_POWER_CAP = 0x4


def num(field):
    return float(field.strip().split()[0])


def read_monitor(path):
    T, P, M, S, CAP = [], [], [], [], []
    with open(path) as f:
        rows = list(csv.reader(f))[1:]
    for r in rows:
        try:
            t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            vals = num(r[4]), num(r[2]), num(r[6]), int(r[8].strip(), 16)
        except (IndexError, ValueError):
            continue  # truncated last line or [N/A] fields
        T.append(t)
        P.append(vals[0])
        M.append(vals[1])
        S.append(vals[2])
        CAP.append(bool(vals[3] & SW_POWER_CAP))
    return T, P, M, S, CAP


def read_phases(run_dir):
    phases = []
    for path in glob.glob(os.path.join(run_dir, "c*.json")):
        with open(path) as f:
            res = json.load(f)
        end = dt.datetime.strptime(res["date"], "%Y%m%d-%H%M%S")
        phases.append((res["max_concurrency"], end - dt.timedelta(seconds=res["duration"]), end))
    return sorted(phases)


def read_smi_q(run_dir):
    """Power limit, max SM clock and product name from nvidia-smi -q."""
    with open(os.path.join(run_dir, "nvidia-smi-q.txt")) as f:
        text = f.read()
    limit = re.search(r"Current Power Limit\s*:\s*([\d.]+) W", text).group(1)
    sm_max = re.search(r"Max Clocks\s*\n(?:.*\n)*?\s*SM\s*:\s*(\d+) MHz", text).group(1)
    name = re.search(r"Product Name\s*:\s*(.+)", text).group(1).strip()
    return float(limit), float(sm_max), name


def run_tag(run_dir, runs_path):
    group = os.path.basename(os.path.normpath(run_dir))
    with open(runs_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["group_id"] == group:
                    return rec["tag"]
    return group


def draw(mon, phases, refs, t0, t1, out, title):
    T, P, M, S, CAP = mon
    sel = [i for i, t in enumerate(T) if t0 <= t <= t1]
    limit, sm_max = refs
    specs = [(P, "Power draw, W", limit, f"power limit {limit:.0f} W"),
             (M, "Memory utilization, %", None, None),
             (S, "SM clock, MHz", sm_max, f"max {sm_max:.0f} MHz")]
    shown = [(c, max(a, t0), min(b, t1)) for c, a, b in phases if a < t1 and b > t0]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True, facecolor=SURF)
    for ax, (Y, label, ref, ref_label) in zip(axes, specs, strict=True):
        style_axis(ax, label)
        for _, a, b in shown:
            ax.axvspan(a, b, color=PHASE, zorder=0)
        ax.plot([T[i] for i in sel], [Y[i] for i in sel], color=BLUE, lw=2, zorder=2)
        capped = [i for i in sel if CAP[i]]
        ax.scatter([T[i] for i in capped], [Y[i] for i in capped],
                   s=46, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=3)
        if ref:
            ax.axhline(ref, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(1.0, ref, " " + ref_label, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=9, color=INK2)
    for c, a, b in shown:
        axes[0].text(a + (b - a) / 2, 1.02, f"C={c}", transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    axes[-1].xaxis.set_major_formatter(md.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("Time, UTC (1 sample per second)", color=INK2, fontsize=10)
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, color=INK, y=0.995)
    fig.text(0.06, 0.945, "Orange dots: clocks_event_reasons has bit 0x4 (SW power cap). "
             "Shaded: benchmark phases.", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 0.93, 0.94))
    fig.savefig(out, dpi=130, facecolor=SURF)
    plt.close(fig)
    print(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="results/raw/<group_id>")
    p.add_argument("--runs", default="results/runs.jsonl", help="for the run tag in default file names")
    p.add_argument("--out-dir", default="experiments/img")
    p.add_argument("--overview", help="overview PNG name (default <date>-<tag>-gpu-overview.png)")
    p.add_argument("--zoom", help="zoom PNG name (default <date>-<tag>-gpu-c<max>.png)")
    a = p.parse_args()

    mon = read_monitor(os.path.join(a.run_dir, "gpu-monitor.csv"))
    phases = read_phases(a.run_dir)
    limit, sm_max, gpu = read_smi_q(a.run_dir)
    c_max, z0, z1 = phases[-1]
    prefix = f"{phases[0][2]:%Y-%m-%d}-{run_tag(a.run_dir, a.runs)}"
    os.makedirs(a.out_dir, exist_ok=True)
    pad = dt.timedelta(seconds=30)
    draw(mon, phases, (limit, sm_max), phases[0][1] - pad, phases[-1][2] + pad / 6,
         os.path.join(a.out_dir, a.overview or f"{prefix}-gpu-overview.png"),
         f"{gpu} during the run: whole benchmark")
    draw(mon, phases, (limit, sm_max), z0 - pad / 7, z1 + pad / 10,
         os.path.join(a.out_dir, a.zoom or f"{prefix}-gpu-c{c_max}.png"),
         f"Zoom on C={c_max}: decode plateaus vs prefill bursts")


if __name__ == "__main__":
    main()
