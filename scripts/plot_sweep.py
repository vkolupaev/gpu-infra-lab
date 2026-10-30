#!/usr/bin/env python3
"""plot_sweep.py - plot sweep points: one panel per metric, shared X axis.

Input is the same JSONL as scripts/table.py (runs.jsonl or points.jsonl), paths are
dotted. Each --y is one panel; several metrics join one panel with `+` only when
they share a unit: `--y 'results.ttft_ms.p50=p50@TTFT+results.ttft_ms.p95=p95@TTFT'`.
The unit comes from a label suffix (`=p50@TTFT, ms`) or from the key name
(`*_ms`, `*_tps`, `*_mib`, ...). Never two Y scales on one panel. Several groups
give one line per group with a legend.

    python3 scripts/plot_sweep.py results/runs.jsonl --group <group_id> --logx --out sweep.png
    python3 scripts/plot_sweep.py results/runs.jsonl --group <g1> --group <g2> \\
        --x results.output_throughput_tps --y results.ttft_ms.p95 --out curve.png
"""
import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

from plotstyle import INK, INK2, SERIES, SURF, style_axis
from table import BENCH_X, get, is_bench, load

BENCH_Y = [
    "results.output_throughput_tps=output, tok/s",
    "results.ttft_ms.p50=p50@TTFT, ms+results.ttft_ms.p95=p95@TTFT, ms",
    "results.tpot_ms.p50=p50@TPOT, ms+results.itl_ms.p50=p50@ITL, ms",
    "derived.usd_per_1m_output_tokens=$/1M out, USD",
]
UNITS = [(r"(^|_)tps$|tokens_per_s$", "tok/s"), (r"(^|_)ms$", "ms"), (r"(^|_)s$", "s"),
         (r"(^|_)mib$", "MiB"), (r"(^|_)gib$", "GiB"), (r"(^|_)w$", "W"), (r"(^|_)pct$", "%"),
         (r"(^|_)gbps$", "GB/s"), (r"(^|_)tflops$", "TFLOP/s"), (r"^usd_", "USD")]
STYLES = ("-", (0, (4, 2)), ":", "-.")
MARKERS = ("o", "s", "^", "D")


def unit_of(path):
    """Unit from the key name: the last path segment with a known suffix."""
    for seg in reversed(path.split(".")):
        for pat, unit in UNITS:
            if re.search(pat, seg):
                return unit
    return None


def parse_metric(item):
    """`path[=label]` -> (path, name, unit); `label, unit` overrides the key unit."""
    path, _, label = item.strip().partition("=")
    name, sep, unit = label.rpartition(", ")
    if not sep:
        name, unit = label or path, unit_of(path)
    return path, name, unit


def parse_panel(spec):
    metrics = [parse_metric(m) for m in spec.split("+")]
    units = {m[2] for m in metrics}
    if len(metrics) > 1 and (len(units) > 1 or None in units):
        sys.exit(f"--y {spec!r}: metrics on one panel need the same known unit, got {units}; "
                 f"put them on separate panels or give 'label, unit'")
    return metrics


def axis_label(names, unit):
    return ", ".join([" / ".join(names)] + ([unit] if unit else []))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", help="results/runs.jsonl or results/<experiment>/points.jsonl")
    p.add_argument("--group", action="append", help="group_id to keep (repeatable)")
    p.add_argument("--x", help=f"dotted path[=label] for X (default {BENCH_X} for bench)")
    p.add_argument("--y", action="append", metavar="PATH[=LABEL][+PATH[=LABEL]...]",
                   help="one panel (repeatable); `+` joins metrics of the same unit")
    p.add_argument("--sort", help=f"dotted path that orders points along a line "
                                  f"(default {BENCH_X} for bench, else X)")
    p.add_argument("--logx", action="store_true", help="log2 X axis")
    p.add_argument("--title")
    p.add_argument("--out", required=True, help="PNG path")
    a = p.parse_args()

    recs = load(a.jsonl, a.group)
    if not recs:
        sys.exit("no records")
    bench = all(is_bench(r) for r in recs)
    if not bench and not (a.x and a.y):
        sys.exit("--x and --y are required for non-bench records")
    xpath, xname, xunit = parse_metric(a.x or f"{BENCH_X}=concurrency (C)")
    panels = [parse_panel(s) for s in (a.y or BENCH_Y)]
    order = a.sort or (BENCH_X if bench else xpath)
    label_c = bench and xpath != BENCH_X  # X is a metric: name each point by its C

    groups = sorted({r.get("group_id") for r in recs}, key=str)
    lines = {g: sorted((r for r in recs if r.get("group_id") == g),
                       key=lambda r: (get(r, order) is None, get(r, order) or 0)) for g in groups}

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.6 * len(panels) + 0.8),
                             sharex=True, squeeze=False, facecolor=SURF)
    axes = axes[:, 0]
    xs_all = set()
    for ax, metrics in zip(axes, panels, strict=True):
        style_axis(ax, axis_label([m[1] for m in metrics], metrics[0][2]))
        for gi, g in enumerate(groups):
            for mi, (path, name, _) in enumerate(metrics):
                pts = [(get(r, xpath), get(r, path), get(r, BENCH_X)) for r in lines[g]]
                pts = [t for t in pts if t[0] is not None and t[1] is not None]
                if not pts:
                    continue
                xs_all.update(t[0] for t in pts)
                color = SERIES[(gi if len(groups) > 1 else mi) % len(SERIES)]
                label = " ".join(([str(g)] if len(groups) > 1 else [])
                                 + ([name] if len(metrics) > 1 else []))
                ax.plot([t[0] for t in pts], [t[1] for t in pts], color=color,
                        ls=STYLES[mi % len(STYLES)] if len(groups) > 1 else "-", lw=2,
                        marker=MARKERS[mi % len(MARKERS)], ms=6, mec=SURF, mew=1.5,
                        label=label or None, zorder=2)
                if label_c:
                    for x, y, c in pts:
                        ax.annotate(f"C={c}", (x, y), xytext=(5, 5), textcoords="offset points",
                                    fontsize=8, color=INK2)
        if any(h.get_label() and not h.get_label().startswith("_") for h in ax.get_lines()):
            ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
        ax.set_ylim(bottom=0)
    ax = axes[-1]
    if a.logx:
        ax.set_xscale("log", base=2)
    if xs_all and len(xs_all) <= 12 and all(isinstance(v, int) for v in xs_all):
        ax.set_xticks(sorted(xs_all))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.set_xlabel(axis_label([xname], xunit), color=INK2, fontsize=10)

    r0 = recs[0]
    title = a.title or (f"{get(r0, 'model.id')} on {get(r0, 'hardware.gpu_name')}: "
                        f"{get(r0, 'tag') or 'sweep'}" if bench else os.path.basename(a.jsonl))
    sub = ", ".join(str(g) for g in groups) if bench else ""
    if bench:
        w = r0["workload"]
        sub += (f"  |  input {w.get('input_len')} / output {w.get('output_len')} tokens, "
                f"{w.get('dataset')} dataset")
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.15 / h, title, va="top", fontsize=13, color=INK)
    if sub:
        fig.text(0.02, 1 - 0.45 / h, sub, va="top", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.7 if sub else 0.45) / h))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130, facecolor=SURF)
    plt.close(fig)
    print(a.out)


if __name__ == "__main__":
    main()
