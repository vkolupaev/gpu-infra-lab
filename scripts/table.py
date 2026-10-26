#!/usr/bin/env python3
"""table.py - sweep points side by side as one markdown or CSV table.

Input is a JSONL file, one point per line: results/runs.jsonl (bench.py, schema v1)
or results/<experiment>/points.jsonl. Columns are dotted paths into a record,
optionally labelled: `results.ttft_ms.p50=p50@TTFT, ms`. A `raw:<key>` path reads
<key> from the raw vllm result (results/raw/<group>/c<C>.json) of a bench record;
a missing file or key gives an empty cell.

Bench records get a default column set and, below the markdown table, a physics
check of p50@TPOT at C=1 against the spec and measured-bandwidth decode floors.

    python3 scripts/table.py results/runs.jsonl --group <group_id>
    python3 scripts/table.py results/<experiment>/points.jsonl --col batch --col 'mem_peak_mib=peak mem, MiB'
"""
import argparse
import csv
import json
import os
import re
import sys

BENCH_X = "workload.concurrency"
MULTI_GROUP = [("group_id", "group"), ("extra.machine_id", "machine"),
               ("hardware.power_limit_w", "power limit, W")]


def load(path, groups=None):
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if groups:
        recs = [r for r in recs if r.get("group_id") in groups]
    return recs


def is_bench(rec):
    return "schema_version" in rec


def raw_result(rec):
    """The raw vllm bench result of a bench record, or {} if it is not there."""
    raw = rec.get("raw") or {}
    name = raw.get("vllm_result") or f"c{get(rec, BENCH_X)}.json"
    path = os.path.join(raw.get("dir") or os.path.join("results", "raw", rec["group_id"]), name)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def get(rec, path):
    """Value at a dotted path, `raw:<key>` from the raw result; None if absent."""
    if path.startswith("raw:"):
        return raw_result(rec).get(path[4:])
    v = rec
    for k in path.split("."):
        if not isinstance(v, dict) or k not in v:
            return None
        v = v[k]
    return v


def ratio(a, b):
    return lambda r: (get(r, a) / get(r, b) * 100
                      if get(r, a) is not None and get(r, b) else None)


# (label, path or function, format)
BENCH_COLS = [
    ("C", BENCH_X, "d"),
    ("n", "results.completed", "d"),
    ("out tok/s", "results.output_throughput_tps", ".0f"),
    ("total tok/s", "results.total_throughput_tps", ".0f"),
    ("p50@TTFT, ms", "results.ttft_ms.p50", ".1f"),
    ("mean@TTFT, ms", "results.ttft_ms.mean", ".1f"),
    ("p95@TTFT, ms", "results.ttft_ms.p95", ".1f"),
    ("p50@TPOT, ms", "results.tpot_ms.p50", ".2f"),
    ("p50@ITL, ms", "results.itl_ms.p50", ".2f"),
    ("p99@ITL, ms", "results.itl_ms.p99", ".2f"),
    ("p50@E2EL, ms", "results.e2el_ms.p50", ".0f"),
    ("std@E2EL, ms", "raw:std_e2el_ms", ".1f"),
    ("p50 TTFT/E2EL, %", ratio("results.ttft_ms.p50", "results.e2el_ms.p50"), ".1f"),
    ("$/1M out", "derived.usd_per_1m_output_tokens", ".3f"),
]


def split_cols(spec):
    """`a.b=label,c.d`; a comma followed by a space stays in the label (`p50@TTFT, ms`)."""
    return re.split(r",(?!\s)", spec)


def parse_col(item):
    path, _, label = item.strip().partition("=")
    return (label or path, path, None)


def fmt(v, spec):
    if v is None:
        return ""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if spec:
        return format(v, spec)
    if isinstance(v, int):
        return str(v)
    return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def sort_key(rec, x):
    v = get(rec, x) if x else None
    return (str(rec.get("group_id", "")), v is None, v if isinstance(v, (int, float)) else str(v))


def physics_lines(recs):
    """p50@TPOT at C=1 as a multiple of the spec and measured-bandwidth floors, with % of bandwidth achieved."""
    out = []
    for r in recs:
        if get(r, BENCH_X) != 1:
            continue
        tpot = get(r, "results.tpot_ms.p50")
        spec = get(r, "derived.decode_floor_ms_per_token")
        gbps = get(r, "hardware.hbm_copy_gbps")
        if gbps is None:
            try:
                with open(os.path.join(r["raw"]["dir"], "hwcheck.json")) as f:
                    gbps = json.load(f).get("hbm_copy_gbps")
            except (OSError, ValueError, KeyError):
                pass
        params = get(r, "model.params_b")
        parts = [f"C=1: p50@TPOT = {tpot:.2f} ms"]
        if spec:
            parts.append(f"{tpot / spec:.2f}x spec floor {spec:.2f} ms "
                         f"({spec / tpot:.0%} of spec bandwidth)")
        if gbps and params:
            floor = params * 2 / gbps * 1000
            parts.append(f"{tpot / floor:.2f}x measured-bandwidth floor {floor:.2f} ms "
                         f"({floor / tpot:.0%} of measured bandwidth)")
        prefix = f"{r['group_id']}: " if len({x.get('group_id') for x in recs}) > 1 else ""
        out.append(prefix + "; ".join(parts))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", help="results/runs.jsonl or results/<experiment>/points.jsonl")
    p.add_argument("--group", action="append", help="group_id to keep (repeatable)")
    p.add_argument("--col", dest="cols", action="extend", type=lambda s: [s], metavar="PATH[=LABEL]",
                   help="one column: a dotted path or raw:<key>, optionally =label "
                        "(repeatable; e.g. --col 'results.ttft_ms.p50=p50@TTFT, ms')")
    p.add_argument("--cols", dest="cols", action="extend", type=split_cols, metavar="LIST",
                   help="shorthand for several --col: items split on a comma NOT followed by "
                        "whitespace, so 'results.ttft_ms.p50=p50@TTFT, ms,raw:std_e2el_ms' is two "
                        "columns; a label with a comma and no space after it needs --col. "
                        "--col and --cols mix, columns keep command-line order")
    p.add_argument("--sort", help=f"dotted path to sort by within a group "
                                  f"(default {BENCH_X} for bench, else the first column)")
    p.add_argument("--format", choices=("md", "csv"), default="md")
    a = p.parse_args()

    recs = load(a.jsonl, a.group)
    if not recs:
        sys.exit("no records")
    bench = all(is_bench(r) for r in recs)
    if a.cols:
        cols = [parse_col(c) for c in a.cols]
    elif bench:
        cols = BENCH_COLS
    else:
        sys.exit("--col/--cols is required for non-bench records")
    x = a.sort or (BENCH_X if bench else cols[0][1])
    recs.sort(key=lambda r: sort_key(r, x))
    if len({r.get("group_id") for r in recs}) > 1:
        cols = [(label, path, None) for path, label in MULTI_GROUP] + list(cols)

    header = [c[0] for c in cols]
    rows = [[fmt(c[1](r) if callable(c[1]) else get(r, c[1]), c[2]) for c in cols] for r in recs]
    if a.format == "csv":
        w = csv.writer(sys.stdout, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
        return
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    if bench:
        lines = physics_lines(recs)
        if lines:
            print()
            print("\n".join(lines))


if __name__ == "__main__":
    main()
