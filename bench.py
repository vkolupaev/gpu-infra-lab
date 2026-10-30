#!/usr/bin/env python3
"""bench.py - thin, reproducible wrapper around `vllm serve` + `vllm bench serve`.

What this script does (and vllm bench does NOT):
  * pins the model revision and downloads it once (snapshot_download)
  * launches `vllm serve` from a YAML config, so recorded engine params == used params
  * runs a warmup, then one `vllm bench serve` per concurrency level
  * captures environment: GPU (nvidia-smi), driver, vLLM/torch/CUDA versions, image, git commit
  * runs scripts/hwcheck.py before the model download and logs nvidia-smi once per second
  * normalizes vLLM's result JSON into ONE fixed record schema (SCHEMA_VERSION)
  * computes derived economics: $ per 1M output / total tokens, decode roofline floor
  * appends records to results/runs.jsonl, keeps raw files, drafts an experiment note

What it leaves to `vllm bench serve`: load generation, timing, percentiles.

Usage:
  python3 bench.py run --config configs/qwen3-8b-bf16.yaml \
      --provider vast --price-per-hour 1.10 --tag first-number
  python3 bench.py validate results/runs.jsonl
  python3 bench.py tocsv results/runs.jsonl > results/runs.csv
"""

from __future__ import annotations

import argparse
import atexit
import csv
import datetime as dt
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request

import yaml

SCHEMA_VERSION = 1
ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXPERIMENTS = ROOT / "experiments"

# Peak memory bandwidth (GB/s) for the decode roofline sanity check.
# Spec-sheet values; extend when a new card shows up.
GPU_MEM_BW_GBPS = {
    "A100-SXM4-80GB": 2039,
    "A100 80GB PCIe": 1935,
    "A100-SXM4-40GB": 1555,
    "A100-PCIE-40GB": 1555,
    "H100 80GB HBM3": 3350,
    "H100 PCIe": 2000,
    "H100 NVL": 3900,
    "H200": 4800,
    "H200 NVL": 4800,
    "A30": 933,
    "Tesla T4": 320,
    "L40S": 864,
}
BYTES_PER_PARAM = {"bfloat16": 2, "float16": 2, "half": 2, "fp8": 1, "int8": 1, "awq": 0.5, "gptq": 0.5}

REQUIRED = {  # dotted paths that must be present (value may be null only where noted)
    "schema_version",
    "run_id",
    "group_id",
    "timestamp_utc",
    "git.commit",
    "provider.name",
    "provider.price_per_hour_usd",
    "provider.gpu_sharing",
    "provider.gpu_fraction",
    "hardware.gpu_name",
    "hardware.form_factor",
    "hardware.memory_total_mib",
    "software.vllm_version",
    "software.image",
    "model.id",
    "model.revision",
    "model.precision",
    "engine.max_num_seqs",
    "engine.max_num_batched_tokens",
    "engine.max_model_len",
    "engine.gpu_memory_utilization",
    "engine.enable_prefix_caching",
    "workload.input_len",
    "workload.output_len",
    "workload.concurrency",
    "workload.num_prompts",
    "results.completed",
    "results.output_throughput_tps",
    "results.total_throughput_tps",
    "results.ttft_ms.p50",
    "results.e2el_ms.p50",
    "results.e2el_ms.p95",
    "results.e2el_ms.p99",
    "derived.usd_per_1m_output_tokens",
    "derived.usd_per_1m_total_tokens",
}
NON_NULL = {  # must also be non-empty (None/0 means the vLLM result keys did not match)
    "results.completed",
    "results.output_throughput_tps",
    "results.total_throughput_tps",
    "results.ttft_ms.p50",
    "results.ttft_ms.p95",
    "results.tpot_ms.p50",
    "results.e2el_ms.p50",
    "results.e2el_ms.p95",
    "derived.usd_per_1m_output_tokens",
    "derived.usd_per_1m_total_tokens",
}


# ----------------------------------------------------------------------------- helpers
def sh(cmd: list[str], **kw) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_info() -> dict:
    commit = sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    if commit:
        dirty = bool(sh(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]))
        return {"commit": commit, "dirty": dirty}
    # no git binary (e.g. plain vLLM image): read .git directly, dirty state unknown
    head = ROOT / ".git" / "HEAD"
    if head.exists():
        ref = head.read_text().strip()
        if ref.startswith("ref: "):
            ref_name = ref[5:]
            loose = ROOT / ".git" / ref_name
            if loose.exists():
                return {"commit": loose.read_text().strip(), "dirty": None}
            packed = ROOT / ".git" / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref_name):
                        return {"commit": line.split()[0], "dirty": None}
        else:
            return {"commit": ref, "dirty": None}
    return {"commit": os.environ.get("GIT_COMMIT", "unknown"), "dirty": None}


def form_factor(name: str) -> str:
    n = name.upper()
    if "SXM" in n or "HBM3" in n:  # "H100 80GB HBM3" is the SXM part
        return "SXM"
    if "NVL" in n:
        return "NVL"
    if "PCIE" in n or n.endswith("T4") or "A30" in n or "L40" in n:
        return "PCIe"
    return "unknown"


def hardware_info(override_ff: str | None) -> dict:
    fields = [
        "name",
        "memory.total",
        "power.limit",
        "power.max_limit",
        "clocks.max.sm",
        "clocks.max.memory",
        "pcie.link.gen.max",
        "pcie.link.width.max",
        "driver_version",
        "mig.mode.current",
    ]
    out = sh(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    rows = [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]
    if not rows:
        return {
            "gpu_name": "unknown",
            "gpu_count": 0,
            "form_factor": override_ff or "unknown",
            "memory_total_mib": None,
        }
    r = dict(zip(fields, rows[0], strict=True))

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "gpu_name": r["name"],
        "gpu_count": len(rows),
        "form_factor": override_ff or form_factor(r["name"]),
        "memory_total_mib": num(r["memory.total"]),
        "power_limit_w": num(r["power.limit"]),
        "power_max_limit_w": num(r["power.max_limit"]),
        "sm_clock_max_mhz": num(r["clocks.max.sm"]),
        "mem_clock_max_mhz": num(r["clocks.max.memory"]),
        "pcie_gen_max": num(r["pcie.link.gen.max"]),
        "pcie_width_max": num(r["pcie.link.width.max"]),
        "driver_version": r["driver_version"],
        "mig_mode": r["mig.mode.current"],
        "cpu_model": next(
            (line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), None
        ),
        "vcpus": os.cpu_count(),
    }


def software_info(image: str | None) -> dict:
    probe = (
        "import json,torch,vllm;print(json.dumps({'vllm':vllm.__version__,"
        "'torch':torch.__version__,'cuda':torch.version.cuda}))"
    )
    try:
        v = json.loads(sh([sys.executable, "-c", probe]) or "{}")
    except json.JSONDecodeError:
        v = {}
    image = image or os.environ.get("LAB_IMAGE") or "unknown"
    if "gpu-infra-lab" not in image:
        print(
            f"WARNING: lab image is probably not set (LAB_IMAGE={image!r}); "
            "pass --image or export LAB_IMAGE=ghcr.io/vkolupaev/gpu-infra-lab:<tag>@sha256:<digest>",
            file=sys.stderr,
        )
    elif "@sha256:" not in image:
        # the image cannot know its own digest before push: it comes from LAB_IMAGE on the host
        print(
            f"WARNING: lab image has no digest (LAB_IMAGE={image!r}); "
            "export LAB_IMAGE=<tag>@sha256:<digest> from notes/images.md",
            file=sys.stderr,
        )
    digest = image.split("@", 1)[1] if "@sha256:" in image else None
    return {
        "vllm_version": v.get("vllm", "unknown"),
        "torch_version": v.get("torch"),
        "cuda_runtime": v.get("cuda"),
        "image": image,
        "image_digest": digest,
        "base_image": os.environ.get("VLLM_BASE_IMAGE") or "unknown",
    }


def serve_flags(serve: dict) -> list[str]:
    """YAML dict -> CLI flags. True -> --flag, False -> --no-flag, list -> repeated values."""
    flags: list[str] = []
    for k, v in serve.items():
        if isinstance(v, bool):
            flags.append(f"--{k}" if v else f"--no-{k}")
        elif isinstance(v, (list, tuple)):
            flags += [f"--{k}", *map(str, v)]
        elif v is not None:
            flags += [f"--{k}", str(v)]
    return flags


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: int, log: pathlib.Path) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            tail = "".join(log.read_text(errors="replace").splitlines(True)[-40:])
            raise SystemExit(f"vllm serve exited with code {proc.returncode}. Log tail:\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except OSError:
            pass
        time.sleep(2)
    raise SystemExit(f"server not ready after {timeout_s}s, see {log}")


def stop_group(proc: subprocess.Popen, timeout_s: int = 60) -> None:
    """SIGTERM the process group, SIGKILL after timeout. A group that already exited is fine."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=timeout_s)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


MONITOR_FIELDS = (
    "timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,"
    "temperature.gpu,clocks.sm,clocks.mem,clocks_throttle_reasons.active"
)


def run_hwcheck(raw_dir: pathlib.Path, skip: bool, allow_suspicious: bool) -> dict:
    """scripts/hwcheck.py -> raw_dir/hwcheck.json; stop unless the verdict is OK."""
    if skip:
        print("WARNING: hwcheck skipped (--skip-hwcheck)", file=sys.stderr)
        return {"hbm_copy_gbps": None, "bf16_gemm_tflops": None, "hwcheck_verdict": None}
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "hwcheck.py")], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"hwcheck failed with code {p.returncode}:\n{p.stderr[-2000:]}")
    (raw_dir / "hwcheck.json").write_text(p.stdout)
    hc = json.loads(p.stdout)
    out = {
        "hbm_copy_gbps": hc.get("hbm_copy_gbps"),
        "bf16_gemm_tflops": hc.get("bf16_gemm_tflops"),
        "hwcheck_verdict": hc.get("verdict"),
    }  # no verdict: GPU not in hwcheck SPEC
    print(
        f"hwcheck: HBM {out['hbm_copy_gbps']} GB/s, BF16 GEMM {out['bf16_gemm_tflops']} TFLOPS, "
        f"verdict {out['hwcheck_verdict']}"
    )
    if out["hwcheck_verdict"] != "OK" and not allow_suspicious:
        raise SystemExit(
            f"hwcheck verdict is {out['hwcheck_verdict']!r}, see {raw_dir / 'hwcheck.json'}. "
            "Rent another host or pass --allow-suspicious."
        )
    return out


def start_monitor(path: pathlib.Path) -> subprocess.Popen | None:
    try:
        with path.open("w") as f:
            return subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={MONITOR_FIELDS}", "--format=csv", "-l", "1"],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as e:
        print(f"WARNING: GPU monitor not started: {e}", file=sys.stderr)
        return None


def stop_monitor(mon: subprocess.Popen | None) -> None:
    if mon is None:
        return
    if mon.poll() is not None:
        print(
            f"WARNING: nvidia-smi monitor exited by itself with code {mon.returncode}; "
            "gpu-monitor.csv may be incomplete",
            file=sys.stderr,
        )
        return
    stop_group(mon, timeout_s=10)


def pct(res: dict, metric: str) -> dict:
    g = lambda k: res.get(k)  # noqa: E731
    return {
        "mean": g(f"mean_{metric}_ms"),
        "p50": g(f"p50_{metric}_ms") if g(f"p50_{metric}_ms") is not None else g(f"median_{metric}_ms"),
        "p95": g(f"p95_{metric}_ms"),
        "p99": g(f"p99_{metric}_ms"),
    }


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and k not in ("serve_args", "extra"):
            out.update(flatten(v, key + "."))
        else:
            out[key] = json.dumps(v) if isinstance(v, (dict, list)) else v
    return out


def validate(rec: dict) -> list[str]:
    flat = flatten(rec)
    missing = sorted(p for p in REQUIRED if p not in flat)
    missing += sorted(f"{p}=empty" for p in NON_NULL if p not in missing and not flat.get(p))
    if rec.get("schema_version") != SCHEMA_VERSION:
        missing.append(f"schema_version!={SCHEMA_VERSION}")
    return missing


# ----------------------------------------------------------------------------- run
def build_record(cfg, args, res, *, c, n, group_id, git, hw, sw, raw_dir, ready_s) -> dict:
    eng, wl = cfg["serve"], cfg["workload"]
    out_tps = res.get("output_throughput") or 0.0
    tot_tps = res.get("total_token_throughput") or 0.0
    price = args.price_per_hour * args.gpu_fraction

    def usd_per_1m(tps):
        return round(price / (tps * 3600) * 1e6, 4) if tps else None

    precision = eng.get("dtype", "auto")
    quant = eng.get("quantization")
    bw = GPU_MEM_BW_GBPS.get(hw["gpu_name"].replace("NVIDIA ", ""))
    params_b = cfg.get("model_params_b")
    floor_ms = None
    if bw and params_b:
        bpp = BYTES_PER_PARAM.get(quant or precision, 2)
        floor_ms = round(params_b * 1e9 * bpp / (bw * 1e9) * 1e3, 2)
    completed = res.get("completed", 0)
    model_short = cfg["model"].split("/")[-1].lower()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{group_id}_c{c}",
        "group_id": group_id,
        "timestamp_utc": utcnow(),
        "tag": args.tag,
        "note": args.note,
        "git": git,
        "provider": {
            "name": args.provider,
            "instance_type": args.instance_type,
            "price_per_hour_usd": args.price_per_hour,
            "price_basis": args.price_basis,
            "gpu_sharing": args.gpu_sharing,
            "gpu_fraction": args.gpu_fraction,
        },
        "hardware": hw,
        "software": sw,
        "model": {
            "id": cfg["model"],
            "short": model_short,
            "revision": cfg["revision"],
            "precision": quant or precision,
            "dtype": precision,
            "quantization": quant,
            "params_b": params_b,
        },
        "engine": {
            "max_num_seqs": eng.get("max-num-seqs"),
            "max_num_batched_tokens": eng.get("max-num-batched-tokens"),
            "max_model_len": eng.get("max-model-len"),
            "gpu_memory_utilization": eng.get("gpu-memory-utilization"),
            "enable_prefix_caching": eng.get("enable-prefix-caching"),
            "enable_chunked_prefill": eng.get("enable-chunked-prefill"),
            "tensor_parallel_size": eng.get("tensor-parallel-size", 1),
            "server_ready_s": round(ready_s, 1),
            "serve_args": eng,
        },
        "workload": {
            "dataset": "random",
            "input_len": wl["input_len"],
            "output_len": wl["output_len"],
            "concurrency": c,
            "num_prompts": n,
            "request_rate": "inf",
            "ignore_eos": wl.get("ignore_eos", True),
            "seed": wl.get("seed", 0),
            "warmup_prompts": wl.get("warmup_prompts", 0),
        },
        "results": {
            "completed": completed,
            "failed": n - completed,
            "duration_s": res.get("duration"),
            "request_throughput_rps": res.get("request_throughput"),
            "output_throughput_tps": out_tps,
            "total_throughput_tps": tot_tps,
            "ttft_ms": pct(res, "ttft"),
            "tpot_ms": pct(res, "tpot"),
            "itl_ms": pct(res, "itl"),
            "e2el_ms": pct(res, "e2el"),
        },
        "derived": {
            "usd_per_1m_output_tokens": usd_per_1m(out_tps),
            "usd_per_1m_total_tokens": usd_per_1m(tot_tps),
            "output_tokens_per_usd": round(out_tps * 3600 / price) if price and out_tps else None,
            "decode_floor_ms_per_token": floor_ms,  # weights / peak bandwidth, batch=1 lower bound
            "p99_reliable": completed >= 200,
        },
        "raw": {"dir": str(raw_dir.relative_to(ROOT)), "vllm_result": f"c{c}.json", "server_log": "server.log"},
        "extra": args.extra or {},
    }


def bench_cmd(cfg, model_path, port, c, n, wl, out_dir=None, name=None) -> list[str]:
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--endpoint",
        "/v1/completions",
        "--model",
        cfg["model"],
        "--tokenizer",
        model_path,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(wl["input_len"]),
        "--random-output-len",
        str(wl["output_len"]),
        "--num-prompts",
        str(n),
        "--max-concurrency",
        str(c),
        "--request-rate",
        "inf",
        "--seed",
        str(wl.get("seed", 0)),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
    ]
    if wl.get("ignore_eos", True):
        cmd.append("--ignore-eos")
    if out_dir:
        cmd += ["--save-result", "--result-dir", str(out_dir), "--result-filename", name]
    return cmd


def cmd_run(args) -> None:
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    rev = cfg.get("revision")
    if not isinstance(rev, str) or not rev or rev.startswith("REPLACE"):
        raise SystemExit("pin model revision in the config as a QUOTED commit sha before running")
    wl = cfg["workload"]
    levels = [int(x) for x in (args.concurrency.split(",") if args.concurrency else wl["concurrency"])]
    git = git_info()
    if git["dirty"]:
        print("WARNING: git tree is dirty - record will carry dirty=true. Commit code before runs.")
    hw, sw = hardware_info(args.form_factor), software_info(args.image)
    group_id = (
        f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{cfg['model'].split('/')[-1].lower()}_{git['commit'][:7]}"
    )
    raw_dir = RESULTS / "raw" / group_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    # atexit runs after Python prints the error (SystemExit message or traceback) and keeps its exit code
    def remind() -> None:
        sys.stdout.flush()  # a piped stdout is block-buffered and flushed only after atexit
        print(
            f"raw outputs kept in results/raw/{group_id}/ - copy them off the host before destroying it",
            file=sys.stderr,
            flush=True,
        )

    atexit.register(remind)
    (raw_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (raw_dir / "nvidia-smi-q.txt").write_text(sh(["nvidia-smi", "-q"]))
    # 0. is the GPU what the listing says? (before the download, so a bad host costs minutes)
    hw.update(run_hwcheck(raw_dir, args.skip_hwcheck, args.allow_suspicious))

    # 1. pinned model snapshot
    from huggingface_hub import snapshot_download

    t0 = time.time()
    model_path = snapshot_download(cfg["model"], revision=cfg["revision"])
    print(f"model ready in {time.time() - t0:.0f}s: {model_path}")

    # 2. server
    port = args.port
    log = raw_dir / "server.log"
    serve = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        cfg["model"],
        "--port",
        str(port),
        *serve_flags(cfg["serve"]),
    ]
    (raw_dir / "serve_cmd.txt").write_text(" ".join(serve) + "\n")
    mon = start_monitor(raw_dir / "gpu-monitor.csv")
    proc = None
    try:
        with log.open("w") as lf:
            proc = subprocess.Popen(serve, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        ready_s = wait_ready(port, proc, args.ready_timeout, log)
        print(f"server ready in {ready_s:.0f}s")
        # 3. warmup (not recorded)
        warm = wl.get("warmup_prompts", 0)
        if warm:
            subprocess.run(
                bench_cmd(cfg, model_path, port, max(levels), warm, wl), check=True, stdout=subprocess.DEVNULL
            )
        # 4. sweep
        records = []
        for c in levels:
            n = args.num_prompts or max(wl.get("min_prompts", 50), wl.get("prompts_per_concurrency", 10) * c)
            name = f"c{c}.json"
            print(f"--- concurrency={c} num_prompts={n}")
            subprocess.run(bench_cmd(cfg, model_path, port, c, n, wl, raw_dir, name), check=True)
            res = json.loads((raw_dir / name).read_text())
            rec = build_record(
                cfg, args, res, c=c, n=n, group_id=group_id, git=git, hw=hw, sw=sw, raw_dir=raw_dir, ready_s=ready_s
            )
            problems = validate(rec)
            if problems:
                print(f"WARNING: record incomplete: {problems}")
            records.append(rec)
            with (RESULTS / "runs.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
    finally:
        if proc is not None:
            stop_group(proc)  # a crashed server must surface its own error, not ProcessLookupError
        stop_monitor(mon)
    atexit.unregister(remind)
    note = draft_note(records, cfg, args)
    print(f"\n{len(records)} records -> results/runs.jsonl\nexperiment note -> {note}")


# ----------------------------------------------------------------------------- notes
def fmt(v, nd=0):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))


def draft_note(records: list[dict], cfg: dict, args) -> pathlib.Path:
    EXPERIMENTS.mkdir(exist_ok=True)
    r0 = records[0]
    slug = args.tag or r0["model"]["short"]
    path = EXPERIMENTS / f"{dt.date.today():%Y-%m-%d}-{slug}.md"
    i = 2
    while path.exists():
        path = EXPERIMENTS / f"{dt.date.today():%Y-%m-%d}-{slug}-{i}.md"
        i += 1
    hw, e, w = r0["hardware"], r0["engine"], r0["workload"]
    rows = "\n".join(
        f"| {r['workload']['concurrency']} | {fmt(r['results']['output_throughput_tps'])} "
        f"| {fmt(r['results']['total_throughput_tps'])} | {fmt(r['results']['ttft_ms']['p50'])} "
        f"/ {fmt(r['results']['ttft_ms']['p95'])} | {fmt(r['results']['tpot_ms']['p50'], 1)} "
        f"| {fmt(r['results']['e2el_ms']['p50'])} / {fmt(r['results']['e2el_ms']['p95'])} "
        f"/ {fmt(r['results']['e2el_ms']['p99'])}{'' if r['derived']['p99_reliable'] else '*'} "
        f"| {fmt(r['derived']['usd_per_1m_output_tokens'], 3)} "
        f"| {fmt(r['derived']['usd_per_1m_total_tokens'], 3)} | {r['results']['completed']} |"
        for r in records
    )
    raw_dir = ROOT / r0["raw"]["dir"]
    raw_files = (
        ", ".join(f"`{f}`" for f in ("gpu-monitor.csv", "nvidia-smi-q.txt", "hwcheck.json") if (raw_dir / f).exists())
        or "none"
    )
    hwcheck = (
        (
            f"HBM copy {hw.get('hbm_copy_gbps')} GB/s, BF16 GEMM {hw.get('bf16_gemm_tflops')} TFLOPS, "
            f"verdict {hw.get('hwcheck_verdict')}"
        )
        if hw.get("hbm_copy_gbps") is not None
        else "skipped"
    )
    floor = r0["derived"]["decode_floor_ms_per_token"]
    tpot_c1 = next((r["results"]["tpot_ms"]["p50"] for r in records if r["workload"]["concurrency"] == 1), None)
    sanity = (
        (
            f"Decode floor (weights / peak HBM bandwidth): **{floor} ms/token**. "
            f"Measured TPOT p50 at concurrency 1: **{fmt(tpot_c1, 1)} ms** "
            f"-> {fmt(floor / tpot_c1 * 100 if floor and tpot_c1 else None)}% of peak bandwidth "
            "(expect ~60-85%; >100% = measurement error, <40% = investigate)."
        )
        if floor
        else "Decode floor: unknown GPU in GPU_MEM_BW_GBPS or model_params_b missing."
    )
    path.write_text(f"""# {slug}: {r0["model"]["short"]} on {hw["gpu_name"]}

- Date: {dt.date.today():%Y-%m-%d}
- Run group: `{r0["group_id"]}`
- Commit: `{r0["git"]["commit"][:12]}` (dirty: {r0["git"]["dirty"]})
- Status: draft

## Question
TODO: one question this run answers. One variable changes per experiment.

## Hypothesis
TODO: expected outcome and why, written BEFORE looking at the results.

## Setup (auto)
| | |
|---|---|
| Provider | {r0["provider"]["name"]} {r0["provider"]["instance_type"] or ""}, ${r0["provider"]["price_per_hour_usd"]}/h ({r0["provider"]["price_basis"]}), sharing: {r0["provider"]["gpu_sharing"]} |
| GPU | {hw["gpu_count"]}x {hw["gpu_name"]} ({hw["form_factor"]}), {fmt(hw.get("memory_total_mib"))} MiB, power limit {fmt(hw.get("power_limit_w"))} W, driver {hw.get("driver_version")} |
| Software | vLLM {r0["software"]["vllm_version"]}, torch {r0["software"]["torch_version"]}, CUDA {r0["software"]["cuda_runtime"]}, image `{r0["software"]["image"]}` (base `{r0["software"].get("base_image", "unknown")}`) |
| Model | `{r0["model"]["id"]}` @ `{r0["model"]["revision"][:12]}`, precision {r0["model"]["precision"]} |
| Engine | max_num_seqs={e["max_num_seqs"]}, max_num_batched_tokens={e["max_num_batched_tokens"]}, max_model_len={e["max_model_len"]}, gpu_mem_util={e["gpu_memory_utilization"]}, prefix_caching={e["enable_prefix_caching"]} |
| Workload | random, {w["input_len"]} in / {w["output_len"]} out tokens, ignore_eos={w["ignore_eos"]}, closed loop (max-concurrency), seed={w["seed"]} |
| Monitoring | raw files {raw_files} (nvidia-smi 1 sample/s); hwcheck: {hwcheck}; plots: `python3 scripts/plot_gpu_monitor.py {r0["raw"]["dir"]}` |

## Results (auto)
| Concurrency | Out tok/s | Total tok/s | TTFT p50/p95 ms | TPOT p50 ms | E2E p50/p95/p99 ms | $/1M out | $/1M total | n |
|---|---|---|---|---|---|---|---|---|
{rows}

`*` p99 from fewer than 200 requests - treat as approximate max, not a percentile.
$/1M = price_per_hour x gpu_fraction / (throughput x 3600) x 1e6, at 100% utilization of the GPU.

## Sanity check (auto)
{sanity}

## Observations
TODO: what the numbers show, no interpretation yet.

## Interpretation
TODO: why. Which resource is the bottleneck at each concurrency level?

## What did not work / surprises
TODO

## Context: reference points
TODO: put the number next to section 7 reference points (e.g. how many such GPUs a
650B tokens/month load would need at this config and an assumed utilization).

## Next
TODO: the single next experiment this result suggests.
""")
    return path


# ----------------------------------------------------------------------------- cli
def cmd_validate(args) -> None:
    bad = 0
    for i, line in enumerate(pathlib.Path(args.file).read_text().splitlines(), 1):
        problems = validate(json.loads(line))
        if problems:
            bad += 1
            print(f"line {i}: missing {problems}")
    print("OK" if not bad else f"{bad} invalid record(s)")
    sys.exit(1 if bad else 0)


def cmd_tocsv(args) -> None:
    rows = [flatten(json.loads(line)) for line in pathlib.Path(args.file).read_text().splitlines()]
    cols = sorted({k for r in rows for k in r})
    w = csv.DictWriter(sys.stdout, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--provider", required=True, help="vast | runpod | crusoe | work-vm ...")
    r.add_argument("--price-per-hour", type=float, required=True, help="USD per hour for the whole instance")
    r.add_argument("--price-basis", default="on-demand", help="on-demand | interruptible | internal-estimate")
    r.add_argument("--instance-type", default=None)
    r.add_argument("--gpu-sharing", default="none", help="none | mig:3g.40gb | mps:50 | timeslice:2")
    r.add_argument("--gpu-fraction", type=float, default=1.0, help="share of the price attributed to this run")
    r.add_argument("--form-factor", default=None, help="override autodetect: SXM | PCIe | NVL")
    r.add_argument(
        "--image", default=None, help="lab image ref, e.g. ghcr.io/vkolupaev/gpu-infra-lab:<tag> (default: $LAB_IMAGE)"
    )
    r.add_argument("--concurrency", default=None, help="e.g. 1,8,32 (default: from config)")
    r.add_argument("--num-prompts", type=int, default=None, help="fixed count for every level")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--ready-timeout", type=int, default=1200)
    r.add_argument("--tag", default=None)
    r.add_argument("--note", default=None)
    r.add_argument("--skip-hwcheck", action="store_true", help="no scripts/hwcheck.py (mock runs only)")
    r.add_argument("--allow-suspicious", action="store_true", help="continue when the hwcheck verdict is not OK")
    r.add_argument("--extra", type=json.loads, default=None, help="free-form JSON, e.g. '{\"host_id\": 123}'")
    r.set_defaults(func=cmd_run)
    v = sub.add_parser("validate")
    v.add_argument("file")
    v.set_defaults(func=cmd_validate)
    c = sub.add_parser("tocsv")
    c.add_argument("file")
    c.set_defaults(func=cmd_tocsv)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
