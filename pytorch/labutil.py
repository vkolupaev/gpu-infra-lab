"""labutil.py - shared helpers for the Block 1 lesson scripts in pytorch/.

- context of every point: GPU, versions, git commit (reuses bench.py, so records agree);
- timing that respects CUDA asynchrony: wall time with synchronize, or CUDA events;
- GPU kernel list of one call from a torch.profiler trace;
- one JSON line per sweep point in results/<experiment>/points.jsonl (format: README, "Compare points").
"""

import itertools
import json
import os
import pathlib
import re
import statistics
import sys
import tempfile
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEMM = re.compile(r"gemm|xmma|cutlass|s16816|s1688|cublas|tem_fused_mm", re.I)
ATTN = re.compile(r"flash|fmha|attention", re.I)


def _bench():
    """bench.py from the repo root: one source for git_info, utcnow and sh."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import bench

    return bench


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def time_ms(fn, device: str, iters: int = 20, warmup: int = 3) -> float:
    """Median wall time of fn() in ms, with a synchronize after every call.

    Without the synchronize this measures how long Python takes to LAUNCH the work,
    not how long the GPU takes to DO it (lesson 1.1).
    """
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


class Marks:
    """Timestamps between phases of one iteration without forcing a sync.

    GPU: CUDA events recorded into the stream (read after one synchronize at the end).
    CPU: perf_counter (CPU work is synchronous anyway).
    """

    def __init__(self, device: str):
        self.cuda = device == "cuda"
        self.m = []

    def mark(self) -> None:
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.m.append(e)
        else:
            self.m.append(time.perf_counter())

    def intervals_ms(self) -> list[float]:
        if self.cuda:
            return [a.elapsed_time(b) for a, b in itertools.pairwise(self.m)]
        return [(b - a) * 1e3 for a, b in itertools.pairwise(self.m)]


def median_phases(marks: list[Marks]) -> list[float]:
    """Per-phase median over iterations; call after a synchronize."""
    rows = [m.intervals_ms() for m in marks]
    return [statistics.median(col) for col in zip(*rows, strict=True)]


def trace_kernels(fn, device: str, path=None) -> list[dict]:
    """Run fn() once under torch.profiler and return its GPU activities in launch order.

    Kineto marks them in the Chrome trace with cat "kernel", "gpu_memcpy", "gpu_memset";
    `dur` is in microseconds. With `path` the trace is kept (open it in ui.perfetto.dev).
    """
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    keep = path is not None
    if path is None:
        with tempfile.NamedTemporaryFile(suffix=".trace.json", delete=False) as f:
            path = f.name
    with profile(activities=acts) as prof:
        fn()
        sync(device)
    prof.export_chrome_trace(str(path))
    events = json.loads(pathlib.Path(path).read_text())["traceEvents"]
    if not keep:
        os.unlink(path)
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    gpu.sort(key=lambda e: e["ts"])
    return [{"name": e["name"], "cat": e["cat"], "dur_us": float(e.get("dur", 0))} for e in gpu]


def kernel_class(name: str) -> str:
    return "gemm" if GEMM.search(name) else "attention" if ATTN.search(name) else "other"


def kernel_summary(kernels: list[dict]) -> dict:
    """Total GPU time and launch count, split into GEMM / attention / other kernels."""
    out = {"gpu_us": sum(k["dur_us"] for k in kernels), "n_kernels": len(kernels)}
    for cls in ("gemm", "attention", "other"):
        out[f"{cls}_us"] = sum(k["dur_us"] for k in kernels if kernel_class(k["name"]) == cls)
    return out


def top_kernels(kernels: list[dict], n: int = 25) -> str:
    """Kernels grouped by name, sorted by total time: a plain-text table for the note."""
    agg = {}
    for k in kernels:
        t, c = agg.get(k["name"], (0.0, 0))
        agg[k["name"]] = (t + k["dur_us"], c + 1)
    total = sum(t for t, _ in agg.values()) or 1.0
    lines = [f"{'total us':>10} {'share':>6} {'count':>6}  class      name"]
    for name, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:n]:
        lines.append(f"{t:10.1f} {t / total:6.1%} {c:6d}  {kernel_class(name):<9}  {name[:140]}")
    return "\n".join(lines) + "\n"


def context(device: str) -> dict:
    """What a point needs to be reproducible: hardware, versions, code."""
    b = _bench()
    ctx = {
        "device": device,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "image": os.environ.get("LAB_IMAGE"),
        "git": b.git_info(),
    }
    try:
        import transformers

        ctx["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        ctx.update(gpu_name=p.name, gpu_mem_total_mib=p.total_memory // 2**20, sm_count=p.multi_processor_count)
        smi = b.sh(
            [
                "nvidia-smi",
                "--query-gpu=power.limit,clocks.max.sm,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        if smi:
            try:
                power, clock, driver = (x.strip() for x in smi.splitlines()[0].split(","))
                ctx.update(power_limit_w=float(power), sm_clock_max_mhz=int(clock), driver_version=driver)
            except ValueError:  # e.g. "[N/A]" or another column count: keep the raw line, do not stop the run
                ctx["nvidia_smi_raw"] = smi
    else:
        ctx.update(gpu_name="cpu", cpu_threads=torch.get_num_threads())
    return ctx


class Points:
    """results/<experiment>/points.jsonl: one line per sweep point, plus env.json once."""

    def __init__(self, out_dir, experiment: str, device: str, append: bool = False):
        self.dir = pathlib.Path(out_dir)
        self.path = self.dir / "points.jsonl"
        if self.path.exists() and not append:
            raise SystemExit(f"{self.path} exists: pass --append or a new --out")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.ctx = context(device)
        env = self.dir / ("env.json" if not append else f"env-{int(time.time())}.json")
        env.write_text(json.dumps(self.ctx, indent=2) + "\n")

    def add(self, **point) -> None:
        rec = {
            "experiment": self.experiment,
            **point,
            "gpu_name": self.ctx["gpu_name"],
            "git": self.ctx["git"],
            "timestamp_utc": _bench().utcnow(),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(point), flush=True)

    def write_text(self, name: str, text: str) -> None:
        (self.dir / name).write_text(text)
