#!/usr/bin/env python3
"""compile_speedup.py - torch.compile with and without, and what got fused (lesson 1.5).

Part A, TinyGPT forward (inference), points = (shape, attention, mode):
  B=8,  S=1024, sdpa  : eager, default            compute-heavy: GEMMs dominate
  B=8,  S=1024, naive : eager, default            how much of naive attention the compiler fuses
  B=1,  S=128,  sdpa  : eager, default, reduce-overhead   launch-bound: CUDA graphs matter
For every point: first call (includes compilation), steady time, kernels per forward and
their split GEMM / attention / other; kernels-<point>.txt lists kernels by name
(inductor names them after the fused ops: triton_poi_fused_add_native_layer_norm_...).
Part B (--hf): Qwen3-8B decode at P=1024 in HF transformers: dynamic cache eager,
static cache eager, static cache + torch.compile(mode="reduce-overhead") -> ITL vs
vLLM first-number p50@ITL. Time-box it: the run_session script wraps it in `timeout`.

    TORCH_LOGS=output_code python3 pytorch/compile_speedup.py --out results/2026-11-07-compile \\
        2> results/2026-11-07-compile/output_code.txt
    python3 pytorch/compile_speedup.py --out results/2026-11-07-compile --append --hf --revision <sha>
    python3 pytorch/compile_speedup.py --out /tmp/cs --quick     # CPU dry run (needs gcc)
"""

import argparse
import statistics
import time

import torch

from labutil import Points, default_device, kernel_summary, sync, time_ms, top_kernels, trace_kernels
from tinygpt import Config, TinyGPT

CASES = [
    (8, 1024, "sdpa", ["eager", "default"]),
    (8, 1024, "naive", ["eager", "default"]),
    (1, 128, "sdpa", ["eager", "default", "reduce-overhead"]),
]


def part_a(a, dev, pts):
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    cases = [(2, 64, at, ["eager", "default"]) for _, _, at, _ in CASES[:2]] if a.quick else CASES
    for B, S, attn, modes in cases:
        for mode in modes:
            torch._dynamo.reset()
            torch.manual_seed(0)
            c = Config(attn=attn) if not a.quick else Config(vocab=1024, d=128, heads=4, layers=2, seq=64, attn=attn)
            model = TinyGPT(c).to(dev, dtype).eval()
            f = model if mode == "eager" else torch.compile(model, mode=mode)
            idx = torch.randint(0, c.vocab, (B, S), device=dev)
            with torch.inference_mode():
                sync(dev)
                t0 = time.perf_counter()
                f(idx)
                sync(dev)
                first_ms = (time.perf_counter() - t0) * 1e3
                ms = time_ms(lambda f=f, i=idx: f(i), dev, iters=20, warmup=3)
                ks = trace_kernels(lambda f=f, i=idx: f(i), dev)
            name = f"b{B}-s{S}-{attn}-{mode}"
            pts.write_text(f"kernels-{name}.txt", top_kernels(ks, n=40))
            pts.add(
                group_id=f"b{B}-s{S}-{attn}",
                part="tinygpt",
                batch=B,
                seq=S,
                attn=attn,
                mode=mode,
                first_call_ms=round(first_ms, 1),
                forward_ms=round(ms, 3),
                **kernel_summary(ks),
            )
            del model, f


def part_b(a, dev, pts):
    from prefill_decode import generate_timed, load

    a.tiny = a.quick
    model, info = load(a, dev)
    eager_forward = model.forward
    P, N = (1024, 128) if not a.quick else (32, 8)
    variants = [
        ("dynamic-eager", "dynamic", False),
        ("static-eager", "static", False),
        ("static-compiled", "static", True),
    ]
    for label, cache, compiled in variants:
        torch._dynamo.reset()
        model.forward = (
            torch.compile(eager_forward, mode="reduce-overhead", fullgraph=True) if compiled else eager_forward
        )
        t0 = time.perf_counter()
        generate_timed(model, P, N, dev, cache)  # first call: compiles prefill and decode graphs
        first_s = time.perf_counter() - t0
        generate_timed(model, P, N, dev, cache)
        ttfts, itls = [], []
        for r in range(3):
            ttft, itl = generate_timed(model, P, N, dev, cache, seed=r)
            ttfts.append(ttft)
            itls += itl
        pts.add(
            group_id="hf-decode",
            part="hf",
            mode=label,
            prompt_len=P,
            new_tokens=N,
            first_call_s=round(first_s, 1),
            ttft_ms=round(statistics.median(ttfts), 2),
            itl_p50_ms=round(statistics.median(itls), 3),
            model=info["model"],
            revision=info["revision"],
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--hf", action="store_true", help="part B instead of part A")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--revision")
    p.add_argument("--append", action="store_true")
    p.add_argument("--quick", action="store_true", help="tiny shapes: CPU dry run")
    a = p.parse_args()
    dev = default_device()
    pts = Points(a.out, "compile", dev, append=a.append)
    if a.hf:
        if not a.quick and not a.revision:
            raise SystemExit("--revision is required with --hf")
        part_b(a, dev, pts)
    else:
        part_a(a, dev, pts)


if __name__ == "__main__":
    main()
