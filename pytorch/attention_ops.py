#!/usr/bin/env python3
"""attention_ops.py - attention as a set of operations: paper vs measurement (lesson 1.3).

The core of ONE Qwen3-8B attention layer at prefill, batch 1, forward only: 32 query
heads, 8 KV heads, head_dim 128 (the Q/K/V/O projections are not included). Variants:
  naive       transformers' eager pattern: repeat_kv -> Q @ K^T * scale -> + causal mask
              -> softmax in fp32 -> cast to bf16 -> P @ V. Materializes S x S per head.
  naive-bf16  the same with softmax in bf16: identical FLOPs, fewer bytes. If time drops
              with the bytes, the step is memory-bound (a substitute for ncu counters).
  sdpa        F.scaled_dot_product_attention(is_causal=True), FlashAttention backend forced.
For every point: time, extra peak memory over the inputs, FLOPs from FlopCounterMode.
With --profile-seq the per-kernel split of every variant at that S goes to kernels-*.txt.

    python3 pytorch/attention_ops.py --out results/2026-11-07-attention-ops --profile-seq 4096
    python3 pytorch/attention_ops.py --out /tmp/ao --quick       # CPU dry run
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from labutil import Points, default_device, kernel_summary, sync, time_ms, top_kernels, trace_kernels

MIB = 2**20


def core_flops(S: int, hq: int, hd: int) -> int:
    """Q @ K^T and P @ V over the full S x S: 2 matmuls x 2*S*S*hd per query head."""
    return 4 * hq * S * S * hd


def naive(q, k, v, mask, softmax_dtype):
    rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(rep, dim=1)  # repeat_kv: GQA heads expanded to query heads (a copy)
    v = v.repeat_interleave(rep, dim=1)
    w = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    w = w + mask
    w = torch.softmax(w, dim=-1, dtype=softmax_dtype).to(q.dtype)
    return w @ v


GQA = {}  # does the forced backend accept enable_gqa? decided on the first call


def sdpa(q, k, v):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = [SDPBackend.FLASH_ATTENTION] if q.is_cuda else [SDPBackend.MATH]
    with sdpa_kernel(backends):
        if GQA.get("native", True):
            try:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                GQA["native"] = True
                return out
            except RuntimeError:  # this backend has no native GQA: expand K/V like repeat_kv
                GQA["native"] = False
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--seqs", default="512,1024,2048,4096,8192,16384")
    p.add_argument("--hq", type=int, default=32)
    p.add_argument("--hkv", type=int, default=8)
    p.add_argument("--hd", type=int, default=128)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--profile-seq", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="short sequences, fp32: CPU dry run")
    a = p.parse_args()
    dev = default_device()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    if a.quick:
        a.seqs, a.iters = "128,256,512", 3
    pts = Points(a.out, "attention-ops", dev)
    variants = {
        "naive": lambda q, k, v, m: naive(q, k, v, m, torch.float32),
        "naive-bf16": lambda q, k, v, m: naive(q, k, v, m, dtype),
        "sdpa": lambda q, k, v, m: sdpa(q, k, v),
    }

    for S in (int(s) for s in a.seqs.split(",")):
        torch.manual_seed(0)
        q = torch.randn(1, a.hq, S, a.hd, device=dev, dtype=dtype)
        k = torch.randn(1, a.hkv, S, a.hd, device=dev, dtype=dtype)
        v = torch.randn(1, a.hkv, S, a.hd, device=dev, dtype=dtype)
        # built once per forward and shared by all layers in transformers, so not timed here
        mask = torch.full((S, S), float("-inf"), device=dev, dtype=dtype).triu(1)
        for name, fn in variants.items():
            try:
                with torch.inference_mode():
                    with FlopCounterMode(display=False) as fc:
                        fn(q, k, v, mask)
                    sync(dev)
                    base = torch.cuda.memory_allocated() if dev == "cuda" else 0
                    if dev == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    out = fn(q, k, v, mask)
                    sync(dev)
                    peak = torch.cuda.max_memory_allocated() - base if dev == "cuda" else None
                    del out
                    ms = time_ms(lambda f=fn, x=(q, k, v, mask): f(*x), dev, iters=a.iters, warmup=2)
                    if S == a.profile_seq:
                        ks = trace_kernels(lambda f=fn, x=(q, k, v, mask): f(*x), dev)
                        pts.write_text(f"kernels-{name}-S{S}.txt", top_kernels(ks))
                f = core_flops(S, a.hq, a.hd)
                pts.add(
                    group_id=name,
                    variant=name,
                    seq=S,
                    time_ms=round(ms, 3),
                    peak_extra_mib=round(peak / MIB, 1) if peak is not None else None,
                    scores_bf16_mib=round(a.hq * S * S * 2 / MIB, 1),
                    flops_core_full=f,
                    flops_counted=fc.get_total_flops(),
                    tflops_full=round(f / (ms * 1e-3) / 1e12, 2),
                    gqa_native=GQA.get("native") if name == "sdpa" else None,
                    **({"profile": kernel_summary(ks)} if S == a.profile_seq else {}),
                )
            except torch.OutOfMemoryError:
                pts.add(group_id=name, variant=name, seq=S, oom=True)
            if dev == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
