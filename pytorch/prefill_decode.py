#!/usr/bin/env python3
"""prefill_decode.py - prompt processing vs per-token generation in HF transformers (lesson 1.4).

Batch 1, greedy, random prompt tokens, a fixed number of new tokens (min_new_tokens =
max_new_tokens, the analogue of --ignore-eos). A streamer stamps every token as it
reaches the CPU: TTFT = first stamp - call, ITL = gaps between later stamps.
Swept variable: prompt length P. For every P: TTFT, ITL p50/p90 over all decode steps.
--busy adds GPU time per decode step from a profiler trace (kernel time of a 33-token
generate minus a 1-token generate, divided by 32) and the number of kernels per step:
busy / ITL = the share of a decode step the GPU actually works.
At P=1024 the trace of that 33-token generate (trace-busy-<model>-P1024.trace.json, not in git)
and its top kernels (kernels-busy-<model>-P1024.txt) are kept next to the points.
--nsys-decode: one P=1024 run, cudaProfilerStart/Stop around the decode steps and an
NVTX range per step; run it under `nsys profile --capture-range=cudaProfilerApi`.

    python3 pytorch/prefill_decode.py --model Qwen/Qwen3-8B --revision <sha> \\
        --out results/2026-11-07-prefill-decode --busy
    python3 pytorch/prefill_decode.py --tiny --out /tmp/pd --busy   # CPU dry run, random tiny Qwen3
"""

import argparse
import itertools
import statistics
import time

import torch

from labutil import Points, default_device, kernel_summary, sync, top_kernels, trace_kernels

try:
    from transformers.generation.streamers import BaseStreamer
except ImportError:  # moved or removed: generate() only calls put() and end()
    BaseStreamer = object


class Clock(BaseStreamer):
    """Streamer that only stamps time: generate() calls put() with the prompt, then per token."""

    def __init__(self, nvtx=False, n_new=None):
        super().__init__()
        self.t, self.nvtx, self.n_new, self.open = [], nvtx, n_new, False

    def put(self, value):
        self.t.append(time.perf_counter())
        if not self.nvtx:
            return
        if self.open:
            torch.cuda.nvtx.range_pop()
            self.open = False
        n = len(self.t) - 1  # tokens so far; t[0] is the prompt
        # a range per decode step: none after the last token, there is no step left to cover
        if n >= 1 and (self.n_new is None or n < self.n_new):
            torch.cuda.nvtx.range_push(f"decode_step_{n}")
            self.open = True

    def end(self):
        if self.open:
            torch.cuda.nvtx.range_pop()
            self.open = False


def load(a, dev):
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    if a.tiny:
        from transformers import Qwen3Config

        cfg = Qwen3Config(
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=8192,
        )
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
        path = "tiny-random-qwen3"
    else:
        from huggingface_hub import snapshot_download

        path = snapshot_download(a.model, revision=a.revision)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
    model = model.to(dev, dtype).eval()
    c = model.config
    hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    bytes_per = torch.finfo(dtype).bits // 8
    storages = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in model.parameters()}
    info = {
        "model": a.model if not a.tiny else path,
        "revision": a.revision,
        "layers": c.num_hidden_layers,
        "hidden": c.hidden_size,
        "q_heads": c.num_attention_heads,
        "kv_heads": c.num_key_value_heads,
        "head_dim": hd,
        "vocab": c.vocab_size,
        "tie_word_embeddings": bool(getattr(c, "tie_word_embeddings", False)),
        "weight_bytes": sum(storages.values()),  # tied embedding counted once
        "kv_bytes_per_token": c.num_hidden_layers * c.num_key_value_heads * hd * 2 * bytes_per,
        "attn_implementation": getattr(c, "_attn_implementation", None),
    }
    return model, info


def make_gen_config(model, n_new, cache):
    from transformers import GenerationConfig

    eos = model.generation_config.eos_token_id
    eos = eos if eos is not None else 0
    pad = eos[0] if isinstance(eos, list) else eos
    kw = {"cache_implementation": "static"} if cache == "static" else {}
    # disable_compile: with a static cache on cuda, generate() would torch.compile the forward by itself
    # (transformers 5.x auto-compile), so "eager" would not be eager; compile_speedup.py compiles explicitly
    return GenerationConfig(
        max_new_tokens=n_new,
        min_new_tokens=n_new,
        do_sample=False,
        eos_token_id=eos,
        pad_token_id=pad,
        disable_compile=True,
        **kw,
    )


def generate_timed(model, prompt_len, n_new, dev, cache="dynamic", nvtx=False, seed=0):
    """One generate(); returns (ttft_ms, [itl_ms, ...])."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), generator=g).to(dev)
    clock = Clock(nvtx=nvtx, n_new=n_new)
    gc = make_gen_config(model, n_new, cache)
    sync(dev)
    t0 = time.perf_counter()
    with torch.inference_mode():
        model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=gc, streamer=clock)
    sync(dev)
    t = clock.t[1:]  # clock.t[0] is the prompt
    return (t[0] - t0) * 1e3, [(b - a) * 1e3 for a, b in itertools.pairwise(t)]


def busy_per_step(model, prompt_len, dev, steps=32, trace=None):
    """GPU time and kernel count per decode step: trace(1 + steps tokens) - trace(1 token).

    With `trace` the trace of the (1 + steps)-token generate is kept at that path.
    Returns the per-step sums and that generate's kernels.
    """
    one = kernel_summary(trace_kernels(lambda: generate_timed(model, prompt_len, 1, dev), dev))
    ks = trace_kernels(lambda: generate_timed(model, prompt_len, 1 + steps, dev), dev, path=trace)
    many = kernel_summary(ks)
    return {k: (many[k] - one[k]) / steps for k in many}, ks


def pctl(xs, q):
    return statistics.quantiles(xs, n=100)[q - 1] if len(xs) > 1 else xs[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--revision", help="commit sha; for Qwen3-8B the one in configs/qwen3-8b-bf16.yaml")
    p.add_argument("--prompt-lens", default="16,32,64,128,256,512,1024,2048,4096,8192")
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--cache", default="dynamic", choices=["dynamic", "static"])
    p.add_argument("--busy", action="store_true")
    p.add_argument("--nsys-decode", action="store_true")
    p.add_argument("--append", action="store_true", help="add to an existing points.jsonl (second model)")
    p.add_argument("--tiny", action="store_true", help="random tiny Qwen3, no download: CPU dry run")
    a = p.parse_args()
    if not a.tiny and not a.revision:
        raise SystemExit("--revision is required: pin the model like configs/*.yaml")
    dev = default_device()
    if a.tiny:
        a.prompt_lens, a.new_tokens, a.reps = "16,64,256", 8, 2
    model, info = load(a, dev)
    short = info["model"].split("/")[-1]

    generate_timed(model, 1024 if not a.tiny else 64, 8, dev, a.cache)  # warmup: cuBLAS, allocator
    generate_timed(model, 1024 if not a.tiny else 64, 8, dev, a.cache)
    if a.nsys_decode:
        torch.cuda.profiler.start()
        generate_timed(model, 1024, 33, dev, a.cache, nvtx=True)
        torch.cuda.profiler.stop()
        return

    pts = Points(a.out, "prefill-decode", dev, append=a.append)
    for P in (int(x) for x in a.prompt_lens.split(",")):
        ttfts, itls = [], []
        for r in range(a.reps):
            ttft, itl = generate_timed(model, P, a.new_tokens, dev, a.cache, seed=r)
            ttfts.append(ttft)
            itls += itl
        point = {
            "group_id": short,
            "prompt_len": P,
            "new_tokens": a.new_tokens,
            "reps": a.reps,
            "cache": a.cache,
            "ttft_ms": round(statistics.median(ttfts), 2),
            "itl_p50_ms": round(statistics.median(itls), 3),
            "itl_p90_ms": round(pctl(itls, 90), 3),
            "context_mean": P + a.new_tokens // 2,
            **info,
        }
        if a.busy and (a.tiny or P in (1024, 4096)):
            # one P keeps its trace and kernel names, so GEMM / attention / other can be recomputed later
            keep = P == (256 if a.tiny else 1024)
            stem = f"busy-{short}-P{P}"
            b, ks = busy_per_step(model, P, dev, trace=pts.dir / f"trace-{stem}.trace.json" if keep else None)
            if keep:
                pts.write_text(f"kernels-{stem}.txt", top_kernels(ks))
            point.update(
                busy_ms_per_step=round(b["gpu_us"] / 1e3, 3),
                kernels_per_step=round(b["n_kernels"], 1),
                gemm_ms_per_step=round(b["gemm_us"] / 1e3, 3),
                gpu_busy_share=round(b["gpu_us"] / 1e3 / point["itl_p50_ms"], 3),
            )
        pts.add(**point)


if __name__ == "__main__":
    main()
