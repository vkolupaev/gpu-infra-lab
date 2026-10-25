# first-number: pre-registered question and hypothesis

- Date written: 2026-10-25 (before the run)
- Config: `configs/qwen3-8b-bf16.yaml`
- Inputs for the estimates: Qwen3-8B config.json (36 layers, 8 KV heads, head_dim 128,
  BF16, 8.2B params); Vast host hwcheck 2026-10-22 (HBM 1766 GB/s, BF16 GEMM 265 TFLOPS);
  price $1.114/h; 1024 in / 256 out tokens.

## Question
How do output throughput, latency and $/1M output tokens of untuned vLLM 0.32
(Qwen3-8B BF16, 1x A100 80GB SXM, 1024 in / 256 out) change from concurrency
1 to 8 to 32, and is decode at C=1 bound by HBM bandwidth?

## Hypothesis
Model: decode is memory-bound (weights 16.4 GB read once per step, shared by all
requests in the batch); prefill is compute-bound (2 x 8.2e9 x 1024 = 16.8 TFLOP per
request, ~90 ms at an assumed 65% of measured GEMM). Per closed-loop cycle:
time = C x T_prefill + 255 x T_step(C); KV cache = 144 KB/token.

| C | TTFT p50 | TPOT p50 | Out tok/s | $/1M out | Prefill share of GPU time |
|---|---|---|---|---|---|
| 1 | ~100 ms (80-120) | ~11 ms | ~88 | ~$3.5 | ~3% |
| 8 | ~150 ms | ~14 ms | ~540 (500-650) | ~$0.57 | ~19% |
| 32 | ~0.2-0.8 s | ~20-25 ms | ~1300 (1100-1600) | ~$0.24 | ~47% |

- C=1: TPOT floor 9.3 ms at measured 1766 GB/s (8.0 ms at spec 2039 GB/s);
  expect ~85% of measured bandwidth.
- C=8 -> C=32: throughput grows ~2.4x, not 4x: decode is shared across the batch,
  prefill FLOPs are per token and grow linearly with C.
- TPOT in the table is vLLM's measured TPOT = (E2E - TTFT) / (output_tokens - 1):
  it includes other requests' prefill chunks mixed into decode steps.
  The cycle formula uses the pure decode step instead.
  TPOT ~= T_step + (C - 1) x T_prefill / 255:
  C=8: 12 + 7 x 90 / 255 = ~14 ms; C=32: 13 + 31 x 90 / 255 = ~24 ms.
  Plugging TPOT into the cycle formula would count prefill twice.
- C=32 TTFT p95 ~1.5-3 s: all 32 requests start at once (32 x 1024 tokens, prefilled
  in ~4 chunks of 8192 tokens), the first wave is 10% of 320 requests.
- C=32 total throughput ~6600 tok/s = ~63% of the compute ceiling
  (~10,500 tok/s at 172 TFLOPS effective), MFU ~35% of spec 312 TFLOPS; C=1 MFU ~2%.

Checks:
- total_throughput / output_throughput = 1280 / 256 = 5.0 at every C.
- `GPU KV cache size` in server.log ~350k tokens
  (80 GB x 0.90 - 16.4 GB weights - a few GB overhead = ~50 GB / 144 KB).

Falsified if: C=1 TPOT < 9 ms (measurement error) or > 20 ms (wrong card,
throttling, eager mode); C=32 output throughput >= 4x C=8 (prefill cheaper than
assumed); total/output ratio != 5.0 (workload not as configured).
