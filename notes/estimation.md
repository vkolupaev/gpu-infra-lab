# Back-of-envelope estimation for LLM inference

- Status: draft skeleton (2026-10-26). Rewrite in your own words; keep the numbers tied to run groups.
- Purpose: predict throughput, latency and cost before renting a GPU, and check every measurement
  against a first-principles bound after the run.
- Rule: a coefficient is either **assumed** (say so) or **measured** (give the run group).
  Reference run: `20261025T101616Z_qwen3-8b_0798e2f` (`experiments/2026-10-25-first-number.md`).

## 1. Inputs and where they come from

| Input | Source | Qwen3-8B / A100 SXM4 80GB example |
|---|---|---|
| Layers, KV heads, head_dim | model `config.json` at the pinned revision: `num_hidden_layers`, `num_key_value_heads`, `head_dim` | 36, 8, 128 |
| hidden_size, vocab_size, tie_word_embeddings | same `config.json` | 4096, 151936, `false` |
| Parameters, dtype | model card / `config.json` `torch_dtype` | 8.2B, bfloat16 |
| Peak HBM bandwidth | GPU spec sheet | 2039 GB/s |
| Peak dense BF16 | GPU spec sheet (**dense**, not "with sparsity") | 312 TFLOPS |
| Measured HBM / GEMM | `scripts/hwcheck.py` on the rented host (`hwcheck.json`) | 1759 GB/s, 265.9 TFLOPS (Vast machine 142924) |
| Power limit | `nvidia-smi` (`power_limit_w`, `power_max_limit_w` in the record) | 400 W of 500 W max |
| Engine limits | config YAML: `max-num-batched-tokens`, `gpu-memory-utilization`, `max-model-len` | 8192, 0.90, 4096 |
| Workload | config YAML + bench flags: input/output length, concurrency C, closed/open loop | 1024 / 256, C = 1/8/32, closed loop |
| Price | provider listing, basis (on-demand / interruptible) | $1.114/h on-demand |

## 2. Derived quantities

| Quantity | Formula | Example |
|---|---|---|
| Weight bytes | params x bytes/param (BF16 2, FP8 1, INT4 ~0.5) | 8.2e9 x 2 = 16.4 GB |
| Embedding / lm_head size | vocab_size x hidden_size (each, when untied) | 151936 x 4096 = 0.62B params |
| Weights read per decode step | weights minus the embedding table (a lookup reads one row per token) | ~15.2 GB (the first-number analysis used 16.4 GB) |
| KV bytes per token | layers x kv_heads x head_dim x 2 (K, V) x bytes | 36 x 8 x 128 x 2 x 2 = 147,456 B = 144 KiB |
| KV per request per decode step | KV/token x average context, context_avg = in + out / 2 | 147,456 B x 1152 = ~0.17 GB |
| KV capacity, tokens | (VRAM x gpu_mem_util - weights - overhead) / KV per token | measured: 54.76 GiB -> 398,720 tokens |
| FLOPs per prefill token | 2 x matmul params; no embedding (lookup), lm_head only for the last token | 2 x (8.2 - 0.62 - 0.62)e9 = ~13.9 GFLOP |
| FLOPs per decode token | 2 x matmul params; lm_head runs every step | 2 x (8.2 - 0.62)e9 = ~15.2 GFLOP |

Attention FLOPs are ignored at short context: ~4% of prefill at 1024 tokens. Revisit at 8k+.

## 3. Roofline: memory-bound or compute-bound

- Arithmetic intensity I = FLOPs done / bytes moved from HBM. Property of the work.
- Ridge point = peak FLOPS / peak bandwidth. Property of the card:
  265.9 TFLOPS / 1759 GB/s = **~151 FLOP/byte** (measured, A100 SXM4 80GB).
- I < ridge: memory-bound, time = bytes / bandwidth. I > ridge: compute-bound, time = FLOPs / FLOPS.

| Work | Intensity | Regime |
|---|---|---|
| Decode, C = 1 | 2 FLOP per param / 2 bytes per param = ~1 FLOP/byte | memory-bound |
| Decode, batch C | ~C on weight reads; KV reads stay ~1 (each request has its own cache) | memory-bound until C ~150+, later in practice |
| Prefill, N tokens | ~N on weight reads | compute-bound for N above ~150 |

## 4. Latency

- E2E = TTFT + (out - 1) x TPOT. vLLM defines TPOT = (E2E - TTFT) / (out - 1).
- Decode step floor: T_step = (weights_read + C x KV per request) / bandwidth.
- Decode step, measured: p50@ITL (typical gap between tokens, i.e. a pure decode step).
  p50@TPOT also contains other requests' prefill chunks: TPOT ~= T_step + (C - 1) x T_prefill / (out - 1)
  when prefills interleave with decode. Never plug TPOT into the throughput model (prefill counted twice).
- TTFT ~= prefill time + queueing. Prefill time = tokens x FLOPs per prefill token / (FLOPS x efficiency).
- Chunked prefill: a burst of C prompts is split into ceil(C x in / max_num_batched_tokens) chunks;
  requests get their first token chunk by chunk.

## 5. Throughput (closed loop)

- Wave time = prefill of C prompts + (out - 1) x T_step(C).
- Output throughput = C x out / wave time (equals C x out / p50@E2EL when requests run in waves).
- Total throughput = output x (in + out) / out. For 1024 / 256 the ratio is exactly 5.0.
- Compute ceiling on total tokens = FLOPS x efficiency / FLOPs per token; output ceiling = that x out / (in + out).
- Why throughput grows slower than C: batching shares only weight reads. Prefill (compute) and the
  KV cache (memory) belong to each request and grow linearly with C.

Check against first-number (p50@ITL as T_step, 0.485 s per 8192-token chunk):

| C | Wave prefill | Decode | Sum | Measured p50@E2EL |
|---|---|---|---|---|
| 8 | 1 chunk = 0.485 s | 255 x 12.05 ms = 3.07 s | 3.56 s | 3.605 s |
| 32 | 4 chunks = 1.94 s | 255 x 15.9 ms = 4.06 s | 6.00 s | 6.085 s |

## 6. Economics

- $/1M tokens = price_per_hour x gpu_fraction / (throughput x 3600) x 1e6, at 100% utilization.
  Real cost = that / average utilization.
- Two prices: per output token (compare with API price lists) and per total token (capacity planning).
- GPUs for a load = tokens per month / (month seconds) / total throughput per GPU / utilization.
  State the month: 730 h (matches $/h x 730) or 30 days (2.592e6 s); they differ by ~1.4%.
- Example: 650B tokens/month = ~247,000 tok/s (730 h). At 6731 total tok/s per GPU (C = 32):
  37 GPUs at 100% utilization, 92 at 40%; ~$30,100 and ~$74,800 per month at $1.114/h.

## 7. Coefficients: assumed -> measured

| Coefficient | Assumed (hypothesis 2026-10-25) | Measured | Run group | Note |
|---|---|---|---|---|
| Decode HBM efficiency, C = 1 | ~85% | 83% (16.4 GB) / ~78% (15.2 GB + KV) | 20261025T101616Z | p50@TPOT 11.21 ms vs 9.32 / 8.74 ms floor |
| Decode HBM efficiency, C = 8 / 32 | - | 84% / 78% (16.4 GB + KV) | 20261025T101616Z | p50@ITL 12.05 / 15.9 ms vs 10.1 / 12.4 ms |
| Prefill efficiency vs measured GEMM | 65% | 62% at 1024 tokens, 88% at 8192-token chunks | 20261025T101616Z | 13.9 GFLOP/token |
| 8192-token prefill chunk | - | 0.485 s | 20261025T101616Z | p50@TTFT at C = 8 |
| KV per token | 144 KiB | 144 KiB (exact) | 20261025T101616Z | 54.76 GiB / 398,720 tokens |
| KV capacity at gpu_mem_util 0.90 | ~350k tokens | 398,720 tokens | 20261025T101616Z | free-memory estimate was too low |
| Power during prefill | - | hits 400 W cap, SM clock -10% (1410 -> ~1275 MHz) | 20261025T101616Z | decode stays at ~315-345 W |
| MFU at C = 32 | ~35% | ~35% (6731 x 16.4 GFLOP / 312 TFLOPS) | 20261025T101616Z | C = 1: ~2% |

Add a row per new card, model, precision or engine setting. One host so far: treat as one sample.

## 8. Checklist after every run

- [ ] `bench.py validate` OK, `git.dirty: false`, image with digest, `hwcheck_verdict: OK`.
- [ ] total / output throughput = (in + out) / out.
- [ ] TPOT at C = 1 vs memory floor (spec and measured bandwidth): ~60-85% is normal.
- [ ] `GPU KV cache size` in `server.log` vs the formula in section 2.
- [ ] p50@ITL vs decode-step floor at every C.
- [ ] TTFT vs prefill estimate; std@E2EL tiny -> synchronized waves (closed-loop artifact).
- [ ] Share of the compute ceiling at the highest C.
- [ ] `gpu-monitor.csv`: power vs limit, `clocks_event_reasons` bits (0x4 power cap, 0x20/0x40 thermal).
- [ ] `p99_reliable`: below 200 requests p99 is an approximate max.

## 9. Pitfalls

- GiB vs GB: 144 KiB is 147.5 kB, a ~2% error if mixed.
- Spec vs measured bandwidth / TFLOPS: always say which one a percentage is relative to.
- Dense vs sparse TFLOPS: A100 BF16 is 312 dense, 624 "with sparsity".
- Bytes per parameter by precision; quantized formats add scales.
- Counting embedding and lm_head in prefill FLOPs gives speeds above the card's peak.
- Percentiles do not add: use p50 or mean for "share of time".
- TPOT includes interleaved prefill; p50@ITL is the pure decode step.
- `utilization.gpu` is "a kernel was running", not load: ~100% even for memory-bound decode.
- Closed loop + `request-rate inf` + fixed lengths = synchronized waves: worst-case TTFT.
- `nproc` and `free` in containers show the host, not the allocation.
- Two "A100 80GB" with different power limits are different measurement points.
