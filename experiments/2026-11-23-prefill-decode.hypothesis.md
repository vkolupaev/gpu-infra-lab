# prefill-decode: pre-registered question and hypothesis

- Written 2026-11-17..2026-11-23, committed 2026-11-23, before the run. Planned run:
  2026-11-24 (session 1, part `a`), Vast A100-SXM4-80GB.
- Script: `pytorch/prefill_decode.py` (eager, SDPA, bf16, B = 1, greedy, DynamicCache, P =
  16...8192, 64 new tokens, `--busy` at P = 1024 and 4096). Qwen3-8B at the revision in
  `configs/qwen3-8b-bf16.yaml`; Qwen3-0.6B at revision
  "c1899de289a04d12100db370d81485cdf75e47ca" (passed as `R06` to `run_session.sh`).
- Baseline: [first-number](2026-10-25-first-number.md) - Qwen3-8B in vLLM 0.32.0 on the
  same card type; constants from [notes/estimation.md](../notes/estimation.md).

## Question
How do TTFT and inter-token latency of Qwen3-8B in HF transformers (eager, SDPA, bf16,
batch 1, 1x A100-SXM4-80GB) depend on prompt length, which of them agree with vLLM 0.32.0
on the same card type (our baseline run first-number: p50@TTFT = 86 ms at 1024 tokens,
p50@ITL = 11.2 ms at C = 1), and what limits a 14x smaller model?

## Hypothesis
**Inputs:** Qwen3-8B config.json (36 layers, hidden 4096, 32 q heads, 8 kv heads, head_dim
128, vocab 151936, untied lm_head); estimation.md: 13.9 GFLOP per prefill token, weights
16.4 GB, ridge ~151 FLOP/byte, prefill efficiency eta = 0.62 at 1024 tokens; measured
265.9 TFLOPS / 1759 GB/s.
- Bytes read per decode step: 16.4 GB minus `embed_tokens` (one 8 KiB row is gathered per
  token, not the 1.24 GB matrix; `lm_head` is read in full) = 15.2 GB, T_w = 8.64 ms.
- KV per token: 36 layers x 2 (K, V) x 8 kv heads x 128 x 2 B = 147456 B (144 KiB).
- Kernels per decode step n_k = 2312: 63 per layer x 36 = 2268, + 22 in the model head
  (embedding, rotary, mask, final norm, lm_head), + 22 in the `generate()` loop outside
  the model (logits copy, min-new-tokens processor, argmax, eos mask, cat, stopping
  criteria). 2 host syncs per step (`tokens.cpu()` for the streamer, `bool(max == 0)`).
  Counted from modeling_qwen3.py and generation/utils.py; checked by `kernels_per_step`.
- Launch cost t_l = 11-19 us per op (ASSUMED): Python + dispatcher ~4.5 us (eager `x + 1`,
  i9-11900K) plus driver launch ~3-5 us, scaled by 1.45-2.04 for the single-thread speed
  of the EPYC 7513 host. Any value above T_gpu / n_k ~ 4.6 us makes HF eager launch-bound.
  Views and reshapes cost Python time but launch no kernel, so t_l fitted per kernel is an
  effective cost.
- T_cpu = n_k x t_l = 25-44 ms. It does not depend on P: one kernel processes the whole
  (1, P, ...) tensor, so a prefill forward launches as many kernels as a decode step.

**Prefill.** TTFT(P) = max(T_cpu, T_w, F(P) / (eta x 265.9 TFLOPS)), F(P) = 13.9e9 x P + 2
x 36 x 32 x 128 x P^2 (causal attention; assumes SDPA takes the causal path - with an
explicit full mask the attention term doubles):
- flat at T_cpu = 25-44 ms (plus a few ms of `generate()` setup) up to P* = T_cpu x eta x
  peak / 13.9e9 = 300-520 tokens, 2-3.5x right of the ridge (151): the high launch floor
  takes longer to catch up;
- then roughly linear (eta may rise with P), slightly convex at 8192: attention adds 2.2%
  to the GEMM FLOPs at 1024, 17% at 8192;
- TTFT(1024) = 88 ms vs vLLM 86 ms (compute-bound, same GEMMs). Partly built in: eta =
  0.62 is derived from vLLM's 86 ms, so this tests that HF adds no large setup or launch
  cost;
- TTFT(8192) = 0.59-0.81 s, above vLLM's 0.485 s for an 8192-token chunk (8 prompts x
  1024): one 8192-token prompt carries 8x the attention, +105 ms at eta = 0.62. 0.59 s if
  HF's GEMMs run as efficiently as vLLM's 8192-token chunk (eta 0.88, estimation.md,
  section 7); 0.81 s if eta stays 0.62 at all P.

**Decode.** GPU time per step T_gpu(ctx) = (15.2 GB + r x ctx x 147456 B) / (0.83 x 1759
GB/s). r = 3: DynamicCache concatenates (`torch.cat`) every step - reads the old cache,
writes a new one a row longer, attention reads the new one. 0.83 is vLLM's share of HBM
bandwidth in first-number, so agreement with 11.2 ms is partly built in;
`busy_ms_per_step` is the check.
- busy_ms_per_step: 10.7 ms at P = 1024 (vLLM's p50@ITL - 4%), 11.7 ms at P = 4096.
- ITL fork: max(T_cpu, T_gpu) ... T_cpu + T_gpu = 25 ... 55 ms. Launches are asynchronous,
  so the GPU works on kernel i while the CPU enqueues i+1: GPU work hides under launching,
  and ITL sits at the lower end. The only part not hidden is the tail behind the last
  large kernel before the sync: lm_head (1.24 GB, ~0.85 ms) minus the 22 small launches
  after it (0.24-0.42 ms) = 0.4-0.6 ms. Inside a layer the queue drains (GPU ~0.27 ms vs
  0.7-1.2 ms of launches). p50@ITL = T_cpu + tail = 25-45 ms; gpu_busy_share = 0.24-0.43.
- ITL slope vs context ~ 0. KV traffic grows by ~0.31 ms of GPU time per 1k tokens of
  context, but it lands inside the layers and hides under launching. Expected: busy grows
  ~0.31 ms per 1k (r = 3; r = 1 would give ~0.10), ITL stays flat. ITL starts to grow only
  where T_gpu reaches T_cpu: ctx* = (T_cpu - T_gpu(0)) / 0.31 ms ~ 47k-108k tokens, beyond
  Qwen3-8B's native 40k context (an extrapolation). A busy slope with a flat ITL is the
  signature of a launch-bound decode; a memory-bound engine (vLLM) would show the slope in
  ITL itself.
- t_l from the run: (p50@ITL - tail) / kernels_per_step; nsys `cuda_api_sum` gives the
  `cudaLaunchKernel` part only.
- Qwen3-0.6B (config.json: 28 layers, hidden 1024, 16 q heads, 8 kv heads, head_dim 128,
  MLP 3072, vocab 151936, tied embeddings; ~1.19 GB read per step): same 63 ops per layer,
  n_k = 63 x 28 + 22 + 22 = 1808, T_cpu = 19.9-34.4 ms; KV 114688 B per token, T_gpu =
  1.06 ms at ctx 1024. No tail: its lm_head (~0.31 GB, ~0.21 ms) finishes before the 22
  launches after it (0.24-0.42 ms), so the queue is empty at the sync. p50@ITL ~ T_cpu =
  20-34 ms; gpu_busy_share = 0.03-0.05. ITL_8B / ITL_0.6B at P = 1024 ~ 1.3 = the ratio of
  kernel counts (2312 / 1808), while the weight ratio is ~13: the small model is limited
  by launching kernels from the CPU (their number and cost per launch), not by memory.

**Falsified if:**
- TTFT(P) has no flat part: TTFT(256) / TTFT(16) > 1.2;
- 8B p50@ITL < T_w = 8.64 ms (faster than the memory floor: a measurement error);
- TTFT(1024) is off vLLM's 86 ms by > 25%, or TTFT(8192) <= 0.485 s (no extra attention);
- busy_ms_per_step at P = 1024 is off vLLM's p50@ITL (11.2 ms) by > 25% (then HF's kernels
  themselves differ, not the overhead);
- ITL_8B / ITL_0.6B at P = 1024 > 1.45: launching and GPU work do not overlap (serial
  would give 1.54-1.73, overlap ~1.3 = the ratio of `kernels_per_step`); much higher - the
  small model is not launch-bound;
- p50@ITL(8192) - p50@ITL(1024) > 1.1 ms (half of the ~2.2 ms of extra GPU work) and
  beyond the spread of the 3 reps: decode is not launch-bound at these lengths.
