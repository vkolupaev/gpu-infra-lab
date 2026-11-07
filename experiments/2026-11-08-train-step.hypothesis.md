# train-step: pre-registered question and hypothesis

- Date written: 2026-11-04 (draft), numbers filled 2026-11-08 (before the run). Planned run: 2026-11-14
  (moved from 2026-11-08), Vast A100-SXM4-80GB.
- Script: `pytorch/train_step.py` (fp32 / TF32 / bf16 autocast; 50 steps after 10 warmup).

## Question
Where does one training step of a 4-layer MLP (1024 -> 4096 -> 4096 -> 4096 -> 10, batch 4096)
go - host->device copy, forward, backward, optimizer step - and how does the split change
from fp32 to TF32 to bf16 autocast?

Secondary: what does a timer around the forward pass measure without `torch.cuda.synchronize()`?

## Hypothesis
Inputs: N = matmul parameters = sum(in x out) = 37.79 M (biases add 0.03%, ignored); B = 4096;
A100 dense peak: fp32 19.5, TF32 156, BF16 312 TFLOPS (measured BF16 GEMM 265.9); HBM 1759 GB/s;
pinned H2D 18.8 GB/s from h2d-copy.
- F_fwd = 2 x B x N = 309.6 GFLOP. F_bwd = 2 x F_fwd - 2 x B x 1024 x 4096 = 584.8 GFLOP (no dX for
  layer 1: its input X is data, not a parameter, and there is no layer before it to pass dX to).
- t_fwd = F_fwd / (peak x eff), eff assumed per precision - fp32 0.90, TF32 0.56, bf16 0.56:
  fp32 17.6 ms, TF32 3.5 ms, bf16 1.8 ms; t_bwd = 1.89 x t_fwd (F_bwd / F_fwd, same eff as forward).
  Why eff differs by precision: it is lower on Tensor Cores (TF32, bf16) than on CUDA cores (fp32).
  Tensor Cores process the same data several times faster, so the data has to be delivered several
  times faster, over the same path into the SM: HBM -> L2 -> shared memory -> registers.
  How the values were chosen: fp32 0.90 is a guess (CUDA cores are easy to keep fed on a large
  compute-bound matmul; no fp32 measurement is available). bf16 = fp32 / 1.6, from the expected ~10x
  forward speed-up fp32 -> bf16 (16x peak ratio); it is below the hwcheck 0.85 because the layers are
  8-32x smaller than the 8192^3 GEMM and ReLU and the loss are inside the forward phase. TF32 = bf16:
  same Tensor Cores, and a 4096^3 layer at 4 B per number stays above the TF32 ridge (683 vs 89 FLOP/B).
- t_opt = 28 B x N / 1759 GB/s = 0.60 ms (AdamW in fp32: reads param, grad, m, v; writes param, m, v),
  the same at every precision. It is memory-bound: ~10 FLOP per 28 B = 0.36 FLOP/B, far below the fp32
  ridge of ~11 FLOP/B. Autocast does not change the dtype of param, grad, m, v: they stay fp32, so the
  bytes do not change. 28 B is the minimum (one pass over memory); an optimizer split into several
  kernels moves more bytes, so 0.60 ms is a lower bound.
- t_h2d = (B x 1024 x 4 + B x 8) bytes / BW_pinned = 0.89 ms; t0 would add ~3%, below the spread of BW
  between hosts, ignored. X is fp32 on the host and is copied before autocast, so h2d does not change
  with precision either.
- Step: fp32 52.5 ms, bf16 6.6 ms - 7.9x faster, not ~10x, because h2d and optimizer do not speed up.
- Share of h2d + optimizer in the step: fp32 2.9 %, bf16 22.6 %.
- forward_ms_no_sync < 1 ms at every precision: without a synchronize only the launch is timed.

Falsified if:
- backward/forward is outside 1.7-2.3;
- optimizer time differs between precisions by more than 20%;
- h2d time differs between precisions by more than 20%;
- loss_last >= loss_first.

## Expectations recorded earlier, before any measurement (not revised)

Source: lesson 1.1, train_step.py part D, question 3, 2026-11-03.

- The naive measurement window on the GPU includes enqueueing, so forward_ms_no_sync will be much smaller than forward_ms.
