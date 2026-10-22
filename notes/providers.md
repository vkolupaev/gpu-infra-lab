# GPU provider comparison (Block 0.1)

Date: 2026-10-22. Card class: A100 80GB, 1 GPU. Raw data: `results/provider-check/`.

| Criterion | Vast.ai (container) | RunPod (container) | <VM provider> |
|---|---|---|---|
| Listed card / price $/h (basis) | A100 SXM4 80GB / $1.114/h (on-demand) | | |
| Time create -> SSH ready, min | 20.9 (1252 s; upper bound, includes my connect delay) | | |
| Time to model on disk (Qwen3-8B, 16 GB), min / MB/s | 2.2 (133 s, 16397461266 bytes) / 123.3 MB/s [^vast-hf] | | |
| GPU as listed? (name, SXM/PCIe, memory) | Yes: NVIDIA A100-SXM4-80GB, SXM4, 81920 MiB | | |
| Power limit / max SM clock | 400 W (max 500 W) / 1410 MHz | | |
| HBM copy GB/s (% of spec) | 1766 (86.6% of 2039) | | |
| BF16 GEMM TFLOPS (% of spec) | 265.4 (85.1% of 312) | | |
| Host->device GB/s (PCIe link) | 19.8 pinned (Gen4 x16, running at max) | | |
| CPU model / vCPUs | AMD EPYC 7513 / 128 vCPU (503 GB RAM) | | |
| ncu counters | No (ERR_NVGPUCTRPERM; RmProfilingAdminOnly: 1) | | |
| nsys CUDA trace / GPU metrics | Yes / No (GPU metrics: insufficient privilege, ERR_NVGPUCTRPERM) | | |
| MIG switchable by user | No (Insufficient Permissions) | | |
| Disk write MB/s | 2859 (dd 4 GiB, oflag=direct, overlay root) | | |
| Billing: what is charged when stopped | Storage (host-set rate, typically higher than while running); compute is not; not observed, instance was destroyed right away [^vast-billing] | | |
| Payment from UZ: method that worked | Visa card in USD issued by a major Uzbek bank; worked on the first attempt | | |
| MI300X listed? price (for 5.3) | No: no AMD MI* cards in the listing on 2026-10-22 (only B300, B300 PC, GB300 came up) | | |

[^vast-hf]: Size from `du -sbL` on the snapshot path (`hf-du.txt`). The `hf download` line in
`summary.txt` (4 MB) came from the old script, which measured symlinks only; fixed in 21efb00.
[^vast-billing]: Vast.ai docs, https://docs.vast.ai/guides/instances/pricing (read 2026-10-22): "Storage charges continue even when instances are stopped. Delete instances completely to cease storage billing."

## Decision
- Benchmark provider (default for Blocks 1-2): ... because ...
- Profiling provider (ncu, MIG): ... because ...
- Excluded: ... because ...

## Rules that follow
- Destroy, never stop, at the end of a session (stopped storage is billed).
- ...
