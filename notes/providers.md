# GPU provider comparison (Block 0.1)

Date: YYYY-MM-DD. Card class: A100 80GB, 1 GPU. Raw data: `results/provider-check/`.

| Criterion | Vast.ai (container) | RunPod (container) | <VM provider> |
|---|---|---|---|
| Listed card / price $/h (basis) | | | |
| Time create -> SSH ready, min | | | |
| Time to model on disk (Qwen3-8B, 16 GB), min / MB/s | | | |
| GPU as listed? (name, SXM/PCIe, memory) | | | |
| Power limit / max SM clock | | | |
| HBM copy GB/s (% of spec) | | | |
| BF16 GEMM TFLOPS (% of spec) | | | |
| Host->device GB/s (PCIe link) | | | |
| CPU model / vCPUs | | | |
| ncu counters | | | |
| nsys CUDA trace / GPU metrics | | | |
| MIG switchable by user | | | |
| Disk write MB/s | | | |
| Billing: what is charged when stopped | | | |
| Payment from UZ: method that worked | | | |
| MI300X listed? price (for 5.3) | | | |

## Decision
- Benchmark provider (default for Blocks 1-2): ... because ...
- Profiling provider (ncu, MIG): ... because ...
- Excluded: ... because ...

## Rules that follow
- Destroy, never stop, at the end of a session (stopped storage is billed).
- ...
