# GPU provider comparison (Block 0.1)

Date: 2026-10-22. Card class: A100 80GB, 1 GPU. Raw data: `results/provider-check/`.

| Criterion | Vast.ai (container) | RunPod (container) | <VM provider> |
|---|---|---|---|
| Listed card / price $/h (basis) | A100 SXM4 80GB / $1.114/h (on-demand) | A100 SXM 80GB / $1.804/h (on-demand: GPU $1.79 + 100 GB container disk $0.014) [^runpod-listing] | |
| Time create -> SSH ready, min | 20.9 (1252 s; upper bound, includes my connect delay) | 22.9 (1372 s; inflated: pod restarted once to expose TCP 22, and sshd was started by hand; not directly comparable with Vast) [^runpod-ssh] | |
| Time to model on disk (Qwen3-8B, 16 GB), min / MB/s | 2.2 (133 s, 16397461266 bytes) / 123.3 MB/s [^vast-hf] | 1.1 (63 s, 16397 MB) / 260.3 MB/s | |
| GPU as listed? (name, SXM/PCIe, memory) | Yes: NVIDIA A100-SXM4-80GB, SXM4, 81920 MiB | Yes: NVIDIA A100-SXM4-80GB, SXM4, 81920 MiB | |
| Power limit / max SM clock | 400 W (max 500 W) / 1410 MHz | 400 W (max 400 W) / 1410 MHz | |
| HBM copy GB/s (% of spec) | 1766 (86.6% of 2039) | 1771 (86.9% of 2039) | |
| BF16 GEMM TFLOPS (% of spec) | 265.4 (85.1% of 312) | 237.3 (76.1% of 312) | |
| Host->device GB/s (PCIe link) | 19.8 pinned (Gen4 x16, running at max) | 19.8 pinned (Gen4 x16, running at max) | |
| CPU model / vCPUs | AMD EPYC 7513 / 32 vCPU listed (129 GB RAM); `nproc` sees 128 vCPU, 503 GB RAM (host, not the allocation) | AMD EPYC 7763 / 16 vCPU listed (117 GB RAM); `nproc` sees 256 vCPU, 2003 GB RAM (host, not the allocation) [^runpod-cpu] | |
| ncu counters | No (ERR_NVGPUCTRPERM; RmProfilingAdminOnly: 1) | No (ERR_NVGPUCTRPERM; RmProfilingAdminOnly: 1) | |
| nsys CUDA trace / GPU metrics | Yes / No (GPU metrics: insufficient privilege, ERR_NVGPUCTRPERM) | Yes / No (GPU metrics: insufficient privilege, ERR_NVGPUCTRPERM) | |
| MIG switchable by user | No (Insufficient Permissions) | No (Insufficient Permissions) | |
| Disk write MB/s | 2859 (dd 4 GiB, oflag=direct, overlay root) | 3855 (dd 4 GiB, oflag=direct, overlay root) | |
| Billing: what is charged when stopped | Storage (host-set rate, typically higher than while running); compute is not; not observed, instance was destroyed right away [^vast-billing] | Storage: volume disk $0.20/GB/month (vs $0.10 running), container disk not charged (erased on stop), network volume $0.07/GB/month; not observed, pod was terminated [^runpod-billing] | |
| Payment from UZ: method that worked | Visa card in USD issued by a major Uzbek bank; worked on the first attempt | Visa card in USD issued by a major Uzbek bank; worked on the first attempt | |
| MI300X listed? price (for 5.3) | No: no AMD MI* cards in the listing on 2026-10-22 (only B300, B300 PC, GB300 came up) | Yes: MI300X 192 GB VRAM, $2.39/h per GPU as listed on 2026-10-22 (basis not recorded), up to 8 GPUs per pod; also MI350X 294 GB VRAM, $5.49/h per GPU, up to 8 per pod | |

[^vast-hf]: Size from `du -sbL` on the snapshot path (`hf-du.txt`). The `hf download` line in
`summary.txt` (4 MB) came from the old script, which measured symlinks only; fixed in 21efb00.
[^vast-billing]: Vast.ai docs, https://docs.vast.ai/guides/instances/pricing (read 2026-10-22): "Storage charges continue even when instances are stopped. Delete instances completely to cease storage billing."
[^runpod-listing]: Listing: 1x A100 SXM 80GB, 117 GB RAM, 16 vCPU, host CUDA 13.2 (driver 595.91.07 in `nvidia-smi-q.txt`).
[^runpod-ssh]: RunPod setup gotchas: proxy SSH (ssh.runpod.io) has no pipe/PTY for tar -> expose TCP 22 and use direct
SSH; PUBLIC_KEY was empty in the custom image, sshd did not start (fixed in 8579dd9, image not rebuilt yet).
[^runpod-cpu]: Allocated vCPU: 16 on RunPod vs 32 on Vast (Vast listing: "32.0/128 CPU, 129/516 GB"), 2x; this
matters for high-concurrency benchmarks (client and tokenizer threads share the CPU with vLLM). In both
containers `nproc` and `free` show the whole host, not the allocation.
[^runpod-billing]: RunPod docs, https://docs.runpod.io/pods/pricing and https://docs.runpod.io/accounts-billing/billing
(read 2026-10-22): "Storage charges continue to accrue on stopped Pods." Container disk $0.10/GB/month while running
matches the listed $0.014/h for 100 GB (100 x 0.10 / 730 h = $0.0137/h).

## Decision
- Benchmark provider (default for Blocks 1-2): **Vast.ai**, because same A100 SXM 80GB
  as RunPod at ~38% lower price ($1.114/h vs $1.804/h), 2x more vCPU (32 vs 16 allocated), SSH
  worked out of the box. RunPod BF16 GEMM was 9 pp lower (76% vs 85% of spec); both ran at a
  400 W power limit, so the gap is not explained by power settings. HBM was equal (~87%),
  so decode is unaffected, prefill and large batches are. HF download is not an advantage: ~123 MB/s vs ~260 MB/s on
  RunPod, but still ~2 min for 16 GB. Condition: marketplace hosts vary,
  so run hwcheck at the start of every session and record the machine id per run;
  SUSPICIOUS -> new host.
- Profiling provider (ncu, MIG): **none yet**. Both container providers return
  ERR_NVGPUCTRPERM for ncu and cannot switch MIG; nsys CUDA trace works on both.
  Candidate: Crusoe (full VM, documented perf counters and MIG) - sign-up and
  1-GPU availability not checked; decide before the first block that needs ncu.
- Excluded: Lambda (no Nsight Compute, card payment limited by region).
  RunPod is kept as the fallback for when Vast has no suitable host.

## Rules that follow
- Destroy, never stop, at the end of a session (stopped storage is billed).
- RunPod: expose TCP 22 at deploy and use direct SSH, not the ssh.runpod.io proxy.
- Pass PRICE as a bare number (`PRICE=1.114`): the shell expands "$1" in "$1.114".
- Check the hwcheck verdict before any measurement; SUSPICIOUS -> new host.
