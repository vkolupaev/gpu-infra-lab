#!/usr/bin/env python3
"""hwcheck.py - is the rented GPU what the listing says it is? (~1 minute)

Measures device memory bandwidth, BF16 GEMM throughput and host->device bandwidth,
and compares them with spec-sheet peaks. Prints one JSON object.
"""

import json
import subprocess
import time

import torch

SPEC = {  # name substring -> (HBM GB/s, dense BF16 TFLOPS)
    "A100-SXM4-80GB": (2039, 312),
    "A100 80GB PCIe": (1935, 312),
    "A100-SXM4-40GB": (1555, 312),
    "A100-PCIE-40GB": (1555, 312),
    "H100 80GB HBM3": (3350, 989),
    "H100 PCIe": (2000, 756),
    "H100 NVL": (3900, 835),
    "H200 NVL": (4800, 835),
    "H200": (4800, 989),
    "A30": (933, 165),
    "L40S": (864, 362),
}


def timed(fn, iters):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    name = torch.cuda.get_device_name(0)
    # device memory bandwidth: copy 2 GiB (read + write)
    n = 2 * 1024**3
    a = torch.empty(n, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    hbm = 2 * n / timed(lambda: b.copy_(a), 20) / 1e9  # noqa: F821 - timed() calls the lambda before `del a, b`
    del a, b
    # BF16 GEMM
    m = 8192
    x = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
    tflops = 2 * m**3 / timed(lambda: x @ y, 20) / 1e12  # noqa: F821 - timed() calls the lambda before `del x, y`
    del x, y
    # host -> device, pinned memory (PCIe link health)
    h = torch.empty(1024**3, dtype=torch.uint8).pin_memory()
    d = torch.empty(1024**3, dtype=torch.uint8, device="cuda")
    h2d = 1024**3 / timed(lambda: d.copy_(h, non_blocking=True), 10) / 1e9
    link = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pcie.link.gen.current,pcie.link.width.current,"
            "clocks.sm,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    spec = next((v for k, v in SPEC.items() if k in name), None)
    out = {
        "gpu_name": name,
        "hbm_copy_gbps": round(hbm),
        "bf16_gemm_tflops": round(tflops, 1),
        "h2d_pinned_gbps": round(h2d, 1),
        "nvidia_smi_after_load": link,
    }
    if spec:
        out["hbm_pct_of_spec"] = round(100 * hbm / spec[0])
        out["gemm_pct_of_spec"] = round(100 * tflops / spec[1])
        out["verdict"] = (
            "OK"
            if out["hbm_pct_of_spec"] >= 75 and out["gemm_pct_of_spec"] >= 55
            else "SUSPICIOUS: check power limit, clocks, throttle reasons"
        )
    out["h2d_note"] = "PCIe Gen4 x16 ~20-25 GB/s, Gen5 x16 ~40-50, Gen3 x16 or Gen4 x8 ~11-13"
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
