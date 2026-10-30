#!/usr/bin/env python3
"""h2d_copy.py - host<->device copy time vs size, pinned vs pageable host memory (lesson 1.1).

Model under test: t(n) = t0 + n / BW. Small copies are latency-bound, large ones
bandwidth-bound; half of the peak bandwidth is reached at n_half = t0 x BW.
Time is wall time with a synchronize, so t0 includes the Python call and the sync.

    python3 pytorch/h2d_copy.py --out results/2026-11-07-h2d-copy
"""

import argparse

import torch

from labutil import Points, time_ms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--min-log2", type=int, default=10, help="smallest copy, log2 bytes (10 = 1 KiB)")
    p.add_argument("--max-log2", type=int, default=30, help="largest copy, log2 bytes (30 = 1 GiB)")
    p.add_argument("--iters", type=int, default=20)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU: on CPU there is no host<->device copy to measure")

    pts = Points(a.out, "h2d-copy", "cuda")
    for memory in ("pinned", "pageable"):
        for direction in ("h2d", "d2h"):
            for lg in range(a.min_log2, a.max_log2 + 1, 2):
                n = 2**lg
                host = torch.empty(n, dtype=torch.uint8, pin_memory=memory == "pinned")
                dev = torch.empty(n, dtype=torch.uint8, device="cuda")
                if direction == "h2d":

                    def fn(d=dev, h=host):
                        d.copy_(h, non_blocking=True)
                else:

                    def fn(d=dev, h=host):
                        h.copy_(d, non_blocking=True)

                ms = time_ms(fn, "cuda", iters=a.iters, warmup=3)
                pts.add(
                    group_id=f"{direction}-{memory}",
                    direction=direction,
                    memory=memory,
                    size_kib=n // 1024,
                    time_us=round(ms * 1e3, 2),
                    gbps=round(n / (ms * 1e-3) / 1e9, 3),
                )
                del host, dev


if __name__ == "__main__":
    main()
