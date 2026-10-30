#!/usr/bin/env python3
"""backward_cost.py - why backward costs about twice the forward (lesson 1.2).

A stack of L bias-free Linear(d, d) layers applied to T tokens. FLOPs are counted
three ways: on paper (paper_flops), by FlopCounterMode, and through time.
The swept variable: does the INPUT require grad (it does not for the first layer
of a real network: there is nothing upstream to send dX to).

    python3 pytorch/backward_cost.py --out results/2026-11-07-backward-cost
    python3 pytorch/backward_cost.py --out /tmp/bc --quick       # CPU dry run
"""

import argparse

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from labutil import Marks, Points, default_device, median_phases, sync


def paper_flops(T: int, d: int, L: int, input_grad: bool) -> tuple[int, int]:
    """(forward, backward) FLOPs of L Linear(d, d) layers on T tokens, 2*M*N*K per matmul.

    Forward per layer: Y = X @ W^T. Backward per layer: dX = dY @ W and dW = dY^T @ X.
    The first layer skips dX when its input does not require grad.
    """
    one = 2 * T * d * d
    return L * one, 2 * L * one - (0 if input_grad else one)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--d", type=int, default=4096)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="small sizes, fp32: CPU dry run")
    a = p.parse_args()
    dev = default_device()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    if a.quick:
        a.tokens, a.d, a.iters = 1024, 1024, 5
    pts = Points(a.out, "backward-cost", dev)
    torch.manual_seed(0)
    net = nn.Sequential(*[nn.Linear(a.d, a.d, bias=False) for _ in range(a.layers)]).to(dev, dtype)

    for input_grad in (False, True):
        x = torch.randn(a.tokens, a.d, device=dev, dtype=dtype, requires_grad=input_grad)
        g = torch.randn(a.tokens, a.d, device=dev, dtype=dtype)

        with FlopCounterMode(display=False) as fc:
            y = net(x)
        f_fwd = fc.get_total_flops()
        with FlopCounterMode(display=False) as fc:
            y.backward(g)
        f_bwd = fc.get_total_flops()
        net.zero_grad(set_to_none=True)

        marks = []
        for i in range(a.iters + 3):
            m = Marks(dev)
            m.mark()
            y = net(x)
            m.mark()
            y.backward(g)
            m.mark()
            net.zero_grad(set_to_none=True)
            if i >= 3:  # warmup: cuBLAS heuristics, allocator
                marks.append(m)
        sync(dev)
        fwd_ms, bwd_ms = median_phases(marks)
        p_fwd, p_bwd = paper_flops(a.tokens, a.d, a.layers, input_grad)
        pts.add(
            group_id="input-grad" if input_grad else "no-input-grad",
            input_requires_grad=input_grad,
            tokens=a.tokens,
            d=a.d,
            layers=a.layers,
            dtype=str(dtype).replace("torch.", ""),
            paper_flops_ratio=round(p_bwd / p_fwd, 4),
            counted_flops_ratio=round(f_bwd / f_fwd, 4),
            counted_equals_paper=(f_fwd, f_bwd) == (p_fwd, p_bwd),
            forward_ms=round(fwd_ms, 3),
            backward_ms=round(bwd_ms, 3),
            time_ratio=round(bwd_ms / fwd_ms, 3),
            forward_tflops=round(p_fwd / (fwd_ms * 1e-3) / 1e12, 2),
            backward_tflops=round(p_bwd / (bwd_ms * 1e-3) / 1e12, 2),
        )


if __name__ == "__main__":
    main()
