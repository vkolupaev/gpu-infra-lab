#!/usr/bin/env python3
"""train_step.py - where the time of one training step goes (lesson 1.1).

Toy task: an MLP learns a fixed random linear "teacher" (if the loss does not go down,
the loop is broken and its timing means nothing). One step = copy the batch
host->device, forward + loss, backward, optimizer step. Phases are timed with CUDA
events, without a sync between phases, so the step stays asynchronous as in real
training. The swept variable is the matmul precision: fp32, tf32, bf16 (autocast).

    python3 pytorch/train_step.py --out results/2026-11-07-train-step --trace
    python3 pytorch/train_step.py --out /tmp/ts --quick          # CPU dry run
"""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F
from torch import nn

from labutil import Marks, Points, default_device, median_phases, sync, trace_kernels

PEAK_TFLOPS = {"fp32": 19.5, "tf32": 156.0, "bf16": 312.0}  # A100 dense, spec sheet


def build(dim_in: int, hidden: int, n_cls: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim_in, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_cls),
    )


def matmul_params(model: nn.Module) -> int:
    return sum(m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear))


class Stepper:
    """One training step with phase marks; NVTX ranges on request (visible in nsys)."""

    def __init__(self, a, dev, prec, model, opt, X, Y):
        self.a, self.dev, self.prec, self.model, self.opt, self.X, self.Y = a, dev, prec, model, opt, X, Y

    def _range(self, name, nvtx):
        if nvtx:
            if name != "h2d":
                torch.cuda.nvtx.range_pop()
            if name:
                torch.cuda.nvtx.range_push(name)

    def __call__(self, i, nvtx=False):
        a, dev, m = self.a, self.dev, Marks(self.dev)
        j = (i * a.batch) % (len(self.X) - a.batch)
        m.mark()
        self._range("h2d", nvtx)
        x = self.X[j : j + a.batch].to(dev, non_blocking=True)
        y = self.Y[j : j + a.batch].to(dev, non_blocking=True)
        m.mark()
        self._range("forward", nvtx)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=self.prec == "bf16"):
            loss = F.cross_entropy(self.model(x), y)
        m.mark()
        self._range("backward", nvtx)
        loss.backward()
        m.mark()
        self._range("optimizer", nvtx)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        m.mark()
        self._range("", nvtx)
        return loss.detach(), m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--precision", action="append", choices=list(PEAK_TFLOPS), help="repeatable; default all")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--dim-in", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--pageable", action="store_true", help="keep the dataset in pageable (not pinned) memory")
    p.add_argument("--trace", action="store_true", help="save a torch.profiler trace of 3 steps per precision")
    p.add_argument("--nsys-range", action="store_true", help="cudaProfilerStart/Stop around 5 steps (for nsys)")
    p.add_argument("--quick", action="store_true", help="tiny sizes for a CPU dry run")
    a = p.parse_args()
    dev = default_device()
    if a.quick:
        a.batch, a.dim_in, a.hidden, a.steps, a.warmup = 256, 128, 256, 10, 2
    precisions = a.precision or (list(PEAK_TFLOPS) if dev == "cuda" else ["fp32"])

    torch.manual_seed(0)
    n_cls, n_data = 10, a.batch * 8
    X = torch.randn(n_data, a.dim_in)
    Y = (X @ torch.randn(a.dim_in, n_cls)).argmax(1)
    if dev == "cuda" and not a.pageable:
        X, Y = X.pin_memory(), Y.pin_memory()
    pts = Points(a.out, "train-step", dev)

    for prec in precisions:
        torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
        torch.backends.cudnn.allow_tf32 = prec == "tf32"
        model = build(a.dim_in, a.hidden, n_cls).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        n_mm = matmul_params(model)
        first = model[0]

        step = Stepper(a, dev, prec, model, opt, X, Y)

        losses = [step(i)[0] for i in range(a.warmup)]
        sync(dev)
        marks, t0 = [], time.perf_counter()
        for i in range(a.warmup, a.warmup + a.steps):
            loss, m = step(i)
            losses.append(loss)
            marks.append(m)
        sync(dev)
        wall_step_ms = (time.perf_counter() - t0) * 1e3 / a.steps
        h2d, fwd, bwd, optim = median_phases(marks)

        # the same forward timed the naive way: no synchronize -> only launch time
        x = X[: a.batch].to(dev)
        sync(dev)
        naive = []
        for _ in range(5):
            t = time.perf_counter()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=prec == "bf16"):
                model(x)
            naive.append((time.perf_counter() - t) * 1e3)
            sync(dev)

        if a.trace:
            trace_kernels(lambda s=step: [s(i) for i in range(3)], dev, path=f"{a.out}/trace-{prec}.trace.json")
        if a.nsys_range and dev == "cuda":
            torch.cuda.profiler.start()
            for i in range(5):
                step(i, nvtx=True)
            sync(dev)
            torch.cuda.profiler.stop()

        f_fwd = 2 * a.batch * n_mm
        f_bwd = 2 * f_fwd - 2 * a.batch * first.in_features * first.out_features  # no dX for layer 1
        pts.add(
            group_id=f"mlp-b{a.batch}",
            precision=prec,
            peak_tflops=PEAK_TFLOPS[prec],
            batch=a.batch,
            matmul_params=n_mm,
            pinned=dev == "cuda" and not a.pageable,
            h2d_ms=round(h2d, 3),
            forward_ms=round(fwd, 3),
            backward_ms=round(bwd, 3),
            optimizer_ms=round(optim, 3),
            step_ms=round(h2d + fwd + bwd + optim, 3),
            wall_step_ms=round(wall_step_ms, 3),
            forward_ms_no_sync=round(statistics.median(naive), 3),
            backward_over_forward=round(bwd / fwd, 3),
            forward_tflops=round(f_fwd / (fwd * 1e-3) / 1e12, 2),
            backward_tflops=round(f_bwd / (bwd * 1e-3) / 1e12, 2),
            loss_first=round(losses[0].item(), 4),
            loss_last=round(losses[-1].item(), 4),
        )
        del model, opt


if __name__ == "__main__":
    main()
