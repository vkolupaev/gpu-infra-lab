#!/usr/bin/env python3
"""activation_memory.py - activation memory vs batch size, with and without checkpointing (lesson 1.2).

TinyGPT (pytorch/tinygpt.py) in pure bf16 with AdamW: a measurement of memory and time,
not of convergence. For every (checkpointing, batch) point:
  saved_mib      allocator: memory_allocated after forward minus before forward
                 = what autograd keeps for backward (+ the loss scalar);
  saved_hooks_mib the same counted tensor by tensor with saved_tensors_hooks (ckpt off only);
  peak_step_mib  max_memory_allocated during the step minus before forward (+ grads, temporaries);
  static_mib     weights + optimizer state, before forward (grads are None at that moment).
For batch 1 the list of saved tensors is written to saved-tensors-b1.txt.

    python3 pytorch/activation_memory.py --out results/2026-11-07-activation-memory
    python3 pytorch/activation_memory.py --out /tmp/am --quick   # CPU: hooks and timing only
"""

import argparse

import torch

from labutil import Marks, Points, default_device, median_phases, sync
from tinygpt import Config, TinyGPT, count_saved_bytes, matmul_params

MIB = 2**20


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--batches", default="1,2,4,8,16,32")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "naive"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--quick", action="store_true", help="tiny model, fp32: CPU dry run")
    a = p.parse_args()
    dev = default_device()
    cuda = dev == "cuda"
    dtype = torch.bfloat16 if cuda else torch.float32
    base = Config(attn=a.attn)
    if a.quick:
        base = Config(vocab=1024, d=128, heads=4, layers=2, seq=128, attn=a.attn)
        a.batches, a.iters = "1,2", 3
    pts = Points(a.out, "activation-memory", dev)

    for ckpt in (False, True):
        for B in (int(b) for b in a.batches.split(",")):
            c = Config(**{**base.__dict__, "ckpt": ckpt})
            torch.manual_seed(0)
            model = TinyGPT(c).to(dev, dtype).train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            idx = torch.randint(0, c.vocab, (B, c.seq), device=dev)
            tgt = torch.randint(0, c.vocab, (B, c.seq), device=dev)
            try:
                hooks_bytes = None
                if not ckpt:
                    hooks_bytes, items = count_saved_bytes(model, idx, tgt)
                    if B == 1:
                        lines = [f"{b:>14,d}  {dt:<9} {list(s)}" for s, dt, b in items]
                        pts.write_text(f"saved-tensors-b1-{a.attn}.txt", "\n".join(lines) + "\n")
                marks, saved, peak, static = [], [], [], []
                for i in range(a.iters + 2):  # 2 warmup steps create grads and optimizer state
                    sync(dev)
                    m0 = torch.cuda.memory_allocated() if cuda else 0
                    if cuda:
                        torch.cuda.reset_peak_memory_stats()
                    m = Marks(dev)
                    m.mark()
                    loss = model(idx, tgt)
                    m.mark()
                    m_fwd = torch.cuda.memory_allocated() if cuda else 0
                    loss.backward()
                    m.mark()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    m.mark()
                    if i >= 2:
                        marks.append(m)
                        if cuda:
                            saved.append(m_fwd - m0)
                            peak.append(torch.cuda.max_memory_allocated() - m0)
                            static.append(m0)
                sync(dev)
                fwd, bwd, optim = median_phases(marks)
                pts.add(
                    group_id=f"ckpt-{'on' if ckpt else 'off'}",
                    checkpointing=ckpt,
                    attn=a.attn,
                    batch=B,
                    seq=c.seq,
                    tokens=B * c.seq,
                    d=c.d,
                    layers=c.layers,
                    vocab=c.vocab,
                    matmul_params=matmul_params(model),
                    saved_mib=round(max(saved) / MIB, 1) if saved else None,
                    saved_hooks_mib=round(hooks_bytes / MIB, 1) if hooks_bytes else None,
                    peak_step_mib=round(max(peak) / MIB, 1) if peak else None,
                    static_mib=round(static[-1] / MIB, 1) if static else None,
                    forward_ms=round(fwd, 2),
                    backward_ms=round(bwd, 2),
                    optimizer_ms=round(optim, 2),
                    step_ms=round(fwd + bwd + optim, 2),
                    loss=round(loss.item(), 3),
                )
            except torch.OutOfMemoryError:
                pts.add(group_id=f"ckpt-{'on' if ckpt else 'off'}", checkpointing=ckpt, batch=B, oom=True)
            del model, opt, idx, tgt
            if cuda:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
