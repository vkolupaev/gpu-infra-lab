"""Paper FLOP formulas of the Block 1 lesson scripts vs what PyTorch counts itself.

CPU, tiny shapes, about a second. Skipped where torch is not installed.
"""

import sys

import pytest

from conftest import ROOT

torch = pytest.importorskip("torch")
sys.path.insert(0, str(ROOT / "pytorch"))

from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from attention_ops import core_flops, naive  # noqa: E402
from backward_cost import paper_flops  # noqa: E402


@pytest.mark.parametrize("input_grad", [False, True])
def test_linear_stack_flops_match_counter(input_grad):
    T, d, L = 64, 32, 4
    net = torch.nn.Sequential(*[torch.nn.Linear(d, d, bias=False) for _ in range(L)])
    x = torch.randn(T, d, requires_grad=input_grad)
    with FlopCounterMode(display=False) as fc:
        y = net(x)
    fwd = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        y.backward(torch.randn_like(y))
    assert (fwd, fc.get_total_flops()) == paper_flops(T, d, L, input_grad)


def test_naive_attention_flops_match_counter():
    S, hq, hkv, hd = 64, 8, 2, 16
    q, k, v = torch.randn(1, hq, S, hd), torch.randn(1, hkv, S, hd), torch.randn(1, hkv, S, hd)
    mask = torch.full((S, S), float("-inf")).triu(1)
    with FlopCounterMode(display=False) as fc:
        naive(q, k, v, mask, torch.float32)
    assert fc.get_total_flops() == core_flops(S, hq, hd)
