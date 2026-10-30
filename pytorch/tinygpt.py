"""tinygpt.py - a small GPT-style decoder whose every tensor shape is known (lessons 1.2, 1.3, 1.5).

Pre-norm blocks: LayerNorm -> fused QKV projection -> causal attention -> output projection
-> residual; LayerNorm -> Linear(d, 4d) -> GELU -> Linear(4d, d) -> residual. No biases in
Linear layers, no dropout, learned positions, untied lm_head.
attn="sdpa": F.scaled_dot_product_attention (FlashAttention on A100);
attn="naive": Q @ K^T -> + causal mask -> softmax in fp32 -> cast -> @ V, the pattern of
transformers' eager attention. It materializes the S x S scores of every head.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class Config:
    vocab: int = 32768
    d: int = 1024
    heads: int = 16
    layers: int = 12
    seq: int = 1024
    attn: str = "sdpa"  # sdpa | naive
    ckpt: bool = False  # recompute each block in backward instead of saving its activations


def naive_attention(q, k, v):
    S = q.shape[-2]
    scores = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    mask = torch.full((S, S), float("-inf"), device=q.device, dtype=q.dtype).triu(1)
    p = torch.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q.dtype)
    return p @ v


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.heads, self.attn = c.heads, c.attn
        self.ln1 = nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.proj = nn.Linear(c.d, c.d, bias=False)
        self.ln2 = nn.LayerNorm(c.d)
        self.fc1 = nn.Linear(c.d, 4 * c.d, bias=False)
        self.fc2 = nn.Linear(4 * c.d, c.d, bias=False)

    def forward(self, x):
        B, S, d = x.shape
        # (B, S, 3, heads, hd) -> 3 x (B, heads, S, hd): views of one buffer, no copy
        q, k, v = self.qkv(self.ln1(x)).view(B, S, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if self.attn == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = naive_attention(q, k, v)
        x = x + self.proj(y.transpose(1, 2).reshape(B, S, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyGPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.seq, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln_f = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)

    def forward(self, idx, targets=None):
        S = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(S, device=idx.device))
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if self.c.ckpt and self.training else b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def matmul_params(model: nn.Module) -> int:
    """Parameters that take part in matmuls (embedding lookups excluded)."""
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))


def count_saved_bytes(model: nn.Module, idx, targets):
    """Bytes autograd keeps for backward after one forward, parameters excluded.

    Each saved tensor goes through the pack hook; tensors that share a storage (views)
    are counted once. Under checkpointing the inner blocks use their own hooks, so this
    count is only meaningful with ckpt=False; use the allocator for ckpt=True.
    Returns (total_bytes, [(shape, dtype, bytes), ...] largest first).
    """
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen = {}

    def pack(t):
        st = t.untyped_storage()
        ptr = st.data_ptr()
        if ptr not in params and ptr not in seen:
            seen[ptr] = (tuple(t.shape), str(t.dtype).replace("torch.", ""), st.nbytes())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = model(idx, targets)
    loss.backward()  # frees the graph
    model.zero_grad(set_to_none=True)
    items = sorted(seen.values(), key=lambda s: -s[2])
    return sum(s[2] for s in items), items
