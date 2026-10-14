"""Tiny MoE transformer, Qwen3-MoE-style block, with optional fake quantization (QAT).

Block: RMSNorm -> MHA (RoPE, real softmax, causal) -> residual
       RMSNorm -> router (top-k) -> SwiGLU experts -> weighted combine -> residual

Every dimension is a Config field so the same code scales to whatever SRAM budget we end up with.
Quantization is fake-quant with a straight-through estimator:
  int8    : per-tensor symmetric weights, per-tensor dynamic int8 activations
  ternary : BitNet b1.58 style weights in {-1,0,+1} * mean|w|, int8 activations
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 32
    d: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 128          # per-expert hidden size
    n_experts: int = 4
    top_k: int = 1
    max_len: int = 64
    dense_q: str = "fp"      # attention / router / head weights: fp | int8 | ternary
    expert_q: str = "fp"     # expert weights: fp | int8 | ternary
    act_q: bool = False      # int8 fake-quant of linear-layer inputs
    aux_coef: float = 0.01   # load-balancing loss weight


# ---------------------------------------------------------------- quantization
def _ste(x, q):
    return x + (q - x).detach()


def fq_int8(w):
    s = w.abs().max().clamp(min=1e-8) / 127.0
    return _ste(w, (w / s).round().clamp(-127, 127) * s)


def fq_ternary(w):
    s = w.abs().mean().clamp(min=1e-8)
    return _ste(w, (w / s).round().clamp(-1, 1) * s)


class QLinear(nn.Linear):
    def __init__(self, i, o, wq="fp", act_q=False):
        super().__init__(i, o, bias=False)
        self.wq, self.act_q = wq, act_q

    def forward(self, x):
        if self.act_q:
            x = fq_int8(x)
        w = self.weight
        if self.wq == "int8":
            w = fq_int8(w)
        elif self.wq == "ternary":
            w = fq_ternary(w)
        return F.linear(x, w)


# ---------------------------------------------------------------- layers
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_tables(head_dim, max_len, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(max_len).float(), inv)  # (T, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); rotate pairs (even, odd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[:T], sin[:T]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg.n_heads, cfg.d // cfg.n_heads
        self.wq = QLinear(cfg.d, cfg.d, cfg.dense_q, cfg.act_q)
        self.wk = QLinear(cfg.d, cfg.d, cfg.dense_q, cfg.act_q)
        self.wv = QLinear(cfg.d, cfg.d, cfg.dense_q, cfg.act_q)
        self.wo = QLinear(cfg.d, cfg.d, cfg.dense_q, cfg.act_q)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        sp = lambda t: t.view(B, T, self.h, self.hd).transpose(1, 2)
        q, k, v = sp(self.wq(x)), sp(self.wk(x)), sp(self.wv(x))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, -1))


class Expert(nn.Module):  # SwiGLU
    def __init__(self, cfg):
        super().__init__()
        self.gate = QLinear(cfg.d, cfg.d_ff, cfg.expert_q, cfg.act_q)
        self.up = QLinear(cfg.d, cfg.d_ff, cfg.expert_q, cfg.act_q)
        self.down = QLinear(cfg.d_ff, cfg.d, cfg.expert_q, cfg.act_q)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.E, self.k = cfg.n_experts, cfg.top_k
        self.router = QLinear(cfg.d, cfg.n_experts, cfg.dense_q, cfg.act_q)
        self.experts = nn.ModuleList(Expert(cfg) for _ in range(cfg.n_experts))
        self.last_idx = None  # (B, T, k) expert ids from the last forward, for logging

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        probs = F.softmax(self.router(flat), dim=-1)           # (N, E)
        topv, topi = probs.topk(self.k, dim=-1)                # (N, k)
        topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, ex in enumerate(self.experts):
            tok, slot = (topi == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, ex(flat[tok]) * topv[tok, slot].unsqueeze(-1))
        # Switch-style load balancing: E * sum_e (dispatch fraction_e * mean router prob_e)
        frac = F.one_hot(topi, self.E).float().sum(1).mean(0) / self.k
        aux = self.E * (frac * probs.mean(0)).sum()
        self.last_idx = topi.detach().view(B, T, self.k)
        return out.view(B, T, D), aux


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.attn, self.moe = Attention(cfg), MoE(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        m, aux = self.moe(self.n2(x))
        return x + m, aux


class TinyMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)  # kept fp for now
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d)
        self.head = QLinear(cfg.d, cfg.vocab, cfg.dense_q, cfg.act_q)
        cos, sin = rope_tables(cfg.d // cfg.n_heads, cfg.max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):
        x = self.emb(idx)
        aux = 0.0
        for b in self.blocks:
            x, a = b(x, self.cos, self.sin)
            aux = aux + a
        return self.head(self.norm(x)), aux / len(self.blocks)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
