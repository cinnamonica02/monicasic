"""float32 golden model of one Qwen3 dense decoder layer (numpy only).

Matches transformers' Qwen3DecoderLayer:
  x -> RMSNorm -> GQA attention (per-head QK-norm, rotate-half RoPE, causal softmax) -> +x
    -> RMSNorm -> SwiGLU MLP -> +x
Weight names follow the HF state_dict of one layer (see LAYER_KEYS).

Note: HF RoPE pairs dimension i with i + head_dim/2 (rotate_half). The toy model in
tiny_moe/model/moe.py pairs even/odd dims instead; the two are not interchangeable with real weights.
"""
from dataclasses import dataclass

import numpy as np

LAYER_KEYS = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)


@dataclass
class Cfg:
    hidden: int = 2560
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    inter: int = 9728
    rope_theta: float = 1e6
    eps: float = 1e-6


def rmsnorm(x, w, eps):
    x = x.astype(np.float32)
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_cos_sin(positions, head_dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    ang = np.outer(positions.astype(np.float32), inv)           # (T, hd/2)
    ang = np.concatenate([ang, ang], axis=-1)                    # (T, hd)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rope(x, cos, sin):
    # x: (H, T, hd); cos/sin: (T, hd)
    return x * cos + rotate_half(x) * sin


def attention(x, w, c, cos, sin):
    T = x.shape[0]
    q = (x @ w["self_attn.q_proj.weight"].T).reshape(T, c.n_heads, c.head_dim)
    k = (x @ w["self_attn.k_proj.weight"].T).reshape(T, c.n_kv_heads, c.head_dim)
    v = (x @ w["self_attn.v_proj.weight"].T).reshape(T, c.n_kv_heads, c.head_dim)
    q = rmsnorm(q, w["self_attn.q_norm.weight"], c.eps).transpose(1, 0, 2)   # (H, T, hd)
    k = rmsnorm(k, w["self_attn.k_norm.weight"], c.eps).transpose(1, 0, 2)   # (KV, T, hd)
    v = v.transpose(1, 0, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    rep = c.n_heads // c.n_kv_heads
    k, v = np.repeat(k, rep, axis=0), np.repeat(v, rep, axis=0)
    s = q @ k.transpose(0, 2, 1) * (c.head_dim ** -0.5)                      # (H, T, T)
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    o = softmax(s) @ v                                                       # (H, T, hd)
    o = o.transpose(1, 0, 2).reshape(T, c.n_heads * c.head_dim)
    return o @ w["self_attn.o_proj.weight"].T


def mlp(x, w):
    g = x @ w["mlp.gate_proj.weight"].T
    u = x @ w["mlp.up_proj.weight"].T
    return (silu(g) * u) @ w["mlp.down_proj.weight"].T


def decoder_layer(x, w, c, positions=None):
    """x: (T, hidden) float32 -> (T, hidden) float32."""
    x = x.astype(np.float32)
    positions = np.arange(x.shape[0]) if positions is None else positions
    cos, sin = rope_cos_sin(positions, c.head_dim, c.rope_theta)
    x = x + attention(rmsnorm(x, w["input_layernorm.weight"], c.eps), w, c, cos, sin)
    x = x + mlp(rmsnorm(x, w["post_attention_layernorm.weight"], c.eps), w)
    return x.astype(np.float32)


def random_weights(c, seed=0):
    r = np.random.default_rng(seed)
    n = lambda *s: (r.standard_normal(s) * 0.05).astype(np.float32)
    h, qd, kd = c.hidden, c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim
    return {
        "input_layernorm.weight": 1 + n(h),
        "self_attn.q_proj.weight": n(qd, h), "self_attn.k_proj.weight": n(kd, h),
        "self_attn.v_proj.weight": n(kd, h), "self_attn.o_proj.weight": n(h, qd),
        "self_attn.q_norm.weight": 1 + n(c.head_dim), "self_attn.k_norm.weight": 1 + n(c.head_dim),
        "post_attention_layernorm.weight": 1 + n(h),
        "mlp.gate_proj.weight": n(c.inter, h), "mlp.up_proj.weight": n(c.inter, h),
        "mlp.down_proj.weight": n(h, c.inter),
    }
