"""Integer-only Qwen3 reference for the frozen A2 check.

Checkpoint conversion uses floating point once to make integer weights and LUTs.
The forward path uses integer arrays only: Q10 activations, Q14 trig/norm
constants, Q15 attention probabilities, and Q20 weight scales.
"""
import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .build_quality_set import ids_sha
from .check_qwen3_full import PINNED, Checkpoint, cfg_from_json, head_name

ACT_FRAC = 10
NORM_FRAC = 14
SCALE_FRAC = 20
PROB_FRAC = 15
LUT_STEP = 256
LUT_LIMIT = 8 * LUT_STEP
LUT_FRAC = 16
GROUP = 64


def round_shift(x, bits):
    """Signed round-to-nearest-even right shift, using integer arithmetic."""
    if bits == 0:
        return x
    x = np.asarray(x, dtype=np.int64)
    a = np.abs(x)
    q, r = np.divmod(a, 1 << bits)
    half = 1 << (bits - 1)
    q += (r > half) | ((r == half) & ((q & 1) == 1))
    return np.where(x < 0, -q, q).astype(np.int64)


def fixed_from_float(x, frac):
    return np.rint(np.asarray(x, dtype=np.float32) * (1 << frac)).astype(np.int64)


def _exp_lut():
    """Build exp(-i/256) with integer recurrence; output is Q16."""
    out = np.empty(LUT_LIMIT + 1, dtype=np.int64)
    out[0] = 1 << LUT_FRAC
    # Integer approximation of exp(-1/256) * 2^16.
    factor = 65280
    for i in range(1, len(out)):
        out[i] = (out[i - 1] * factor + (1 << (LUT_FRAC - 1))) >> LUT_FRAC
    return out


EXP_LUT = _exp_lut()


def _lut_index(x, frac):
    """Index exp(-abs(x)) where x is non-negative fixed-point."""
    i = (np.asarray(x, dtype=np.int64) * LUT_STEP + (1 << (frac - 1))) >> frac
    return np.clip(i, 0, LUT_LIMIT).astype(np.int64)


def integer_silu(x, frac=ACT_FRAC):
    """SiLU using the integer exp LUT, returning the same fixed-point format."""
    x = np.asarray(x, dtype=np.int64)
    neg = x < 0
    e = EXP_LUT[_lut_index(np.abs(x), frac)]
    den = (1 << LUT_FRAC) + e
    sig = np.where(
        neg,
        (e * (1 << LUT_FRAC) + den // 2) // den,
        ((1 << LUT_FRAC) * (1 << LUT_FRAC) + den // 2) // den,
    )
    return round_shift(x * sig, LUT_FRAC)


def integer_rmsnorm(x, norm_weight, eps_scaled, frac=ACT_FRAC):
    """RMSNorm with integer sum-of-squares and integer square root."""
    x = np.asarray(x, dtype=np.int64)
    n = x.shape[-1]
    denom2 = (np.sum(x * x, axis=-1, dtype=np.int64) + n * eps_scaled) // n
    roots = np.maximum(1, np.vectorize(math.isqrt, otypes=[np.int64])(denom2))
    y = (x * (1 << frac) + roots[..., None] // 2) // roots[..., None]
    return round_shift(y * norm_weight, NORM_FRAC)


def integer_rope(x, cos_q14, sin_q14, frac=ACT_FRAC):
    """Hugging Face rotate-half RoPE using Q14 tables."""
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return round_shift(x * cos_q14 + rotated * sin_q14, NORM_FRAC)


def integer_softmax(scores, score_frac):
    """Causal softmax: integer exp LUT and Q15 reciprocal normalization."""
    mx = scores.max(axis=-1, keepdims=True)
    idx = _lut_index(mx - scores, score_frac)
    e = EXP_LUT[idx]
    total = e.sum(axis=-1, keepdims=True, dtype=np.int64)
    return ((e * (1 << PROB_FRAC) + total // 2) // total).astype(np.int64)


@dataclass
class QuantizedWeight:
    q: np.ndarray
    scales_q20: np.ndarray
    group: int = GROUP

    @classmethod
    def from_float(cls, w, bits, group=GROUP):
        w = np.asarray(w, dtype=np.float32)
        out, ins = w.shape
        assert ins % group == 0, (w.shape, group)
        qmax = (1 << (bits - 1)) - 1
        g = w.reshape(out, ins // group, group)
        scale = np.abs(g).max(axis=-1) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        q = np.clip(np.rint(g / scale[..., None]), -qmax, qmax).astype(np.int64)
        scales_q20 = np.maximum(1, np.rint(scale * (1 << SCALE_FRAC))).astype(np.int64)
        return cls(q.reshape(out, ins), scales_q20, group)

    def matmul(self, x):
        """x is fixed-point; return a fixed-point output with the same frac."""
        x = np.asarray(x, dtype=np.int64)
        # Expand the group scales once so the whole projection is one integer
        # GEMM. Rounding after the accumulated dot product is the hardware-like
        # requantization point and avoids one GEMM per scale group.
        effective = self.q.astype(np.int64) * np.repeat(self.scales_q20, self.group, axis=1)
        return round_shift(x @ effective.T, SCALE_FRAC)

    def rows_fixed(self):
        """Return quantized rows as fixed-point values, for embeddings."""
        out = np.empty_like(self.q, dtype=np.int64)
        for g0 in range(0, self.q.shape[1], self.group):
            g1 = g0 + self.group
            out[:, g0:g1] = round_shift(
                self.q[:, g0:g1] * self.scales_q20[:, g0 // self.group, None], SCALE_FRAC
            )
        return out


@dataclass
class KVCache:
    k: np.ndarray | None = None
    v: np.ndarray | None = None

    def write_prefill(self, k, v):
        self.k = np.asarray(k, dtype=np.int64).copy()
        self.v = np.asarray(v, dtype=np.int64).copy()


class IntegerLayer:
    def __init__(self, weights, cfg):
        self.cfg = cfg
        self.w = {k: QuantizedWeight.from_float(weights[k], 8) for k in (
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "mlp.down_proj.weight")}
        self.norm = {k: fixed_from_float(weights[k], NORM_FRAC) for k in (
            "input_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "post_attention_layernorm.weight")}
        self.eps_scaled = max(1, int(round(cfg.eps * (1 << (2 * ACT_FRAC)))))
        self.cos_q14, self.sin_q14 = self._rope(np.arange(0), cfg.head_dim, cfg.rope_theta)

    @staticmethod
    def _rope(positions, head_dim, theta):
        inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        ang = np.outer(np.asarray(positions, dtype=np.float32), inv)
        ang = np.concatenate([ang, ang], axis=-1)
        return fixed_from_float(np.cos(ang), NORM_FRAC), fixed_from_float(np.sin(ang), NORM_FRAC)

    def forward(self, x, positions):
        x = np.asarray(x, dtype=np.int64)
        self.cos_q14, self.sin_q14 = self._rope(positions, self.cfg.head_dim, self.cfg.rope_theta)
        h = integer_rmsnorm(x, self.norm["input_layernorm.weight"], self.eps_scaled)
        q = self.w["self_attn.q_proj.weight"].matmul(h).reshape(-1, self.cfg.n_heads, self.cfg.head_dim)
        k = self.w["self_attn.k_proj.weight"].matmul(h).reshape(-1, self.cfg.n_kv_heads, self.cfg.head_dim)
        v = self.w["self_attn.v_proj.weight"].matmul(h).reshape(-1, self.cfg.n_kv_heads, self.cfg.head_dim)
        q = integer_rmsnorm(q, self.norm["self_attn.q_norm.weight"], self.eps_scaled).transpose(1, 0, 2)
        k = integer_rmsnorm(k, self.norm["self_attn.k_norm.weight"], self.eps_scaled).transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        q, k = integer_rope(q, self.cos_q14, self.sin_q14), integer_rope(k, self.cos_q14, self.sin_q14)
        rep = self.cfg.n_heads // self.cfg.n_kv_heads
        k, v = np.repeat(k, rep, axis=0), np.repeat(v, rep, axis=0)
        cache = KVCache()
        cache.write_prefill(k, v)
        scores = q @ cache.k.transpose(0, 2, 1)
        scores = round_shift(scores * int(round((1 / math.sqrt(self.cfg.head_dim)) * (1 << NORM_FRAC))), NORM_FRAC)
        # Keep the masked value far below any real score but avoid overflow when
        # the softmax indexes (max - score) into the LUT.
        scores = np.where(np.tril(np.ones((x.shape[0], x.shape[0]), dtype=bool)), scores, -(1 << 40))
        probs = integer_softmax(scores, 2 * ACT_FRAC)
        o = round_shift(probs @ cache.v, PROB_FRAC).transpose(1, 0, 2).reshape(x.shape[0], -1)
        x = x + self.w["self_attn.o_proj.weight"].matmul(o)
        h = integer_rmsnorm(x, self.norm["post_attention_layernorm.weight"], self.eps_scaled)
        g = self.w["mlp.gate_proj.weight"].matmul(h)
        u = self.w["mlp.up_proj.weight"].matmul(h)
        m = round_shift(integer_silu(g) * u, ACT_FRAC)
        return x + self.w["mlp.down_proj.weight"].matmul(m)


class IntegerQwen3:
    def __init__(self, ck, cfg, n_layers):
        self.ck, self.cfg, self.n_layers = ck, cfg, n_layers
        self.eps_scaled = max(1, int(round(cfg.eps * (1 << (2 * ACT_FRAC)))))

    def embeddings(self, ids):
        w = QuantizedWeight.from_float(self.ck.get("model.embed_tokens.weight", rows=ids), 8)
        return w.rows_fixed()

    def final_hidden(self, ids, log=print):
        x = self.embeddings(np.asarray(ids, dtype=np.int64))
        positions = np.arange(len(ids), dtype=np.int64)
        for li in range(self.n_layers):
            t0 = time.time()
            weights = {k: self.ck.get(f"model.layers.{li}.{k}") for k in (
                "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight",
                "self_attn.k_norm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight",
                "mlp.up_proj.weight", "mlp.down_proj.weight")}
            x = IntegerLayer(weights, self.cfg).forward(x, positions)
            log(f"layer {li + 1}/{self.n_layers}  {time.time() - t0:.1f}s")
        norm = fixed_from_float(self.ck.get("model.norm.weight"), NORM_FRAC)
        return integer_rmsnorm(x, norm, self.eps_scaled)[:-1]


def integer_head_stats(ck, h, targets, head_chunk=8192):
    """Stream an INT16 LM head; metrics are float only after integer logits exist."""
    h = np.asarray(h, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    m = len(h)
    head = head_name(ck)
    vocab = ck.t[head][2][0]
    run_max = np.full(m, -np.inf)
    run_sum = np.zeros(m)
    top1 = np.full(m, np.iinfo(np.int64).min, dtype=np.int64)
    top2 = np.full(m, np.iinfo(np.int64).min, dtype=np.int64)
    arg = np.zeros(m, dtype=np.int64)
    tgt = np.zeros(m, dtype=np.int64)
    for r0 in range(0, vocab, head_chunk):
        w = QuantizedWeight.from_float(ck.get(head, rows=slice(r0, min(r0 + head_chunk, vocab))), 16)
        lg = w.matmul(h)
        vals = lg.astype(np.float64) / (1 << ACT_FRAC)
        lg64 = vals
        m_new = np.maximum(run_max, lg64.max(1))
        run_sum = run_sum * np.exp(run_max - m_new) + np.exp(lg64 - m_new[:, None]).sum(1)
        i = lg.argmax(1)
        c1 = lg[np.arange(m), i]
        better = c1 > top1
        arg = np.where(better, i + r0, arg)
        c2 = np.partition(lg, -2, axis=1)[:, -2] if lg.shape[1] > 1 else np.full(m, -np.inf, dtype=np.int64)
        both = np.sort(np.stack([top1, top2, c1, c2]), axis=0)
        top1, top2 = both[-1], both[-2]
        inside = (targets >= r0) & (targets < r0 + lg.shape[1])
        tgt[inside] = lg[np.nonzero(inside)[0], targets[inside] - r0]
    return {"argmax": arg, "top1": top1, "top2": top2, "lse": run_max + np.log(run_sum), "target_logit": tgt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir")
    p.add_argument("--windows", default="out/quality_set/windows.npz")
    p.add_argument("--recipe", default="data/quality_set_recipe.json")
    p.add_argument("--float-ref", default="out/quality_ref/float_ref.npz")
    p.add_argument("--out", default="out/integer_ref")
    a = p.parse_args()
    windows = np.load(a.windows)["windows"]
    recipe = json.load(open(a.recipe))
    assert [ids_sha(w) for w in windows] == [m["token_ids_sha256"] for m in recipe["windows"]]
    ref = dict(np.load(a.float_ref))
    d = a.dir
    if not d:
        from huggingface_hub import snapshot_download
        d = snapshot_download("Qwen/Qwen3-4B", revision=PINNED, allow_patterns=["*.safetensors", "*.json"])
    cfg, n_layers = cfg_from_json(Path(d) / "config.json")
    model = IntegerQwen3(Checkpoint(d), cfg, n_layers)
    t0 = time.time()
    hidden = np.concatenate([model.final_hidden(w) for w in windows])
    targets = np.concatenate([w[1:] for w in windows]).astype(np.int64)
    s = integer_head_stats(model.ck, hidden, targets)
    nll = s["lse"] - s["target_logit"].astype(np.float64) / (1 << ACT_FRAC)
    ppl = float(np.exp(nll.mean()))
    ppl_float = float(np.exp((ref["lse"] - ref["target_logit"].astype(np.float64)).mean()))
    agree = s["argmax"] == ref["argmax"]
    result = {"purpose": "integer-only reference on frozen A2 quality set", "revision": PINNED,
              "positions": int(len(targets)), "seconds": round(time.time() - t0, 1),
              "top1_agreement": float(agree.mean()), "ppl": ppl, "ppl_float": ppl_float,
              "ppl_rel_diff": ppl / ppl_float - 1, "mismatches": int((~agree).sum()),
              "gate_agreement_ge_97": bool(agree.mean() >= 0.97),
              "gate_ppl_within_0.5pct": bool(abs(ppl / ppl_float - 1) <= 0.005),
              "format": {"weights": "int8 symmetric group64", "activations": "int16 Q10", "lm_head": "int16 group64", "rounding": "nearest-even"},
              "numpy": np.__version__, "python": platform.python_version()}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / "integer_ref.npz", targets=targets, **s)
    json.dump(result, open(Path(a.out) / "result.json", "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
