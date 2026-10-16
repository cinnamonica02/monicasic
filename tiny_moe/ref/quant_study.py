"""Quantization study on the FROZEN A2 quality set: simulated ("fake") quantization of Qwen3-4B, numpy only.

STUDY, not the integer reference. Weights and the activations feeding each linear layer are rounded onto an integer grid and
dequantized again, then multiplied in float32 BLAS. This tells us what the integer formats cost in quality. What it does NOT cover
(all stay float here, and are the next step toward the true integer-only reference of docs/spec/NUMERICS_CONTRACT.md):
  RMSNorm, softmax, SiLU, RoPE, attention scores/values and KV cache, embeddings, norm weights, the LM head.

Quantizers (rounding = round-half-to-even via np.rint; saturating, never wrapping):
  weights:     symmetric, per group of `group` input channels (and per output row), scale = absmax / qmax
  activations: symmetric, per token (dynamic absmax) at the input of every linear layer (q/k/v, o, gate/up, down)

  uv run --group ref python -m tiny_moe.ref.quant_study --configs w8a8        # ~15 min on a 2-core laptop, per config
"""
import argparse
import json
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .build_quality_set import ids_sha
from .check_qwen3_full import PINNED, Checkpoint, cfg_from_json, head_name
from .qwen3_ref import LAYER_KEYS, apply_rope, rmsnorm, rope_cos_sin, silu, softmax
from .score_quality import head_stats

LINEAR = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
          "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


@dataclass(frozen=True)
class QCfg:
    name: str
    w_bits: int | None = 8        # None = float weights
    group: int = 64
    a_bits: int | None = 8        # None = float activations


CONFIGS = {c.name: c for c in (
    QCfg("w8a8", 8, 64, 8),          # the INT8 baseline (A2 gate candidate)
    QCfg("w8a16", 8, 64, 16),        # wider activations
    QCfg("w8a8_g32", 8, 32, 8),      # finer weight groups
    QCfg("w4a8", 4, 64, 8),          # cheaper weights; needs its own pre-agreed bar before it is measured
    QCfg("float", None, 64, None),   # identity: must reproduce the float reference
)}


def quant_sym(x, bits, axis=-1):
    """Symmetric absmax fake-quant along `axis`; returns (dequantized float32, integer levels int32, scale float32)."""
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(x).max(axis=axis, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s).astype(np.float32)
    q = np.clip(np.rint(x / s), -qmax, qmax)
    return (q * s).astype(np.float32), q.astype(np.int32), s


def quant_weight(w, bits, group):
    """w: (out, in) -> dequantized (out, in); scales are per (output row, group of `group` input channels)."""
    o, i = w.shape
    assert i % group == 0, f"in={i} not divisible by group={group}"
    d, _, _ = quant_sym(w.reshape(o, i // group, group), bits, axis=-1)
    return d.reshape(o, i)


def make_qa(bits):
    if bits is None:
        return lambda x: x
    return lambda x: quant_sym(x, bits, axis=-1)[0]


def qlayer(x, w, c, qa):
    """Same math and operation order as qwen3_ref.decoder_layer, with `qa` applied at the input of each linear layer."""
    x = x.astype(np.float32)
    T = x.shape[0]
    cos, sin = rope_cos_sin(np.arange(T), c.head_dim, c.rope_theta)
    h = qa(rmsnorm(x, w["input_layernorm.weight"], c.eps))
    q = (h @ w["self_attn.q_proj.weight"].T).reshape(T, c.n_heads, c.head_dim)
    k = (h @ w["self_attn.k_proj.weight"].T).reshape(T, c.n_kv_heads, c.head_dim)
    v = (h @ w["self_attn.v_proj.weight"].T).reshape(T, c.n_kv_heads, c.head_dim)
    q = rmsnorm(q, w["self_attn.q_norm.weight"], c.eps).transpose(1, 0, 2)
    k = rmsnorm(k, w["self_attn.k_norm.weight"], c.eps).transpose(1, 0, 2)
    v = v.transpose(1, 0, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    rep = c.n_heads // c.n_kv_heads
    k, v = np.repeat(k, rep, axis=0), np.repeat(v, rep, axis=0)
    s = q @ k.transpose(0, 2, 1) * (c.head_dim ** -0.5)
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    o = (softmax(s) @ v).transpose(1, 0, 2).reshape(T, c.n_heads * c.head_dim)
    x = x + qa(o) @ w["self_attn.o_proj.weight"].T
    h = qa(rmsnorm(x, w["post_attention_layernorm.weight"], c.eps))
    m = silu(h @ w["mlp.gate_proj.weight"].T) * (h @ w["mlp.up_proj.weight"].T)
    return (x + qa(m) @ w["mlp.down_proj.weight"].T).astype(np.float32)


def run_configs(ck, cfg, n_layers, windows, qcfgs, log=print):
    """Teacher-forced pass over the windows for several configs at once (each layer's weights are read once).
    Returns {name: final hidden (n_windows * (L-1), hidden) float32 after the final RMSNorm}."""
    L = windows.shape[1]
    emb = ck.get("model.embed_tokens.weight", rows=windows.reshape(-1).astype(np.int64)).reshape(len(windows), L, -1)
    xs = {q.name: [emb[i].copy() for i in range(len(windows))] for q in qcfgs}
    qas = {q.name: make_qa(q.a_bits) for q in qcfgs}
    for li in range(n_layers):
        t0 = time.time()
        w = {k: ck.get(f"model.layers.{li}.{k}") for k in LAYER_KEYS}
        for q in qcfgs:
            wq = dict(w)
            if q.w_bits is not None:
                for k in LINEAR:
                    wq[k] = quant_weight(w[k], q.w_bits, q.group)
            xs[q.name] = [qlayer(x, wq, cfg, qas[q.name]) for x in xs[q.name]]
        log(f"layer {li + 1}/{n_layers}  {time.time() - t0:.1f}s")
    nrm = ck.get("model.norm.weight")
    return {n: np.concatenate([rmsnorm(x[:L - 1], nrm, cfg.eps) for x in v]) for n, v in xs.items()}


def compare(s, ref, kinds, window):
    """Agreement and perplexity of stats `s` (from head_stats) against the stored float reference `ref`."""
    nll = s["lse"] - s["target_logit"].astype(np.float64)
    ppl = float(np.exp(nll.mean()))
    ppl_f = float(np.exp((ref["lse"] - ref["target_logit"].astype(np.float64)).mean()))
    agree = s["argmax"] == ref["argmax"]
    gap = (ref["top1"] - ref["top2"]).astype(np.float64)
    out = {"top1_agreement": float(agree.mean()), "ppl": ppl, "ppl_float": ppl_f, "ppl_rel_diff": ppl / ppl_f - 1.0,
           "agreement_per_window": [float(agree[window == i].mean()) for i in range(window.max() + 1)],
           "agreement_prose": float(agree[np.isin(window, np.nonzero(kinds == "prose")[0])].mean()),
           "agreement_code": float(agree[np.isin(window, np.nonzero(kinds == "code")[0])].mean()),
           "mismatches": int((~agree).sum()), "mismatches_with_float_gap_below_0.25": int(((~agree) & (gap < 0.25)).sum()),
           "mean_abs_target_logit_diff": float(np.abs(s["target_logit"].astype(np.float64) - ref["target_logit"]).mean())}
    out["gate_agreement_ge_97"] = out["top1_agreement"] >= 0.97
    out["gate_ppl_within_0.5pct"] = abs(out["ppl_rel_diff"]) <= 0.005
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="w8a8", help="comma list from: " + ",".join(CONFIGS))
    p.add_argument("--dir", help="local Qwen3-4B snapshot dir (default: pinned commit from the HF cache)")
    p.add_argument("--windows", default="out/quality_set/windows.npz")
    p.add_argument("--recipe", default="data/quality_set_recipe.json")
    p.add_argument("--float-ref", default="out/quality_ref/float_ref.npz")
    p.add_argument("--out", default="out/quant_study")
    a = p.parse_args()

    windows = np.load(a.windows)["windows"]
    recipe = json.load(open(a.recipe))
    assert [ids_sha(w) for w in windows] == [m["token_ids_sha256"] for m in recipe["windows"]], "windows do not match the frozen recipe"
    ref = dict(np.load(a.float_ref))
    kinds = np.array([m["kind"] for m in recipe["windows"]])
    d = a.dir
    if not d:
        from huggingface_hub import snapshot_download
        d = snapshot_download("Qwen/Qwen3-4B", revision=PINNED, allow_patterns=["*.safetensors", "*.json"])
    cfg, n_layers = cfg_from_json(Path(d) / "config.json")
    ck = Checkpoint(d)
    qcfgs = [CONFIGS[n] for n in a.configs.split(",")]

    t0 = time.time()
    hs = run_configs(ck, cfg, n_layers, windows, qcfgs)
    t1 = time.time()
    targets = np.concatenate([w[1:] for w in windows]).astype(np.int64)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for q in qcfgs:
        s = head_stats(ck, hs[q.name], targets)
        res = {"config": asdict(q), "revision": PINNED, "recipe_status": recipe["status"], "positions": int(len(targets)),
               "seconds_layers_all_configs": round(t1 - t0, 1), "numpy": np.__version__, "python": platform.python_version(),
               "NOT_QUANTIZED": ["rmsnorm", "softmax", "silu", "rope", "attention scores/values", "kv cache", "embeddings", "norm weights", "lm head"],
               **compare(s, ref, kinds, ref["window"])}
        json.dump(res, open(f"{a.out}/{q.name}.json", "w"), indent=2)
        np.savez_compressed(f"{a.out}/{q.name}_stats.npz", **s)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
