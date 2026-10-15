"""Full-model float32 numpy forward of Qwen3 on REAL weights, streamed layer by layer, compared with the HF baseline (A1, full model).

No torch needed. bf16 safetensors are read directly (bf16 -> fp32 is a 16-bit shift, exact). One layer is ~0.4 GB in fp32, so
7 GB of RAM is enough. The 4 prompts are run teacher-forced: sequence = prompt + the baseline's generated tokens, one causal pass,
and the logits at the last prompt position and each generated position equal the decode-step logits of baseline_dump.py
(up to float rounding). Each layer's weights are therefore read once for all prompts.

  uv run --group ref python -m tiny_moe.ref.check_qwen3_full                 # downloads ~8 GB (pinned commit), then compares
  uv run --group ref python -m tiny_moe.ref.check_qwen3_full --dir <local snapshot dir>

A1 in docs/spec/ACCEPTANCE.md only says "full-model top-1 agree on test set"; there is no logit threshold, so this prints numbers
and does not invent a pass/fail. Do not tune anything to the numbers.
"""
import argparse
import hashlib
import json
import platform
import struct
import time
from pathlib import Path

import numpy as np

from .qwen3_ref import LAYER_KEYS, Cfg, decoder_layer, rmsnorm

PINNED = "1cfa9a7208912126459214e8b04321603b3df60c"   # the commit the run1 baselines used
_DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}


def to_f32(a, dt):
    if dt == "BF16":
        return (a.astype(np.uint32) << 16).view(np.float32)   # bf16 is the top 16 bits of a float32
    return a.astype(np.float32)


class Checkpoint:
    """Minimal safetensors reader (all *.safetensors in a directory); no open handles are kept."""

    def __init__(self, d):
        self.t = {}   # name -> (file, dtype, shape, abs start, abs end)
        for f in sorted(Path(d).glob("*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    s, e = v["data_offsets"]
                    self.t[k] = (f, v["dtype"], tuple(v["shape"]), 8 + n + s, 8 + n + e)
        if not self.t:
            raise FileNotFoundError(f"no *.safetensors in {d}")

    def get(self, name, rows=None):
        """Tensor as float32. rows: None (all), a slice, or an int array (row gather, e.g. embeddings)."""
        f, dt, shape, s, e = self.t[name]
        code = _DT[dt]
        isz = np.dtype(code).itemsize
        rowlen = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        with open(f, "rb") as fh:
            if rows is None:
                fh.seek(s)
                a = np.frombuffer(fh.read(e - s), dtype=code).reshape(shape)
            elif isinstance(rows, slice):
                r0, r1, _ = rows.indices(shape[0])
                fh.seek(s + r0 * rowlen * isz)
                a = np.frombuffer(fh.read((r1 - r0) * rowlen * isz), dtype=code).reshape((r1 - r0,) + shape[1:])
            else:
                parts = []
                for r in rows:
                    fh.seek(s + int(r) * rowlen * isz)
                    parts.append(np.frombuffer(fh.read(rowlen * isz), dtype=code))
                a = np.stack(parts).reshape((len(rows),) + shape[1:])
        return to_f32(a, dt)


def cfg_from_json(path):
    j = json.load(open(path))
    rope = j.get("rope_theta") or (j.get("rope_parameters") or {}).get("rope_theta")
    assert rope, "rope_theta not found in config.json"
    assert not j.get("rope_scaling"), "rope scaling is not implemented in qwen3_ref"
    assert not j.get("use_sliding_window"), "sliding-window attention is not implemented in qwen3_ref"
    return Cfg(hidden=j["hidden_size"], n_heads=j["num_attention_heads"], n_kv_heads=j["num_key_value_heads"],
               head_dim=j.get("head_dim") or j["hidden_size"] // j["num_attention_heads"],
               inter=j["intermediate_size"], rope_theta=float(rope), eps=float(j["rms_norm_eps"])), j["num_hidden_layers"]


def head_name(ck):
    return "lm_head.weight" if "lm_head.weight" in ck.t else "model.embed_tokens.weight"   # Qwen3-4B ties them


def final_hidden(ck, cfg, n_layers, seqs, take, log=print):
    """Run all layers + final RMSNorm. seqs: list of int token arrays; take: list of position-index arrays.
    Returns (sum(len(take_i)), hidden) float32, rows in order of take."""
    xs = [ck.get("model.embed_tokens.weight", rows=np.asarray(s, dtype=np.int64)) for s in seqs]
    for L in range(n_layers):
        t0 = time.time()
        w = {k: ck.get(f"model.layers.{L}.{k}") for k in LAYER_KEYS}
        xs = [decoder_layer(x, w, cfg) for x in xs]
        log(f"layer {L + 1}/{n_layers}  {time.time() - t0:.1f}s")
    nrm = ck.get("model.norm.weight")
    return np.concatenate([rmsnorm(x[np.asarray(t)], nrm, cfg.eps) for x, t in zip(xs, take)])


def forward(ck, cfg, n_layers, seqs, take, head_chunk=8192, log=print):
    """Returns a list of (len(take_i), vocab) float32 logits. Holds all logits in memory: use head_stats for big sets."""
    h = final_hidden(ck, cfg, n_layers, seqs, take, log)
    head = head_name(ck)
    vocab = ck.t[head][2][0]
    out = np.empty((h.shape[0], vocab), np.float32)
    for r0 in range(0, vocab, head_chunk):
        out[:, r0:r0 + head_chunk] = h @ ck.get(head, rows=slice(r0, r0 + head_chunk)).T
    cuts = np.cumsum([len(t) for t in take])[:-1]
    return np.split(out, cuts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--revision", default=PINNED)
    p.add_argument("--dir", help="local snapshot dir (skips the download)")
    p.add_argument("--baseline", default="out/run1/baseline_fp32")
    p.add_argument("--compare", default="out/run1/baseline_fp16", help="second baseline, for scale (optional)")
    p.add_argument("--out", default="out/numpy_ref_run1")
    a = p.parse_args()

    d = a.dir
    if not d:
        from huggingface_hub import snapshot_download
        d = snapshot_download(a.model, revision=a.revision, allow_patterns=["*.safetensors", "*.json"])
    print("snapshot:", d)
    cfg, n_layers = cfg_from_json(Path(d) / "config.json")
    ck = Checkpoint(d)

    files = sorted(Path(a.baseline).glob("prompt*.npz"))
    assert files, f"no baseline files in {a.baseline}"
    base = [np.load(f) for f in files]
    seqs, take = [], []
    for b in base:
        ids, gen = b["prompt_ids"], b["gen_ids"]
        seqs.append(np.concatenate([ids, gen[:-1]]))
        take.append(np.arange(len(ids) - 1, len(ids) - 1 + len(gen)))   # step j predicts gen[j] from prompt + gen[:j]

    t0 = time.time()
    logits = forward(ck, cfg, n_layers, seqs, take)
    secs = time.time() - t0

    other = [np.load(Path(a.compare) / f.name) for f in files] if Path(a.compare).exists() else None
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows, agree, total = [], 0, 0
    for i, (lg, b) in enumerate(zip(logits, base)):
        ref = b["logits"].astype(np.float64)
        d_abs = np.abs(lg.astype(np.float64) - ref)
        ok = int((lg.argmax(1) == ref.argmax(1)).sum())
        agree, total = agree + ok, total + len(lg)
        row = {"prompt": i, "steps": len(lg), "top1_agree": ok, "max_abs": float(d_abs.max()), "mean_abs": float(d_abs.mean()),
               "tokens_equal_gen_ids": bool(np.array_equal(lg.argmax(1), b["gen_ids"])),
               "logits_sha256": hashlib.sha256(lg.tobytes()).hexdigest()}
        if other is not None:
            row["fp16_vs_fp32_max_abs"] = float(np.abs(other[i]["logits"].astype(np.float64) - ref).max())
        rows.append(row)
        np.savez_compressed(f"{a.out}/prompt{i}.npz", logits=lg)
        print(row)
    res = {"snapshot": str(d), "model": a.model, "revision": a.revision, "baseline": a.baseline, "seconds": round(secs, 1),
           "numpy": np.__version__, "python": platform.python_version(), "top1_agree_total": f"{agree}/{total}", "prompts": rows}
    json.dump(res, open(f"{a.out}/result.json", "w"), indent=2)
    print(f"\nfull-model top-1 agreement vs fp32 baseline: {agree}/{total}   ({secs:.0f}s)   -> {a.out}/result.json")


if __name__ == "__main__":
    main()
