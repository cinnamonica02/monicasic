"""Plumbing tests for the streamed full-model reference on a tiny fake bf16 checkpoint (numpy only, no download)."""
import json
import struct

import numpy as np

from tiny_moe.ref.check_qwen3_full import Checkpoint, cfg_from_json, forward, to_f32
from tiny_moe.ref.qwen3_ref import LAYER_KEYS, Cfg, decoder_layer, random_weights, rmsnorm

C = Cfg(hidden=64, n_heads=4, n_kv_heads=2, head_dim=32, inter=128)
VOCAB, NL = 50, 2


def _bf16(a):
    return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)   # truncate to bf16


def _write(path, tensors):
    hdr, off, blobs = {}, 0, []
    for k, a in tensors.items():
        b = a.astype("<u2").tobytes()
        hdr[k] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _make(tmp_path):
    rng = np.random.default_rng(0)
    t = {"model.embed_tokens.weight": _bf16(rng.standard_normal((VOCAB, 64)) * 0.5),
         "model.norm.weight": _bf16(1 + 0.05 * rng.standard_normal(64))}
    for L in range(NL):
        for k, v in random_weights(C, seed=L).items():
            t[f"model.layers.{L}.{k}"] = _bf16(v)
    _write(tmp_path / "model.safetensors", t)
    (tmp_path / "config.json").write_text(json.dumps(
        {"hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "intermediate_size": 128,
         "rope_theta": 1e6, "rms_norm_eps": 1e-6, "num_hidden_layers": NL, "rope_scaling": None}))
    return t


def test_bf16_roundtrip_and_row_gather(tmp_path):
    t = _make(tmp_path)
    ck = Checkpoint(tmp_path)
    name = "model.embed_tokens.weight"
    full = to_f32(t[name], "BF16")
    np.testing.assert_array_equal(ck.get(name), full)
    np.testing.assert_array_equal(ck.get(name, rows=np.array([3, 0, 49])), full[[3, 0, 49]])
    np.testing.assert_array_equal(ck.get(name, rows=slice(10, 17)), full[10:17])
    cfg, n = cfg_from_json(tmp_path / "config.json")
    assert n == NL and cfg.n_kv_heads == 2 and cfg.head_dim == 32


def test_forward_matches_manual_chain_and_is_causal(tmp_path):
    t = _make(tmp_path)
    ck = Checkpoint(tmp_path)
    cfg, n = cfg_from_json(tmp_path / "config.json")
    seq = np.array([5, 1, 7, 7, 2, 40, 3, 9])
    take = np.array([2, 5, 7])
    (out,) = forward(ck, cfg, n, [seq], [take], head_chunk=16, log=lambda *_: None)   # chunk < vocab: tests the head loop
    f = lambda k: to_f32(t[k], "BF16")
    x = f("model.embed_tokens.weight")[seq]
    for L in range(NL):
        x = decoder_layer(x, {k: f(f"model.layers.{L}.{k}") for k in LAYER_KEYS}, cfg)
    want = rmsnorm(x[take], f("model.norm.weight"), cfg.eps) @ f("model.embed_tokens.weight").T   # tied head
    np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-5)
    seq2 = seq.copy()
    seq2[6:] = [11, 12]
    (out2,) = forward(ck, cfg, n, [seq2], [take], log=lambda *_: None)
    np.testing.assert_array_equal(out[:2], out2[:2])      # positions before the edit cannot change
    assert np.abs(out[2] - out2[2]).max() > 1e-4          # position 7 does
