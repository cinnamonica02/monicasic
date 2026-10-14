"""Standalone sanity tests for the numpy Qwen3 layer reference (numpy only; no torch needed).
The comparison against transformers lives in tiny_moe/ref/check_qwen3_layer.py (needs torch)."""
import numpy as np

from tiny_moe.ref.qwen3_ref import Cfg, apply_rope, decoder_layer, random_weights, rope_cos_sin

C = Cfg(hidden=64, n_heads=4, n_kv_heads=2, head_dim=32, inter=128)


def test_shape_and_finite():
    x = np.random.default_rng(1).standard_normal((10, 64)).astype(np.float32)
    y = decoder_layer(x, random_weights(C), C)
    assert y.shape == (10, 64) and np.isfinite(y).all()


def test_causal():
    w = random_weights(C)
    x = np.random.default_rng(1).standard_normal((10, 64)).astype(np.float32)
    x2 = x.copy()
    x2[7:] += 1.0
    y, y2 = decoder_layer(x, w, C), decoder_layer(x2, w, C)
    assert np.abs(y[:7] - y2[:7]).max() == 0.0          # earlier rows cannot see later tokens
    assert np.abs(y[7:] - y2[7:]).max() > 1e-3           # later rows do change


def test_rope_rotates_and_preserves_norm():
    cos, sin = rope_cos_sin(np.arange(10), 32, 1e6)
    v = np.random.default_rng(2).standard_normal((1, 10, 32)).astype(np.float32)
    r = apply_rope(v, cos, sin)
    assert np.abs(r - v).max() > 1.0
    np.testing.assert_allclose(np.linalg.norm(r, axis=-1), np.linalg.norm(v, axis=-1), rtol=1e-5)
