"""Quantization-study building blocks on small shapes (numpy only)."""
import numpy as np
from test_qwen3_full import C, NL, VOCAB, _make

from tiny_moe.ref.check_qwen3_full import Checkpoint, cfg_from_json, final_hidden
from tiny_moe.ref.qwen3_ref import decoder_layer, random_weights
from tiny_moe.ref.quant_study import CONFIGS, LINEAR, QCfg, make_qa, qlayer, quant_sym, quant_weight, run_configs


def test_qlayer_identity_equals_reference():
    w = random_weights(C, seed=4)
    x = np.random.default_rng(1).standard_normal((9, 64)).astype(np.float32)
    np.testing.assert_array_equal(qlayer(x, w, C, make_qa(None)), decoder_layer(x, w, C))


def test_quant_sym_levels_and_error_bound():
    x = np.random.default_rng(0).standard_normal((5, 128)).astype(np.float32) * 3
    d, q, s = quant_sym(x, 8)
    assert q.min() >= -127 and q.max() <= 127 and np.abs(q).max() == 127           # saturating grid, absmax maps to qmax
    assert (np.abs(d - x) <= s / 2 + 1e-6).all()                                    # error at most half a step
    d4, q4, _ = quant_sym(x, 4)
    assert np.abs(q4).max() == 7 and np.abs(d4 - x).max() > np.abs(d - x).max()
    z, qz, sz = quant_sym(np.zeros((2, 8), np.float32), 8)                          # all-zero row: no divide by zero
    assert (z == 0).all() and (sz == 1).all()


def test_quant_weight_groups_are_independent():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((6, 128)).astype(np.float32)
    w[:, :64] *= 100                                                                # a huge first group must not coarsen the second
    d = quant_weight(w, 8, 64)
    assert np.abs(d[:, 64:] - w[:, 64:]).max() < 0.02 * np.abs(w[:, 64:]).max()
    assert np.abs(d[:, :64] - w[:, :64]).max() < 0.02 * np.abs(w[:, :64]).max()


def test_run_configs_float_matches_final_hidden_and_a8_is_close(tmp_path):
    _make(tmp_path)
    ck = Checkpoint(tmp_path)
    cfg, n = cfg_from_json(tmp_path / "config.json")
    rng = np.random.default_rng(5)
    windows = rng.integers(0, VOCAB, (2, 12))
    take = [np.arange(11)] * 2
    quiet = lambda *_: None
    want = final_hidden(ck, cfg, n, list(windows), take, log=quiet)
    cfgs = [CONFIGS["float"], QCfg("w8a8_small", 8, 32, 8)]                          # group 32 divides the toy dims (64, 128)
    got = run_configs(ck, cfg, n, windows, cfgs, log=quiet)
    np.testing.assert_array_equal(got["float"], want)
    err = np.abs(got["w8a8_small"] - want).max() / np.abs(want).max()
    assert 0 < err < 0.2
