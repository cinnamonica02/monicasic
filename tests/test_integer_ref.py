"""Small-shape checks for the integer Qwen3 reference."""
import numpy as np

from test_qwen3_full import C
from tiny_moe.ref.integer_qwen3 import IntegerLayer, integer_silu, integer_softmax, round_shift
from tiny_moe.ref.qwen3_ref import random_weights


def test_round_shift_is_nearest_even_for_signed_values():
    x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    np.testing.assert_array_equal(round_shift(x, 1), [-2, -2, -2, -1, 0, 0, 0, 1, 2, 2, 2])


def test_integer_lut_outputs_are_fixed_point_and_finite():
    x = np.arange(-8192, 8193, 257, dtype=np.int64)
    y = integer_silu(x)
    assert y.dtype == np.int64 and np.isfinite(y).all()
    scores = np.zeros((2, 5, 5), dtype=np.int64)
    scores[:, :, 0] = -1 << 40
    p = integer_softmax(scores, 20)
    assert p.dtype == np.int64 and (p >= 0).all()


def test_integer_layer_is_deterministic_and_shape_correct():
    w = random_weights(C, seed=8)
    x = np.random.default_rng(9).standard_normal((7, C.hidden)).astype(np.float32)
    xi = np.rint(x * (1 << 10)).astype(np.int64)
    layer = IntegerLayer(w, C)
    y1 = layer.forward(xi, np.arange(len(x)))
    y2 = layer.forward(xi, np.arange(len(x)))
    assert y1.shape == xi.shape and y1.dtype == np.int64
    np.testing.assert_array_equal(y1, y2)
