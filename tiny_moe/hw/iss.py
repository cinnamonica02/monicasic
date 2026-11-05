"""Integer ISS for the toy MoE route/dispatch/combine command slice."""

import numpy as np

from .export import validate_schedule
from ..ref.integer_qwen3 import EXP_LUT, LUT_LIMIT, PROB_FRAC

ROUTER_SCORE_FRAC = 8


def topk_indices(scores, k):
    """Select highest scores, breaking ties by the lowest expert index."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or not 1 <= k <= scores.shape[1]:
        raise ValueError("scores must be 2-D and 1 <= k <= number of experts")
    # Stable descending sort preserves expert index order for equal scores.
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def router_weights_q15(scores, k):
    """Return top-k routes and integer-renormalized Q15 gate weights.

    Scores are signed integer logits in Q8. Score differences are clamped to
    eight before indexing the shared integer exponential LUT.
    """
    scores = np.asarray(scores)
    if not np.issubdtype(scores.dtype, np.integer):
        raise ValueError("router scores must be integer logits")
    routes = topk_indices(scores, k)
    selected = np.take_along_axis(scores, routes, axis=1).astype(np.int64)
    delta = np.maximum(0, scores.max(axis=1, keepdims=True) - selected)
    exp = EXP_LUT[np.minimum(delta, LUT_LIMIT)]
    total = exp.sum(axis=1, dtype=np.int64)
    scaled = exp * (1 << PROB_FRAC)
    weights = scaled // total[:, None]
    remainder = (1 << PROB_FRAC) - weights.sum(axis=1)
    fractions = scaled % total[:, None]
    for row, count in enumerate(remainder):
        order = np.argsort(-fractions[row], kind="stable")
        weights[row, order[:count]] += 1
    return routes, weights.astype(np.int64)


def _round_shift_nearest_even(x, bits):
    x = np.asarray(x, dtype=np.int64)
    if bits == 0:
        return x
    a = np.abs(x)
    q, r = np.divmod(a, 1 << bits)
    half = 1 << (bits - 1)
    q += (r > half) | ((r == half) & ((q & 1) == 1))
    return np.where(x < 0, -q, q)


def run_router_schedule(schedule, router_scores, expert_outputs, weights_q15):
    """Execute route/dispatch/expert/combine for integer toy-MoE tensors.

    `router_scores` has shape (layers, tokens, experts), `expert_outputs` has
    shape (layers, tokens, experts, hidden), and `weights_q15` has shape
    (layers, tokens, top_k). Outputs are int64 hidden states.
    """
    validate_schedule(schedule)
    scores = np.asarray(router_scores)
    expert_outputs = np.asarray(expert_outputs, dtype=np.int64)
    weights = np.asarray(weights_q15, dtype=np.int64)
    layers, tokens, experts = scores.shape
    if (layers, experts) != (schedule["n_layers"], schedule["n_experts"]):
        raise ValueError("router_scores do not match schedule")
    if expert_outputs.shape[:3] != (layers, tokens, experts):
        raise ValueError("expert_outputs do not match router_scores")
    if weights.shape != (layers, tokens, schedule["top_k"]):
        raise ValueError("weights_q15 must have shape (layers, tokens, top_k)")
    if np.any(weights < 0) or np.any(weights.sum(axis=-1) != 1 << 15):
        raise ValueError("weights_q15 must be non-negative and sum to Q15 one")

    out = np.empty((layers, tokens, expert_outputs.shape[-1]), dtype=np.int64)
    routes = np.empty((layers, tokens, schedule["top_k"]), dtype=np.int64)
    for layer in range(layers):
        routes[layer] = topk_indices(scores[layer], schedule["top_k"])
        selected = expert_outputs[layer, np.arange(tokens)[:, None], routes[layer]]
        acc = (selected * weights[layer][..., None]).sum(axis=1, dtype=np.int64)
        out[layer] = _round_shift_nearest_even(acc, 15)
    return out, routes
