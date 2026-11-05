import json

import numpy as np

from tiny_moe.hw import (export_schedule, required_combine_accumulator_bits,
                         router_weights_q15, run_router_schedule, topk_indices)


def test_topk_ties_choose_lowest_expert_index():
    scores = np.array([[4, 7, 7, 1], [3, 3, 2, 3]])
    np.testing.assert_array_equal(topk_indices(scores, 2), [[1, 2], [0, 1]])


def test_router_iss_dispatches_and_combines_integer_outputs():
    schedule = export_schedule(n_layers=1, n_experts=3, top_k=2)
    scores = np.array([[[1, 9, 9], [8, 2, 8]]])
    expert_outputs = np.array([[[[10, 0], [20, 2], [30, 4]],
                                [[1, 10], [2, 20], [3, 30]]]], dtype=np.int64)
    weights = np.array([[[16384, 16384], [24576, 8192]]], dtype=np.int64)
    out, routes = run_router_schedule(schedule, scores, expert_outputs, weights)
    np.testing.assert_array_equal(routes, [[[1, 2], [0, 2]]])
    np.testing.assert_array_equal(out, [[[25, 3], [2, 15]]])


def test_router_weights_are_integer_q15_and_tie_deterministic():
    routes, weights = router_weights_q15(np.array([[7, 7, 1, 0]], dtype=np.int32), 2)
    np.testing.assert_array_equal(routes, [[0, 1]])
    assert weights.dtype == np.int64
    np.testing.assert_array_equal(weights.sum(axis=1), [1 << 15])
    np.testing.assert_array_equal(weights, [[16384, 16384]])


def test_router_weights_reject_float_scores():
    with np.testing.assert_raises(ValueError):
        router_weights_q15(np.array([[1.0, 2.0]]), 1)


def test_int16_q10_top2_combine_fits_signed_int32():
    assert required_combine_accumulator_bits() == 32


def test_schedule_has_fixed_commands_for_dynamic_routes():
    schedule = export_schedule(2, 3, 2)
    schedule = json.loads(json.dumps(schedule))
    assert [c["op"] for c in schedule["commands"]] == [
        "ROUTE_TOPK", "DISPATCH", "EXPERT", "EXPERT", "EXPERT", "COMBINE",
        "ROUTE_TOPK", "DISPATCH", "EXPERT", "EXPERT", "EXPERT", "COMBINE",
    ]


def test_iss_rejects_reordered_commands():
    schedule = export_schedule(1, 2, 1)
    schedule["commands"][0], schedule["commands"][1] = schedule["commands"][1], schedule["commands"][0]
    with np.testing.assert_raises(ValueError):
        run_router_schedule(schedule, np.zeros((1, 1, 2)), np.zeros((1, 1, 2, 1)), np.array([[[1 << 15]]]))


def test_schedule_rejects_hardware_unsupported_router_sizes():
    with np.testing.assert_raises(ValueError):
        export_schedule(1, 9, 1)
    with np.testing.assert_raises(ValueError):
        export_schedule(1, 8, 3)
