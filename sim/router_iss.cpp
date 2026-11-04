#include "router_iss.hpp"

#include <algorithm>
#include <stdexcept>

namespace tiny_moe {

Schedule export_schedule(int layers, int experts, int top_k) {
    if (layers < 1 || experts < 1 || top_k < 1 || top_k > experts)
        throw std::invalid_argument("invalid schedule dimensions");
    Schedule schedule{layers, experts, top_k, {}};
    for (int layer = 0; layer < layers; ++layer) {
        schedule.commands.push_back({Op::RouteTopK, layer});
        schedule.commands.push_back({Op::Dispatch, layer});
        for (int expert = 0; expert < experts; ++expert)
            schedule.commands.push_back({Op::Expert, layer, expert});
        schedule.commands.push_back({Op::Combine, layer});
    }
    return schedule;
}

std::vector<int> topk_indices(const std::vector<std::int64_t>& scores,
                              int tokens, int experts, int top_k) {
    if (tokens < 0 || experts < 1 || top_k < 1 || top_k > experts ||
        scores.size() != static_cast<std::size_t>(tokens * experts))
        throw std::invalid_argument("invalid router scores");
    std::vector<int> routes;
    routes.reserve(tokens * top_k);
    for (int token = 0; token < tokens; ++token) {
        std::vector<int> order(experts);
        for (int expert = 0; expert < experts; ++expert) order[expert] = expert;
        const auto begin = scores.begin() + token * experts;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return begin[a] > begin[b];
        });
        routes.insert(routes.end(), order.begin(), order.begin() + top_k);
    }
    return routes;
}

static std::int64_t round_shift_even(std::int64_t value, int bits) {
    if (bits == 0) return value;
    const bool negative = value < 0;
    const std::uint64_t magnitude = negative ? -static_cast<std::uint64_t>(value)
                                             : static_cast<std::uint64_t>(value);
    const std::uint64_t divisor = std::uint64_t{1} << bits;
    std::uint64_t quotient = magnitude / divisor;
    const std::uint64_t remainder = magnitude % divisor;
    const std::uint64_t half = divisor >> 1;
    if (remainder > half || (remainder == half && (quotient & 1))) ++quotient;
    const auto result = static_cast<std::int64_t>(quotient);
    return negative ? -result : result;
}

RouteResult run_router(const Schedule& schedule,
                       const std::vector<std::int64_t>& scores,
                       const std::vector<std::int64_t>& expert_outputs,
                       const std::vector<std::int64_t>& weights_q15,
                       int tokens, int hidden) {
    if (tokens < 0 || hidden < 1 || scores.size() != static_cast<std::size_t>(tokens * schedule.experts) ||
        expert_outputs.size() != static_cast<std::size_t>(tokens * schedule.experts * hidden) ||
        weights_q15.size() != static_cast<std::size_t>(tokens * schedule.top_k))
        throw std::invalid_argument("router tensors do not match schedule");

    RouteResult result;
    result.routes = topk_indices(scores, tokens, schedule.experts, schedule.top_k);
    result.output.assign(tokens * hidden, 0);
    for (int token = 0; token < tokens; ++token) {
        std::vector<std::int64_t> sums(hidden, 0);
        std::int64_t weight_sum = 0;
        for (int slot = 0; slot < schedule.top_k; ++slot) {
            const int expert = result.routes[token * schedule.top_k + slot];
            const auto weight = weights_q15[token * schedule.top_k + slot];
            if (weight < 0) throw std::invalid_argument("negative route weight");
            weight_sum += weight;
            const auto base = (token * schedule.experts + expert) * hidden;
            for (int j = 0; j < hidden; ++j) sums[j] += expert_outputs[base + j] * weight;
        }
        if (weight_sum != (std::int64_t{1} << 15))
            throw std::invalid_argument("route weights must sum to Q15 one");
        for (int j = 0; j < hidden; ++j) result.output[token * hidden + j] = round_shift_even(sums[j], 15);
    }
    return result;
}

} // namespace tiny_moe
