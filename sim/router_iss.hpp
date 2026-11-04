#pragma once

#include <cstdint>
#include <vector>

namespace tiny_moe {

enum class Op { RouteTopK, Dispatch, Expert, Combine };

struct Command {
    Op op;
    int layer;
    int expert = -1;
};

struct Schedule {
    int layers;
    int experts;
    int top_k;
    std::vector<Command> commands;
};

Schedule export_schedule(int layers, int experts, int top_k);
std::vector<int> topk_indices(const std::vector<std::int64_t>& scores,
                              int tokens, int experts, int top_k);

struct RouteResult {
    std::vector<std::int64_t> output;
    std::vector<int> routes;
};

RouteResult run_router(const Schedule& schedule,
                       const std::vector<std::int64_t>& scores,
                       const std::vector<std::int64_t>& expert_outputs,
                       const std::vector<std::int64_t>& weights_q15,
                       int tokens, int hidden);

} // namespace tiny_moe
