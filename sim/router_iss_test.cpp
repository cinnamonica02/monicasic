#include "router_iss.hpp"

#include <cassert>

int main() {
    using namespace tiny_moe;
    const auto schedule = export_schedule(1, 3, 2);
    assert(schedule.commands.size() == 6);

    const auto routes = topk_indices({4, 7, 7, 3, 3, 2, 3, 0}, 2, 4, 2);
    assert((routes == std::vector<int>{1, 2, 0, 1}));

    const auto result = run_router(
        schedule,
        {1, 9, 9, 8, 2, 8},
        {10, 0, 20, 2, 30, 4, 1, 10, 2, 20, 3, 30},
        {16384, 16384, 24576, 8192}, 2, 2);
    assert((result.routes == std::vector<int>{1, 2, 0, 2}));
    assert((result.output == std::vector<std::int64_t>{25, 3, 2, 15}));
    return 0;
}
