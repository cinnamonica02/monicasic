# Toy MoE router numerics

Status: **initial hardware-facing draft**. This defines the router slice used
by `tiny_moe.hw`; it is not the frozen Qwen3 A2 contract.

## Router

- Router logits are signed integer values in Q8 (`ROUTER_SCORE_FRAC = 8`).
- Top-k supports `k = 1..2`; expert count supports `1..8`.
- Larger score wins. Equal scores select the lower expert index.
- For the selected experts, `delta = max(score) - score` is clamped to
  `[0, 8]` real units and evaluated with the shared integer exponential LUT.
- Selected exponential values are renormalized to non-negative Q15 weights
  summing exactly to `32768`.
- Integer normalization uses floor division. Remaining Q15 units go to the
  largest fractional remainders, with lower slot index breaking ties.

The router weights are multiplied with integer expert outputs, accumulated in
the reference width, then rounded-to-nearest-even back to the output format.
