# Acceptance criteria

Status: **DRAFT.** Values marked PROPOSED are suggestions, not decisions. Freeze (commit) before the first measured run, then do not edit thresholds after seeing results.
Precedent: Melten's N10 note (`docs/background/SUCCESS_STORY.md`) fixed targets, prompts and quality bounds before measuring. Their numbers are self-reported.

## Targets

| id | what | model | pass condition | status |
|---|---|---|---|---|
| A1 | float golden matches `transformers` | Qwen3-4B (layer, then full) | max abs diff < 1e-4 per layer; full-model top-1 agree on test set | STATUS (thresholds unchanged): layer check run, small random-weight config, max abs diff 2.4e-07; full 36-layer real-weight check (`tiny_moe/ref/check_qwen3_full.py`, run1 fp32 baseline, 4 prompts, 128 positions): top-1 agreement 128/128, max abs logit diff 4e-05..1.5e-04 (fp16 vs fp32: 0.03..0.16) |
| A2 | integer reference vs float golden (quality) | Qwen3-4B | top-1 next-token agreement >= 97% on the frozen set AND perplexity within 0.5% of float (N10: 95.3-98.4% agreement; N11: 97.2% agreement, perplexity within 0.45%) | **FROZEN 2026-10-16 (user confirmed 97% / 0.5%)** for the INT8 baseline, on the frozen quality set. Looser bars for cheaper formats (INT4, ternary) must be agreed BEFORE measuring them. Float reference: ppl 6.802818 (`out/quality_ref/float_ref.json`) |
| A3 | simulator / RTL vs integer reference (exactness) | toy MoE, then Qwen3-4B | 100% of steps equal: token id AND digest of full output logits | PROPOSED (N10 method) |
| A4 | bandwidth efficiency | FPGA or memory-model sim | tokens/s x bytes read per token / sustained read bandwidth >= 60% at 2K context (N10's target). Predict it with a memory-controller simulation BEFORE the build and report predicted vs measured (N11: 45.3 predicted, 45.37 GB/s measured) | PROPOSED |
| A5 | negative tests | all | corrupted data, malformed commands and an invalid expert index are detected and fail cleanly | PROPOSED |

## Frozen test set
FROZEN 2026-10-15 (user approved), before any quantized result existed. Do not edit the entries below; a change means a new versioned set.
- Model: **base** `Qwen/Qwen3-4B` at commit `1cfa9a7208912126459214e8b04321603b3df60c`, raw text, no chat template.
- Prompts (A1 and generation checks): 4 fixed prompts, greedy decoding, batch 1, 32 new tokens, in `data/frozen_prompts.json` (sha256 `f9a2c1f2b5b7859f049f8eb83533f8e129e984310e635f1a36555e716a437195`). Baselines: `run1`.
- Quality set (A2): 4 windows x 1,024 Qwen3 tokens = **4,092 scored positions** (replaces the earlier proposed 256, which was too few to resolve a 97% gate: about +-2 points at 256 vs about +-0.5 at 4,092). 2 prose windows (WikiText-2 test, `Salesforce/wikitext` revision `b08601e0...`) and 2 code windows (CPython 3.12.10 stdlib `argparse.py`, `dataclasses.py`). Exact sources, offsets, file sha256 and token-id digests: `data/quality_set_recipe.json`; rebuild and check with `uv run --group ref python -m tiny_moe.ref.build_quality_set --verify`. The text itself is not committed (third-party).
- Scoring convention (A2): each window is fed alone, no BOS (Qwen3 adds none). Logits at positions 0..1022 are scored against tokens 1..1023. The reference "float" is the numpy fp32 forward (`tiny_moe/ref/check_qwen3_full.py`), not HF. **Top-1 agreement** = fraction of the 4,092 pooled positions where argmax(integer logits) == argmax(float logits), ties to the lowest index. **Perplexity** = exp(mean negative log-likelihood of the true next token), log-softmax in float64; "within 0.5%" means |ppl_integer / ppl_float - 1| <= 0.005. The gate applies to the pooled set; per-window and prose-vs-code numbers are reported separately (see Rules) and are not gates.
- Known limitation: Qwen has probably seen this text in training, so absolute perplexity is low. A2 compares integer against float on the same text, so this does not invalidate it; agreement may be slightly optimistic compared with unseen text.
- Context qualified: 2,048 tokens (N10's figure). TBD for us. A2 itself uses 1,024-token windows; a 2K-context check comes later with the same text joined.

## Rules
- Targets are written before results. A failed check triggers another design/measure cycle, not a threshold change.
- Every reported result carries a run manifest (`runs/MANIFEST_TEMPLATE.json`).
- A result is accepted only after a separate check (a fresh reviewer with only the evidence), not by the author of the change.
- Post-synthesis (netlist) simulation is part of A3 for any RTL that goes to FPGA/OpenLane. N10 found two tool-translation bugs this way.
- Decision rules for choosing between alternatives (e.g. two implementations of the same block, a faster path that fails a fault test) are written and approved before the comparison is run. N11 rejected a faster DMA engine this way.
- Evidence sets are reported separately: frozen corpus positions, long-generation runs, and (later) multi-turn chat. N11 used all three.
- Against NVIDIA/HF output we measure closeness (A2). Against our integer reference we demand equality (A3). Do not mix them.
