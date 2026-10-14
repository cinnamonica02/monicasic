# Acceptance criteria

Status: **DRAFT.** Values marked PROPOSED are suggestions, not decisions. Freeze (commit) before the first measured run, then do not edit thresholds after seeing results.
Precedent: Melten's N10 note (`docs/background/SUCCESS_STORY.md`) fixed targets, prompts and quality bounds before measuring. Their numbers are self-reported.

## Targets

| id | what | model | pass condition | status |
|---|---|---|---|---|
| A1 | float golden matches `transformers` | Qwen3-4B (layer, then full) | max abs diff < 1e-4 per layer; full-model top-1 agree on test set | layer check written (`tiny_moe/ref/check_qwen3_layer.py`), not yet run |
| A2 | integer reference vs float golden (quality) | Qwen3-4B | top-1 next-token agreement >= 97% on the frozen set AND perplexity within 0.5% of float (N10: 95.3-98.4% agreement; N11: 97.2% agreement, perplexity within 0.45%) | PROPOSED |
| A3 | simulator / RTL vs integer reference (exactness) | toy MoE, then Qwen3-4B | 100% of steps equal: token id AND digest of full output logits | PROPOSED (N10 method) |
| A4 | bandwidth efficiency | FPGA or memory-model sim | tokens/s x bytes read per token / sustained read bandwidth >= 60% at 2K context (N10's target). Predict it with a memory-controller simulation BEFORE the build and report predicted vs measured (N11: 45.3 predicted, 45.37 GB/s measured) | PROPOSED |
| A5 | negative tests | all | corrupted data, malformed commands and an invalid expert index are detected and fail cleanly | PROPOSED |

## Frozen test set (to fill in, then freeze)
- Prompts: 4 fixed prompts, greedy decoding, batch 1. TBD: store verbatim in `data/frozen_prompts.json`. Decide base vs instruct; if instruct, apply the chat template.
- Quality set: 256 scored positions from a fixed long-context text. TBD source.
- Context qualified: 2,048 tokens (N10's figure). TBD for us.

## Rules
- Targets are written before results. A failed check triggers another design/measure cycle, not a threshold change.
- Every reported result carries a run manifest (`runs/MANIFEST_TEMPLATE.json`).
- A result is accepted only after a separate check (a fresh reviewer with only the evidence), not by the author of the change.
- Post-synthesis (netlist) simulation is part of A3 for any RTL that goes to FPGA/OpenLane. N10 found two tool-translation bugs this way.
- Decision rules for choosing between alternatives (e.g. two implementations of the same block, a faster path that fails a fault test) are written and approved before the comparison is run. N11 rejected a faster DMA engine this way.
- Evidence sets are reported separately: frozen corpus positions, long-generation runs, and (later) multi-turn chat. N11 used all three.
- Against NVIDIA/HF output we measure closeness (A2). Against our integer reference we demand equality (A3). Do not mix them.
