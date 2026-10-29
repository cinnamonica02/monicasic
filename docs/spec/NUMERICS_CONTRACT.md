# Numerics contract

Status: **DRAFT.** Everything marked PROPOSED/TBD must be decided by measurement (A2) or by the user. The integer reference (to be written in `tiny_moe/ref/`) is the authority for the chip once this is frozen.

## Authority
float32 golden (`tiny_moe/ref/qwen3_ref.py`, matches HF) -> integer reference (this contract) -> RTL/simulator, bit-exact.

## Formats
| tensor | format | status |
|---|---|---|
| dense weights (attn, FFN) | INT8, symmetric group-scaled, group size 64 | selected by simulated A2 study (W8A16/group-64) |
| toy MoE weights | INT8 per-tensor baseline; ternary {-1,0,+1} x scale as variant | decided (see `tiny_moe/model/moe.py`) |
| activations | INT16, symmetric per-token at linear inputs | selected by simulated A2 study (99.36% agreement, perplexity +0.209%); exact fixed-point scale is defined below |
| LM head | INT16 | PROPOSED (N10: all-INT8 head missed their quality gate at 7B) |
| accumulators | INT64 reference accumulators; INT32-or-wider hardware accumulators | reference width selected to prevent overflow; hardware width TBD from range analysis |
| KV cache | INT16 fixed-point K/V cache | selected for the first integer reference; per-head scale compression is a later variant |
| embeddings | INT8, symmetric group-scaled, group size 64 | selected for the first integer reference |
| norm weights, scales | signed fixed-point INT16 | selected for the first integer reference |

## Operations needing a defined integer/LUT form
- RMSNorm: integer sum-of-squares plus integer square-root division; normalized values use Q10.
- Softmax: integer exp LUT over score differences clamped to [-8, 0], output probabilities use Q15. Keep real softmax in hardware (not ReLU as in transformer-tpu-int4).
- SiLU (SwiGLU gate): integer sigmoid/exp LUT over [-8, 8], output uses Q10.
- RoPE: Hugging Face pairing (i, i+hd/2); cos/sin tables use Q14.
- Rounding: round-to-nearest-even; saturate on quantization, never wrap.
- Linear scales: positive fixed-point Q20 values; linear outputs and residuals use Q10.
- Qwen3.5 stretch adds sigmoid, softplus, wide-range exp, L2-norm, depthwise conv1d; recurrent state format open (fp32 in the reference). Not part of the first contract.

## Additional requirements (from N11, SUCCESS_STORY / N11 note)
- The integer reference uses integer arithmetic only during forward execution (numpy int32/int64), no floating point, so it is bit-exact across machines. Weight/table conversion happens once when the checkpoint is exported. N11's reference was identical on three CPU types and the board.
- ARGMAX / token selection is part of the contract: ties break to the lowest index (PROPOSED), and the chip feeds the token back without the host.
- Router: gate precision and top-k selection (ties, renormalization of the selected weights) must be defined. The toy default is top-1 of 4; raise to top-2 of 8 so combining weights and ties are exercised (PROPOSED). N11 routed 8 of 128.
- Invalid expert index must produce a defined error, not silent wrong output.

## Rules
- Any change to this file requires re-running A2 and regenerating the integer reference outputs.
- The exporter writes weights in exactly these formats; the manifest records the contract version (git hash of this file).
