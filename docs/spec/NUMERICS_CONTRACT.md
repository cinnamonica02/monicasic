# Numerics contract

Status: **DRAFT.** Everything marked PROPOSED/TBD must be decided by measurement (A2) or by the user. The integer reference (to be written in `tiny_moe/ref/`) is the authority for the chip once this is frozen.

## Authority
float32 golden (`tiny_moe/ref/qwen3_ref.py`, matches HF) -> integer reference (this contract) -> RTL/simulator, bit-exact.

## Formats
| tensor | format | status |
|---|---|---|
| dense weights (attn, FFN) | INT8, group-scaled, group size 64 (N10 precedent; llama.cpp uses 32) | PROPOSED; group size TBD, scale format TBD |
| toy MoE weights | INT8 per-tensor baseline; ternary {-1,0,+1} x scale as variant | decided (see `tiny_moe/model/moe.py`) |
| activations | INT8 vs INT16 | TBD, measure A2 both (N10 used 16-bit) |
| LM head | INT16 | PROPOSED (N10: all-INT8 head missed their quality gate at 7B) |
| accumulators | INT32 minimum | PROPOSED; width TBD per layer |
| KV cache | INT8 per-head scale | TBD |
| embeddings | same as dense weights | TBD |
| norm weights, scales | fixed-point, width TBD | TBD |

## Operations needing a defined integer/LUT form
- RMSNorm: needs rsqrt (LUT or Newton). Input range and output format TBD.
- Softmax: exp + reciprocal. Keep real softmax in hardware (not ReLU as in transformer-tpu-int4).
- SiLU (SwiGLU gate): LUT, input range TBD.
- RoPE: Hugging Face pairing (i, i+hd/2). cos/sin tables: precision TBD.
- Rounding: TBD (round-half-away vs round-half-even); saturate on overflow, never wrap.
- Qwen3.5 stretch adds sigmoid, softplus, wide-range exp, L2-norm, depthwise conv1d; recurrent state format open (fp32 in the reference). Not part of the first contract.

## Additional requirements (from N11, SUCCESS_STORY / N11 note)
- The integer reference uses integer arithmetic only (numpy int32/int64), no floating point, so it is bit-exact across machines. N11's reference was identical on three CPU types and the board.
- ARGMAX / token selection is part of the contract: ties break to the lowest index (PROPOSED), and the chip feeds the token back without the host.
- Router: gate precision and top-k selection (ties, renormalization of the selected weights) must be defined. The toy default is top-1 of 4; raise to top-2 of 8 so combining weights and ties are exercised (PROPOSED). N11 routed 8 of 128.
- Invalid expert index must produce a defined error, not silent wrong output.

## Rules
- Any change to this file requires re-running A2 and regenerating the integer reference outputs.
- The exporter writes weights in exactly these formats; the manifest records the contract version (git hash of this file).
