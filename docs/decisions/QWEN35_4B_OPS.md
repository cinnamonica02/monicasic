# Qwen3.5-4B — what the chip has to compute (text path only)

Sources: `Qwen/Qwen3.5-4B` `config.json` and `transformers` `models/qwen3_5/modeling_qwen3_5.py`, fetched 2026-10-12.
The fetch tool summarized the files, so line-level details (e.g. how Q/K are repeated to 32 heads) are **not confirmed** and should be checked against the source when we write the reference.
Vision encoder (24-layer ViT, GELU-tanh) and the MTP head are ignored: text-only inference.

## Dimensions
| | value |
|---|---|
| layers | 32 = 8 × (3 × linear_attention + 1 × full_attention); full attention at layers 4, 8, ..., 32 |
| hidden / FFN | 2560 / 9216 (SwiGLU, SiLU, no bias) |
| vocab | 248,320, tied embedding and head |
| RMSNorm eps | 1e-6; weights are zero-centered: scale is `(1 + w)`, computed in fp32 |
| max context | 262,144 |

### Gated DeltaNet layer (24 of them)
- Q/K: 16 heads × 128 (key_dim 2048). V: 32 heads × 128 (value_dim 4096).
- `in_proj_qkv`: 2560 → 8192. `in_proj_z`: 2560 → 4096. `in_proj_b`, `in_proj_a`: 2560 → 32 each. `out_proj`: 4096 → 2560.
- Depthwise causal conv1d over the 8192 qkv channels, kernel 4, then SiLU. Split into q, k, v after the conv.
- Q and K are L2-normalised per head (eps 1e-6, fp32). Q is scaled by 128^-0.5.
- Gates, per value head per token: `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`, `decay = exp(g)`.
- State S per value head: 128 × 128, kept in **fp32**.
- Decode (recurrent) update, per head:
  ```
  S      = S * decay
  kv_mem = S^T k            # matvec
  delta  = (v - kv_mem) * beta
  S      = S + k (outer) delta   # rank-1 update
  out    = S^T q            # matvec
  ```
- Output: `rmsnorm(out) * w * silu(z)`, then `out_proj`. This gated norm uses plain `w`, not `1 + w` (from the summary; verify).
- Prefill in the reference uses a **chunked** form with `solve_triangular`. We can instead run the recurrent form token by token, and avoid a triangular solver in hardware.

### Gated Attention layer (8 of them)
- 16 Q heads, 4 KV heads, head_dim 256. `q_proj` outputs 2 × 256 per head: query and an output gate.
- Per-head RMSNorm on Q and K before RoPE.
- RoPE on 64 of the 256 dims (partial_rotary_factor 0.25), theta 10,000,000. The config lists mrope sections [11, 11, 10]; for text-only input I expect this to reduce to ordinary RoPE (unverified).
- Scale 256^-0.5, causal softmax, then `attn * sigmoid(gate)`, then `o_proj` (4096 → 2560).

## Parameter and memory arithmetic (mine)
| item | value |
|---|---|
| Embedding (tied head) | 248,320 × 2560 = 635.7 M |
| FFN, 32 layers | 3 × 2560 × 9216 = 70.8 M per layer, 2.27 B total |
| DeltaNet projections | about 42.1 M per layer, about 1.0 B over 24 layers |
| Attention projections | about 36.7 M per layer, about 0.29 B over 8 layers |
| Sum | about 4.2 B (card says 4B; I did not chase the 0.2 B difference) |
| Recurrent state | 32 × 128 × 128 = 524,288 values per layer; fp32 = 2 MB per layer, 50 MB over 24 layers |
| Conv state | 8192 channels × 3 = 24.6 K values per layer |
| KV cache | 8 layers × 2 × 4 × 256 = 16,384 values per token (INT8: 16 KB/token, 64 MB at 4K context) |
| Decode weight traffic | about 2.3 GB at 4-bit with group scales; head alone is 0.32 GB |
| Recurrent update MACs | about 3 × 128² per head → 38 M MAC per token over all heads/layers (tiny vs ~4 G for weights) |

## What is new compared with the toy MoE block
1. Depthwise conv1d (k=4) with SiLU.
2. L2-norm per head (rsqrt).
3. Sigmoid, softplus, exp: three LUT functions with wide input ranges.
4. The state update: elementwise multiply and a rank-1 outer-product add on a 128×128 matrix, plus two matvecs. The elementwise parts suit the vector unit, not the systolic array.
5. Gated RMSNorm, and output gating by sigmoid.
6. Partial RoPE (64 of 256 dims), per-head QK-norm, head_dim 256.
7. Zero-centered RMSNorm weights `(1 + w)`.
8. Head dimension (256 and 128) differs from hidden size; GQA ratios 4:1 (attention) and 2:1 (DeltaNet value heads vs Q/K heads).

## Design consequences
- **Numerics are fp32 in the reference.** State, gates and norms are computed in fp32. A pure integer pipeline has to choose fixed-point formats for them, and the state is recurrent, so errors accumulate across tokens. Whether the state can be quantized to INT8/INT16 is an open question and likely a research result in itself.
- **So bit-exact against Hugging Face is not possible.** The plan would be: (a) a float32 golden that matches HF closely; (b) an integer reference with defined formats, measured against (a) on logits and perplexity; (c) the chip bit-exact against (b).
- **Dataflow for the recurrent layers:** process one value head at a time with its 128×128 state held in scratchpad (64 KB in fp32). Stream q, k, v and the gates through it. For prefill this keeps the state out of DRAM; otherwise 100 MB of state read + write per token would dominate.
- **Memory footprint is friendlier than Qwen3-4B for long context:** 24 layers have a constant-size state and only 8 have a KV cache.
