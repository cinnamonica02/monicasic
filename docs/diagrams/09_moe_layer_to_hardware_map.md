# 09 — MoE transformer block → hardware unit map, and prefill/decode flow

Method borrowed from `TINY_TPU_BLOG.md`: write the layer math, then give every operation a unit.
Block style is Qwen3-MoE-like (RMSNorm, RoPE, GQA, softmax attention, SwiGLU experts, top-k router) so the op set scales to real models.
Only the first two are in the int4 project. The others are what we add.

## Op → unit

```mermaid
flowchart TB
    X["x (residual stream)"] --> N1["RMSNorm"]
    N1 --> QKV["Q,K,V projections"]
    QKV --> ROPE["RoPE"]
    ROPE --> QK["QK^T"]
    QK --> SM["softmax (+ causal mask)"]
    SM --> PV["P x V"]
    PV --> O["output projection"]
    O --> ADD1["residual add"]
    X --> ADD1
    ADD1 --> N2["RMSNorm"]
    N2 --> RT["router: x W_r"]
    RT --> TOPK["softmax + top-k"]
    TOPK --> EXP["selected expert(s):<br/>gate, up → SiLU(gate)*up → down"]
    N2 --> EXP
    EXP --> COMB["weighted combine"]
    COMB --> ADD2["residual add"]
    ADD1 --> ADD2
    ADD2 --> OUT["next layer / LM head → argmax"]

    classDef mxu fill:#cfe8ff,stroke:#1b5fa8
    classDef vpu fill:#ffe9c7,stroke:#a8701b
    classDef lut fill:#e0f5d8,stroke:#2f7d1b
    classDef ctl fill:#f5d8e8,stroke:#a81b6e
    class QKV,QK,PV,O,RT,EXP mxu
    class N1,N2,ROPE,ADD1,ADD2,COMB vpu
    class SM,TOPK lut
    class TOPK ctl
```

Legend: blue = systolic array (matmul), orange = vector unit, green = LUT (exp / reciprocal / rsqrt / SiLU), pink = sequencer decision (expert select).

| Op | Unit | Notes |
|---|---|---|
| Q,K,V,O, router, expert gate/up/down | array | about 90% of MACs. Reads weights from SRAM. |
| QKᵀ | array | read K transposed, as in the int4 MXU (no physical transpose) |
| softmax | LUT unit | max-subtract, exp LUT, sum, reciprocal LUT. The int4 project avoided it by using ReLU. |
| RMSNorm | vector + rsqrt LUT | int4 project used tanh-norm |
| RoPE | vector | sin/cos table in ROM |
| SiLU | LUT | shares the exp LUT |
| top-k | sequencer | writes expert-base register, then dispatches |

## Prefill vs decode with expert residency

```mermaid
sequenceDiagram
    participant H as Host
    participant S as Sequencer
    participant A as Array
    participant R as Router/LUT
    participant E as Expert SRAM banks
    participant K as KV bank

    H->>S: load program + weights (once)
    Note over S,E: all experts resident, no re-streaming
    rect rgb(235,245,255)
    Note over H,K: PREFILL (compute-bound, T tokens at once)
    H->>S: prompt tokens
    loop each layer
        S->>A: QKV matmul (T rows)
        A->>K: write K,V
        S->>A: attention (flash-style tiles)
        S->>R: router for T tokens
        R-->>S: expert id per token
        S->>E: group tokens by expert
        S->>A: expert FFN per group
    end
    end
    rect rgb(255,243,230)
    Note over H,K: DECODE (bandwidth-bound, 1 token)
    loop each new token
        S->>A: QKV matmul (1 row)
        S->>K: read cached K,V
        S->>R: router
        R-->>S: expert id
        S->>E: read ONLY chosen expert weights
        S->>A: expert FFN
    end
    S-->>H: next token
    end
```

In decode the weight read dominates (the int4 project measured DMA at 3–4x MXU time). Because an MoE reads only the chosen experts, it cuts weight traffic per token. This is the actual specialization advantage.
