# 06 — Ternary (BitNet b1.58) + static VLIW + StableHLO variant (from INSPO.md addendum)

Source: the "Software Map" section added to `INSPO.md`. The VLIW layout is a sketch from that chat and has gaps, which are marked below.

```mermaid
flowchart TB
    W["Open weights / own QAT model"] --> Q["Quantized graph<br/>ternary weights {-1,0,+1}, int8 activations"]
    Q --> HLO["StableHLO (optional frontend)"]
    HLO --> BE["Custom backend: static schedule → VLIW binary"]
    BE --> ROM["Instruction memory"]

    subgraph CHIP["Chip"]
        VLIW["VLIW decode<br/>64-bit word"]
        subgraph F["Fields (sketch)"]
            F1["MXU opcode 16b<br/>reset/run, accumulate, sparsity"]
            F2["DMA src 16b<br/>bank, base, size"]
            F3["DMA dst 16b"]
            F4["LUT / softmax 16b<br/>activation, scale, step"]
        end
        MXU["Ternary array<br/>add / sub / skip, no multipliers<br/>(linear layers only)"]
        MULT["INT8 multipliers<br/>still needed for QK^T and PV"]
        LUT["LUT block: exp, SiLU"]
        SRAM["Banked scratchpad SRAM"]
    end

    ROM --> VLIW --> F
    F1 --> MXU
    F1 --> MULT
    F2 --> SRAM
    F3 --> SRAM
    F4 --> LUT
    MXU <--> SRAM
    MULT <--> SRAM
    LUT <--> SRAM

    GAP["GAP: data-dependent MoE routing<br/>needs an expert-base register or branch.<br/>The sketched word has no such field."]
    VLIW -.-> GAP
```
