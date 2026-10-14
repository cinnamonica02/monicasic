# 04 — ChipFoundry-scale quantized MoE (the plan's draft target)

Source: `CHIPFOUNDRY_MOE_PLAN.md`. Draft, open to revision. Blocks marked (?) are open questions in the plan.

```mermaid
flowchart TB
    HOST["Host CPU<br/>(FPGA: LiteX / OpenFrame: GPIO, 44 pins)"]

    subgraph CHIP["Custom MoE chip (~15 mm² user area)"]
        SEQ["Own sequencer + ISA<br/>matmul dispatch, expert-select, softmax trigger"]

        subgraph COMPUTE["Compute"]
            ARR["Systolic array NxN, INT8<br/>start 4x4, grow to 8x8 (?)"]
            SMX["Softmax unit<br/>LUT exp + reciprocal normalise"]
            VPU["Vector unit<br/>norm, activation, residual add"]
            RTR["Router<br/>small matmul + argmax / top-k (?)"]
        end

        subgraph SRAM["On-chip scratchpad SRAM (banked)"]
            ATT["attention weights"]
            E0["expert 0"]
            E1["expert 1"]
            EN["expert 2..3 (?)"]
            KV["KV cache"]
            ACT["activations"]
        end

        DMA["DMA / I/O interface"]
    end

    HOST <-->|"commands, tokens, weights at load time"| DMA
    DMA <--> SRAM
    SEQ --> ARR
    SEQ --> SMX
    SEQ --> VPU
    SEQ --> RTR
    ARR <--> SRAM
    SMX <--> SRAM
    VPU <--> SRAM
    RTR -->|"expert id"| SEQ
    SEQ -->|"point array at resident expert"| E0
    SEQ -.-> E1
    SEQ -.-> EN
```

Differences from #01: the experts stay resident and nothing is re-streamed. Softmax is real, in hardware. The sequencer is on chip.
Open: whether the routing decision stays host-side or on chip (the plan lists this as open).
