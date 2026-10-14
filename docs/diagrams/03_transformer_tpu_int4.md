# 03 — transformer-tpu-int4 (FPGA, Cmod A7-35T)

Source: `transformer-tpu-int4/README.md`, `writeup/writeup.md`.
Model: decoder-only int4 adder, d=128, d_ff=512, 4 layers, 4 heads. Prefill then decode with a KV cache.

```mermaid
flowchart TB
    HOSTPC["Host PC<br/>infer/run.py over UART"]

    subgraph FPGA["Artix-7 FPGA"]
        CPU["PicoRV32 softcore<br/>C firmware: tiled matmul, fused ops, flash-style attention"]

        subgraph QUEUES["Per-unit command queues (async, AXI MMIO)"]
            QM["MXU queue"]
            QV["VPU queue"]
            QD["DMA queue"]
        end

        MXU["MXU: 8x8 output-stationary systolic<br/>int4 x int4, transposed-read trick for QK^T"]
        VPU["VPU: add / ReLU / hardtanh<br/>(softmax→ReLU, LayerNorm→tanh)"]
        DMA["DMA: base addr, stride, 2-D dims"]
        SPAD["Scratchpad (BRAM) 64 KB<br/>= 'VMEM', region-banked,<br/>multiple ports if regions differ"]
        PERF["Perf counters:<br/>MXU / VPU / DMA busy, issue overhead"]
    end

    EXT["External async SRAM 512 KB<br/>= stands in for HBM/DDR<br/>1 B/clk read, 0.5 B/clk write"]

    HOSTPC <-->|UART| CPU
    CPU --> QM --> MXU
    CPU --> QV --> VPU
    CPU --> QD --> DMA
    MXU <--> SPAD
    VPU <--> SPAD
    DMA <--> SPAD
    DMA <--> EXT
    MXU -.-> PERF
    VPU -.-> PERF
    DMA -.-> PERF
```

Measured (12 MHz): prefill-only 64 tokens ≈ 249 ms (fused + double-buffered). Prefill+decode ≈ 4.06 s.
Decode is over 90% of total time, and DMA-bound by weight loads.
