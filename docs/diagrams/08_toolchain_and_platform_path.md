# 08 — Software toolchain and platform path (agreed pipeline + open-source options)

Source: `CHAT_RECAP.md` (platform path), `GUIDE_2.md`, plus the StableHLO idea in INSPO. Items in dashed boxes are optional or proposed.

```mermaid
flowchart TB
    MODEL["Model definition + QAT<br/>PyTorch / JAX<br/>int8 baseline, ternary experts as a variant"]
    GOLD["Golden model<br/>numpy, bit-exact integer reference"]
    EXPORT["export.py<br/>quantized weights + static schedule<br/>+ expert-select hooks"]

    subgraph OPT["Optional frontends (dashed)"]
        SHLO["StableHLO / ONNX import"]
        MLIR["MLIR / IREE custom backend"]
    end

    ISS["ISS: bit-exact instruction-set simulator"]
    RTLSIM["RTL sim: cocotb + Verilator / Icarus<br/>VCD in Surfer / GTKWave"]
    FPGA["FPGA prototype<br/>LiteX SoC + our accelerator"]
    SYN["OpenLane / LibreLane<br/>area, timing, power"]
    OF["ChipFoundry OpenFrame<br/>~15 mm², 44 GPIO"]
    TT["Tiny Tapeout baseline<br/>(kept as separate repo)"]

    MODEL --> GOLD
    MODEL --> EXPORT
    GOLD -. "compare outputs" .-> ISS
    EXPORT --> ISS --> RTLSIM --> FPGA
    RTLSIM --> SYN --> OF
    FPGA -. "live demo" .-> OF
    TT -. "verified 2x2 MXU, proven routing pattern" .-> RTLSIM
    MODEL -.-> SHLO -.-> MLIR -.-> EXPORT

    style OPT stroke-dasharray: 5 5
```

Every arrow into the ISS or RTL is checked bit-for-bit against the golden model. This is the `transformer-tpu-int4` test-suite pattern, with `-b iss`, `-b rtl` and `-b board` as the backends.
