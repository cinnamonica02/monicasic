# 01 — Tiny Tapeout 2x2 MXU, host-streamed MoE (current silicon baseline)

Source: `CHIPFOUNDRY_MOE_PLAN.md` ("What's already proven"), `CHAT_RECAP.md`.
No on-chip memory, no ISA change. The host does routing and streams only the chosen expert's weights.
`MOE_ATTENTION_ROADMAP.md` is referenced by the plan but was not in the folder, so this is built from the plan's description of it.

```mermaid
flowchart LR
    subgraph HOST["Host (Python / C++ runtime, Verilator or TT board)"]
        TOK["token activations"]
        RT["router: small matmul"]
        ARG["host argmax"]
        STREAM["weight streamer<br/>(only chosen expert)"]
    end

    subgraph TT["Tiny Tapeout tile (project.v)"]
        IO["8-bit I/O + opcodes<br/>RESET / LOAD_A / LOAD_B / EXECUTE / READ_OUT"]
        MXU["2x2 INT8 MXU<br/>(no local memory)"]
    end

    TOK --> RT --> ARG --> STREAM
    STREAM -->|"expert weights, every use"| IO
    TOK -->|activations| IO
    IO --> MXU
    MXU -->|results| IO
    IO -->|READ_OUT| HOST

    NOTE["Agree case: 1 of 2 experts streamed, 60 cycles<br/>Disagree case: both streamed, 80 cycles, no savings"]
    ARG -.-> NOTE
```
