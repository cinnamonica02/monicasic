# 07 — NVDLA (reference only)

Source: `NVDLA_fork/hw/vmod/nvdla/` directory names (cdma, cbuf, csc, cmac, cacc, sdp, pdp, cdp, rubik, bdma, glb, csb_master, apb2csb, nocif).
Roles are from my knowledge of NVDLA and I have not read the RTL. It is conv-oriented and has no attention or softmax path.

```mermaid
flowchart LR
    CPU["Host CPU"] -->|APB| A2C["apb2csb → csb_master"]
    A2C --> REGS["per-block config regs<br/>+ GLB (interrupts)"]

    MEM["System DRAM<br/>(+ optional SRAM)"] <--> NOC["nocif / memory interface"]

    subgraph PIPE["Convolution pipeline"]
        CDMA["CDMA<br/>fetch weights + features"]
        CBUF["CBUF<br/>convolution buffer"]
        CSC["CSC<br/>sequence controller"]
        CMAC["CMAC<br/>MAC array (int8/fp16)"]
        CACC["CACC<br/>accumulator"]
    end

    subgraph POST["Post-processing"]
        SDP["SDP: bias, scale, activation,<br/>element-wise"]
        PDP["PDP: pooling"]
        CDP["CDP: LRN"]
    end

    BDMA["BDMA: bulk copy"]
    RUBIK["Rubik: reshape / contract"]

    NOC --> CDMA --> CBUF --> CSC --> CMAC --> CACC --> SDP
    SDP --> PDP
    SDP --> CDP
    SDP --> NOC
    PDP --> NOC
    CDP --> NOC
    NOC <--> BDMA
    NOC <--> RUBIK
    REGS -.-> PIPE
    REGS -.-> POST
```

Reusable ideas: the fetch → buffer → sequence → MAC → accumulate → post-process split, the register-programmed layer descriptors, and keeping the post-ops after the accumulator (same idea as tiny-tpu's VPU).
Not reusable as a base: the convolution-specific buffer and sequencer, and the size.
