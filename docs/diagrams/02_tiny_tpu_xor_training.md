# 02 — tiny-tpu (XOR inference + training)

Source: `tiny-tpu/README.md`, `TINY_TPU_BLOG.md`.
Weight-stationary systolic array, 94-bit ISA, trains a 2→2→1 MLP. The design is derived from the layer math (bias, activation, loss, derivative).

```mermaid
flowchart TB
    CU["Control Unit<br/>94-bit instruction"]

    subgraph UB["Unified Buffer (dual-port scratchpad)"]
        UBX["inputs X"]
        UBW["weights W"]
        UBB["bias / leak factor"]
        UBH["post-activations H<br/>(kept for backprop)"]
    end

    subgraph SA["Systolic array (2x2, weight-stationary)"]
        STAG["input + weight FIFOs<br/>staggered by 1 clk per row"]
        PE["PE grid: psum = in * W + psum_in<br/>double-buffered weights<br/>(shadow reg + switch / accept flags)"]
    end

    subgraph VPU["Vector Processing Unit (pipelined, pathway bits select stages)"]
        direction LR
        B["bias add"] --> A["Leaky ReLU"] --> L["MSE loss"] --> D["Leaky ReLU derivative"]
    end

    GD["Gradient descent module<br/>W = W - lr * dW"]

    CU --> UB
    CU --> SA
    CU --> VPU
    UBX --> STAG
    UBW --> STAG
    STAG --> PE --> VPU
    UBB --> VPU
    VPU -->|"H"| UBH
    VPU -->|"forward result"| UBX
    UBH -->|"backward pass: dL/dZ^T x H, tiled"| STAG
    PE -->|"weight gradients"| GD
    GD -->|"updated W"| UBW
```

Takeaway for us: forward and backward both use the same matmul array. Everything non-matmul sits in a streaming VPU behind it.
