# Model choice and open-source stack — proposal

Diagrams are in `docs/diagrams/` (01–09). This doc answers: what did the reference projects actually run, which model should we target, and what can we reuse.

## 1. What the reference projects targeted

**transformer-tpu-int4** (`reference/transformer-tpu-int4/model/transformer.py`, `.../writeup/writeup.md`)
- A tiny decoder-only transformer for **multi-digit addition** (up to 31 digits, 64-token prefill + 64-token decode).
- Live config `adder_int4_wide`: d=128, d_ff=512, 4 layers, 4 heads (head_dim 32), no biases, int4 weights and activations via QAT, no positional encoding (the causal mask is the only position signal).
- Deliberate simplifications: **softmax → ReLU, LayerNorm → tanh, GELU → ReLU**.
- It is a fixed architecture, but the hardware is programmable through PicoRV32 firmware.
- Results: decode is >90% of runtime and DMA-bound. Fusion gave about 24% and double-buffering about 18%.

**tiny-tpu / blog** targets a 2→2→1 XOR MLP. The method matters more than the model.
1. Write the layer math: Z = XWᵀ + b, H = act(Z), loss, derivatives.
2. Give each term a unit: matmul → systolic array, bias/act/loss/derivative → a streaming VPU.
3. Add memory (the unified buffer) only when a real need shows up, such as keeping H for backprop.
4. Size the ISA to the signals that exist.

We should do the same for the MoE block (diagram 09), not copy their specific ops. Their activation choices are what our plan is trying to fix.

## 2. Should we decide on their model? No

The addition model is a fine **regression and benchmark baseline**, but it has no experts and no real softmax. Use it to check our toolchain against a known result, not as the target.

## 3. Kimi / Qwen / Gemma at 3B? Not for the chip

| | Weight bytes |
|---|---|
| 3B int8 | ~3 GB |
| 3B int4 | ~1.5 GB |
| 3B ternary (1.58-bit) | ~0.6 GB |

A ~15 mm² die in an open PDK holds well under a few MB of SRAM. The exact figure depends on which SRAM macros are available and must be measured in OpenLane. It is roughly 3 orders of magnitude short, even ternary.
- **MoE saves compute, not memory.** Qwen3-30B-A3B has ~3B *active* params but ~30B *total*, and all experts must be resident for our thesis to hold. Kimi's flagship models are far larger still.
- Real models also need RoPE, RMSNorm, GQA, and long-context KV storage.

So there are three tiers:

| Tier | Model | Where it runs | Purpose |
|---|---|---|---|
| A (target) | **ChipMoE-Tiny**: our own tiny MoE, Qwen3-MoE-style block | ISS → RTL → FPGA → OpenFrame | The thing we tape out |
| B (validation) | A real small open checkpoint (e.g. Qwen3-0.6B-class dense, or OLMoE-class MoE) | ISS and a larger FPGA with real DRAM only | Shows the op set and ISA are complete for real architectures |
| C (analysis) | 3B-class MoE and dense (Qwen3-30B-A3B, Gemma 3, etc.) | Paper roofline using the same cost model | Shows what the architecture would need to scale (SRAM, bandwidth) |

I haven't checked current model lineups past my June 2026 knowledge cutoff, so confirm tier B/C candidates before committing.

### ChipMoE-Tiny sizing

Block: RMSNorm, RoPE, MHA/GQA with real softmax, SwiGLU experts, top-1 or top-2 router, 2–4 experts, char-level or small vocab.

Params per layer ≈ 4·d² (attention) + E·3·d·d_ff (experts) + d·E (router).

| Config | Total params | int8 | ternary |
|---|---|---|---|
| d=128, d_ff=256, E=4, 4 layers | ~1.84 M | ~1.8 MB | ~0.4 MB |
| d=64, d_ff=128, E=4, 4 layers | ~0.46 M | ~0.46 MB | ~0.09 MB |

The first likely does **not** fit on chip at int8, and the second likely does. We'll **let the measured SRAM budget pick the config**, and treat the ternary experts as the knob that fits more experts. That is the first number to get from OpenLane.

### Task

Don't train the MoE on addition alone. With one narrow task, routing often collapses to one expert, and the "MoE" does nothing. Use a small multi-task mix (addition + copy/reverse + simple char-level text) so experts have something to specialize on, and log router utilization.

## 4. Open-source stack to reuse

| Layer | Option | Why |
|---|---|---|
| Accelerator generator | **Gemmini** (Berkeley, Chipyard) | Open systolic-array generator with scratchpad, DMA and a RoCC interface. The closest existing thing to our plan. Good for study and a possible baseline. Chisel-based. |
| Control core | **PicoRV32** (already proven in the int4 project), Ibex, VexRiscv (LiteX) | Small; PicoRV32/Ibex are easy to fit. VexRiscv comes with LiteX. |
| SoC / FPGA integration | **LiteX** | Already in the platform plan. |
| Compiler frontend | **IREE** (MLIR-based; supports custom HAL backends), **torch-mlir**, **ONNX-MLIR**, **StableHLO** | Real route from PyTorch/JAX to a custom target. Heavy, so only after the hand-written exporter works. |
| Code generation for the controller | **LLVM RISC-V backend** | If we keep a RISC-V controller, firmware kernels compile with stock clang/gcc, as the int4 project does. |
| HW-side MLIR | **CIRCT** | Optional, for generating RTL from higher-level descriptions. High effort. Skip for now. |
| Alternative tensor compiler | **Apache TVM** (BYOC; VTA as a template accelerator) | Another way to target a custom accelerator. |
| Quantized-format reference | **llama.cpp / GGUF**, HF `transformers` | Golden outputs and block-quant formats for tier B/C. |
| Simulation / verification | **Verilator, Icarus, cocotb**, Surfer/GTKWave | Already in use. |
| Physical flow | **OpenLane / LibreLane, Yosys, OpenROAD, OpenRAM (SRAM macros), KLayout**, ChipFoundry `cf` CLI | OpenFrame path. |
| Reference RTL | **NVDLA** (study only) | See diagram 07. |

Recommended minimum: **PyTorch → our exporter → numpy ISS → cocotb/Verilator → LiteX FPGA → OpenLane.** Study Gemmini before finalizing our array and scratchpad interface. IREE/StableHLO is a later, optional frontend.

## 5. Decisions needed from you

1. Config tier: start with the small config (d=64) and scale up if SRAM allows? (Recommended.)
2. Expert weight format: INT8 baseline with ternary as a measured variant? (Recommended.)
3. Controller: PicoRV32 firmware (flexible, as in the int4 project) or a static VLIW stream (smaller, but needs an expert-select mechanism)?
4. Do we get the SRAM budget from OpenLane/OpenRAM first, before fixing model dimensions? (Recommended.)
