# monic_asic

*A tiny TPU / MoE accelerator.* (GitHub repo: `moni_asic`)

A small custom inference chip for a quantized mixture-of-experts model, built to test whether a model-specialized design can beat general-purpose silicon on efficiency (tokens per joule and per area, not absolute speed).

Status: early. A numpy reference of one Qwen3 layer, a toy MoE, frozen test prompts, draft acceptance criteria and a draft numerics contract. No RTL yet.

## Where to start
- `CLAUDE.md` — scope, decisions, working rules, current status (read this first).
- `docs/spec/ACCEPTANCE.md`, `docs/spec/NUMERICS_CONTRACT.md` — what "done" means and the number formats.
- `docs/decisions/` — why Qwen3-4B first, what each reference project ran.
- `docs/diagrams/` — architecture sketches (Mermaid).

## Verification chain
HF Qwen3-4B on a GPU -> float32 numpy golden (`tiny_moe/ref`) -> integer reference -> simulator / RTL / FPGA (bit-exact against the integer reference).

## Develop
Uses [uv](https://docs.astral.sh/uv/). Run everything from the repo root.
```
uv sync                                          # numpy + pytest into .venv (the project itself is not installed)
uv run pytest                                    # numpy-only sanity tests
uv sync --extra torch                            # heavy: CPU torch + transformers on Windows
uv run python -m tiny_moe.ref.check_qwen3_layer  # needs the torch extra
```
On a fresh GPU pod (RunPod, as root) with the repo present, `bash scripts/runpod_session.sh` bootstraps uv, installs CUDA torch with `uv sync --extra torch`, and runs the layer check and the HF baselines. On Colab, where torch is preinstalled, use `uv pip install --system ...` instead of `uv sync`.
C++ (`sim/`, root `CMakeLists.txt`) and RTL (`hw/`) are placeholders.

`reference/` holds third-party projects used as inspiration (NVDLA, tiny-tpu, transformer-tpu-int4); they keep their own licenses and history.
