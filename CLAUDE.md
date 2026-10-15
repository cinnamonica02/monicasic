# tiny_moe_chip

Custom small inference chip for a quantized MoE. Question under test: can a model-specialized design beat general-purpose silicon on efficiency (tokens/joule, tokens/area), not absolute speed?

## Scope
- Inference only. No on-chip training. No fine-tuning unless the quality gate (docs/spec/ACCEPTANCE.md A2) fails after quantization; then QAT/LoRA is the repair tool.
- No upstream integration (vLLM etc.). The "compiler" for a fixed model is a Python exporter plus a command-stream generator, with a host runtime on top. Revisit MLIR only if multiple models need support.

## Decisions so far (change only with the user)
- First validation target: **Qwen3-4B** (dense, standard block). Qwen3.5-4B (Gated DeltaNet, see docs/decisions/QWEN35_4B_OPS.md) is a stretch track.
- On-chip model for the first silicon-sized design: toy MoE (`tiny_moe/model/`, d=64 default). INT8 baseline, ternary experts as a variant.
- Path: simulation -> FPGA -> decide OpenFrame (SKY130, ~15 mm2, 44 GPIO) vs another open shuttle (IHP SG13G2, GF180). Tiny Tapeout is done and not the target.
- Keep options open for Qwen3.5 only through cheap choices: lane-structured vector unit, generic LUT interface. Do NOT size the scratchpad for the 128x128 DeltaNet state.
- Controller (PicoRV32 vs VLIW vs host-directed command queue) is open. Melten N10 used host-directed, no soft CPU.

## Authority chain
1. HF `transformers` Qwen3-4B on an NVIDIA GPU (RunPod/Colab): baseline logits/tokens, saved as files.
2. `tiny_moe/ref/qwen3_ref.py` float32 numpy golden: must match HF (A1).
3. Integer reference (to write, `tiny_moe/ref/`): measured against 2 for quality (A2).
4. Simulator / RTL / FPGA: bit-exact against 3 (A3).
Never claim chip == HF bit-exact. Bit-exact is only against the integer reference.

## Working rules
- Ask before: spending cloud credits, buying hardware, installing heavy packages, publishing anything.
- Scratch compute is **Modal (user paid; logged in on this machine as profile `cinnamonica02-53934`)**: `scripts/modal_baseline.py` (layer check + HF baselines; `--smoke` = Qwen3-0.6B plumbing test for cents). The client is in the `cloud` uv group (`uv sync --group cloud`). Modal's "free $30/month" turned out to require paying in practice, so treat every run as spending the user's money: ask before running, state the GPU and expected cost, and use `--smoke` first. Fallbacks, both user-run and untested: RunPod (`scripts/runpod_session.sh`; the user's bare image lacked curl/apt, so it needs a known base image) and Colab (`scripts/baseline_dump.py`). Compute says nothing about chip speed, power or area. Fetch results with `uv run modal volume get tiny-moe-out /<tag> ./out/<tag>` and keep the manifest.
- Local machine is small (7.7 GB RAM, 2-core i3, no cmake/g++/verilator/iverilog installed; cargo is): fine for numpy references and tests; too small for Qwen3-4B. `scripts/modal_run.py` is unused.
- Freeze thresholds in docs/spec/ACCEPTANCE.md before measuring. Do not change them after seeing results. `data/frozen_prompts.json` is frozen-candidate; its sha256 goes into every manifest.
- Mark unverified numbers as unverified (prices, SRAM density, shuttle terms, anything from a summarized fetch).
- Real Qwen weights use Hugging Face RoPE pairing (i with i+hd/2). `tiny_moe/model/moe.py` uses even/odd pairing; do not mix them.
- Every reported result gets a manifest (`runs/MANIFEST_TEMPLATE.json`).

## Layout
```
docs/spec/        ACCEPTANCE.md, NUMERICS_CONTRACT.md      frozen targets and number formats (DRAFT, many TBD)
docs/decisions/   MODEL_AND_OSS_DECISION.md, QWEN35_4B_OPS.md
docs/diagrams/    01-09 Mermaid sketches of each proposed architecture (not rendered/tested)
docs/background/  source material: plans, chats, blogs, SUCCESS_STORY.md (Melten N10; company claims, self-reported). GITIGNORED on purpose: it holds third-party text and chat exports; the user decides before it is ever published.
tiny_moe/model/   toy MoE + QAT training (never run)
tiny_moe/ref/     float32 Qwen3 layer reference + HF comparison (check_qwen3_layer.py written, not yet run)
tests/            pytest, numpy only:  uv run pytest   (env via `uv sync`; uv.lock is committed; project is package=false)
scripts/          baseline_dump.py, runpod_session.sh, modal_run.py (unused)
data/ runs/       frozen prompts; run manifests (outputs are gitignored)
hw/ sim/          empty placeholders for RTL and the C++ simulator (CMakeLists.txt at the root)
reference/        NVDLA_fork; tiny-tpu and transformer-tpu-int4 are STILL at the repo root (moves were refused, see Status)
```
Run Python from the repo root with uv: `uv run python -m tiny_moe.ref.check_qwen3_layer`, `uv run python -m tiny_moe.model.train` (both need `uv sync --extra torch`). User prefers uv over pip/conda. Local Python via uv is 3.12.10; the setuptools editable build crashed on this Windows box, hence `package = false`.

## Status
Done: diagrams, model decision, toy MoE + Modal script (unrun), numpy Qwen3 layer reference (sanity-tested via tests/), frozen-candidate prompts, baseline scripts (untested), repo reorganized.
Baseline run `run1` (2026-10-15, Modal, NVIDIA A10 22.1 GiB, torch 2.14.1+cu130, transformers 5.19.0, Qwen/Qwen3-4B commit 1cfa9a7208912126459214e8b04321603b3df60c, base model raw text, greedy, 32 new tokens, prompts sha f9a2c1f2...7195): layer check passed (max abs diff 2.4e-07 on a small RANDOM-weight layer, CPU, not real weights); fp16 and fp32 both ran; all 4 prompts produced identical tokens in fp16 and fp32 (equal tokens_sha256). Logits npz files (float32) are in Modal volume `tiny-moe-out` under /run1 and downloaded to `out/run1/` (gitignored); fetch with `uv run python scripts/modal_fetch.py` because `modal volume get` on a directory failed with Errno 21 on modal 1.6.1. fp16 vs fp32 noise floor on the 4x32 steps: top-1 agreement 100%, mean |dlogit| ~0.004, max 0.16, mean KL ~1e-5 (smallest top1-top2 gap 0.011, so near-ties exist). Logs and manifests are in the WSL checkout `~/projects/monicasic_verif/monicasic/runs/modal_run1_20261015_185504/` (runs/* is gitignored). fp16 digests are tied to this GPU and library stack.
Next: download run1 logits -> full 36-layer numpy reference on real weights, compared with the fp32 logits (real A1) -> integer reference -> rough SRAM budget -> simulator.
N11 (Melten, Gemma 4 26B-A4B, 8 of 128 experts) was pasted by the user and folded into the spec docs; its source is the pasted text only (company claims, self-reported). The arXiv paper is not out yet.
Open: toy MoE router config (train top-2 of 8, verify 8 of 128 in simulation?); base vs instruct; quality-set text; activation width (INT8 vs INT16) and weight grouping untested; `git init` not done.
Pending move: `tiny-tpu/` and `transformer-tpu-int4/` -> `reference/` (Windows refused; VS Code likely has them open). Close editors first, then rename each as ONE operation (PowerShell `Move-Item`) and check `git status` afterwards. Do not use bash `mv` on a locked folder: it falls back to per-file moves and once split tiny-tpu's `.git` away from its working tree (repaired and verified on 2026-10-14).
