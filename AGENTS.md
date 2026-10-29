# Repository Guidelines

## Project Structure & Module Organization

Read `CLAUDE.md` first for scope, architecture decisions, and verification rules. The main Python code is under `tiny_moe/model/` (toy MoE and training utilities) and `tiny_moe/ref/` (float and quantized reference implementations). Tests live in `tests/`. `scripts/` contains local, GPU-pod, and Modal helpers. Specifications and design decisions are in `docs/spec/` and `docs/decisions/`; frozen inputs are in `data/`. `hw/` and `sim/` are reserved for RTL and simulator work. `reference/` contains third-party projects and their own licenses. Generated results belong in `out/` or `runs/` and should remain uncommitted.

## Build, Test, and Development Commands

Run commands from the repository root with `uv`:

```text
uv sync                                      # Install the normal development environment
uv run pytest                                # Run the numpy-only test suite
uv sync --extra torch                        # Install optional HF/torch tooling
uv run python -m tiny_moe.ref.check_qwen3_layer  # Compare the reference layer with HF
cmake -S . -B build && cmake --build build   # Build C++ parts when simulator code exists
```

Use `uv sync --group ref` for checkpoint and quality-set reference tooling. GPU or Modal scripts can incur cost; ask before running paid compute.

## Coding Style & Naming Conventions

Use Python 3.10+ with four-space indentation, clear `snake_case` functions and variables, `PascalCase` classes, and focused modules matching the existing layout. Prefer NumPy operations and the repository's existing helpers over new dependencies. Keep numerical behavior explicit and deterministic; document any intentional shortcut with `shortcut: ...`. Preserve third-party code and licenses under `reference/`.

## Testing Guidelines

Add or update pytest tests in `tests/` for non-trivial numerical changes. Name files `test_*.py` and test functions `test_*`. Run `uv run pytest` before submitting changes; tests should be reproducible and avoid network, GPU, or paid services unless explicitly marked as integration checks.

## Commit & Pull Request Guidelines

Use a short, imperative commit subject describing one change (for example, `Add quantization study`). Pull requests should explain the motivation, summarize behavior and verification, identify any changed numerical assumptions, and link relevant specification or decision documents. Include reproducible commands and call out unverified hardware, cloud, or quality results.

## Security & Configuration Tips

Never commit credentials, downloaded model weights, generated run outputs, or private source material. Treat cloud execution as user-funded and verify manifests, frozen-prompt hashes, and acceptance thresholds before reporting results.
