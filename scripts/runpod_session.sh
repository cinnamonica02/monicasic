#!/usr/bin/env bash
# One-shot GPU session for a FRESH Linux image (RunPod/other, run as root). UNTESTED. Stops on the first error; logs to session.log.
# Prereq: the repo is already on the pod (git clone or unzip). Then:   bash scripts/runpod_session.sh
# Afterwards: download out.tar.gz and TERMINATE the pod (setup time and storage are billed).
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root, wherever the script is called from
exec > >(tee -a session.log) 2>&1

date -u
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "WARNING: nvidia-smi failed"

# 1. system basics (skipped if already present)
if ! command -v curl >/dev/null || ! command -v git >/dev/null; then
  apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git build-essential unzip
fi

# 2. uv (downloads its own Python; no apt python needed)
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
uv --version

# 3. keep large caches on the persistent disk when /workspace exists
if [ -d /workspace ]; then export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache; fi

# 4. environment: numpy + CUDA torch + transformers into .venv
uv sync --extra torch

# 5. fail fast, before the big downloads, if torch cannot see the GPU (set ALLOW_CPU=1 to override)
uv run python - <<'EOF'
import os, torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
    raise SystemExit("torch cannot see the GPU (driver/CUDA wheel mismatch?). Not continuing; see CLAUDE.md / README.")
print(torch.cuda.get_device_name(0))
EOF

echo "== A1 layer check (seconds) =="
uv run python -m tiny_moe.ref.check_qwen3_layer

echo "== baseline fp16 =="
uv run python scripts/baseline_dump.py --out out/baseline_fp16 --dtype fp16
echo "== baseline fp32 =="
uv run python scripts/baseline_dump.py --out out/baseline_fp32 --dtype fp32

uv pip freeze > out/pip_freeze.txt
cp session.log out/ 2>/dev/null || true
tar czf out.tar.gz out
ls -lh out.tar.gz
date -u
echo "DONE: download out.tar.gz, then terminate the pod."
