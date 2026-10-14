#!/usr/bin/env bash
# One-shot GPU session (RunPod / Colab terminal). UNTESTED. Stops on the first error; logs to session.log.
# Layout expected in the working dir: tiny_moe/  scripts/  data/   (upload them, e.g. as a zip via Jupyter)
# Usage:  bash scripts/runpod_session.sh        then download out.tar.gz and TERMINATE the pod.
set -euo pipefail
exec > >(tee -a session.log) 2>&1

date -u; nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# torch is assumed to come with the template; install into the system python so it is not replaced
if command -v uv >/dev/null 2>&1; then
  uv pip install --system -q "transformers>=4.51" accelerate numpy
else
  pip -q install "transformers>=4.51" accelerate numpy
fi

echo "== A1 layer check (seconds) =="
python -m tiny_moe.ref.check_qwen3_layer

echo "== baseline fp16 =="
python scripts/baseline_dump.py --out out/baseline_fp16 --dtype fp16
echo "== baseline fp32 =="
python scripts/baseline_dump.py --out out/baseline_fp32 --dtype fp32

pip freeze > out/pip_freeze.txt
cp session.log out/ 2>/dev/null || true
tar czf out.tar.gz out
ls -lh out.tar.gz
date -u
echo "DONE: download out.tar.gz, then terminate the pod."
