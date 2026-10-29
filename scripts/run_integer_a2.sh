#!/usr/bin/env bash
# Rebuild the checkout-local A2 inputs, score the float reference, then run the integer reference.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Python =="
PYVER="$(uv run python -c 'import platform; print(platform.python_version())')"
echo "$PYVER"
if [[ "$PYVER" != "3.12.10" ]]; then
    echo "ERROR: the frozen code windows require CPython 3.12.10; create that uv environment first." >&2
    exit 2
fi

echo "== Build frozen quality windows =="
uv run --group ref python -m tiny_moe.ref.build_quality_set

echo "== Verify frozen recipe =="
uv run --group ref python -m tiny_moe.ref.build_quality_set --verify

echo "== Score float reference =="
uv run --group ref python -m tiny_moe.ref.score_quality

echo "== Run integer A2 reference =="
uv run --group ref python -m tiny_moe.ref.integer_qwen3

echo "DONE: see out/quality_ref/ and out/integer_ref/"
