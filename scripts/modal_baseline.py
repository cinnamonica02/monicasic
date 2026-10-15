"""Modal job: layer check against transformers + HF baseline dump (tokens/logits) for the frozen prompts.

UNTESTED until first run. Costs money: ask before running.

  uv sync --group cloud                       # installs the modal client (once)
  uv run modal run scripts/modal_baseline.py --smoke              # Qwen3-0.6B, fp16, short: a few cents, tests the plumbing
  uv run modal run scripts/modal_baseline.py                      # Qwen3-4B, fp16 + fp32 (needs a 24 GB GPU for fp32)
  uv run modal run scripts/modal_baseline.py --tag run2 --dtypes fp16

Results land in the Modal volume 'tiny-moe-out' under /<tag>/; fetch with
  uv run modal volume get tiny-moe-out /<tag> ./out/<tag>
The Hugging Face download is cached in the volume 'tiny-moe-hf-cache', so repeat runs do not re-download.
GPU: default L4 (24 GB). Override with the env var TINY_MOE_GPU (e.g. A10G, T4 cannot hold fp32 Qwen3-4B).
"""
import os
import subprocess
import sys

import modal

app = modal.App("tiny-moe-baseline")
out_vol = modal.Volume.from_name("tiny-moe-out", create_if_missing=True)
hf_vol = modal.Volume.from_name("tiny-moe-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=4.51", "accelerate", "numpy")
    .env({"HF_HOME": "/cache/hf"})
    # local files must be added last so the image layers above stay cached
    .add_local_python_source("tiny_moe")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_dir("data", remote_path="/root/data")
)


@app.function(
    image=image,
    gpu=os.environ.get("TINY_MOE_GPU", "L4"),
    timeout=3000,
    volumes={"/out": out_vol, "/cache": hf_vol},
)
def run(model: str, dtypes: list, tag: str, max_new: int, layer_check: bool) -> dict:
    def sh(cmd):
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd="/root")

    if layer_check:
        sh([sys.executable, "-m", "tiny_moe.ref.check_qwen3_layer"])
    for dt in dtypes:
        sh([sys.executable, "scripts/baseline_dump.py", "--model", model, "--dtype", dt,
            "--max-new", str(max_new), "--prompts", "data/frozen_prompts.json",
            "--out", f"/out/{tag}/baseline_{dt}"])
    os.makedirs(f"/out/{tag}", exist_ok=True)
    with open(f"/out/{tag}/pip_freeze.txt", "w") as f:  # exact package versions, for the run manifest
        subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, stdout=f)
    out_vol.commit()
    hf_vol.commit()
    return {"tag": tag, "model": model, "dtypes": dtypes}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-4B", dtypes: str = "fp16,fp32", tag: str = "run1",
         max_new: int = 32, smoke: bool = False):
    if smoke:  # cheap plumbing test
        model, dtypes, tag, max_new = "Qwen/Qwen3-0.6B", "fp16", "smoke", 8
    res = run.remote(model, dtypes.split(","), tag, max_new, True)
    print(res)
    print(f"fetch with:  uv run modal volume get tiny-moe-out /{tag} ./out/{tag}")
