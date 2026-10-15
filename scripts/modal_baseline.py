"""Modal batch job: layer check against transformers + HF baseline dump (tokens/logits) for the frozen prompts.

UNTESTED until first run. Costs money: ask before running.

  uv sync --group cloud                                           # installs the modal client (once)
  uv run modal run scripts/modal_baseline.py --smoke              # Qwen3-0.6B, fp16, short: cents, tests the plumbing
  uv run modal run scripts/modal_baseline.py                      # Qwen3-4B, fp16 + fp32
  uv run modal run scripts/modal_baseline.py --tag run2 --dtypes fp16

Design (borrowed from the earlier vLLM validation runner): batch `modal run`, never `modal shell`; steps run in order and the
job stops at the first failure so GPU time is not burned on a doomed run; failures do NOT raise inside the container, so the
logs always come back, and the local entrypoint saves them to runs/ and then raises.

Big outputs (npz logits) stay in the Modal volume 'tiny-moe-out' under /<tag>/ :
  uv run modal volume get tiny-moe-out /<tag> ./out/<tag>
The HF download is cached in the volume 'tiny-moe-hf-cache'.
GPU: env TINY_MOE_GPU, comma-separated fallback list, default "L4,A10G" (both 24 GB; fp32 Qwen3-4B needs ~20 GB, so T4 is out).
"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("tiny-moe-baseline")
out_vol = modal.Volume.from_name("tiny-moe-out", create_if_missing=True)
hf_vol = modal.Volume.from_name("tiny-moe-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=4.51", "accelerate", "numpy")  # torch wheels bundle CUDA; no devel image needed
    .env({"HF_HOME": "/cache/hf"})
    # local files must be added last so the image layers above stay cached
    .add_local_python_source("tiny_moe")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_dir("data", remote_path="/root/data")
)

_gpus = [g.strip() for g in os.environ.get("TINY_MOE_GPU", "L4,A10G").split(",") if g.strip()]
GPU = _gpus if len(_gpus) > 1 else _gpus[0]
MIN_GIB = {"fp16": 10, "bf16": 10, "fp32": 20}  # rough VRAM need for a 4B model; skip instead of OOM mid-run


def _step(name, cmd, logs):
    """Run one step, stream its output, keep it in logs[name]. Returns True on success (never raises)."""
    print(f"\n=== {name} ===\n$ {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, cwd="/root", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    p.wait()
    logs[name] = "".join(lines) + f"\n[exit code {p.returncode}]\n"
    return p.returncode == 0


@app.function(image=image, gpu=GPU, timeout=3000, volumes={"/out": out_vol, "/cache": hf_vol})
def run(model: str, dtypes: list, tag: str, max_new: int) -> dict:
    import torch

    logs, ok = {}, True
    props = torch.cuda.get_device_properties(0)
    gib = props.total_memory / 2**30
    logs["gpu"] = f"{props.name}  {gib:.1f} GiB  compute {props.major}.{props.minor}\n"
    print(logs["gpu"], flush=True)
    try:
        ok = _step("layer_check", [sys.executable, "-m", "tiny_moe.ref.check_qwen3_layer"], logs)
        for dt in dtypes if ok else []:
            if gib < MIN_GIB.get(dt, 20):
                logs[f"baseline_{dt}"] = f"SKIPPED: {gib:.1f} GiB < {MIN_GIB.get(dt, 20)} GiB needed for {dt}\n"
                ok = False
                break
            ok = _step(f"baseline_{dt}",
                       [sys.executable, "scripts/baseline_dump.py", "--model", model, "--dtype", dt,
                        "--max-new", str(max_new), "--prompts", "data/frozen_prompts.json",
                        "--out", f"/out/{tag}/baseline_{dt}"], logs)
            if not ok:
                break
        os.makedirs(f"/out/{tag}", exist_ok=True)
        with open(f"/out/{tag}/pip_freeze.txt", "w") as f:  # exact versions, for the run manifest
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f)
    finally:
        out_vol.commit()
        hf_vol.commit()
    manifests = {}
    for dt in dtypes:
        mp = Path(f"/out/{tag}/baseline_{dt}/manifest.json")
        if mp.exists():
            manifests[dt] = mp.read_text()
    return {"ok": ok, "logs": logs, "manifests": manifests}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-4B", dtypes: str = "fp16,fp32", tag: str = "run1",
         max_new: int = 32, smoke: bool = False):
    if smoke:  # cheap plumbing test
        model, dtypes, tag, max_new = "Qwen/Qwen3-0.6B", "fp16", "smoke", 8
    res = run.remote(model, dtypes.split(","), tag, max_new)
    out_dir = Path("runs") / f"modal_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in res["logs"].items():
        (out_dir / f"{name}.log").write_text(text, encoding="utf-8")
    for dt, text in res["manifests"].items():
        (out_dir / f"manifest_{dt}.json").write_text(text, encoding="utf-8")
    print(f"\nLogs and manifests saved to {out_dir}/")
    if not res["ok"]:
        raise RuntimeError(f"baseline job failed or was skipped - see {out_dir}/*.log")
    print(f"Big outputs are in the volume. Fetch with:  uv run modal volume get tiny-moe-out /{tag} ./out/{tag}")
