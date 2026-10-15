"""Dump HF baseline tokens + logits for the frozen prompts (A1/A2 baseline). UNTESTED: torch/transformers not installed locally.

Run in a Colab cell (GPU optional):
  !pip -q install "transformers>=4.51" accelerate
  %run scripts/baseline_dump.py --out /content/baseline   # from the repo root; then copy /content/baseline to Drive or download it

Greedy, batch 1. Saves per prompt: ids, generated ids, per-step logits (float32) and sha256 digests, plus a manifest
following runs/MANIFEST_TEMPLATE.json. Record the dtype: a T4 has no native bf16, so use fp16 there; fp32 needs ~16 GB.
Raw text is used by default; pass --chat to wrap prompts in the model's chat template (instruct models).
"""
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--revision", default=None, help="pin a commit hash for reproducibility")
    p.add_argument("--prompts", default="data/frozen_prompts.json")
    p.add_argument("--out", default="baseline")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--chat", action="store_true")
    a = p.parse_args()

    dt = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=dt).to(dev).eval()
    prompts = json.load(open(a.prompts))["prompts"]
    os.makedirs(a.out, exist_ok=True)

    runs = []
    for i, text in enumerate(prompts):
        if a.chat:
            text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)  # Qwen3 flag; ignored by other templates
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        gen, step_logits, past, cur = [], [], None, ids
        with torch.no_grad():
            for _ in range(a.max_new):
                out = model(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                lg = out.logits[0, -1].float()
                step_logits.append(lg.cpu().numpy())  # float32: lossless for fp16/bf16 runs too, exact for fp32 runs
                nxt = int(lg.argmax())          # greedy; torch picks the lowest index on exact ties
                gen.append(nxt)
                if nxt == tok.eos_token_id:
                    break
                cur = torch.tensor([[nxt]], device=dev)
        logits = np.stack(step_logits)
        np.savez_compressed(f"{a.out}/prompt{i}.npz", prompt_ids=ids[0].cpu().numpy(),
                            gen_ids=np.array(gen), logits=logits)
        runs.append({"prompt_index": i, "n_prompt_tokens": ids.shape[1], "n_generated": len(gen),
                     "tokens_sha256": sha(np.array(gen, dtype=np.int64).tobytes()),
                     "logits_digest_sha256": sha(logits.tobytes())})
        print(i, tok.decode(gen)[:80].replace("\n", " "))

    manifest = {
        "purpose": "baseline", "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"name": a.model, "revision": a.revision, "dtype": a.dtype,
                  "resolved_commit": getattr(model.config, "_commit_hash", None)},  # what an unpinned download actually got
        "inputs": {"prompts_file_sha256": sha(open(a.prompts, "rb").read()), "chat_template_applied": a.chat,
                   "decoding": "greedy", "batch": 1, "max_new_tokens": a.max_new},
        "environment": {"where": "colab" if "COLAB_GPU" in os.environ or os.path.exists("/content") else "other",
                        "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "numpy": np.__version__, "python": platform.python_version()},
        "runs": runs,
    }
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
