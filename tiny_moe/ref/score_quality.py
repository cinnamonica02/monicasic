"""Float reference scores on the FROZEN A2 quality set (docs/spec/ACCEPTANCE.md): the numbers the integer reference is compared to.

For every scored position it stores: float argmax (ties -> lowest index), top-1 and top-2 logit, log-sum-exp over the vocabulary
(float64), the true next token and its logit. Perplexity = exp(mean(lse - target_logit)). The vocabulary is streamed in chunks,
so the full 4092 x 151936 logits are never held in memory.

  uv run --group ref python -m tiny_moe.ref.score_quality          # after build_quality_set; ~10-20 min on a 2-core laptop

Refuses to run if the windows do not match the committed recipe digests.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from .build_quality_set import ids_sha
from .check_qwen3_full import PINNED, Checkpoint, cfg_from_json, final_hidden, head_name


def head_stats(ck, h, targets, head_chunk=8192):
    """h: (M, hidden) float32 final-normed states; targets: (M,) true next-token ids. Streams the LM head over the vocab."""
    M = h.shape[0]
    head = head_name(ck)
    vocab = ck.t[head][2][0]
    run_max = np.full(M, -np.inf)
    run_sum = np.zeros(M)
    top1 = np.full(M, -np.inf, np.float32)
    top2 = np.full(M, -np.inf, np.float32)
    arg = np.zeros(M, np.int64)
    tgt = np.zeros(M, np.float32)
    for r0 in range(0, vocab, head_chunk):
        lg = h @ ck.get(head, rows=slice(r0, r0 + head_chunk)).T               # (M, c) float32
        c = lg.shape[1]
        lg64 = lg.astype(np.float64)
        m_new = np.maximum(run_max, lg64.max(1))
        run_sum = run_sum * np.exp(run_max - m_new) + np.exp(lg64 - m_new[:, None]).sum(1)
        run_max = m_new
        i = lg.argmax(1)
        c1 = lg[np.arange(M), i]
        better = c1 > top1                                                      # strict: ties keep the earlier (lower) index
        arg = np.where(better, i + r0, arg)
        c2 = np.partition(lg, -2, axis=1)[:, -2] if c > 1 else np.full(M, -np.inf, np.float32)
        both = np.sort(np.stack([top1, top2, c1, c2]), axis=0)
        top1, top2 = both[-1], both[-2]
        inside = (targets >= r0) & (targets < r0 + c)
        tgt[inside] = lg[np.nonzero(inside)[0], targets[inside] - r0]
    return {"argmax": arg, "top1": top1, "top2": top2, "lse": run_max + np.log(run_sum), "target_logit": tgt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", help="local Qwen3-4B snapshot dir (default: the pinned commit from the HF cache)")
    p.add_argument("--windows", default="out/quality_set/windows.npz")
    p.add_argument("--recipe", default="data/quality_set_recipe.json")
    p.add_argument("--out", default="out/quality_ref")
    a = p.parse_args()

    windows = np.load(a.windows)["windows"]
    recipe = json.load(open(a.recipe))
    assert [ids_sha(w) for w in windows] == [m["token_ids_sha256"] for m in recipe["windows"]], \
        "windows.npz does not match the frozen recipe; rebuild with tiny_moe.ref.build_quality_set"
    d = a.dir
    if not d:
        from huggingface_hub import snapshot_download
        d = snapshot_download("Qwen/Qwen3-4B", revision=PINNED, allow_patterns=["*.safetensors", "*.json"])
    cfg, n_layers = cfg_from_json(Path(d) / "config.json")
    ck = Checkpoint(d)

    L = windows.shape[1]
    take = [np.arange(L - 1)] * len(windows)                                    # positions 0..L-2 predict tokens 1..L-1
    targets = np.concatenate([w[1:] for w in windows]).astype(np.int64)
    t0 = time.time()
    h = final_hidden(ck, cfg, n_layers, list(windows), take)
    t1 = time.time()
    s = head_stats(ck, h, targets)
    secs = time.time() - t0
    nll = s["lse"] - s["target_logit"].astype(np.float64)
    win = np.repeat(np.arange(len(windows)), L - 1)

    def ppl(mask):
        return float(np.exp(nll[mask].mean()))

    kinds = np.array([m["kind"] for m in recipe["windows"]])
    res = {"purpose": "float reference scores on the frozen A2 quality set", "windows_recipe_status": recipe["status"],
           "snapshot": str(d), "revision": PINNED, "positions": int(len(nll)),
           "ppl_pooled": ppl(np.ones_like(win, bool)), "ppl_per_window": [ppl(win == i) for i in range(len(windows))],
           "ppl_prose": ppl(np.isin(win, np.nonzero(kinds == "prose")[0])), "ppl_code": ppl(np.isin(win, np.nonzero(kinds == "code")[0])),
           "float_argmax_equals_true_next_token": float((s["argmax"] == targets).mean()),
           "min_top1_top2_gap": float((s["top1"] - s["top2"]).min()),
           "positions_with_gap_below_0.05": int(((s["top1"] - s["top2"]) < 0.05).sum()),
           "seconds_layers": round(t1 - t0, 1), "seconds_head": round(secs - (t1 - t0), 1),
           "numpy": np.__version__, "python": platform.python_version()}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{a.out}/float_ref.npz", targets=targets, window=win, **s)
    json.dump(res, open(f"{a.out}/float_ref.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
