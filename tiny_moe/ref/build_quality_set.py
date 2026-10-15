"""Build the A2 quality set (4 windows x 1024 Qwen3 tokens: 2 prose, 2 code) from pinned public sources.

Prose: WikiText-2 raw, test split (Salesforce/wikitext, pinned dataset revision; CC-BY-SA-3.0 / GFDL per the Hub tags).
Code:  two files of the CPython standard library (PSF license), identified by file name + sha256 of their bytes.
The recipe (sources, revisions, offsets, token-id digests) is committed; the text and token ids are NOT (out/quality_set/, gitignored).
Files and offsets below were chosen BEFORE any quantized result exists. Do not change them after measuring.

  uv run --group ref python -m tiny_moe.ref.build_quality_set           # builds out/quality_set/windows.npz + data/quality_set_recipe.json
  uv run --group ref python -m tiny_moe.ref.build_quality_set --verify  # rebuild and compare with the committed recipe

Scoring convention (to be written into ACCEPTANCE.md when frozen): each window is fed alone (no BOS, as Qwen3 adds none);
logits at positions 0..L-2 are scored against tokens 1..L-1, so 1023 scored positions per window, 4092 in total.
"""
import argparse
import hashlib
import json
import sysconfig
from pathlib import Path

import numpy as np

QWEN_REV = "1cfa9a7208912126459214e8b04321603b3df60c"
WIKI_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKI_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
L = 1024
PROSE_STARTS = (0.25, 0.75)                    # fractions of the tokenized test split (rounded down to a multiple of L)
CODE = (("argparse.py", 1024), ("dataclasses.py", 1024))   # (stdlib file, token offset; skips the license/import header)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def ids_sha(ids):
    return sha(np.asarray(ids, dtype="<i8").tobytes())


def build():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, snapshot_download
    from tokenizers import Tokenizer

    snap = snapshot_download("Qwen/Qwen3-4B", revision=QWEN_REV, allow_patterns=["*.json"])
    tok = Tokenizer.from_file(str(Path(snap) / "tokenizer.json"))
    enc = lambda s: np.array(tok.encode(s, add_special_tokens=False).ids, dtype=np.int64)

    pq_path = hf_hub_download("Salesforce/wikitext", WIKI_FILE, repo_type="dataset", revision=WIKI_REV)
    text = "".join(pq.read_table(pq_path).column("text").to_pylist())   # raw lines already end in newlines
    prose = enc(text)

    windows, meta = [], []
    for frac in PROSE_STARTS:
        start = int(len(prose) * frac) // L * L
        windows.append(prose[start:start + L])
        meta.append({"kind": "prose", "source": f"Salesforce/wikitext@{WIKI_REV}:{WIKI_FILE}", "token_offset": start})
    lib = Path(sysconfig.get_paths()["stdlib"])
    for name, off in CODE:
        raw = (lib / name).read_bytes()
        ids = enc(raw.decode("utf-8"))
        windows.append(ids[off:off + L])
        meta.append({"kind": "code", "source": f"cpython stdlib {name}", "file_sha256": sha(raw), "token_offset": off})
    for w, m in zip(windows, meta):
        assert len(w) == L, f"{m['source']}: only {len(w)} tokens available"
        m["n_tokens"] = L
        m["token_ids_sha256"] = ids_sha(w)
    recipe = {"status": "FROZEN-CANDIDATE (not frozen until the user approves)",
              "tokenizer": f"Qwen/Qwen3-4B@{QWEN_REV} tokenizer.json, add_special_tokens=False",
              "window_tokens": L, "scored_positions_per_window": L - 1, "total_scored_positions": len(windows) * (L - 1),
              "python_stdlib_note": "code windows depend on the file bytes; file_sha256 detects a different Python",
              "windows": meta}
    return np.stack(windows), recipe, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--recipe", default="data/quality_set_recipe.json")
    p.add_argument("--out", default="out/quality_set")
    a = p.parse_args()
    windows, recipe, tok = build()
    if a.verify:
        old = json.load(open(a.recipe))
        same = [x["token_ids_sha256"] == y["token_ids_sha256"] for x, y in zip(old["windows"], recipe["windows"])]
        print("verify:", same)
        raise SystemExit(0 if all(same) and len(same) == len(old["windows"]) else 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{a.out}/windows.npz", windows=windows)
    json.dump(recipe, open(a.recipe, "w"), indent=2)
    for w, m in zip(windows, recipe["windows"]):
        print(f"{m['kind']:5s} offset {m['token_offset']:>7}  {m['token_ids_sha256'][:16]}  {tok.decode(list(map(int, w[:40]))).replace(chr(10), ' ')[:110]!r}")
    print(f"{len(windows)} windows x {L} tokens, {recipe['total_scored_positions']} scored positions -> {a.out}/windows.npz, {a.recipe}")


if __name__ == "__main__":
    main()
