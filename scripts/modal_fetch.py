"""Download a directory from a Modal volume, file by file (workaround: `modal volume get <vol> /dir ./dst` failed with
'[Errno 21] Is a directory' on modal 1.6.1 even for a fresh destination). Read-only: no GPU, no compute cost.

  uv run python scripts/modal_fetch.py                       # tiny-moe-out:/run1 -> out/run1
  uv run python scripts/modal_fetch.py --remote /smoke --dest out/smoke
"""
import argparse
from pathlib import Path

import modal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--volume", default="tiny-moe-out")
    p.add_argument("--remote", default="/run1")
    p.add_argument("--dest", default="out/run1")
    a = p.parse_args()

    vol = modal.Volume.from_name(a.volume)
    root = a.remote.rstrip("/")
    entries = [e for e in vol.listdir(root, recursive=True) if e.type.name == "FILE"]
    if not entries:
        raise SystemExit(f"no files under {a.volume}:{root} (check the volume name and the --tag you ran with)")
    total = 0
    for e in entries:
        rel = e.path[len(root.lstrip("/")):].lstrip("/") if e.path.startswith(root.lstrip("/")) else e.path.lstrip("/")
        dst = Path(a.dest) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(dst, "wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
                n += len(chunk)
        total += n
        print(f"{dst}  {n / 2**20:.1f} MiB")
    print(f"{len(entries)} files, {total / 2**20:.1f} MiB -> {a.dest}")


if __name__ == "__main__":
    main()
