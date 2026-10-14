"""Run model.train on Modal.

  modal run scripts/modal_run.py --args "--steps 3000"          (from the repo root)
  modal run scripts/modal_run.py --args "--steps 3000 --dense-q int8 --expert-q ternary --act-q --save /ckpt/tern.pt"
  modal run scripts/modal_run.py --gpu T4 --args "--steps 3000"  # optional GPU; CPU is fine at this size

Checkpoints written under /ckpt persist in the Modal volume 'tiny-moe-ckpt'
(download with: modal volume get tiny-moe-ckpt tern.pt .)
"""
import shlex

import modal

app = modal.App("tiny-moe-chip")
vol = modal.Volume.from_name("tiny-moe-ckpt", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_python_source("tiny_moe")
)


def _train(argv):
    from tiny_moe.model.train import main
    main(argv)
    vol.commit()


@app.function(image=image, cpu=4, timeout=3600, volumes={"/ckpt": vol})
def train_cpu(argv: list):
    _train(argv)


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/ckpt": vol})
def train_gpu(argv: list):
    _train(argv)


@app.local_entrypoint()
def main(args: str = "--steps 3000", gpu: str = ""):
    argv = shlex.split(args)
    (train_gpu if gpu else train_cpu).remote(argv)
