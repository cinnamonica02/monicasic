"""Synthetic multi-task character data: add, reverse, copy.

Examples, one per line:  A12+34=46 | Rabc>cba | Cabc>abc
The leading tag (A/R/C) tells the model which task it is in, which gives the
router something to specialise on. All examples are packed into fixed-length
streams; the model is trained on next-char prediction over everything.
"""
import random

import torch

LETTERS = "abcdefghij"
CHARS = "\nARC+=>" + "0123456789" + LETTERS
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS)
TASKS = ("A", "R", "C")


def make_example(rng, task=None):
    task = task or rng.choice(TASKS)
    if task == "A":
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        return f"A{a}+{b}={a + b}\n"
    w = "".join(rng.choice(LETTERS) for _ in range(rng.randint(2, 6)))
    return f"R{w}>{w[::-1]}\n" if task == "R" else f"C{w}>{w}\n"


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[int(i)] for i in ids)


def batch(rng, B, T, device="cpu"):
    xs = []
    for _ in range(B):
        s = ""
        while len(s) < T + 1:
            s += make_example(rng)
        xs.append(encode(s[: T + 1]))
    t = torch.tensor(xs, device=device)
    return t[:, :-1], t[:, 1:]


def split_prompt(s):
    """'A12+34=46\\n' -> ('A12+34=', '46')"""
    cut = s.index("=") + 1 if s[0] == "A" else s.index(">") + 1
    return s[:cut], s[cut:-1]
