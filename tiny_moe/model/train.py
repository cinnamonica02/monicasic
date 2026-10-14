"""Train the tiny MoE and report per-task accuracy + expert usage.

Locally / anywhere with torch, from the repo root:
  python -m tiny_moe.model.train --steps 3000
  python -m tiny_moe.model.train --steps 3000 --dense-q int8 --expert-q int8 --act-q
  python -m tiny_moe.model.train --steps 3000 --dense-q int8 --expert-q ternary --act-q
On Modal: see scripts/modal_run.py (unused unless Modal access returns)
"""
import argparse
import random
from collections import Counter

import torch
import torch.nn.functional as F

from . import data
from .moe import Config, TinyMoE


@torch.no_grad()
def greedy(model, prompt, max_new, dev):
    ids = data.encode(prompt)
    for _ in range(max_new):
        x = torch.tensor([ids[-model.cfg.max_len:]], device=dev)
        nxt = model(x)[0][0, -1].argmax().item()
        if data.ITOS[nxt] == "\n":
            break
        ids.append(nxt)
    return data.decode(ids[len(prompt):])


@torch.no_grad()
def evaluate(model, dev, n=200, seed=123):
    model.eval()
    rng, res = random.Random(seed), {}
    for t in data.TASKS:
        ok = 0
        for _ in range(n):
            prompt, ans = data.split_prompt(data.make_example(rng, t))
            ok += greedy(model, prompt, 10, dev) == ans
        res[t] = ok / n
    model.train()
    return res


@torch.no_grad()
def expert_usage(model, dev, n=200, seed=7):
    """Fraction of tokens each expert gets, per layer, per task."""
    model.eval()
    rng, out = random.Random(seed), {}
    for t in data.TASKS:
        cnt = [Counter() for _ in model.blocks]
        for _ in range(n):
            ids = data.encode(data.make_example(rng, t))
            model(torch.tensor([ids], device=dev))
            for li, b in enumerate(model.blocks):
                cnt[li].update(b.moe.last_idx.flatten().tolist())
        out[t] = [[c[e] / max(1, sum(c.values())) for e in range(model.cfg.n_experts)] for c in cnt]
    model.train()
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--dense-q", default="fp", choices=["fp", "int8", "ternary"])
    p.add_argument("--expert-q", default="fp", choices=["fp", "int8", "ternary"])
    p.add_argument("--act-q", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="")
    a = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    cfg = Config(vocab=data.VOCAB, d=a.d, d_ff=a.d_ff, n_layers=a.layers, n_experts=a.experts,
                 top_k=a.top_k, dense_q=a.dense_q, expert_q=a.expert_q, act_q=a.act_q)
    model = TinyMoE(cfg).to(dev)
    print(f"device={dev} params={model.n_params():,} cfg={cfg}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    rng = random.Random(a.seed)
    for step in range(1, a.steps + 1):
        x, y = data.batch(rng, a.bs, cfg.max_len, dev)
        logits, aux = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        (loss + cfg.aux_coef * aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 250 == 0 or step == a.steps:
            print(f"step {step:5d} loss {loss.item():.4f} aux {float(aux):.3f}", flush=True)

    print("exact-match accuracy per task:", evaluate(model, dev))
    for t, layers in expert_usage(model, dev).items():
        print(f"task {t} expert usage per layer:")
        for li, fr in enumerate(layers):
            print(f"  L{li}: " + " ".join(f"{f:.2f}" for f in fr))
    if a.save:
        torch.save({"cfg": cfg.__dict__, "state": model.state_dict()}, a.save)
        print("saved", a.save)


if __name__ == "__main__":
    main()
