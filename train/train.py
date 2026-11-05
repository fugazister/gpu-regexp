"""Train the role tagger on freshly generated samples. Works on CPU or CUDA.

Data generation dominates: these models barely need a GPU, but scans over a large batch do benefit.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from features import N_ROWS, SLOTS, featurize, tokenize
from grammar import MAX_TOKENS, ROLES, Gen, build_fixed
from model import ScanTagger

ROLE_ID = {r: i for i, r in enumerate(ROLES)}
_w = {}


def _init(fixed, seed):
    _w["g"] = Gen(random.Random(seed + os.getpid()), fixed=fixed)


def _batch(n):
    rows, labels, lengths = [], [], []
    for _ in range(n):
        s = _w["g"].sample()
        f = [featurize(t) for t in tokenize(s["text"])]
        lengths.append(len(f))
        rows.append(f + [[0] * SLOTS] * (MAX_TOKENS - len(f)))
        labels.append([ROLE_ID[r] for r in s["roles"]] + [-100] * (MAX_TOKENS - len(f)))
    return rows, labels, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="train/out")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    fixed = build_fixed()
    model = ScanTagger(N_ROWS, len(ROLES), a.d, a.layers, a.hidden).to(a.device)
    print(f"device={a.device} params={sum(p.numel() for p in model.parameters())} roles={len(ROLES)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)

    pool = mp.Pool(a.workers, _init, (fixed, a.seed))
    batches = pool.imap(_batch, (a.bs for _ in range(a.steps)), chunksize=4)

    def save():
        torch.save(dict(state=model.state_dict(), cfg=model.cfg, roles=ROLES), f"{a.out}/tagger.pt")

    t0, ema = time.time(), None
    for step, (rows, labels, lengths) in enumerate(batches, 1):
        L = max(lengths)
        rows = torch.tensor(rows, device=a.device)[:, :L]
        labels = torch.tensor(labels, device=a.device)[:, :L]
        lengths = torch.tensor(lengths, device=a.device)
        logits = model(rows, lengths)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 500 == 0 or step == a.steps:
            pred = logits.argmax(-1)
            ok = ((pred == labels) | (labels == -100)).all(1).float().mean().item()
            print(f"step {step}/{a.steps} loss {ema:.4f} seq_acc {ok:.3f} {step / (time.time() - t0):.1f} it/s", flush=True)
            save()
    pool.terminate()
    save()
    with open(f"{a.out}/done.json", "w") as f:
        json.dump(dict(steps=a.steps, seconds=time.time() - t0, loss=ema, **model.cfg), f)


if __name__ == "__main__":
    main()
