"""Train the role tagger on freshly generated samples. Works on CPU or CUDA.

Data generation dominates: these models barely need a GPU, but scans over a large batch do benefit.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from features import N_ROWS, SLOTS, featurize, tokenize
from grammar import MAX_TOKENS, ROLES, Gen, build_fixed
from model import ScanTagger

ROLE_ID = {r: i for i, r in enumerate(ROLES)}
MAX_RSS_GB = 6  # abort rather than pressure a shared machine
REAL_LABELS = os.path.join(os.path.dirname(__file__), "real_labels.jsonl")
REAL_RATE = 0.15  # share of each batch taken from inferred labels on real descriptions
NUM_WORDS = "one two three four five six seven eight nine ten".split()


def load_real():
    """Descriptions by other people, with roles inferred by train/label_real.mjs. Tokenizers must agree on them."""
    if not os.path.exists(REAL_LABELS):
        return []
    real = []
    with open(REAL_LABELS) as f:
        for line in f:
            s = json.loads(line)
            toks = tokenize(s["text"])
            if len(toks) == len(s["roles"]) and len(toks) <= MAX_TOKENS:
                real.append((toks, s["roles"]))
    return real


def vary(r, toks, roles):
    """A few hundred sentences are easy to memorise, so swap the exact counts, which the roles don't depend on.
    Quoted literals already reach the model without their spelling, so they need no swapping."""
    out = []
    for t, role in zip(toks, roles):
        t = dict(t)
        if role == "NUM" and r.random() < 0.5:
            text = t["text"]
            if text.isdigit():
                t["text"] = str(r.randint(1, 20))
            elif text.lower() in NUM_WORDS:
                t["text"] = r.choice(NUM_WORDS)
            t["end"] = t["start"] + len(t["text"])
        out.append(t)
    return out, roles


def make_batch(g, n, real=(), rng=None):
    rows = np.zeros((n, MAX_TOKENS, SLOTS), dtype=np.int16)
    labels = np.full((n, MAX_TOKENS), -100, dtype=np.int8)
    lengths = np.zeros(n, dtype=np.int16)
    for b in range(n):
        if real and rng.random() < REAL_RATE:
            toks, roles = vary(rng, *rng.choice(real))
        else:
            s = g.sample()
            toks, roles = tokenize(s["text"]), s["roles"]
        lengths[b] = len(toks)
        rows[b, : len(toks)] = [featurize(t) for t in toks]
        labels[b, : len(toks)] = [ROLE_ID[r] for r in roles]
    return rows, labels, lengths


def worker(q, fixed, seed, n, real):
    """put() blocks while the bounded queue is full, so workers can't outrun training."""
    rng = random.Random(seed)
    g = Gen(rng, fixed=fixed)
    while True:
        q.put(make_batch(g, n, real, rng))


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
    real = load_real()
    print(f"real descriptions with inferred labels: {len(real)}", flush=True)
    model = ScanTagger(N_ROWS, len(ROLES), a.d, a.layers, a.hidden).to(a.device)
    print(f"device={a.device} params={sum(p.numel() for p in model.parameters())} roles={len(ROLES)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)

    q = mp.Queue(maxsize=2 * a.workers)
    procs = [mp.Process(target=worker, args=(q, fixed, a.seed * 1000 + i, a.bs, real), daemon=True) for i in range(a.workers)]
    for p in procs:
        p.start()

    def save():
        torch.save(dict(state=model.state_dict(), cfg=model.cfg, roles=ROLES), f"{a.out}/tagger.pt")

    t0, ema = time.time(), None
    for step in range(1, a.steps + 1):
        rows, labels, lengths = q.get()
        L = int(lengths.max())
        rows = torch.from_numpy(rows[:, :L].astype(np.int64)).to(a.device)
        labels = torch.from_numpy(labels[:, :L].astype(np.int64)).to(a.device)
        lengths = torch.from_numpy(lengths.astype(np.int64)).to(a.device)
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
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            print(f"step {step}/{a.steps} loss {ema:.4f} seq_acc {ok:.3f} {step / (time.time() - t0):.1f} it/s peak_rss {rss:.2f}GB", flush=True)
            save()
            if rss > MAX_RSS_GB:
                raise SystemExit(f"peak RSS {rss:.1f}GB is over the {MAX_RSS_GB}GB guard, stopping")
    for p in procs:
        p.terminate()
    save()
    with open(f"{a.out}/done.json", "w") as f:
        json.dump(dict(steps=a.steps, seconds=time.time() - t0, loss=ema, **model.cfg), f)


if __name__ == "__main__":
    main()
