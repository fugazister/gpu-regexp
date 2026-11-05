"""Scores a checkpoint on fresh generated data: token and sequence accuracy, plus the roles it confuses.

High numbers here only show it learned the generator. They say nothing about real phrasing; for that,
see eval/score.mjs. A big gap between the two is distribution shift.
"""
import json
import os
import random
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(__file__))
from features import featurize, tokenize
from grammar import ROLES, Gen, build_fixed
from model import ScanTagger


def main(ckpt="train/out/tagger.pt", out=None, n=5000):
    c = torch.load(ckpt, weights_only=False, map_location="cpu")
    model = ScanTagger(**{k: c["cfg"][k] for k in ("n_rows", "n_roles", "d", "layers", "hidden")})
    model.load_state_dict(c["state"])
    model.eval()
    g = Gen(random.Random(777), fixed=build_fixed())
    tok_ok = tok_all = seq_ok = 0
    confusion = Counter()
    with torch.no_grad():
        for _ in range(n):
            s = g.sample()
            toks = tokenize(s["text"])
            rows = torch.tensor([[featurize(t) for t in toks]])
            pred = model(rows, torch.tensor([len(toks)]))[0].argmax(-1).tolist()
            right = True
            for p, gold in zip(pred, s["roles"]):
                tok_all += 1
                if ROLES[p] == gold:
                    tok_ok += 1
                else:
                    right = False
                    confusion[f"{gold}->{ROLES[p]}"] += 1
            seq_ok += right
    res = dict(n=n, token_acc=tok_ok / tok_all, seq_acc=seq_ok / n, top_confusions=confusion.most_common(10))
    print(f"generated data: token {res['token_acc']:.4f}, sequence {res['seq_acc']:.4f}; top confusions {res['top_confusions'][:5]}")
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
