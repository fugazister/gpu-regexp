"""Tracks the probability a checkpoint gives the right role on specific words.

Pass/fail moves in steps of one prompt. This shows smaller shifts, and whether a fix changed the
model at all. Probes come from dev and retired prompts only; never add test prompts here.
    python train/probe.py results/E0-s1.pt results/E1.pt ... [--json out.json]
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from features import featurize, tokenize
from model import ScanTagger

PROBES = [
    # prompt, word, right role, the experiment that targets it
    ("only letters", "only", "ONLY", "E1"),
    ("either yes or no", "no", "LIT", "E2"),
    ("thre digits", "thre", "NUM", "E3"),
    ("time in hh:mm format", "format", "O", "E4"),
    ("version number like 1.2.3", "like", "O", "E4"),
    ("a vowel followed by two consonants", "consonants", "ATOM", "E5"),
]


def probe(path):
    c = torch.load(path, weights_only=False, map_location="cpu")
    model = ScanTagger(**{k: c["cfg"][k] for k in ("n_rows", "n_roles", "d", "layers", "hidden")})
    model.load_state_dict(c["state"])
    model.eval()
    out = []
    with torch.no_grad():
        for prompt, word, role, target in PROBES:
            toks = tokenize(prompt)
            i = [t["text"] for t in toks].index(word)
            rows = torch.tensor([[featurize(t) for t in toks]])
            p = model(rows, torch.tensor([len(toks)]))[0].softmax(-1)[i]
            out.append(dict(prompt=prompt, word=word, role=role, target=target, p=round(p[c["roles"].index(role)].item(), 3), top=c["roles"][p.argmax().item()]))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[args.index("--json") + 1] if "--json" in args else None
    paths = [a for a in args if a.endswith(".pt")]
    res = {os.path.basename(p)[:-3]: probe(p) for p in paths}
    print("word".ljust(12) + "target".ljust(8) + "".join(n.ljust(8) for n in res))
    for i, (_, word, role, target) in enumerate(PROBES):
        print(f"{word}/{role}".ljust(12)[:12] + target.ljust(8) + "".join(f"{r[i]['p']:.3f}".ljust(8) for r in res.values()))
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
