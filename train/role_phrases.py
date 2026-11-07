"""Writes train/role_phrases.json: every phrase the grammar uses for each role, for vetting inferred labels."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import tokenize
from grammar import Gen, build_fixed

SKIP = {"O", "LIT", "NUM", "ATOM"}


def main(n=200_000):
    g = Gen(random.Random(7), fixed=build_fixed())
    phrases = {}
    for _ in range(n):
        segs, _, _, typo = g.sample_segs()
        if typo:
            continue
        for text, role, _ in segs:
            if role not in SKIP and text:
                phrases.setdefault(role, set()).add(" ".join(t["text"].lower() for t in tokenize(text)))
    with open(os.path.join(os.path.dirname(__file__), "role_phrases.json"), "w") as f:
        json.dump({k: sorted(v) for k, v in sorted(phrases.items())}, f, indent=0)
    print({k: len(v) for k, v in phrases.items()})


if __name__ == "__main__":
    main()
