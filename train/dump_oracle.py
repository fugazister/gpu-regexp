"""Writes web/lexicon.json and eval/oracle.json (gold labels + features, checked by eval/roundtrip.mjs)."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import featurize, tokenize
from grammar import ROLES, Gen, build_fixed
from lexicon import LEX


def main(n=20000):
    with open("web/lexicon.json", "w") as f:
        json.dump(LEX, f, separators=(",", ":"))
    g = Gen(random.Random(4242), fixed=build_fixed())
    cases = []
    while len(cases) < n:
        s = g.sample()
        if s["typo"]:
            continue  # fuzzy lexicon matching is exercised by the model scores, not by the oracle
        s["features"] = [featurize(t) for t in tokenize(s["text"])]
        cases.append(s)
    with open("eval/oracle.json", "w") as f:
        json.dump(dict(roles=ROLES, cases=cases), f)
    print(len(cases), "oracle cases")


if __name__ == "__main__":
    main()
