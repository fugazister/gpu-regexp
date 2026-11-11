"""Round 10: grammar prompts to be paraphrased. Real requests only, without typos (those stay as the grammar makes them).
   python train/para_prompts.py train/para/prompts.jsonl [n] [seed]"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from grammar import Gen, build_fixed


def main(out, n=2000, seed=10):
    g = Gen(random.Random(seed), fixed=build_fixed())
    seen, rows = set(), []
    while len(rows) < n:
        s = g.sample()
        if s["regex"] is None or s["typo"] or s["regex"] in seen:
            continue  # one prompt per regex, so the paraphrases cover more patterns
        seen.add(s["regex"])
        rows.append(dict(id=len(rows), text=s["text"], regex=s["regex"], flags=s["flags"]))
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(len(rows), "prompts")


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:]))
