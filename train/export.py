"""Export tagger.pt to web/model.bin + web/model.json as int6 weights (stored one per byte; Brotli
packs them), and write eval/fixtures.json with torch logits computed from those same weights."""
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from features import featurize, tokenize
from grammar import MAX_TOKENS, ROLES, Gen, build_fixed
from model import QMAX, ScanTagger


def main(ckpt="train/out/tagger.pt"):
    c = torch.load(ckpt, weights_only=False, map_location="cpu")
    assert c["roles"] == ROLES, "checkpoint was trained with a different role list"
    blobs, tensors, offset = [], {}, 0
    for name, t in c["state"].items():
        a = t.numpy().astype(np.float32)
        scale = max(float(np.abs(a).max()) / QMAX, 1e-8)
        q = np.round(a / scale).astype(np.int8)
        tensors[name] = dict(offset=offset, shape=list(a.shape), scale=scale)
        blobs.append(q.tobytes())
        offset += a.size
    data = b"".join(blobs)
    with open("web/model.bin", "wb") as f:
        f.write(data)
    with open("web/model.json", "w") as f:
        json.dump(dict(cfg=c["cfg"], roles=ROLES, n_weights=offset, tensors=tensors), f, separators=(",", ":"))
    print(f"model.bin {len(data) / 1024:.1f} kB, {offset} params")

    # fq() is idempotent on values already on the int6 grid, so loading the raw weights reproduces the shipped ones.
    model = ScanTagger(**{k: c["cfg"][k] for k in ("n_rows", "n_roles", "d", "layers", "hidden")})
    model.load_state_dict(c["state"])
    model.eval()
    g = Gen(random.Random(99), fixed=build_fixed())
    prompts = [g.sample()["text"] for _ in range(80)] + [
        'starts with "foo bar", then 3-digit', "one or more (digits).", "contains @ and .png!",
        "don't match 'quoted' `tick` it's", "TWENTY 007 7 seven", "digts folowed by a dahs",
    ]
    cases = []
    with torch.no_grad():
        for p in prompts:
            toks = tokenize(p)[:MAX_TOKENS]
            rows = torch.tensor([[featurize(t) for t in toks]])
            logits = model(rows, torch.tensor([len(toks)]))[0]
            cases.append(dict(q=p, rows=rows[0].tolist(), logits=[[round(v, 4) for v in r] for r in logits.tolist()]))
    with open("eval/fixtures.json", "w") as f:
        json.dump(cases, f)


if __name__ == "__main__":
    main(*sys.argv[1:2])
