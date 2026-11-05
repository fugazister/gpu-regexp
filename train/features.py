"""Mechanical tokenizer + hashed sparse features. web/features.js must match (eval/parity.mjs).

There is no vocabulary. A token reaches the model only as hashes of its spelling plus its shape,
so a typo like "digts" shares its consonant skeleton and prefix with "digits".
"""
import re

_SPLIT = re.compile(r"\"[^\"]*\"|`[^`]*`|(?<![^\W_])'[^']*'(?![^\W_])|[^\W\d_]+|[0-9]+|\S")
_VOWELS = re.compile(r"[aeiou]")

SIZES = dict(word=768, skel=192, pre=192, suf=192, shape=28, length=9)
OFFSETS = {}
_at = 0
for _k, _v in SIZES.items():
    OFFSETS[_k] = _at
    _at += _v
N_ROWS = _at
SLOTS = len(SIZES)


def fnv(s):
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def tokenize(text):
    """-> list of dicts: text (without quotes), start, end, quoted, glue_l, glue_r."""
    toks = []
    for m in _SPLIT.finditer(text):
        t = m.group(0)
        quoted = len(t) >= 2 and t[0] in "\"`'" and t[-1] == t[0]
        toks.append(dict(text=t[1:-1] if quoted else t, start=m.start(), end=m.end(), quoted=quoted))
    for i, t in enumerate(toks):
        t["glue_l"] = i > 0 and toks[i - 1]["end"] == t["start"]
        t["glue_r"] = i + 1 < len(toks) and toks[i + 1]["start"] == t["end"]
    return toks


def _case_shape(t):
    if t["quoted"]:
        return 6
    s = t["text"]
    if s.isascii() and s.isdigit():
        return 4
    if not any(c.isalpha() for c in s):
        return 5
    if s == s.lower():
        return 0
    if s == s.upper():
        return 1
    if s[0] == s[0].upper() and s[1:] == s[1:].lower():
        return 2
    return 3


def featurize(t):
    """-> SLOTS row ids into the shared embedding table."""
    shape = OFFSETS["shape"] + _case_shape(t) * 4 + (2 if t["glue_l"] else 0) + (1 if t["glue_r"] else 0)
    length = OFFSETS["length"] + min(len(t["text"]), SIZES["length"] - 1)
    if t["quoted"]:  # quoted text is literal whatever it spells, so its spelling is hidden
        return [OFFSETS["word"], OFFSETS["skel"], OFFSETS["pre"], OFFSETS["suf"], shape, length]
    low = t["text"].lower()
    skel = _VOWELS.sub("", low) or low
    return [
        OFFSETS["word"] + fnv("w" + low) % SIZES["word"],
        OFFSETS["skel"] + fnv("s" + skel) % SIZES["skel"],
        OFFSETS["pre"] + fnv("p" + low[:3]) % SIZES["pre"],
        OFFSETS["suf"] + fnv("x" + low[-3:]) % SIZES["suf"],
        shape,
        length,
    ]
