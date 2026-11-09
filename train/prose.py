"""Sentences from public-domain books (Project Gutenberg), for refusal: none of them asks for a pattern.

    python train/prose.py out.txt --per-book N book.txt ...

Keeps 4-30 word sentences that start with a letter, spread evenly through each book, without the
Project Gutenberg header and licence. The same rules as eval/refuse-real.mjs, so train and test look alike.
"""
import argparse
import re


def body(text):
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF", text)
    return text[start.end() if start else 0:end.start() if end else len(text)]


def sentences(text):
    flat = re.sub(r"\s+", " ", body(text))
    seen, out = set(), []
    for s in re.split(r"(?<=[.;:!?])\s+", flat):
        s = s.strip().replace("_", "")
        words = s.split(" ")
        if 4 <= len(words) <= 30 and re.match(r"[A-Za-z]", s) and s not in seen and "gutenberg" not in s.lower():
            seen.add(s)
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("books", nargs="+")
    ap.add_argument("--per-book", type=int, default=1000)
    a = ap.parse_args()
    keep = []
    for path in a.books:
        with open(path, encoding="utf-8-sig") as f:
            all_ = sentences(f.read())
        step = max(1, len(all_) / a.per_book)
        keep += [all_[int(i * step)] for i in range(min(a.per_book, len(all_)))]
    with open(a.out, "w") as f:
        f.write("\n".join(keep) + "\n")
    print(f"{len(keep)} sentences from {len(a.books)} books")


if __name__ == "__main__":
    main()
