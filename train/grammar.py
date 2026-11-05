"""Synthetic generator: English spans labelled with roles, plus an independently built gold regex.

Generators return (segs, regex) where segs is a list of (text, role, glue). web/compile.js must
turn the labels back into exactly the gold regex; eval/roundtrip.mjs enforces this at 100%.
Otherwise the model would be scored on compiler bugs.
"""
import random
import re

from features import tokenize
from lexicon import CLASSES, COMMON_WORDS, LEX, PRESETS, PUNCT, SYLL, SYMBOL_LITS, compact

ROLES = ("O LIT NUM ATOM Q_PLUS Q_STAR Q_OPT Q_MIN Q_MAX Q_RANGE Q_TIMES OR THEN NEG WORD WORDS LINE "
         "START END ONLY CONTAINS NOT_START NOT_END NOT_CONTAIN CAPTURE ICASE PASSWORD EXT").split()
MAX_TOKENS = 48

NUM_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
             "fifteen sixteen seventeen eighteen nineteen twenty").split()
JOINERS = ["followed by", "then", "and then", "and", "plus", "after that", "next", "with", "and after that", "followed by a"]
PREFIXES = ["", "", "", "", "", "match", "matches", "find", "a regex for", "regex for", "regex that matches", "a regex that matches",
            "a regular expression for", "regexp for", "give me a regex for", "i want to match", "i need to match", "please match",
            "pattern for", "a pattern matching", "a pattern for", "detect", "extract", "search for", "look for",
            "write a regex to match", "how do i match", "validate", "get all", "any", "something that matches", "all",
            "recognize", "parse", "check for", "i want a regex that finds", "need a regex for", "make a regex for"]
# Words that also carry a role elsewhere ("only", "just", "some", "has", "have") must not appear here,
# or the same word gets contradictory labels.
NOISE = ("the a an any my our this that in of for to it please like kind sort thing things stuff text value "
         "values string strings input really basically simply actually valid so also which is are there can be "
         "from at on by must should my your").split()
# People often describe a pattern by example or by format. These tails are filler to the compiler.
EXAMPLES = {
    "<DATE_ISO>": ["2024-01-31", "1999-12-25"], "<DATE_US>": ["12/25/2024", "1/2/2020"], "<TIME>": ["12:30", "09:15", "23:59:59"],
    "<SEMVER>": ["1.2.3", "2.0.1", "10.4.12"], "<IPV4>": ["192.168.0.1", "8.8.8.8"], "<PHONE>": ["555-123-4567", "(555) 123-4567"],
    "<EMAIL>": ["me@example.com", "a.b@c.org"], "<HEXCOLOR>": ["#ff00aa", "#fff"], "<ZIP>": ["90210", "12345-6789"],
    "<URL>": ["https://example.com", "http://a.io/x"], "<PERCENT>": ["50%", "12.5%"], "<MONEY>": ["$9.99", "$5"],
    "<FLOAT>": ["3.14", "0.5"], "<INT>": ["42", "-7"], "<HASHTAG>": ["#cool"], "<MENTION>": ["@bob"], "<NUMBER>": ["42", "1000"],
    "<SSN>": ["123-45-6789"], "<MAC>": ["00:1a:2b:3c:4d:5e"], "<DOMAIN>": ["example.com"],
}
FORMATS = {"<DATE_ISO>": ["yyyy-mm-dd", "iso", "iso 8601"], "<DATE_US>": ["mm/dd/yyyy", "dd/mm/yyyy", "us"], "<TIME>": ["hh:mm", "hh:mm:ss", "24 hour"]}

# Short everyday words people use as literals, even though the grammar also uses some of them.
PLAIN_LITS = "yes no on off in out up down start stop end true false get set new old first last left right top bottom open close none".split()
# Roles that can open a new sentence.
SENTENCE_STARTS = ("THEN", "START", "END", "CONTAINS")
SUBJECT = ["string", "strings", "line", "lines", "text", "input", "value", "values", "anything", "something",
           "any line", "any string", "a line", "a string", "the string", "the line", "a value"]


def esc(s):
    return re.sub(r"[.*+?^${}()|\[\]\\/]", lambda m: "\\" + m.group(0), s)


def S(text, role="O", glue=False):
    return (text, role, glue)


def words(text, role="O"):
    return [S(text, role)] if text else []


class Gen:
    def __init__(self, rng, fixed=None):
        self.r = rng
        self.fixed = fixed  # None => every literal is quoted, used to discover the fixed words

    def c(self, xs):
        return self.r.choice(xs)

    # --- literals and numbers ---
    def lit_word(self):
        r = self.r.random()
        if r < 0.42:
            w = self.c(COMMON_WORDS)
        elif r < 0.80:
            w = "".join(self.c(SYLL) for _ in range(self.r.randint(1, 3)))
        elif r < 0.92:
            w = self.c(SYMBOL_LITS)
        elif r < 0.96:
            w = str(self.r.randint(0, 999))
        else:
            w = f"{self.c(COMMON_WORDS)} {self.c(COMMON_WORDS)}"
        s = self.r.random()
        if s < 0.08:
            w = w.upper()
        elif s < 0.15:
            w = w.capitalize()
        return w

    def lit_seg(self, w, loose=False):
        """Bare literals normally avoid every grammar word. loose only avoids atoms and numbers, for
        positions where context shows it's a literal ("yes or no"), so those words get seen as LIT."""
        if loose:
            clash = lambda t: compact(t) in LEX or t.lower() in NUM_WORDS
        else:
            clash = lambda t: t.lower() in self.fixed
        bare_ok = self.fixed is not None and " " not in w and not w.isdigit() and not any(clash(t["text"]) for t in tokenize(w))
        if bare_ok and self.r.random() < 0.7:
            return S(w, "LIT")
        q = self.c(['"', '"', '"', "'", "`"])
        return S(f"{q}{w}{q}", "LIT")

    def number(self, lo=1, hi=20):
        hi = max(lo, hi)
        n = self.r.randint(lo, hi) if self.r.random() < 0.85 else self.r.randint(lo, max(hi, 64))
        if n <= 20 and self.r.random() < 0.35:
            return NUM_WORDS[n], n
        return str(n), n

    def number_seg(self, lo=1, hi=20):
        t, n = self.number(lo, hi)
        return S(t, "NUM"), n

    # --- atoms ---
    def atom(self):
        if self.r.random() < 0.72:
            src, sg, pl, _ = self.c(CLASSES)
        else:
            src, sg, pl = self.c(PUNCT)
        return src, sg, pl

    def art(self, phrase):
        return "an" if phrase[0] in "aeiou" else "a"

    def bare_plural(self, src, form):
        """English `form` with no explicit quantity: plural forms mean one-or-more."""
        return [S(form, "ATOM")], src + ("+" if LEX[compact(form)].get("plural") else "")

    def quantified(self):
        src, sg_l, pl_l = self.atom()
        sg, pl = self.c(sg_l), self.c(pl_l)
        k = self.r.random()
        A = S(sg, "ATOM")
        P = S(pl, "ATOM")
        if k < 0.20:  # exactly one
            v = self.r.randint(0, 3)
            if v == 0:
                return [S(self.art(sg)), A], src
            if v == 1:
                return [A], src
            if v == 2:
                return [S("one", "NUM"), A], src
            return [S("exactly"), S("one", "NUM"), A], src
        if k < 0.42:  # {n}
            N, n = self.number_seg(2)
            q = "{%d}" % n
            v = self.r.randint(0, 6)
            if v == 0:
                return [N, P], src + q
            if v == 1:
                return [S("exactly"), N, P], src + q
            if v == 2:
                return [A, N, S("times", "Q_TIMES")], src + q
            if v == 3:
                return [A, S("repeated", "Q_TIMES"), N, S("times", "Q_TIMES")], src + q
            if v == 4:
                return [N, A], src + q
            if v == 5:
                return [N, S("-", "O", True), S(sg, "ATOM", True)], src + q
            return [S("a"), N, A, S("sequence")], src + q
        if k < 0.60:  # +
            v = self.r.randint(0, 5)
            if v <= 1:
                return self.bare_plural(src, pl)
            if v == 2:
                return [S(self.c(["one or more", "1 or more", "some", "multiple", "several", "many", "a sequence of", "a run of", "a string of"]), "Q_PLUS"), P], src + "+"
            if v == 3:
                return [S("at least one", "Q_PLUS"), A], src + "+"
            if v == 4:
                return [A, S("one or more", "Q_PLUS"), S("times", "Q_TIMES")], src + "+"
            s, r = self.bare_plural(src, pl)
            return s + [S("of any length")], r
        if k < 0.70:  # *
            if self.r.random() < 0.2:
                return [P, S("if any", "Q_STAR")], src + "*"
            return [S(self.c(["zero or more", "0 or more", "any number of", "optional", "maybe some"]), "Q_STAR"), P], src + "*"
        if k < 0.80:  # ?
            v = self.r.randint(0, 3)
            if v == 0:
                return [S("an"), S("optional", "Q_OPT"), A], src + "?"
            if v == 1:
                return [S(self.c(["optional", "maybe a", "possibly a", "zero or one"]), "Q_OPT"), A], src + "?"
            if v == 2:
                return [A, S("optionally", "Q_OPT")], src + "?"
            return [S(self.art(sg)), A, S("which is"), S("optional", "Q_OPT")], src + "?"
        if k < 0.87:  # {n,}
            N, n = self.number_seg(2)
            q = "{%d,}" % n
            v = self.r.randint(0, 2)
            if v == 0:
                return [S(self.c(["at least", "minimum", "no fewer than", "minimum of"]), "Q_MIN"), N, P], src + q
            if v == 1:
                return [N, S("or more", "Q_MIN"), P], src + q
            return [N, P, S("or more", "Q_MIN")], src + q
        if k < 0.93:  # {0,n}
            N, n = self.number_seg(2)
            q = "{0,%d}" % n
            if self.r.random() < 0.8:
                return [S(self.c(["at most", "up to", "no more than", "maximum", "maximum of"]), "Q_MAX"), N, P], src + q
            return [N, S("or fewer", "Q_MAX"), P], src + q
        # {a,b}
        A1, a = self.number_seg(1, 9)
        B1, b = self.number_seg(a + 1, 20)
        q = "{%d,%d}" % (a, b)
        v = self.r.randint(0, 4)
        if v == 0:
            return [S("between", "Q_RANGE"), A1, S("and", "Q_RANGE"), B1, P], src + q
        if v == 1:
            return [A1, S(self.c(["to", "or", "through"]), "Q_RANGE"), B1, P], src + q
        if v == 2:
            return [S("from", "Q_RANGE"), A1, S("to", "Q_RANGE"), B1, P], src + q
        if v == 3:
            return [S("at least", "Q_RANGE"), A1, S("and at most", "Q_RANGE"), B1, P], src + q
        return [A1, S("-", "Q_RANGE", True), S(B1[0], "NUM", True), P], src + q

    def lit_element(self):
        w = self.c(PLAIN_LITS) if self.r.random() < 0.15 else self.lit_word()
        L = self.lit_seg(w, loose=True)
        e = esc(w)
        k = self.r.random()
        if k < 0.48:
            return [L], e
        if k < 0.72:
            return [S(self.c(["the text", "the string", "the literal", "the characters", "the sequence", "literal"])), L], e
        if k < 0.88:
            return [S(self.c(["the word", "the whole word", "the exact word", "word"]), "WORD"), L], rf"\b{e}\b"
        N, n = self.number_seg(2)
        body = f"(?:{e})" if len(w) > 1 else e
        v = self.r.randint(0, 2)
        if v == 0:
            return [L, N, S("times", "Q_TIMES")], body + "{%d}" % n
        if v == 1:
            return [L, S("repeated", "Q_TIMES"), N, S("times", "Q_TIMES")], body + "{%d}" % n
        return [N, S("copies of"), L], body + "{%d}" % n

    def set_items(self, lo, hi):
        pool = [c for c in CLASSES if c[3]]
        items, n = [], self.r.randint(lo, hi)
        while len(items) < n:
            if self.r.random() < 0.5:
                _, sg, pl, inner = self.c(pool)
            else:
                src, sg, pl = self.c(PUNCT)
                inner = {"-": r"\-"}.get(src, src)
            if inner not in [i[0] for i in items]:
                items.append((inner, sg, pl))
        return items

    def join_items(self, forms):
        segs = []
        for i, f in enumerate(forms):
            if i:
                if i < len(forms) - 1:
                    segs.append(S(",", "OR", True))
                else:
                    segs.append(S(self.c(["or", "and"]), "OR"))
            segs.append(S(f, "ATOM"))
        return segs

    def char_set(self):
        k = self.r.random()
        if k < 0.42:  # plural set: one or more
            items = self.set_items(2, 3)
            segs = self.join_items([self.c(i[2]) for i in items])
            if self.r.random() < 0.3:
                segs = [S(self.c(["one or more", "any mix of", "any combination of"]), "Q_PLUS")] + segs
            elif not all(LEX[compact(s[0])].get("plural") for s in segs if s[1] == "ATOM"):
                segs = [S("one or more", "Q_PLUS")] + segs
            return segs, "[" + "".join(i[0] for i in items) + "]+"
        if k < 0.58:  # singular set: exactly one
            items = self.set_items(2, 3)
            forms = [self.c(i[1]) for i in items]
            if any(LEX[compact(f)].get("plural") for f in forms):
                return self.char_set()
            return [S(self.c(["a", "any of", "either a", "one of"]))] + self.join_items(forms), "[" + "".join(i[0] for i in items) + "]"
        items = self.set_items(1, 3)  # negated
        lead = self.c(["anything except", "anything but", "no", "characters other than", "everything except", "not", "any characters except", "anything other than"])
        return [S(lead, "NEG")] + self.join_items([self.c(i[2]) for i in items]), "[^" + "".join(i[0] for i in items) + "]*"

    def preset_tail(self, tok):
        """ "... like 1.2.3" or "... in hh:mm format": filler that a real prompt often adds."""
        if self.r.random() > 0.2:
            return []
        if tok in FORMATS and self.r.random() < 0.5:
            f = self.c(FORMATS[tok])
            return [S(self.c(["in", "in the", "with the", "using"])), S(f), S(self.c(["format", "style", "form"]))]
        if tok in EXAMPLES:
            return [S(self.c(["like", "such as", "e.g.", "for example", "for instance", "similar to", "looking like"])), S(self.c(EXAMPLES[tok]))]
        return []

    def preset_el(self):
        tok = self.c(list(PRESETS))
        src, forms = PRESETS[tok]
        p = self.c(forms)
        lead = self.c(["", "", self.art(p), "any", "valid", "the", "a valid"])
        return words(lead) + [S(p, "ATOM")] + self.preset_tail(tok), src

    def element(self):
        k = self.r.random()
        if k < 0.60:
            return self.quantified()
        if k < 0.82:
            return self.lit_element()
        if k < 0.93:
            return self.char_set()
        return self.preset_el()

    def seq(self, lo=1, hi=4):
        n = self.r.randint(lo, hi)
        segs, rx = [], ""
        for i in range(n):
            e, r = self.element()
            if i:
                if self.r.random() < 0.2:
                    segs.append(S(",", "THEN", True))
                    if self.r.random() < 0.4:
                        segs.append(S("then", "THEN"))
                else:
                    segs.append(S(self.c(JOINERS), "THEN"))
            segs += e
            rx += r
        return segs, rx

    def anch_seq(self):
        return self.seq(1, 2) if self.r.random() < 0.6 else self.seq(1, 3)

    def subject(self):
        return words(self.c(SUBJECT + [""]))

    # --- top level shapes ---
    def t_seq(self):
        return self.seq(1, 4)

    def t_start(self):
        e, r = self.anch_seq()
        if self.r.random() < 0.25:
            return e + [S(self.c(["at the start of the line", "at the beginning of the string", "at the start", "at the beginning"]), "START")], "^" + r
        v = self.c(["starts with", "starting with", "begins with", "beginning with", "that starts with", "that begin with", "that start with", "which starts with", "prefixed with", "with prefix", "has prefix"])
        return self.subject() + [S(v, "START")] + e, "^" + r

    def t_end(self):
        e, r = self.anch_seq()
        if self.r.random() < 0.25:
            return e + [S(self.c(["at the end of the line", "at the end of the string", "at the end"]), "END")], r + "$"
        v = self.c(["ends with", "ending with", "ending in", "ends in", "that ends with", "that end with", "that end in", "which ends with", "suffixed with", "with suffix", "has suffix", "finishing with"])
        return self.subject() + [S(v, "END")] + e, r + "$"

    def t_start_end(self):
        a, ra = self.element()
        b, rb = self.element()
        v1 = self.c(["starts with", "starting with", "begins with", "beginning with", "that starts with"])
        v2 = self.c(["ends with", "ending with", "ending in", "ends in"])
        link = self.c([[], [S("and", "O")], [S(",", "O", True)], [S(",", "O", True), S("and", "O")]])
        return self.subject() + [S(v1, "START")] + a + link + [S(v2, "END")] + b, f"^{ra}.*{rb}$"

    def t_only(self):
        e, r = self.seq(1, 3)
        k = self.r.random()
        if k < 0.25:
            return e + [S(self.c(["and nothing else", "only"]), "ONLY")], f"^{r}$"
        lead = self.c(["only", "just", "nothing but", "containing only", "with only", "consisting of", "made of", "entire string is",
                       "whole string is", "whole line is", "full match of", "entire line is", "of only", "that is", "that only has", "nothing except"])
        return (self.subject() if k < 0.6 else []) + [S(lead, "ONLY")] + e, f"^{r}$"

    def t_word_affix(self):
        w = self.lit_word()
        L, e = self.lit_seg(w, loose=True), esc(w)
        wd = S(self.c(["word", "words", "any word", "a word", "all words"]), "WORDS")
        k = self.r.random()
        if k < 0.35:
            return [wd, S(self.c(["that start with", "starting with", "beginning with", "that begin with", "that starts with", "with prefix", "begins with", "starts with"]), "START"), L], rf"\b{e}\w*"
        if k < 0.7:
            return [wd, S(self.c(["that end with", "ending with", "ending in", "that end in", "that ends with", "with suffix", "ends with", "ends in"]), "END"), L], rf"\w*{e}\b"
        return [wd, S(self.c(["containing", "that contain", "that contains", "with", "including", "having"]), "CONTAINS"), L], rf"\w*{e}\w*"

    def t_contains(self):
        e, r = self.element()
        k = self.r.random()
        ln = self.c(["line", "lines", "a line", "any line", "the line", "the whole line", "entire line", "string", "a string", "strings"])
        if k < 0.3:
            return [S(ln, "LINE"), S(self.c(["containing", "that contains", "that contain", "with", "which contains", "having", "including"]), "CONTAINS")] + e, f"^.*{r}.*$"
        if k < 0.55:
            return [S(self.c(["contains", "containing", "contain", "includes", "has"]), "CONTAINS")] + e, r
        lead = words(ln) if self.r.random() < 0.7 else []
        if k < 0.75:
            return lead + [S(self.c(["not containing", "that does not contain", "that doesn't contain", "that do not contain", "that don't contain", "without", "not having", "which does not contain", "excluding", "does not contain", "doesn't contain", "with no", "that have no", "that has no", "lacking", "missing"]), "NOT_CONTAIN")] + e, f"^(?!.*{r}).*$"
        if k < 0.88:
            return lead + [S(self.c(["not starting with", "that does not start with", "that doesn't start with", "that do not start with", "that don't start with", "not beginning with", "does not start with", "doesn't start with", "doesn't begin with"]), "NOT_START")] + e, f"^(?!{r}).*$"
        return lead + [S(self.c(["not ending with", "that does not end with", "that doesn't end with", "that do not end with", "that don't end with", "not ending in", "does not end with", "doesn't end with", "doesn't end in"]), "NOT_END")] + e, f"^(?!.*{r}$).*$"

    def t_alt(self):
        n = self.r.randint(2, 4)
        plain = self.r.random() < 0.4
        ws = [self.c(PLAIN_LITS) if plain else self.lit_word() for _ in range(n)]
        Ls = [self.lit_seg(w, loose=True) for w in ws]
        body = "|".join(esc(w) for w in ws)
        comma_style = self.r.random() < 0.5
        segs = []
        for i, L in enumerate(Ls):
            if i:
                if comma_style and i < n - 1:
                    segs.append(S(",", "OR", True))
                else:
                    if comma_style and self.r.random() < 0.4:
                        segs.append(S(",", "OR", True))
                    segs.append(S("or", "OR"))
            segs.append(L)
        lead = words(self.c(["", "", "either", "one of", "any of"]))
        k = self.r.random()
        if k < 0.45:
            return lead + segs, body
        if k < 0.7:
            return [S(self.c(["the word", "the words", "the whole word", "whole words", "word", "words"]), "WORD")] + lead + segs, rf"\b(?:{body})\b"
        if k < 0.85:
            return words(self.c(["", "lines", "a line that"])) + [S(self.c(["starts with", "starting with", "begins with"]), "START")] + lead + segs, f"^(?:{body})"
        return words(self.c(["", "lines", "a line that", "files", "file names", "filenames"])) + [S(self.c(["ends with", "ending with", "ending in"]), "END")] + lead + segs, f"(?:{body})$"

    def t_preset(self):
        tok = self.c(list(PRESETS))
        src, forms = PRESETS[tok]
        p = self.c(forms)
        A = S(p, "ATOM")
        k = self.r.random()
        if k < 0.72 or tok in ("<TRAILWS>", "<LEADWS>", "<EMPTYLINE>", "<BLANKLINE>"):
            return words(self.c(["", "", self.art(p), "a valid", "valid", "any", "all", "the"])) + [A] + self.preset_tail(tok), src
        if k < 0.86:
            lead = self.c(["only", "entire string is", "whole string is", "that is exactly", "with only", "that is"])
            return self.subject() + [S(lead, "ONLY"), S(self.art(p)), A], f"^{src}$"
        if k < 0.93:
            return self.subject() + [S(self.c(["starts with", "starting with", "beginning with"]), "START"), S(self.art(p)), A], f"^{src}"
        return self.subject() + [S(self.c(["ends with", "ending with", "ending in"]), "END"), S(self.art(p)), A], f"{src}$"

    def t_password(self):
        options = [
            (["digit", "numeral"], r"\d"),
            (["uppercase letter", "capital letter", "capital"], "[A-Z]"),
            (["lowercase letter", "small letter"], "[a-z]"),
            (["symbol", "special character", "punctuation mark"], r"[^\w\s]"),
            (["letter"], "[a-zA-Z]"),
        ]
        self.r.shuffle(options)
        reqs, look = [], ""
        for names, t in options[: self.r.randint(1, 4)]:
            lead = self.c([[S("one", "NUM")], [S("a")], [S("at least one", "Q_PLUS")], [S("1", "NUM")], [S("an")]])
            reqs.append(lead + [S(self.c(names), "ATOM")])
            look += f"(?=.*{t})"
        N, n = self.number_seg(4, 20)
        C = S(self.c(["characters", "chars"]), "ATOM")
        length = self.c([[S("at least", "Q_MIN"), N, C], [N, S("or more", "Q_MIN"), C], [S("minimum", "Q_MIN"), N, C],
                         [N, C, S("minimum", "Q_MIN")], [S("minimum of", "Q_MIN"), N, C], [S("at least", "Q_MIN"), N, C, S("long")]])
        parts = reqs + [length] if self.r.random() < 0.3 else [length] + reqs
        segs = words(self.c(["", "a", "strong", "secure"])) + [S("password", "PASSWORD")] + words(self.c(["", "with", "that has", "containing", "requiring"]))
        for i, p in enumerate(parts):
            if i:
                segs.append(self.c([S(",", "THEN", True), S("and", "THEN"), S("with", "THEN")]))
            segs += p
        return segs, f"^{look}.{{{n},}}$"

    def t_capture(self):
        e, r = self.seq(1, 2)
        cap = S(self.c(["capture", "capture group for", "capturing group of", "a capture group of", "capture group containing", "capture group of"]), "CAPTURE")
        if self.r.random() < 0.5:
            return [cap] + e, f"({r})"
        p, rp = self.element()
        return p + [S(self.c(JOINERS), "THEN"), cap] + e, f"{rp}({r})"

    def t_ext(self):
        w = "".join(self.c(SYLL) for _ in range(self.r.randint(1, 2))) if self.r.random() < 0.3 else self.c(["png", "jpg", "pdf", "txt", "csv", "mp3", "js", "ts", "py", "gif", "json", "xml", "zip", "html"])
        lead = self.c(["file with extension", "files with extension", "file extension", "files with the extension", "extension"])
        return [S(lead, "EXT"), self.lit_seg(w)], rf"\.{esc(w)}$"

    def t_digit_noun(self):
        N, n = self.number_seg(2, 16)
        noun = S(self.c(["number", "pin", "code", "year", "zip code", "string", "sequence", "id"]))
        glue = self.r.random() < 0.5
        A = S("digit", "ATOM", glue)
        segs = [N] + ([S("-", "O", True)] if glue else []) + [A, noun]
        return words(self.c(["", "a", "any"])) + segs, r"\d{%d}" % n

    SHAPES = [("t_seq", 24), ("t_start", 11), ("t_end", 11), ("t_start_end", 8), ("t_only", 9), ("t_word_affix", 5),
              ("t_contains", 10), ("t_alt", 7), ("t_preset", 15), ("t_password", 2), ("t_capture", 3), ("t_ext", 1),
              ("t_digit_noun", 2)]

    def typo(self, text, min_len=5):
        """One edit in one long word, so the model learns that misspelt words keep their role."""
        ws = text.split(" ")
        i = self.r.randrange(len(ws))
        w = ws[i]
        if len(w) < min_len or not w.isalpha():
            return text
        j = self.r.randrange(1, len(w) - 1)
        k = self.r.randint(0, 2)
        if k == 0:
            w = w[:j] + w[j + 1:]
        elif k == 1:
            w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        else:
            w = w[:j] + w[j] + w[j:]
        ws[i] = w
        return " ".join(ws)

    def sample_segs(self):
        names, weights = zip(*self.SHAPES)
        segs, rx = getattr(self, self.r.choices(names, weights)[0])()
        segs = words(self.c(PREFIXES)) + segs
        icase = self.r.random() < 0.05
        if icase:
            if self.r.random() < 0.3:
                segs.append(S(",", "O", True))
            segs.append(S(self.c(["case insensitive", "ignoring case", "ignore case", "case-insensitive", "in any case", "regardless of case"]), "ICASE"))
        has_typo = False
        if self.fixed is not None:
            if self.r.random() < 0.15:
                segs.insert(self.r.randrange(len(segs) + 1), S(self.c(NOISE)))
            if self.r.random() < 0.10:
                i = self.r.randrange(len(segs))
                t = segs[i]
                if t[1] != "LIT":
                    new = self.typo(t[0], 4 if t[1] == "NUM" else 5)
                    has_typo = new != t[0]
                    segs[i] = (new, t[1], t[2])
            spelled = [i for i, t in enumerate(segs) if t[1] == "NUM" and t[0].isalpha()]
            if spelled and self.r.random() < 0.10:
                i = self.c(spelled)
                new = self.typo(segs[i][0], 4)
                has_typo = has_typo or new != segs[i][0]
                segs[i] = (new, segs[i][1], segs[i][2])
            if self.r.random() < 0.06 and segs[0][1] not in ("LIT", "NUM"):
                segs[0] = (segs[0][0].capitalize(), segs[0][1], segs[0][2])
            segs = self.sentences(segs)
        return segs, rx, ("im" if icase else "m"), has_typo

    def sentences(self, segs):
        """People write in sentences, so a period closing a clause or the prompt is usually filler."""
        out = []
        for i, t in enumerate(segs):
            if i and t[1] in SENTENCE_STARTS and not t[2] and self.r.random() < 0.15:
                out.append(S(".", "O", True))
                if self.r.random() < 0.7:
                    t = (t[0].capitalize(), t[1], t[2])
            out.append(t)
        if self.r.random() < 0.3:
            out.append(S(".", "O", True))
        return out

    def sample(self):
        """-> dict(text, roles per token, regex, flags, typo). Retries on anything too long."""
        while True:
            segs, rx, flags, typo = self.sample_segs()
            text, spans = "", []
            for t, role, glue in segs:
                if not t:
                    continue
                if text and not glue:
                    text += " "
                spans.append((len(text), len(text) + len(t), role))
                text += t
            toks = tokenize(text)
            if not toks or len(toks) > MAX_TOKENS:
                continue
            roles, i, ok = [], 0, True
            for tk in toks:
                while i < len(spans) and spans[i][1] <= tk["start"]:
                    i += 1
                if i == len(spans) or not (spans[i][0] <= tk["start"] and tk["end"] <= spans[i][1]):
                    ok = False  # a token straddled two segments
                    break
                roles.append(spans[i][2])
            if ok:
                return dict(text=text, roles=roles, regex=rx, flags=flags, typo=typo)


def build_fixed(n=100_000, seed=1):
    """Every word the grammar itself uses; a bare literal must not collide with any of them."""
    g = Gen(random.Random(seed), fixed=None)
    fixed = set(NOISE)
    for _ in range(n):
        s = g.sample()
        for tk, role in zip(tokenize(s["text"]), s["roles"]):
            if role != "LIT":
                fixed.add(tk["text"].lower())
    return fixed


if __name__ == "__main__":
    fixed = build_fixed(20000)
    print(len(fixed), "fixed words,", len(ROLES), "roles,", len(LEX), "lexicon forms")
    g = Gen(random.Random(2), fixed=fixed)
    for _ in range(25):
        s = g.sample()
        print(repr(s["text"]), "\n   ", " ".join(f"{t['text']}/{r}" for t, r in zip(tokenize(s["text"]), s["roles"]) if r != "O"), "\n   =>", f"/{s['regex']}/{s['flags']}")
