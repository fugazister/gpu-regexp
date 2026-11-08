// Deterministic compiler: token roles -> regex. All regex semantics live here, not in the model,
// so the output is always syntactically valid and anything it can't interpret becomes a diagnostic.

const NUMBER_WORDS = new Map(("zero one two three four five six seven eight nine ten eleven twelve thirteen " +
  "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split(" ").map((w, i) => [w, i]));
const QUANTS = new Set(["Q_PLUS", "Q_STAR", "Q_OPT", "Q_MIN", "Q_MAX", "Q_RANGE"]);
const MODES = new Set(["START", "END", "CONTAINS", "NOT_START", "NOT_END", "NOT_CONTAIN"]);

const esc = (s) => s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
export const compact = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, "") || s.toLowerCase();

function editDistance(a, b, cap) {
  if (Math.abs(a.length - b.length) > cap) return cap + 1;
  let before = null, prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      let d = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d = Math.min(d, before[j - 2] + 1);
      cur.push(d);
    }
    before = prev;
    prev = cur;
  }
  return prev[b.length];
}

/** Exact lexicon hit, else the unique closest form within a small edit distance. */
function lookup(lex, text) {
  const key = compact(text);
  if (lex[key]) return { entry: lex[key], form: key, exact: true };
  if (key.length < 4) return null;
  const cap = key.length >= 8 ? 2 : 1;
  let best = null, bestD = cap + 1, tied = false;
  for (const form of Object.keys(lex)) {
    const d = editDistance(key, form, cap);
    if (d < bestD) { best = form; bestD = d; tied = false; }
    else if (d === bestD && best && lex[form].src !== lex[best].src) tied = true;
  }
  return best && !tied ? { entry: lex[best], form: best, exact: false } : null;
}

function numberWord(low, text, diagnostics) {
  if (NUMBER_WORDS.has(low)) return NUMBER_WORDS.get(low);
  if (low.length < 4) return undefined;
  const near = [...NUMBER_WORDS.keys()].filter((w) => editDistance(low, w, 1) <= 1);
  if (near.length !== 1) return undefined;
  diagnostics.push(`read "${text}" as "${near[0]}"`);
  return NUMBER_WORDS.get(near[0]);
}

/** Adjacent tokens with one role become a span. Quoted literals and numbers always stand alone. */
function toSpans(tokens, roles) {
  const spans = [];
  tokens.forEach((t, i) => {
    const role = roles[i], last = spans.at(-1);
    const joinable = last && last.role === role && role !== "NUM" && role !== "O" && !t.quoted && !last.quoted &&
      (role !== "LIT" || t.glueL);
    if (joinable) {
      last.text += (t.glueL ? "" : " ") + t.text;
      last.end = t.end;
      last.tokens.push(t);
    } else {
      spans.push({ role, text: t.text, start: t.start, end: t.end, quoted: t.quoted, tokens: [t] });
    }
  });
  return spans;
}

function quantText(p) {
  const [a, b] = p.nums;
  switch (p.type) {
    case "Q_PLUS": return "+";
    case "Q_STAR": return "*";
    case "Q_OPT": return "?";
    case "Q_MIN": return a === undefined ? null : `{${a},}`;
    case "Q_MAX": return a === undefined ? null : `{0,${a}}`;
    case "Q_RANGE": return b === undefined ? null : `{${a},${b}}`;
    default:
      if (b !== undefined) return `{${a},${b}}`;
      if (a === undefined) return null;
      return a === 1 ? "" : `{${a}}`;
  }
}

function renderElem(e, standalone) {
  let s;
  if (e.set) s = `[${e.neg ? "^" : ""}${e.set.join("")}]`;
  else if (e.alts) s = e.alts.join("|");
  else s = e.src;
  if (e.ext && !s.startsWith("\\.")) s = "\\." + s;
  if (e.word) return `\\b${e.alts ? `(?:${s})` : s}\\b${e.quant ?? ""}`;
  if (e.alts && !standalone) s = `(?:${s})`;
  // "3 numbers" means three digits: a count on \d+ applies to the digit, not to runs of digits.
  if (s === "\\d+" && !e.alts && /^(\+|\*|\{[\d,]+\})$/.test(e.quant ?? "")) return "\\d" + e.quant;
  if (e.quant && e.group) s = `(?:${s})`;
  return s + (e.quant ?? "");
}

/**
 * tokens: from tokenize(). roles: one role name per token. lex: the phrase table.
 * -> { source, flags, diagnostics, spans } with source === null when nothing could be compiled.
 */
export function compile(tokens, roles, lex) {
  const spans = toSpans(tokens, roles);
  const diagnostics = [];
  const flags = { icase: false, only: false, words: false, line: false, password: false };
  let cur = { mode: null, elems: [], captureAt: null };
  const clauses = [cur];
  let pend = null, neg = false, wordNext = false, extNext = false, join = false;

  const contentAfter = (i) => spans.slice(i + 1).some((s) => s.role === "LIT" || s.role === "ATOM");
  const lastElem = () => cur.elems.at(-1);
  const flush = (flushingTimes = false) => {
    if (!pend) return;
    const e = lastElem();
    const text = quantText(pend);
    if (pend.type === null) {
      // A bare count with nothing after it ("a vowel then two ...") belongs to something we couldn't read.
      // Only "<atom> 3 times" counts backwards, and that's applied before reaching here.
      if (!flushingTimes) { diagnostics.push("a count had nothing to apply to, so it was ignored"); pend = null; return; }
    }
    if (e && text !== null) e.quant = text;
    else if (e && pend.type === "Q_MIN" && /^\{\d+\}$/.test(e.quant ?? "")) e.quant = e.quant.replace("}", ",}");
    else if (e && pend.type === "Q_MAX" && /^\{\d+\}$/.test(e.quant ?? "")) e.quant = e.quant.replace("{", "{0,");
    else diagnostics.push("a quantity had nothing to apply to, so it was ignored");
    pend = null;
  };

  const addElem = (e, span) => {
    span.elem = e;
    const last = lastElem();
    if (join && last) {
      join = false;
      if (e.inner && (last.set || last.inner)) {
        last.set ??= [last.inner];
        last.set.push(e.inner);
        if (e.plural && !last.neg && !last.explicitQuant) last.quant = "+";
        return;
      }
      last.alts ??= [last.src ?? ""];
      last.alts.push(e.src);
      last.quant = "";
      return;
    }
    join = false;
    if (neg) {
      if (!e.inner) {
        diagnostics.push(`can't negate "${span.text}" as a character set, so it was matched as written`);
      } else {
        e.neg = true;
        e.set = [e.inner];
        e.quant = "*";
      }
    }
    if (pend) {
      const q = quantText(pend);
      if (q !== null) { e.quant = q; e.explicitQuant = true; pend = null; }
    } else if (e.plural && !e.neg) {
      e.quant = "+";
    }
    e.group = !!(e.preset || e.multi);
    e.word = wordNext;
    e.ext = extNext;
    wordNext = extNext = false;
    cur.elems.push(e);
  };

  const addLiteral = (text, span) => {
    addElem({ src: esc(text), multi: [...text].length > 1 }, span);
  };

  spans.forEach((sp, i) => {
    const role = sp.role;
    const next = spans.slice(i + 1).find((x) => x.role !== "O")?.role; // filler never separates a count from "times"
    if (role === "LIT") return addLiteral(sp.text, sp);
    if (role === "ATOM") {
      const resolved = resolveAtom(sp, lex, diagnostics);
      for (const r of resolved) {
        if (r.literal !== undefined) addLiteral(r.literal, sp);
        else addElem({ src: r.entry.src, inner: r.entry.inner, plural: r.entry.plural, preset: r.entry.preset }, sp);
      }
      return;
    }
    if (role === "NUM") {
      const low = sp.text.toLowerCase();
      const n = /^[0-9]+$/.test(low) ? Number(low) : numberWord(low, sp.text, diagnostics);
      if (n === undefined) {
        diagnostics.push(`"${sp.text}" was read as a number but isn't one, so it was matched as text`);
        return addLiteral(sp.text, sp);
      }
      pend ??= { type: null, nums: [] };
      pend.nums.push(n);
      if (next === "Q_TIMES") flush(true);
      return;
    }
    if (QUANTS.has(role)) {
      pend ??= { type: null, nums: [] };
      pend.type = role;
      if (next === "Q_TIMES") flush(true);
      return;
    }
    if (role === "OR") {
      if (pend?.nums.length) pend.type = "Q_RANGE";
      else join = true;
      return;
    }
    if (role === "THEN") { flush(); neg = wordNext = join = false; return; }
    if (role === "NEG") { neg = true; return; }
    if (role === "WORD") { wordNext = true; return; }
    if (role === "WORDS") { flags.words = true; return; }
    if (role === "LINE") { flags.line = true; return; }
    if (role === "ONLY") { flags.only = true; return; }
    if (role === "ICASE") { flags.icase = true; return; }
    if (role === "PASSWORD") { flags.password = true; return; }
    if (role === "CAPTURE") { flush(); cur.captureAt = cur.elems.length; return; }
    if (role === "EXT" || MODES.has(role)) {
      flush();
      neg = false;
      const mode = role === "EXT" ? "END" : role;
      extNext = role === "EXT";
      if (!contentAfter(i) || !cur.elems.length) {
        cur.mode = mode; // "foo at the start" applies to what came before
      } else {
        cur = { mode, elems: [], captureAt: null };
        clauses.push(cur);
      }
    }
    // Q_TIMES on its own ("repeated") and O carry no meaning.
  });
  flush();

  const out = assemble(clauses, flags, diagnostics);
  return { source: out, flags: flags.icase ? "im" : "m", diagnostics, spans };
}

function resolveAtom(sp, lex, diagnostics) {
  const hit = lookup(lex, sp.text);
  if (hit) {
    if (!hit.exact) diagnostics.push(`read "${sp.text}" as "${hit.form}"`);
    return [hit];
  }
  // A merged span like "digits letters": take the longest known runs of its tokens.
  const toks = sp.tokens.map((t) => t.text), res = [];
  for (let i = 0; i < toks.length;) {
    let found = null;
    for (let j = toks.length; j > i; j--) {
      const h = lookup(lex, toks.slice(i, j).join(" "));
      if (h?.exact || (h && j === i + 1)) { found = { h, j }; break; }
    }
    if (found) { res.push(found.h); i = found.j; } else {
      res.length = 0;
      break;
    }
  }
  if (res.length) return res;
  diagnostics.push(`don't know what "${sp.text}" means, so it was matched as literal text`);
  return [{ literal: sp.text }];
}

function renderClause(c, standalone) {
  const parts = c.elems.map((e) => renderElem(e, standalone));
  if (c.captureAt !== null) {
    const body = parts.slice(c.captureAt).join("");
    return parts.slice(0, c.captureAt).join("") + `(${body})`;
  }
  return parts.join("");
}

function assemble(clauses, flags, diagnostics) {
  const nonEmpty = clauses.filter((c) => c.elems.length);
  if (!nonEmpty.length) {
    diagnostics.push("didn't find anything to match");
    return null;
  }

  if (flags.password) {
    let length = null, look = "";
    for (const c of nonEmpty) {
      for (const e of c.elems) {
        const min = e.quant?.match(/^\{(\d+),\}$/);
        if (e.src === "." && min) length = min[1];
        else look += `(?=.*${renderElem({ ...e, quant: "" }, false)})`;
      }
    }
    if (length === null) diagnostics.push("no minimum length found for the password");
    return `^${look}.${length === null ? "*" : `{${length},}`}$`;
  }

  const of = (mode) => nonEmpty.filter((c) => c.mode === mode);
  const solo = nonEmpty.length === 1 && !flags.only && nonEmpty[0].elems.length === 1 && nonEmpty[0].mode !== "START" &&
    nonEmpty[0].mode !== "END" && !flags.line;
  const text = (cs) => cs.map((c) => renderClause(c, solo)).join("");
  const S = text(of("START")), E = text(of("END")), C = text(of("CONTAINS")), M = text(of(null)) + C;

  for (const [mode, build] of [
    ["NOT_START", (x) => `^(?!${x}).*$`],
    ["NOT_END", (x) => `^(?!.*${x}$).*$`],
    ["NOT_CONTAIN", (x) => `^(?!.*${x}).*$`],
  ]) {
    const n = of(mode);
    if (n.length) {
      if (nonEmpty.length > n.length) diagnostics.push("combined a negative condition with other conditions; only the negative one was kept");
      return build(text(n));
    }
  }
  if (flags.only) return `^${S}${M}${E}$`;
  if (flags.words) {
    if (S) return `\\b${S}\\w*`;
    if (E) return `\\w*${E}\\b`;
    if (M) return `\\w*${M}\\w*`;
  }
  // "starts with a and ends with b" leaves a gap. Listing more parts than that describes the whole string, in order.
  const parts = [...of("START"), ...of(null), ...of("CONTAINS"), ...of("END")].reduce((a, c) => a + c.elems.length, 0);
  if (S && E && parts > 2) return `^${S}${M}${E}$`;
  if (S && E) return `^${S}${M && `.*${M}`}.*${E}$`;
  if (S) return `^${S}${M}`;
  if (E) return `${M}${E}$`;
  if (flags.line) return `^.*${M}.*$`;
  return M;
}
