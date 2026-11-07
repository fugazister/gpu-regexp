// StructuredRegex's notation: parsing it, matching with it exactly, and judging what this project could express.
import { readFileSync } from "node:fs";

// ---- the dataset's notation ----
export function parseDsl(text) {
  let i = 0;
  const number = () => {
    const start = i;
    while (/\d/.test(text[i])) i++;
    return Number(text.slice(start, i));
  };
  const node = () => {
    if (text[i] === "<") {
      // The value can itself be ">" or ",", so the closing bracket is the first ">" after at least one character.
      const end = text.indexOf(">", i + 2);
      const value = text.slice(i + 1, end);
      i = end + 1;
      return { op: "leaf", value };
    }
    const start = i;
    while (text[i] !== "(") i++;
    const op = text.slice(start, i++);
    const kids = [node()];
    const params = [];
    while (text[i] === ",") {
      i++;
      if (/\d/.test(text[i])) params.push(number());
      else kids.push(node());
    }
    if (text[i++] !== ")") throw new Error(`bad spec: ${text}`);
    return { op, kids, params };
  };
  const out = node();
  if (i !== text.length) throw new Error(`trailing text in spec: ${text}`);
  return out;
}

const CLASSES = { num: /[0-9]/, let: /[a-zA-Z]/, low: /[a-z]/, cap: /[A-Z]/, any: /./, spec: /[^a-zA-Z0-9]/ };

/** Every position where a match of n starting at i can end. A real matcher, since `and` and `not` nested in `concat` have no plain regex form. */
function ends(n, s, i) {
  const rest = (from) => Array.from({ length: s.length - from + 1 }, (_, k) => from + k);
  const repeat = (kid, min, max) => {
    const out = new Set();
    let frontier = new Set([i]), seen = new Set();
    for (let count = 0; count <= max && frontier.size; count++) {
      if (count >= min) frontier.forEach((e) => out.add(e));
      const next = new Set();
      for (const e of frontier) for (const f of ends(kid, s, e)) next.add(f);
      // Past the minimum, a position already expanded can't add anything new, which also stops empty-match loops.
      if (count >= min) { frontier.forEach((e) => seen.add(e)); frontier = new Set([...next].filter((e) => !seen.has(e))); }
      else frontier = next;
    }
    return out;
  };
  switch (n.op) {
    case "leaf": {
      if (CLASSES[n.value]) return new Set(i < s.length && CLASSES[n.value].test(s[i]) ? [i + 1] : []);
      return new Set(s.startsWith(n.value, i) ? [i + n.value.length] : []);
    }
    case "const": return ends(n.kids[0], s, i);
    case "concat": {
      const out = new Set();
      for (const e of ends(n.kids[0], s, i)) for (const f of ends(n.kids[1], s, e)) out.add(f);
      return out;
    }
    case "or": return new Set([...ends(n.kids[0], s, i), ...ends(n.kids[1], s, i)]);
    case "and": { const b = ends(n.kids[1], s, i); return new Set([...ends(n.kids[0], s, i)].filter((e) => b.has(e))); }
    case "not": { const a = ends(n.kids[0], s, i); return new Set(rest(i).filter((e) => !a.has(e))); }
    case "notcc": return new Set(i < s.length && !ends(n.kids[0], s, i).has(i + 1) ? [i + 1] : []);
    case "star": return repeat(n.kids[0], 0, Infinity);
    case "optional": return new Set([i, ...ends(n.kids[0], s, i)]);
    case "repeat": return repeat(n.kids[0], n.params[0], n.params[0]);
    case "repeatatleast": return repeat(n.kids[0], n.params[0], Infinity);
    case "repeatrange": return repeat(n.kids[0], n.params[0], n.params[1]);
    case "startwith": { const out = new Set(); for (const e of ends(n.kids[0], s, i)) rest(e).forEach((x) => out.add(x)); return out; }
    case "endwith": { const out = new Set(); for (const from of rest(i)) ends(n.kids[0], s, from).forEach((x) => out.add(x)); return out; }
    case "contain": { const out = new Set(); for (const from of rest(i)) for (const e of ends(n.kids[0], s, from)) rest(e).forEach((x) => out.add(x)); return out; }
    default: throw new Error(`unknown op ${n.op}`);
  }
}
export const goldMatches = (ast, s) => ends(ast, s, 0).has(s.length);

// ---- which regexes this project's grammar could express at all, judged from the gold regex alone ----
const isAtom = (n) => n.op === "leaf" || n.op === "const";
const isItem = (n) => isAtom(n) || (["repeat", "repeatatleast", "repeatrange", "optional", "star"].includes(n.op) && isAtom(n.kids[0]));
const isSeq = (n) => isItem(n) || (n.op === "concat" && n.kids.every(isSeq));
export const inScope = (n) =>
  isSeq(n) || (["startwith", "endwith", "contain"].includes(n.op) && isSeq(n.kids[0])) || (n.op === "or" && n.kids.every((k) => isAtom(k) || (k.op === "or" && inScope(k))));

// ---- the data ----
export function readCases(split) {
  const lines = readFileSync(new URL(`external/streg/${split}.tsv`, import.meta.url), "utf8").trim().split(/\r?\n/).slice(1);
  return lines.map((line) => {
    const [problem, raw, spec, pos, neg] = line.split("\t");
    // Descriptions with quotes inside are CSV-quoted in the raw TSV. Left as is, the whole sentence reads as one quoted literal.
    const q = /^".*"$/.test(raw) ? raw.slice(1, -1).replace(/""/g, '"') : raw;
    return { problem, q, spec, ast: parseDsl(spec), yes: pos.split(" "), no: neg.split(" ") };
  });
}
