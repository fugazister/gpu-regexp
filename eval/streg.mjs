// Scores the model on StructuredRegex (Ye et al.), whose descriptions were written by crowdworkers who never
// saw this project's grammar. Each description comes with strings that should and shouldn't match.
//   mkdir -p eval/external/streg && for s in dev teste testi; do
//     gh api repos/xiye17/StructuredRegex/contents/data/raw/$s.tsv -H 'Accept: application/vnd.github.raw' > eval/external/streg/$s.tsv
//   done
//   node eval/streg.mjs dev [-v] [--json out.json]
// Only dev may be read. teste and testi print totals, so they can stay an independent test.
import { readFileSync, writeFileSync } from "node:fs";
import { loadCpu } from "./lib.mjs";

const args = process.argv.slice(2);
const split = args[0] ?? "dev";
const verbose = args.includes("-v") && split === "dev";
const jsonAt = args.indexOf("--json");
const MAX_TOKENS = 48;

// ---- the dataset's notation ----
function parseDsl(text) {
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
const goldMatches = (ast, s) => ends(ast, s, 0).has(s.length);

// ---- which regexes this project's grammar could express at all, judged from the gold regex alone ----
const isAtom = (n) => n.op === "leaf" || n.op === "const";
const isItem = (n) => isAtom(n) || (["repeat", "repeatatleast", "repeatrange", "optional", "star"].includes(n.op) && isAtom(n.kids[0]));
const isSeq = (n) => isItem(n) || (n.op === "concat" && n.kids.every(isSeq));
const inScope = (n) =>
  isSeq(n) || (["startwith", "endwith", "contain"].includes(n.op) && isSeq(n.kids[0])) || (n.op === "or" && n.kids.every((k) => isAtom(k) || (k.op === "or" && inScope(k))));

// ---- scoring ----
const lines = readFileSync(new URL(`external/streg/${split}.tsv`, import.meta.url), "utf8").trim().split(/\r?\n/).slice(1);
const cases = lines.map((line) => {
  const [problem, raw, spec, pos, neg] = line.split("\t");
  // Descriptions with quotes inside are CSV-quoted in the raw TSV. Left as is, the whole sentence reads as one quoted literal.
  const q = /^".*"$/.test(raw) ? raw.slice(1, -1).replace(/""/g, '"') : raw;
  const ast = parseDsl(spec);
  return { problem, q, spec, ast, yes: pos.split(" "), no: neg.split(" ") };
});
const behaves = (re, c) => c.yes.every((s) => re.test(s)) && c.no.every((s) => !re.test(s));

const { parser } = loadCpu();
let goldOk = 0;
for (const c of cases) {
  c.goldOk = c.yes.every((s) => goldMatches(c.ast, s)) && c.no.every((s) => !goldMatches(c.ast, s));
  goldOk += c.goldOk;
  c.scope = inScope(c.ast);
  const r = await parser.parse(c.q);
  c.source = r.source;
  c.flags = r.flags;
  c.confidence = r.confidence;
  c.truncated = r.diagnostics.some((d) => d.startsWith("only the first"));
  c.none = !r.regexp;
  c.diagnosed = r.diagnostics.length > 0;
  // As produced is what a user gets. Whole-string also forgives a regex that wasn't anchored, since these
  // descriptions usually describe the whole string; it's the most generous reading.
  c.asIs = !!r.regexp && behaves(r.regexp, c);
  c.whole = !!r.regexp && (c.asIs || behaves(new RegExp(`^(?:${r.source})$`, r.flags), c));
}

// Descriptions of one regex share their examples, so the unit of independence is the problem, not the description.
function clusterCi(list, key) {
  const groups = Object.values(Object.groupBy(list, (c) => c.problem));
  let seed = 12345;
  const rand = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const rates = [];
  for (let b = 0; b < 2000; b++) {
    let pass = 0, all = 0;
    for (let g = 0; g < groups.length; g++) {
      const pick = groups[Math.floor(rand() * groups.length) % groups.length];
      all += pick.length;
      pass += pick.filter((c) => c[key]).length;
    }
    rates.push(all ? (100 * pass) / all : 0);
  }
  rates.sort((a, b) => a - b);
  return [rates[50], rates[1949]];
}

const row = (name, list) => {
  const count = (key) => list.filter((c) => c[key]).length;
  if (!list.length) return { name, n: 0 };
  const res = { name, n: list.length, problems: new Set(list.map((c) => c.problem)).size, asIs: count("asIs"), whole: count("whole"), asIsCi: clusterCi(list, "asIs"), wholeCi: clusterCi(list, "whole") };
  const f = (k, ci) => `${String(res[k]).padStart(4)} (${((100 * res[k]) / res.n).toFixed(1)}%, CI ${ci[0].toFixed(1)}–${ci[1].toFixed(1)})`;
  console.log(`${name.padEnd(30)} ${String(res.n).padStart(4)} descr, ${String(res.problems).padStart(3)} regexes | as produced ${f("asIs", res.asIsCi)} | whole-string ${f("whole", res.wholeCi)}`);
  return res;
};

console.log(`StructuredRegex ${split}: gold regexes pass their own examples ${goldOk}/${cases.length}; ${cases.filter((c) => c.none).length} no regex; ${cases.filter((c) => c.truncated).length} truncated at ${MAX_TOKENS} tokens; ${cases.filter((c) => c.diagnosed).length} with diagnostics`);
const valid = cases.filter((c) => c.goldOk);
const scoped = valid.filter((c) => c.scope);
const rows = [
  row("all", valid),
  row("in grammar scope", scoped),
  row("in scope, not truncated", scoped.filter((c) => !c.truncated)),
  row("out of scope", valid.filter((c) => !c.scope)),
];

if (verbose) {
  for (const c of scoped) console.log(`${c.asIs ? "pass" : c.whole ? "WHOLE" : "FAIL"}  ${c.q}\n      got /${c.source}/${c.flags}   gold ${c.spec}`);
  for (const c of cases.filter((c) => !c.goldOk).slice(0, 10)) console.log(`GOLD-MISMATCH ${c.spec}\n   yes ${c.yes.join(" ")}\n   no ${c.no.join(" ")}`);
}
if (jsonAt >= 0) writeFileSync(args[jsonAt + 1], JSON.stringify({ split, n: cases.length, goldOk, rows }, null, 1));
