// Infers word-by-word roles for StructuredRegex descriptions, which only come with a regex and example strings.
// Starting from the model's own roles, a beam search changes as few words as it can until the compiled regex
// behaves like the gold one. A label is kept only if it agrees with the gold regex on the dataset's examples
// and on strings generated from the gold regex, because labels that pass only by luck would teach the wrong thing.
//   node train/label_real.mjs train/real_labels.jsonl [--limit N] [--shard i/n] [-v]
import { readFileSync, writeFileSync } from "node:fs";
import { tokenize } from "../web/features.js";
import { label, stripAnchors } from "./label-lib.mjs";
import { goldMatches, readCases } from "../eval/streg-lib.mjs";

const args = process.argv.slice(2);
const flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const out = args[0];
const limit = Number(flag("--limit", Infinity));
const [shard, shards] = flag("--shard", "0/1").split("/").map(Number);
const verbose = args.includes("-v");

const GENERATED = 120;

// ---- the gold regex, as a JS regex in the lexicon's own spelling so that edit distance is meaningful ----
const CLASS_SRC = { num: "\\d", let: "[a-zA-Z]", low: "[a-z]", cap: "[A-Z]", any: ".", spec: "[^\\w\\s]" };
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
const group = (s) => (/^(\\.|\[[^\]]*\]|.)$/.test(s) ? s : `(?:${s})`);
function toRegex(n) {
  switch (n.op) {
    case "leaf": return CLASS_SRC[n.value] ?? esc(n.value);
    case "const": return toRegex(n.kids[0]);
    case "concat": return n.kids.map((k) => (k.op === "or" ? `(?:${toRegex(k)})` : toRegex(k))).join("");
    case "or": return n.kids.map(toRegex).join("|");
    case "star": return group(toRegex(n.kids[0])) + "*";
    case "optional": return group(toRegex(n.kids[0])) + "?";
    case "repeat": return group(toRegex(n.kids[0])) + `{${n.params[0]}}`;
    case "repeatatleast": return group(toRegex(n.kids[0])) + (n.params[0] === 1 ? "+" : `{${n.params[0]},}`);
    case "repeatrange": return group(toRegex(n.kids[0])) + `{${n.params[0]},${n.params[1]}}`;
    case "notcc": return `[^${toRegex(n.kids[0]).replace(/^\[|\]$/g, "")}]`;
    case "startwith": return toRegex(n.kids[0]);
    case "endwith": return toRegex(n.kids[0]);
    case "contain": return toRegex(n.kids[0]);
    default: return null; // `and` and `not` have no regex this compiler could ever write
  }
}
// What the compiler can write today. Searching anything else only wastes time.
const isAtom = (n) => n.op === "leaf" || n.op === "const" || (n.op === "notcc" && n.kids[0].op === "leaf");
const isChoice = (n) => isAtom(n) || (n.op === "or" && n.kids.every(isChoice));
const isItem = (n) => isChoice(n) || (["repeat", "repeatatleast", "repeatrange", "optional", "star"].includes(n.op) && isChoice(n.kids[0]));
const isSeq = (n) => isItem(n) || (n.op === "concat" && n.kids.every(isSeq));
const reachable = (n) => isSeq(n) || (["startwith", "endwith", "contain"].includes(n.op) && isSeq(n.kids[0]));

// ---- strings for checking a label against the gold regex, beyond the dataset's few examples ----
const CHARS = { num: "0123456789", low: "abcdefghijklmnopqrstuvwxyz", cap: "ABCDEFGHIJKLMNOPQRSTUVWXYZ", spec: "!@#$%^&*-_=+;:,.?" };
CHARS.let = CHARS.low + CHARS.cap;
CHARS.any = CHARS.let + CHARS.num + CHARS.spec;
function seeded(seed) {
  return () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
}
function generate(n, rand) {
  const pick = (s) => s[Math.floor(rand() * s.length) % s.length];
  const noise = () => Array.from({ length: Math.floor(rand() * 4) }, () => pick(CHARS.any)).join("");
  const times = (kid, k) => Array.from({ length: k }, () => generate(kid, rand)).join("");
  switch (n.op) {
    case "leaf": return CHARS[n.value] ? pick(CHARS[n.value]) : n.value;
    case "const": return generate(n.kids[0], rand);
    case "concat": return n.kids.map((k) => generate(k, rand)).join("");
    case "or": return generate(pick(n.kids), rand);
    case "star": return times(n.kids[0], Math.floor(rand() * 4));
    case "optional": return rand() < 0.5 ? generate(n.kids[0], rand) : "";
    case "repeat": return times(n.kids[0], n.params[0]);
    case "repeatatleast": return times(n.kids[0], n.params[0] + Math.floor(rand() * 3));
    case "repeatrange": return times(n.kids[0], n.params[0] + Math.floor(rand() * (n.params[1] - n.params[0] + 1)));
    case "notcc": { for (;;) { const c = pick(CHARS.any); if (!goldMatches(n.kids[0], c)) return c; } }
    case "startwith": return generate(n.kids[0], rand) + noise();
    case "endwith": return noise() + generate(n.kids[0], rand);
    case "contain": return noise() + generate(n.kids[0], rand) + noise();
    default: throw new Error(n.op);
  }
}
function mutate(s, rand) {
  const at = Math.floor(rand() * (s.length + 1)) % (s.length + 1);
  const c = CHARS.any[Math.floor(rand() * CHARS.any.length) % CHARS.any.length];
  const k = rand();
  if (k < 0.25) return s.slice(0, at) + s.slice(at + 1);
  if (k < 0.5) return s.slice(0, at) + c + s.slice(at);
  if (k < 0.75) return s.slice(0, at) + c + s.slice(at + 1);
  return rand() < 0.5 ? s + c : c + s;
}
function checkStrings(c) {
  const rand = seeded(c.q.length * 31 + 7), strings = [];
  for (let i = 0; i < GENERATED / 2; i++) {
    const s = generate(c.ast, rand);
    strings.push(s, mutate(s, rand));
  }
  return [...new Set(strings)].map((s) => ({ s, gold: goldMatches(c.ast, s) }));
}

/** "as-produced" when the regex is right as it stands, "whole" when it's right if the user anchors it, else null. */
function verdict(source, flags, c) {
  if (source === null) return null;
  let re, whole;
  try { re = new RegExp(source, flags); whole = new RegExp(`^(?:${source})$`, flags); } catch { return null; }
  const agrees = (rx) => c.yes.every((s) => rx.test(s)) && c.no.every((s) => !rx.test(s)) && c.checks.every((x) => rx.test(x.s) === x.gold);
  return agrees(re) ? "as-produced" : agrees(whole) ? "whole" : null;
}

// ---- run ----
const heldOut = new Set(["dev", "teste", "testi"].flatMap((s) => readCases(s).map((c) => c.spec)));
const cases = readCases("train");
const usable = cases.filter((c) => !heldOut.has(c.spec) && reachable(c.ast));
console.log(`${cases.length} descriptions, ${cases.filter((c) => heldOut.has(c.spec)).length} share a regex with a held-out split, ${usable.length} usable and reachable`);
const chosen = usable.filter((_, i) => i % shards === shard).slice(0, limit);
const why = {}, labels = [];
const t0 = Date.now();
for (const c of chosen) {
  c.goldRegex = toRegex(c.ast);
  c.checks = checkStrings(c);
  c.verdict = (source, flags) => verdict(source, flags, c);
  const r = await label(c);
  if (r.label) labels.push(r.label);
  else why[r.why] = (why[r.why] ?? 0) + 1;
  if (verbose && r.label) console.log(`${r.label.kind} ${r.label.changed} changed  ${c.q}\n   /${r.label.source}/   gold /${c.goldRegex}/\n   ${tokenize(c.q).map((t, i) => (r.label.roles[i] === "O" ? null : `${t.text}/${r.label.roles[i]}`)).filter(Boolean).join(" ")}`);
  if (verbose && !r.label) console.log(`MISS (${r.why})  ${c.q}\n   gold /${c.goldRegex}/`);
}
const count = (k) => labels.filter((l) => l.kind === k).length;
console.log(`labelled ${labels.length}/${chosen.length} in ${((Date.now() - t0) / 1000).toFixed(0)}s: ${count("as-produced")} as produced, ${count("whole")} whole-string; missed ${JSON.stringify(why)}; ${new Set(labels.map((l) => l.problem)).size} regexes; mean words changed ${(labels.reduce((a, l) => a + l.changed, 0) / (labels.length || 1)).toFixed(1)}`);
if (out && !out.startsWith("-")) writeFileSync(out, labels.map((l) => JSON.stringify(l)).join("\n") + "\n");
