// Infers word-by-word roles for StructuredRegex descriptions, which only come with a regex and example strings.
// Starting from the model's own roles, a beam search changes as few words as it can until the compiled regex
// behaves like the gold one. A label is kept only if it agrees with the gold regex on the dataset's examples
// and on strings generated from the gold regex, because labels that pass only by luck would teach the wrong thing.
//   node train/label_real.mjs train/real_labels.jsonl [--limit N] [--shard i/n] [-v]
import { readFileSync, writeFileSync } from "node:fs";
import { compile as compileProgram } from "../web/program.js";
import { featurize, SLOTS, tokenize } from "../web/features.js";
import { compact, compile as compileRoles } from "../web/compile.js";
import { createCpuBackend } from "../web/backend-cpu.js";
import { decodeWeights } from "../web/gpu-regexp.js";
import { goldMatches, readCases } from "../eval/streg-lib.mjs";

const args = process.argv.slice(2);
const flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const out = args[0];
const limit = Number(flag("--limit", Infinity));
const [shard, shards] = flag("--shard", "0/1").split("/").map(Number);
const verbose = args.includes("-v");

const BEAM = 3, MAX_STEPS = 16, STALL = 3, MAX_TOKENS = 48, GENERATED = 120;
const WINDOW_ROLES = ["O", "THEN", "START", "END", "Q_PLUS", "Q_STAR", "Q_OPT", "Q_MIN", "Q_MAX", "Q_RANGE", "NOT_START", "NOT_END", "NOT_CONTAIN", "OR"];

const root = new URL("../", import.meta.url);
const spec = JSON.parse(readFileSync(new URL("web/model.json", root)));
const lex = JSON.parse(readFileSync(new URL("web/lexicon.json", root)));
const bin = readFileSync(new URL("web/model.bin", root));
const backend = createCpuBackend(decodeWeights(spec, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)));
const prog = compileProgram(spec, SLOTS);

async function logProbs(tokens) {
  const rows = new Uint32Array(tokens.length * SLOTS);
  tokens.forEach((t, i) => rows.set(featurize(t), i * SLOTS));
  const logits = await backend.run(prog, rows, new Uint32Array([tokens.length]), tokens.length);
  return tokens.map((_, i) => {
    const row = Array.from(logits.slice(i * prog.nRoles, (i + 1) * prog.nRoles));
    const top = Math.max(...row);
    const total = Math.log(row.reduce((a, x) => a + Math.exp(x - top), 0));
    return row.map((x) => x - top - total);
  });
}

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

// ---- search ----
function editDistance(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)));
    prev = cur;
  }
  return prev[b.length];
}
const stripAnchors = (s) => s.replace(/^\^/, "").replace(/\$$/, "");

/** "as-produced" when the regex is right as it stands, "whole" when it's right if the user anchors it, else null. */
function verdict(source, flags, c) {
  if (source === null) return null;
  let re, whole;
  try { re = new RegExp(source, flags); whole = new RegExp(`^(?:${source})$`, flags); } catch { return null; }
  const agrees = (rx) => c.yes.every((s) => rx.test(s)) && c.no.every((s) => !rx.test(s)) && c.checks.every((x) => rx.test(x.s) === x.gold);
  return agrees(re) ? "as-produced" : agrees(whole) ? "whole" : null;
}

function moves(tokens, lp) {
  const list = [];
  tokens.forEach((t, i) => {
    const ranked = lp[i].map((p, k) => [p, spec.roles[k]]).sort((a, b) => b[0] - a[0]).slice(0, 3).map((x) => x[1]);
    const singles = new Set([...ranked, "O"]);
    if (t.quoted) singles.add("LIT");
    if (/^[0-9]+$/.test(t.text) || /^(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)$/i.test(t.text)) singles.add("NUM");
    if (lex[compact(t.text)]) singles.add("ATOM");
    if (!/^[a-z]+$/i.test(t.text) || t.text.length <= 3) singles.add("LIT");
    for (const role of singles) list.push({ at: i, len: 1, role });
    for (const len of [2, 3, 4]) {
      if (i + len > tokens.length) continue;
      for (const role of WINDOW_ROLES) list.push({ at: i, len, role });
      if (lex[compact(tokens.slice(i, i + len).map((x) => x.text).join(" "))]) list.push({ at: i, len, role: "ATOM" });
    }
  });
  return list;
}

// A regex can be right for the wrong reason: "starts/START with/END" passes, but it would teach that "with" means
// "ends". A role with no regex text of its own has to sit on a phrase the grammar really uses for that role.
const PHRASES = Object.fromEntries(Object.entries(JSON.parse(readFileSync(new URL("train/role_phrases.json", root)))).map(([k, v]) => [k, new Set(v)]));
// The grammar only says "that start with"; plain imperatives mean the same thing.
for (const [role, words] of [["START", ["start with", "begin with", "start", "begin"]], ["END", ["end with", "end in", "end", "finish with"]]]) words.forEach((w) => PHRASES[role].add(w));
const CORE = new Set(["O", "LIT", "NUM", "ATOM"]);
function saidSo(role, words) {
  const known = PHRASES[role];
  if (!known) return false;
  const ok = [true];
  for (let i = 1; i <= words.length; i++) {
    ok[i] = false;
    for (let j = 0; j < i && !ok[i]; j++) if (ok[j] && known.has(words.slice(j, i).join(" "))) ok[i] = true;
  }
  return ok[words.length];
}
/** Turns every role that the words don't support into filler. Returns null if the regex then stops working. */
function vet(tokens, roles, c) {
  const out = roles.slice();
  for (let i = 0; i < out.length;) {
    let j = i + 1;
    while (j < out.length && out[j] === out[i]) j++;
    if (!CORE.has(out[i]) && !saidSo(out[i], tokens.slice(i, j).map((t) => t.text.toLowerCase()))) out.fill("O", i, j);
    i = j;
  }
  // An ATOM the lexicon can't read is really a literal ("4 q's").
  tokens.forEach((t, i) => { if (out[i] === "ATOM" && t.text.length < 4 && !lex[compact(t.text)] && out[i - 1] !== "ATOM" && out[i + 1] !== "ATOM") out[i] = "LIT"; });
  const r = compileRoles(tokens, out, lex);
  return verdict(r.source, r.flags, c) ? out : null;
}

// Labels that compile to the right regex but would teach something false about common phrases.
function misleading(tokens, roles) {
  const low = tokens.map((t) => t.text.toLowerCase());
  const nextRole = (i) => { while (roles[++i] === "O"); return roles[i]; };
  for (let i = 0; i < low.length; i++) {
    // "ends with" labelled as filler: in this dataset "starts ... and ends ..." can leave no gap, unlike the grammar's reading.
    if (/^(starts?|starting|begins?|beginning|ends?|ending|ended)$/.test(low[i]) && /^(with|in)$/.test(low[i + 1] ?? "") &&
        !(roles[i] === roles[i + 1] && /START|END/.test(roles[i]))) return "drops an anchor";
    if (roles[i] === "Q_PLUS" && low[i] === "string" && ["NUM", "Q_MIN", "Q_MAX", "Q_RANGE", "Q_OPT", "Q_STAR"].includes(nextRole(i))) return "a string of N";
    if (roles[i] === "ONLY" && low[i] === "is" && low[i - 1] === "that" && nextRole(i) === "THEN") return "that is followed by";
  }
  return null;
}

async function label(c) {
  const tokens = tokenize(c.q);
  if (!tokens.length || tokens.length > MAX_TOKENS) return { why: "too long" };
  const lp = await logProbs(tokens);
  const start = lp.map((row) => spec.roles[row.indexOf(Math.max(...row))]);
  const target = stripAnchors(c.goldRegex);
  const logP = (roles) => roles.reduce((a, r, i) => a + lp[i][spec.roles.indexOf(r)], 0);
  const evaluate = (roles) => {
    const r = compileRoles(tokens, roles, lex);
    return { roles, source: r.source, flags: r.flags, dist: r.source === null ? target.length + 5 : editDistance(stripAnchors(r.source), target), lp: logP(roles) };
  };
  const all = moves(tokens, lp);
  let beam = [evaluate(start)], found = null, kind = null, bestDist = beam[0].dist, stalled = 0;
  const seen = new Set([start.join()]);
  const accept = (s) => { kind = verdict(s.source, s.flags, c); if (kind) found = s; return !!kind; };
  if (!accept(beam[0])) {
    for (let step = 0; step < MAX_STEPS && !found; step++) {
      const next = [];
      for (const s of beam) {
        for (const m of all) {
          const roles = s.roles.slice();
          for (let k = m.at; k < m.at + m.len; k++) roles[k] = m.role;
          const key = roles.join();
          if (seen.has(key)) continue;
          seen.add(key);
          next.push(evaluate(roles));
        }
      }
      if (!next.length) break;
      next.sort((a, b) => a.dist - b.dist || b.lp - a.lp);
      for (const s of next) { if (s.dist > next[0].dist + 2 || accept(s)) break; }
      if (next[0].dist < bestDist) { bestDist = next[0].dist; stalled = 0; } else if (++stalled >= STALL) break;
      beam = next.slice(0, BEAM);
    }
  }
  if (!found) return { why: "no labelling found" };
  // Undo every change the regex doesn't need, so a label differs from the model's own view as little as possible.
  const roles = found.roles.slice();
  for (let i = 0; i < roles.length; i++) {
    if (roles[i] === start[i]) continue;
    const was = roles[i];
    roles[i] = start[i];
    const r = compileRoles(tokens, roles, lex);
    if (!verdict(r.source, r.flags, c)) roles[i] = was;
  }
  const vetted = vet(tokens, roles, c);
  if (!vetted) return { why: "needs a role on words that don't say so" };
  const bad = misleading(tokens, vetted);
  if (bad) return { why: bad };
  const r = compileRoles(tokens, vetted, lex);
  return { label: { text: c.q, roles: vetted, problem: c.problem, kind: verdict(r.source, r.flags, c), changed: vetted.filter((x, i) => x !== start[i]).length, source: r.source, gold: c.spec } };
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
  const r = await label(c);
  if (r.label) labels.push(r.label);
  else why[r.why] = (why[r.why] ?? 0) + 1;
  if (verbose && r.label) console.log(`${r.label.kind} ${r.label.changed} changed  ${c.q}\n   /${r.label.source}/   gold /${c.goldRegex}/\n   ${tokenize(c.q).map((t, i) => (r.label.roles[i] === "O" ? null : `${t.text}/${r.label.roles[i]}`)).filter(Boolean).join(" ")}`);
  if (verbose && !r.label) console.log(`MISS (${r.why})  ${c.q}\n   gold /${c.goldRegex}/`);
}
const count = (k) => labels.filter((l) => l.kind === k).length;
console.log(`labelled ${labels.length}/${chosen.length} in ${((Date.now() - t0) / 1000).toFixed(0)}s: ${count("as-produced")} as produced, ${count("whole")} whole-string; missed ${JSON.stringify(why)}; ${new Set(labels.map((l) => l.problem)).size} regexes; mean words changed ${(labels.reduce((a, l) => a + l.changed, 0) / (labels.length || 1)).toFixed(1)}`);
if (out && !out.startsWith("-")) writeFileSync(out, labels.map((l) => JSON.stringify(l)).join("\n") + "\n");
