// Shared by train/label_real.mjs (Round 3) and train/label_para.mjs (Round 10): a beam search over word roles,
// starting from the model's own roles, until the compiled regex is accepted by the case's verdict, plus the
// filters that reject labels which are right for the wrong reason.
import { readFileSync } from "node:fs";
import { compile as compileProgram } from "../web/program.js";
import { featurize, SLOTS, tokenize } from "../web/features.js";
import { compact, compile as compileRoles } from "../web/compile.js";
import { createCpuBackend } from "../web/backend-cpu.js";
import { decodeWeights } from "../web/gpu-regexp.js";

export const BEAM = 3, MAX_STEPS = 16, STALL = 3, MAX_TOKENS = 48;
export const WINDOW_ROLES = ["O", "THEN", "START", "END", "Q_PLUS", "Q_STAR", "Q_OPT", "Q_MIN", "Q_MAX", "Q_RANGE", "NOT_START", "NOT_END", "NOT_CONTAIN", "OR"];

const root = new URL("../", import.meta.url);
export const spec = JSON.parse(readFileSync(new URL("web/model.json", root)));
export const lex = JSON.parse(readFileSync(new URL("web/lexicon.json", root)));
const bin = readFileSync(new URL("web/model.bin", root));
const backend = createCpuBackend(decodeWeights(spec, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)));
const prog = compileProgram(spec, SLOTS);

export async function logProbs(tokens) {
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

// ---- search ----
export function editDistance(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)));
    prev = cur;
  }
  return prev[b.length];
}
export const stripAnchors = (s) => s.replace(/^\^/, "").replace(/\$$/, "");

export function moves(tokens, lp) {
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
export function saidSo(role, words) {
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
export function vet(tokens, roles, c) {
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
  return c.verdict(r.source, r.flags) ? out : null;
}

// Labels that compile to the right regex but would teach something false about common phrases.
// Class words next to a literal can be filler ("the letter S", "3 capital D's"); elsewhere they say what to match.
const ATOM_WORDS = /^(digits?|numbers?|numerals?|letters?|characters?|chars?|capitals?|uppercase|lowercase|vowels?|consonants?|spaces?|symbols?)$/;

export function misleading(tokens, roles) {
  const low = tokens.map((t) => t.text.toLowerCase());
  const nextRole = (i) => { while (roles[++i] === "O"); return roles[i]; };
  for (let i = 0; i < low.length; i++) {
    // "ends with" labelled as filler: in this dataset "starts ... and ends ..." can leave no gap, unlike the grammar's reading.
    if (/^(starts?|starting|begins?|beginning|ends?|ending|ended)$/.test(low[i]) && /^(with|in)$/.test(low[i + 1] ?? "") &&
        !(roles[i] === roles[i + 1] && /START|END/.test(roles[i]))) return "drops an anchor";
    if (roles[i] === "Q_PLUS" && low[i] === "string" && ["NUM", "Q_MIN", "Q_MAX", "Q_RANGE", "Q_OPT", "Q_STAR"].includes(nextRole(i))) return "a string of N";
    if (roles[i] === "ONLY" && low[i] === "is" && low[i - 1] === "that" && nextRole(i) === "THEN") return "that is followed by";
    // E11: words that carry meaning but were labelled as filler, so the label passes only by luck.
    if (roles[i] === "NUM" && !/^\d+$/.test(low[i]) && low[i].length < 3) return "a letter read as a number";
    // "4 more letters" means another 4, so only "more than", "or more", "at least" and the like count.
    if (roles[i] === "O" && (low[i] === "than" || (low[i] === "least" && low[i - 1] === "at") ||
        (/^(more|fewer|less)$/.test(low[i]) && (low[i - 1] === "or" || low[i + 1] === "than")))) return "drops a quantity word";
    // Fine next to a literal ("the letter S"), another class word ("letters in lowercase"), a negation
    // ("characters that are not letters"), or as "a number of".
    if (roles[i] === "O" && ATOM_WORDS.test(low[i]) && !(low[i] === "number" && low[i + 1] === "of") &&
        ![-2, -1, 1, 2].some((d) => roles[i + d] === "LIT" || roles[i + d] === "ATOM") &&
        ![1, 2, 3, 4].some((d) => roles[i + d] === "NEG")) return "drops a character class";
  }
  return null;
}

export async function label(c) {
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
  const accept = (s) => { kind = c.verdict(s.source, s.flags); if (kind) found = s; return !!kind; };
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
    if (!c.verdict(r.source, r.flags)) roles[i] = was;
  }
  const vetted = vet(tokens, roles, c);
  if (!vetted) return { why: "needs a role on words that don't say so" };
  const bad = misleading(tokens, vetted);
  if (bad) return { why: bad };
  const r = compileRoles(tokens, vetted, lex);
  return { label: { text: c.q, roles: vetted, problem: c.problem, kind: c.verdict(r.source, r.flags), changed: vetted.filter((x, i) => x !== start[i]).length, source: r.source, gold: c.spec } };
}

