// Scores prompts by behaviour: a prompt passes if its regex matches every "yes" string and no "no" string.
//   node eval/score.mjs eval/dev-v2.json [-v] [--json out.json]
// Test sets only print totals unless you pass --peek: reading their failures turns them into dev sets.
import { writeFileSync } from "node:fs";
import { loadCpu, readJson } from "./lib.mjs";

const args = process.argv.slice(2);
const file = args[0] ?? "eval/dev-v2.json";
const peek = !/test|human/.test(file) || args.includes("--peek");
const verbose = args.includes("-v") && peek;
const jsonAt = args.indexOf("--json");
const { parser } = loadCpu();
const cases = readJson(file);
if (!cases.length) {
  console.log(`${file}: no prompts yet`);
  process.exit(0);
}

/** 95% Wilson interval: honest about how little a small set can tell you. */
function wilson(k, n) {
  const z = 1.96, p = k / n, d = 1 + (z * z) / n;
  const mid = (p + (z * z) / (2 * n)) / d;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [100 * (mid - half), 100 * (mid + half)];
}

const per = {}, cats = {};
let pass = 0, none = 0;
for (const c of cases) {
  const r = await parser.parse(c.q);
  const ok = !!r.regexp && c.yes.every((s) => r.regexp.test(s)) && c.no.every((s) => !r.regexp.test(s));
  if (!r.regexp) none++;
  if (ok) pass++;
  per[c.q] = ok;
  const k = c.c ?? "-";
  cats[k] ??= [0, 0];
  cats[k][0] += ok ? 1 : 0;
  cats[k][1]++;
  if (peek && (verbose || !ok)) console.log(`${ok ? "pass" : "FAIL"}  ${c.q}\n      /${r.source}/${r.flags}${r.diagnostics.length ? "  | " + r.diagnostics.join("; ") : ""}`);
}
const [lo, hi] = wilson(pass, cases.length);
const pct = (100 * pass) / cases.length;
console.log(`${file}: ${pass}/${cases.length} (${pct.toFixed(1)}%, 95% CI ${lo.toFixed(1)}–${hi.toFixed(1)}), ${none} no regex`);
if (Object.keys(cats).length > 1) console.log("  " + Object.entries(cats).map(([k, [a, b]]) => `${k} ${a}/${b}`).join(", "));
if (jsonAt >= 0) writeFileSync(args[jsonAt + 1], JSON.stringify({ file, pass, total: cases.length, pct, ci: [lo, hi], cats, per }, null, 1));
