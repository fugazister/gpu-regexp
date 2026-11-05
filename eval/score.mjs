// Scores prompts by behaviour: a prompt passes if its regex matches every "yes" string and no "no" string.
// usage: node eval/score.mjs eval/dev.json [-v]
import { loadCpu, readJson } from "./lib.mjs";

const file = process.argv[2] ?? "eval/dev.json";
const verbose = process.argv.includes("-v");
const { parser } = loadCpu();
const cases = readJson(file);
let pass = 0;
for (const c of cases) {
  const r = await parser.parse(c.q);
  const ok = r.regexp && c.yes.every((s) => r.regexp.test(s)) && c.no.every((s) => !r.regexp.test(s));
  if (ok) pass++;
  if (verbose || !ok) console.log(`${ok ? "pass" : "FAIL"}  ${c.q}\n      /${r.source}/${r.flags}${r.error ? "  " + r.error : ""}`);
}
console.log(`${file}: ${pass}/${cases.length} (${((100 * pass) / cases.length).toFixed(1)}%)`);
