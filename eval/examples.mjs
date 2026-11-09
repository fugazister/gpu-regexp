// Does giving a few example strings fix wrong answers? Scored on StructuredRegex with held-out strings:
// each description's strings are split, one part goes to parseWithExamples, and only the rest decide pass/fail.
//   node eval/examples.mjs dev [-v] [--json out.json]
// Only dev may be read. teste and testi print totals.
import { writeFileSync } from "node:fs";
import { loadCpu } from "./lib.mjs";
import { goldMatches, inScope, readCases } from "./streg-lib.mjs";

const args = process.argv.slice(2);
const split = args[0] ?? "dev";
const verbose = args.includes("-v") && split === "dev";
const jsonAt = args.indexOf("--json");
const { parser } = loadCpu();

const cases = readCases(split).filter((c) => c.yes.every((s) => goldMatches(c.ast, s)) && c.no.every((s) => !goldMatches(c.ast, s)));
// Picking strings come from the front, scoring strings from the back, so no string is used for both.
const SETTINGS = { "1+1": 1, "3+3": 3 };
const behaves = (re, yes, no) => !!re && yes.every((s) => re.test(s)) && no.every((s) => !re.test(s));

const per = {};
for (const [name, k] of Object.entries(SETTINGS)) per[name] = [];
per.plain = [];
for (const c of cases) {
  // Scoring always uses the last 3 of each, so every setting is graded on the same strings.
  const scoreYes = c.yes.slice(3), scoreNo = c.no.length > 3 ? c.no.slice(3) : c.no.slice(-1);
  const plain = await parser.parse(c.q);
  per.plain.push(behaves(plain.regexp, scoreYes, scoreNo));
  for (const [name, k] of Object.entries(SETTINGS)) {
    const pickNo = c.no.slice(0, Math.min(k, Math.max(0, c.no.length - scoreNo.length)));
    const r = await parser.parseWithExamples(c.q, { yes: c.yes.slice(0, k), no: pickNo });
    const ok = behaves(r.regexp, scoreYes, scoreNo);
    per[name].push(ok);
    if (verbose && k === 3 && ok !== per.plain.at(-1)) console.log(`${ok ? "FIXED" : "BROKE"}  ${c.q}\n      plain /${plain.source}/  ->  /${r.source}/  ${r.diagnostics.slice(-1)[0] ?? ""}`);
  }
}

const scope = cases.map(inScopeOf);
function inScopeOf(c) { return inScope(c.ast); }
const out = { split, n: cases.length, rows: {} };
for (const [name, list] of Object.entries(per)) {
  const all = list.filter(Boolean).length, sc = list.filter((x, i) => x && scope[i]).length;
  out.rows[name] = { all, inScope: sc };
  let line = `${split} ${name.padEnd(5)} ${String(all).padStart(4)}/${cases.length} all, ${String(sc).padStart(3)}/${scope.filter(Boolean).length} in scope`;
  if (name !== "plain") {
    const gained = list.filter((x, i) => x && !per.plain[i]).length, lost = list.filter((x, i) => !x && per.plain[i]).length;
    line += `  (vs plain: ${gained} gained, ${lost} lost)`;
    out.rows[name].gained = gained; out.rows[name].lost = lost;
  }
  console.log(line);
}
if (jsonAt >= 0) writeFileSync(args[jsonAt + 1], JSON.stringify({ ...out, per }, null, 1));
