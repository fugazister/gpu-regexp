// Scores the model on StructuredRegex (Ye et al.), whose descriptions were written by crowdworkers who never
// saw this project's grammar. Each description comes with strings that should and shouldn't match.
//   mkdir -p eval/external/streg && for s in dev teste testi; do
//     gh api repos/xiye17/StructuredRegex/contents/data/raw/$s.tsv -H 'Accept: application/vnd.github.raw' > eval/external/streg/$s.tsv
//   done
//   node eval/streg.mjs dev [-v] [--json out.json]
// Only dev may be read. teste and testi print totals, so they can stay an independent test.
import { writeFileSync } from "node:fs";
import { loadCpu } from "./lib.mjs";
import { goldMatches, inScope, readCases } from "./streg-lib.mjs";

const args = process.argv.slice(2);
const split = args[0] ?? "dev";
const verbose = args.includes("-v") && split === "dev";
const jsonAt = args.indexOf("--json");
const MAX_TOKENS = 48;

// ---- scoring ----
const cases = readCases(split);
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
if (jsonAt >= 0) writeFileSync(args[jsonAt + 1], JSON.stringify({
  file: `streg-${split}`, pass: valid.filter((c) => c.asIs).length, total: valid.length, goldOk, rows,
  per: Object.fromEntries(valid.map((c) => [`${c.problem} ${c.q}`, c.asIs])),
}, null, 1));
