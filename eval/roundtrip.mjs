// Invariant: compile(gold roles) must equal the generator's gold regex, and JS features must equal
// Python's. Anything below 100% would mean model scores are partly measuring compiler bugs.
import { readFileSync } from "node:fs";
import { tokenize, featurize } from "../web/features.js";
import { compile } from "../web/compile.js";

const read = (p) => JSON.parse(readFileSync(new URL("../" + p, import.meta.url)));
const lex = read("web/lexicon.json");
const { cases } = read("eval/oracle.json");
const show = Number(process.argv[2] ?? 12);
let bad = 0, featBad = 0, invalid = 0, diag = 0;
for (const c of cases) {
  const toks = tokenize(c.text);
  if (JSON.stringify(toks.map(featurize)) !== JSON.stringify(c.features)) {
    if (featBad++ < 3) console.log("FEATURES", JSON.stringify(c.text));
  }
  const r = compile(toks, c.roles, lex);
  if (r.diagnostics.length) diag++;
  if (r.source !== null) { try { new RegExp(r.source, r.flags); } catch { invalid++; } }
  if (r.source !== c.regex || r.flags !== c.flags) {
    if (bad++ < show) {
      console.log(`MISMATCH ${JSON.stringify(c.text)}`);
      console.log("   " + toks.map((t, i) => (c.roles[i] === "O" ? "" : `${t.text}/${c.roles[i]}`)).filter(Boolean).join(" "));
      console.log(`   gold /${c.regex}/${c.flags}\n   got  /${r.source}/${r.flags}  ${r.diagnostics.join("; ")}`);
    }
  }
}
console.log(`${cases.length} cases: ${((100 * (cases.length - bad)) / cases.length).toFixed(2)}% round-trip, ${bad} mismatches, ${featBad} feature mismatches, ${invalid} invalid regexes, ${diag} with diagnostics`);
process.exit(bad || featBad || invalid ? 1 : 0);
