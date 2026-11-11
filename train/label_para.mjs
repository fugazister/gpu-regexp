// Round 10: word roles for paraphrases of grammar prompts. The grammar's gold regex is known exactly, and
// eval/roundtrip.mjs shows the compiler can write it, so a paraphrase is kept only if some labelling compiles to
// exactly that regex and flags. That's stricter than agreeing on sample strings, so no label passes by luck.
// The search and the filters against spurious labels are Round 3's (train/label-lib.mjs).
//   node train/label_para.mjs <paraphrases.jsonl> <out.jsonl> [--shard i/n] [-v]
// Input lines: {id, text, regex, flags} where text is the paraphrase and regex/flags are the original's gold.
import { readFileSync, writeFileSync } from "node:fs";
import { tokenize } from "../web/features.js";
import { label } from "./label-lib.mjs";

const args = process.argv.slice(2);
const flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const [input, out] = args;
const [shard, shards] = flag("--shard", "0/1").split("/").map(Number);
const verbose = args.includes("-v");

const rows = readFileSync(input, "utf8").trim().split("\n").map((l) => JSON.parse(l)).filter((_, i) => i % shards === shard);
const why = {}, labels = [];
const t0 = Date.now();
for (const p of rows) {
  const c = { q: p.text, goldRegex: p.regex, problem: p.id, spec: p.regex };
  c.verdict = (source, flags) => (source === p.regex && flags === p.flags ? "as-produced" : null);
  const r = await label(c);
  if (r.label) labels.push({ ...r.label, id: p.id, flags: p.flags });
  else why[r.why] = (why[r.why] ?? 0) + 1;
  if (verbose) {
    const roles = r.label?.roles;
    console.log(`${roles ? "KEEP" : `MISS (${r.why})`}  ${p.text}\n   gold /${p.regex}/${p.flags}` +
      (roles ? `\n   ${tokenize(p.text).map((t, i) => (roles[i] === "O" ? null : `${t.text}/${roles[i]}`)).filter(Boolean).join(" ")}` : ""));
  }
}
console.log(`kept ${labels.length}/${rows.length} in ${((Date.now() - t0) / 1000).toFixed(0)}s; missed ${JSON.stringify(why)}; ` +
  `mean words changed ${(labels.reduce((a, l) => a + l.changed, 0) / (labels.length || 1)).toFixed(2)}`);
writeFileSync(out, labels.map((l) => JSON.stringify(l)).join("\n") + (labels.length ? "\n" : ""));
