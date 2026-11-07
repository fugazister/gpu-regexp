// Refusal on real prose that nobody wrote for this project, as a check on the hand-written refusal sets.
//   node eval/refuse-real.mjs [-v] [--json out.json] file.txt ...
// Each file is split into sentences. None of them asks for a pattern, so any regex is a wrong answer.
import { readFileSync, writeFileSync } from "node:fs";
import { loadCpu } from "./lib.mjs";

const args = process.argv.slice(2);
const verbose = args.includes("-v");
const jsonAt = args.indexOf("--json");
const json = jsonAt >= 0 ? args[jsonAt + 1] : null;
const files = args.filter((a, i) => a !== "-v" && (jsonAt < 0 || (i !== jsonAt && i !== jsonAt + 1)));
const { parser } = loadCpu();

const LIMIT = 300;
const sentences = (text) => {
  const all = text.replace(/\s+/g, " ").split(/(?<=[.;:!?])\s+/).map((s) => s.trim());
  const usable = [...new Set(all)].filter((s) => {
    const words = s.split(" ");
    return words.length >= 4 && words.length <= 30 && /^[A-Za-z]/.test(s);
  });
  // An even spread through the file, so the cap doesn't only keep the opening.
  const step = Math.max(1, usable.length / LIMIT);
  return Array.from({ length: Math.min(LIMIT, usable.length) }, (_, i) => usable[Math.floor(i * step)]);
};

const THRESHOLDS = [0, 0.5, 0.6, 0.7, 0.8, 0.9];
const LENGTHS = [[4, 8], [9, 15], [16, 30]];
const rows = [];
for (const file of files) {
  for (const q of sentences(readFileSync(file, "utf8"))) {
    const r = await parser.parse(q);
    rows.push({ file: file.split("/").pop(), q, words: q.split(" ").length, none: !r.regexp, source: r.source, confidence: r.confidence });
  }
}
const refused = (x, t) => x.none || x.confidence < t;
const rate = (list, t) => `${list.filter((x) => refused(x, t)).length}/${list.length}`;
const out = { perFile: {}, perLength: {}, sweep: {} };
for (const file of new Set(rows.map((x) => x.file))) out.perFile[file] = rate(rows.filter((x) => x.file === file), 0);
for (const [lo, hi] of LENGTHS) out.perLength[`${lo}-${hi} words`] = rate(rows.filter((x) => x.words >= lo && x.words <= hi), 0);
for (const t of THRESHOLDS) out.sweep[t] = rate(rows, t);
console.log(JSON.stringify(out));
if (verbose) for (const x of rows.filter((x) => !x.none).slice(0, 12)) console.log(`  ANSWERED ${x.q}\n           /${x.source}/  ${x.confidence.toFixed(2)}`);
if (json) writeFileSync(json, JSON.stringify(out, null, 1));
