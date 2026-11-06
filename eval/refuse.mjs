// Measures refusal: how often input that isn't a pattern request gets no regex, and what that costs on real requests.
//   node eval/refuse.mjs dev [-v]             sweep of confidence thresholds, calibration, failures (read freely)
//   node eval/refuse.mjs test --threshold T   totals at one threshold chosen on dev
// A prompt is refused when it compiles to no regex or its confidence is below the threshold.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { loadCpu, readJson } from "./lib.mjs";

const args = process.argv.slice(2);
const split = args[0] ?? "dev";
const flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const threshold = Number(flag("--threshold", 0));
const verbose = args.includes("-v") && split === "dev";
const { parser } = loadCpu();

const readStreg = (name) => {
  const url = new URL(`external/streg/${name}.tsv`, import.meta.url);
  if (!existsSync(url)) return [];
  return readFileSync(url, "utf8").trim().split(/\r?\n/).slice(1).map((line) => {
    const [, raw, , pos, neg] = line.split("\t");
    const q = /^".*"$/.test(raw) ? raw.slice(1, -1).replace(/""/g, '"') : raw;
    return { c: "streg", q, yes: pos.split(" "), no: neg.split(" ") };
  });
};

async function run(cases) {
  const out = [];
  for (const c of cases) {
    const r = await parser.parse(c.q);
    const pass = !!r.regexp && !!c.yes && c.yes.every((s) => r.regexp.test(s)) && c.no.every((s) => !r.regexp.test(s));
    out.push({ ...c, source: r.source, none: !r.regexp, confidence: r.confidence, pass });
  }
  return out;
}
const refused = (x, t) => x.none || x.confidence < t;
const pct = (a, b) => (b ? ((100 * a) / b).toFixed(1) : "-");

const garbage = await run(readJson(`eval/refuse-${split}.json`));
const valid = await run(readJson(`eval/${split}-v2.json`));
const streg = await run(split === "dev" ? readStreg("dev") : [...readStreg("teste"), ...readStreg("testi")]);

function summary(t) {
  const kept = valid.filter((x) => !refused(x, t));
  const answered = streg.filter((x) => !refused(x, t));
  return {
    threshold: t,
    garbageRefused: garbage.filter((x) => refused(x, t)).length, garbageTotal: garbage.length,
    validPassed: kept.filter((x) => x.pass).length, validTotal: valid.length,
    validRefused: valid.length - kept.length, goodAnswersLost: valid.filter((x) => x.pass && refused(x, t)).length,
    stregAnswered: answered.length, stregRight: answered.filter((x) => x.pass).length, stregTotal: streg.length,
  };
}
const line = (s) =>
  `t=${s.threshold.toFixed(2)}  non-requests refused ${s.garbageRefused}/${s.garbageTotal} (${pct(s.garbageRefused, s.garbageTotal)}%) | ` +
  `real requests still passing ${s.validPassed}/${s.validTotal}, refused ${s.validRefused}, good answers lost ${s.goodAnswersLost} | ` +
  `StructuredRegex answered ${s.stregAnswered}/${s.stregTotal}, right ${s.stregRight} (${pct(s.stregRight, s.stregAnswered)}%)`;

const cats = {};
for (const x of garbage) {
  cats[x.c] ??= [0, 0];
  cats[x.c][0] += refused(x, threshold) ? 1 : 0;
  cats[x.c][1]++;
}
console.log(`refused by category at t=${threshold}: ` + Object.entries(cats).map(([k, [a, b]]) => `${k} ${a}/${b}`).join(", "));

const results = {};
if (split === "dev") {
  results.sweep = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995].map(summary);
  results.sweep.forEach((s) => console.log(line(s)));
  // Calibration: when the model reports confidence p, how often is the answer right?
  console.log("calibration on real requests (own dev + StructuredRegex dev):");
  const bins = [[0, 0.5], [0.5, 0.8], [0.8, 0.95], [0.95, 0.99], [0.99, 1.01]];
  const scored = [...valid, ...streg].filter((x) => !x.none);
  results.calibration = bins.map(([lo, hi]) => {
    const inBin = scored.filter((x) => x.confidence >= lo && x.confidence < hi);
    const mean = inBin.reduce((a, x) => a + x.confidence, 0) / (inBin.length || 1);
    const right = inBin.filter((x) => x.pass).length;
    console.log(`  confidence ${lo.toFixed(2)}–${Math.min(hi, 1).toFixed(2)}: ${String(inBin.length).padStart(3)} answers, mean confidence ${mean.toFixed(3)}, actually right ${pct(right, inBin.length)}%`);
    return { lo, hi, n: inBin.length, mean, right };
  });
  results.garbageConfidence = garbage.filter((x) => !x.none).map((x) => x.confidence).sort((a, b) => a - b);
} else {
  results.at = summary(threshold);
  console.log(line(results.at));
}
if (verbose) {
  for (const x of garbage) if (!refused(x, threshold)) console.log(`ANSWERED  ${x.q}\n          /${x.source}/  ${x.confidence.toFixed(2)}`);
  for (const x of valid) if (refused(x, threshold)) console.log(`REFUSED   ${x.q}  ${x.none ? "no regex" : x.confidence.toFixed(2)}`);
}
results.cats = cats;
if (args.includes("--json")) writeFileSync(flag("--json"), JSON.stringify(results, null, 1));
