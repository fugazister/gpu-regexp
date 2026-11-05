// Checks that the JS CPU backend reproduces the torch logits in eval/fixtures.json, singly and batched.
import { compile } from "../web/program.js";
import { createCpuBackend } from "../web/backend-cpu.js";
import { tokenize, featurize, SLOTS } from "../web/features.js";
import { loadCpu, readJson } from "./lib.mjs";

const { spec, weights } = loadCpu();
const cpu = createCpuBackend(weights);
const prog = compile(spec, SLOTS);
const cases = readJson("eval/fixtures.json");
let featBad = 0, maxDiff = 0, batchDiff = 0;

const single = [];
for (const c of cases) {
  const feats = tokenize(c.q).slice(0, 48).map(featurize);
  if (JSON.stringify(feats) !== JSON.stringify(c.rows)) { featBad++; console.log("FEATURES", JSON.stringify(c.q)); }
  const T = feats.length;
  const logits = await cpu.run(prog, new Uint32Array(feats.flat()), new Uint32Array([T]), T);
  single.push(Float32Array.from(logits));
  c.logits.forEach((row, i) => row.forEach((v, k) => {
    maxDiff = Math.max(maxDiff, Math.abs(v - logits[i * prog.nRoles + k]));
  }));
}

// Padding must not change anything: a batch has to give the same logits as one-at-a-time.
const T = Math.max(...cases.map((c) => c.rows.length));
const rows = new Uint32Array(cases.length * T * SLOTS), lengths = new Uint32Array(cases.length);
cases.forEach((c, q) => { lengths[q] = c.rows.length; rows.set(c.rows.flat(), q * T * SLOTS); });
const batched = await cpu.run(prog, rows, lengths, T);
cases.forEach((c, q) => {
  for (let i = 0; i < c.rows.length * prog.nRoles; i++)
    batchDiff = Math.max(batchDiff, Math.abs(batched[q * T * prog.nRoles + i] - single[q][i]));
});

console.log(`${cases.length} cases: ${featBad} feature mismatches, max torch-vs-JS logit diff ${maxDiff.toFixed(5)}, max batch-vs-single diff ${batchDiff.toFixed(6)}`);
process.exit(featBad || maxDiff > 1e-2 || batchDiff > 1e-3 ? 1 : 0);
