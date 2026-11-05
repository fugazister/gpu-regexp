// Paired comparison of two score files on the same prompts (exact sign test / McNemar).
// Only prompts where the runs disagree carry information about which one is better.
//   node eval/compare.mjs results/a.json results/b.json [--show]
import { readFileSync } from "node:fs";

const [a, b] = process.argv.slice(2, 4).map((f) => JSON.parse(readFileSync(f)));
const show = process.argv.includes("--show") && !/test|human/.test(a.file);
if (a.file !== b.file) throw new Error(`different eval sets: ${a.file} vs ${b.file}`);
const wins = [], losses = [];
for (const q of Object.keys(a.per)) {
  if (a.per[q] && !b.per[q]) losses.push(q);
  if (!a.per[q] && b.per[q]) wins.push(q);
}
function choose(n, k) { let r = 1; for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i; return r; }
const n = wins.length + losses.length, k = Math.min(wins.length, losses.length);
let p = 0;
for (let i = 0; i <= k; i++) p += choose(n, i);
p = n ? Math.min(1, (2 * p) / 2 ** n) : 1;
console.log(`${a.pass}/${a.total} -> ${b.pass}/${b.total}: ${wins.length} gained, ${losses.length} lost, two-sided p=${p.toFixed(4)}`);
if (show) {
  for (const q of wins) console.log("  + " + q);
  for (const q of losses) console.log("  - " + q);
}
