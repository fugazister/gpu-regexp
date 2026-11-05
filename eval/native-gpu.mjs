// Runs the shipped WGSL on a real GPU outside the browser, through Dawn (Chrome's WebGPU engine).
// For machines where headless Chrome can't reach the GPU. Setup: npm i --no-save webgpu
import { create, globals } from "webgpu";
import { readFileSync } from "node:fs";
import { decodeWeights, createParser } from "../web/gpu-regexp.js";
import { createCpuBackend } from "../web/backend-cpu.js";
import { createWebGpuBackend } from "../web/backend-webgpu.js";
import { tokenize } from "../web/features.js";

Object.assign(globalThis, globals);
Object.defineProperty(globalThis.navigator, "gpu", { value: create([]) });

const read = (p) => readFileSync(new URL("../" + p, import.meta.url));
const spec = JSON.parse(read("web/model.json"));
const buf = read("web/model.bin");
const weights = decodeWeights(spec, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));

const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
console.log("adapter:", adapter.info.vendor, "|", adapter.info.architecture, "|", adapter.info.device, "|", adapter.info.description);
if (/swiftshader|llvmpipe|lavapipe/i.test(`${adapter.info.description} ${adapter.info.device}`)) {
  console.error("software adapter, refusing to report these as GPU numbers");
  process.exit(3);
}

const t0 = performance.now();
const backends = { cpu: createCpuBackend(weights), webgpu: await createWebGpuBackend(weights, { adapterOptions: { powerPreference: "high-performance" } }) };
const parser = createParser(spec, JSON.parse(read("web/lexicon.json")), backends);
const cases = JSON.parse(read("eval/fixtures.json"));
const lists = cases.map((c) => tokenize(c.q).slice(0, 48));
await parser.tag(lists, backends.webgpu);
console.log(`cold start (device, pipelines, weights, first batch): ${(performance.now() - t0).toFixed(0)} ms`);

const [a, b] = [await parser.tag(lists, backends.cpu), await parser.tag(lists, backends.webgpu)];
let roleDiff = 0, maxP = 0;
b.forEach((seq, q) => seq.forEach((t, i) => {
  if (t.role !== a[q][i].role) roleDiff++;
  maxP = Math.max(maxP, Math.abs(t.p - a[q][i].p));
}));
console.log(`${lists.length} prompts: ${roleDiff} role differences, max confidence diff ${maxP.toFixed(6)}`);

const median = async (batch, backend) => {
  await parser.tag(batch, backend);
  const ts = [];
  for (let k = 0; k < 7; k++) { const t = performance.now(); await parser.tag(batch, backend); ts.push(performance.now() - t); }
  return ts.sort((x, y) => x - y)[3];
};
console.log("prompts   cpu ms   gpu ms   gpu speedup");
for (const n of [1, 32, 256, 1024, 4096, 16384]) {
  const batch = Array.from({ length: n }, (_, i) => lists[i % lists.length]);
  const c = await median(batch, backends.cpu), g = await median(batch, backends.webgpu);
  console.log(`${String(n).padStart(7)} ${c.toFixed(1).padStart(8)} ${g.toFixed(1).padStart(8)} ${(c / g).toFixed(1).padStart(10)}x`);
}
process.exit(roleDiff ? 1 : 0);
