// Runs the real WGSL backend in headless Chrome and compares it to the CPU backend in the same page,
// then times batches on both. No dependencies: a tiny static server + a page that POSTs back its results.
//   node eval/webgpu.mjs            software adapter (SwiftShader): proves correctness, timings mean nothing
//   node eval/webgpu.mjs --real     the machine's real GPU, if Chrome can reach it
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, normalize } from "node:path";

const real = process.argv.includes("--real");
const root = new URL("../", import.meta.url).pathname;
const TYPES = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html", ".bin": "application/octet-stream" };

const PAGE = `<!doctype html><script type=module>
import { load } from "/web/gpu-regexp.js";
import { tokenize } from "/web/features.js";
const post = (o) => fetch("/result", { method: "POST", body: JSON.stringify(o) });
try {
  const gr = await load({ base: new URL("/web/", location.href), adapterOptions: { powerPreference: "high-performance" } });
  if (!gr.backends.webgpu) throw new Error("WebGPU unavailable: " + gr.gpuError);
  const cases = await (await fetch("/eval/fixtures.json")).json();
  const lists = cases.map((c) => tokenize(c.q).slice(0, 48));

  // Correctness: same roles, near-identical confidence, batched and one at a time.
  const [cpuB, gpuB] = [await gr.tag(lists, gr.backends.cpu), await gr.tag(lists, gr.backends.webgpu)];
  let roleDiff = 0, maxP = 0, singleDiff = 0;
  gpuB.forEach((seq, q) => seq.forEach((t, i) => {
    if (t.role !== cpuB[q][i].role) roleDiff++;
    maxP = Math.max(maxP, Math.abs(t.p - cpuB[q][i].p));
  }));
  for (let q = 0; q < lists.length; q++) {
    const one = (await gr.tag([lists[q]], gr.backends.webgpu))[0];
    one.forEach((t, i) => { if (t.role !== gpuB[q][i].role) singleDiff++; });
  }

  // Throughput, warm: tagging only, so compiler time isn't counted.
  const timing = [];
  for (const n of [1, 32, 256, 1024, 4096]) {
    const batch = Array.from({ length: n }, (_, i) => lists[i % lists.length]);
    const time = async (b) => { await gr.tag(batch, b); let best = Infinity;
      for (let k = 0; k < 3; k++) { const t = performance.now(); await gr.tag(batch, b); best = Math.min(best, performance.now() - t); }
      return +best.toFixed(2); };
    timing.push({ prompts: n, cpuMs: await time(gr.backends.cpu), gpuMs: await time(gr.backends.webgpu) });
  }
  await post({ adapter: gr.backends.webgpu.adapter, compared: lists.length, roleDiff, singleDiff, maxConfidenceDiff: +maxP.toFixed(6), timing });
} catch (e) { await post({ error: String(e?.stack ?? e) }); }
</script>`;

const profile = mkdtempSync(join(tmpdir(), "gpu-regexp-chrome-"));
let chrome;
const finish = (code) => {
  chrome?.kill();
  // Chrome keeps writing to its profile for a moment after the kill.
  setTimeout(() => { rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }); process.exit(code); }, 500);
};

const server = createServer(async (req, res) => {
  const path = new URL(req.url, "http://x").pathname;
  if (path === "/") return res.writeHead(200, { "content-type": "text/html" }).end(PAGE);
  if (path === "/result") {
    let body = "";
    for await (const chunk of req) body += chunk;
    res.end();
    const r = JSON.parse(body);
    console.dir(r, { depth: null });
    return finish(!r.error && r.roleDiff === 0 && r.singleDiff === 0 ? 0 : 1);
  }
  try {
    const data = await readFile(join(root, normalize(path)));
    res.writeHead(200, { "content-type": TYPES[path.slice(path.lastIndexOf("."))] ?? "application/octet-stream" }).end(data);
  } catch { res.writeHead(404).end(); }
});

server.listen(0, "127.0.0.1", () => {
  const adapter = real ? ["--use-angle=vulkan", "--ignore-gpu-blocklist"] : ["--use-webgpu-adapter=swiftshader"];
  chrome = spawn(process.env.CHROME ?? "google-chrome-stable", [
    "--headless=new", "--no-first-run", `--user-data-dir=${profile}`,
    "--enable-unsafe-webgpu", "--enable-features=Vulkan", ...adapter,
    `http://127.0.0.1:${server.address().port}/`,
  ], { stdio: "ignore" });
});
setTimeout(() => { console.error("timed out waiting for Chrome"); finish(2); }, 300000);
