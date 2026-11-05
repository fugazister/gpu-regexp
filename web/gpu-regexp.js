// const gr = await load();
// const { source, flags, regexp, diagnostics, spans } = await gr.parse("three digits");
// const many = await gr.parseMany(["email", "only letters"]);
import { compile as compileProgram } from "./program.js";
import { tokenize, featurize, SLOTS } from "./features.js";
import { compile as compileRoles } from "./compile.js";
import { createCpuBackend } from "./backend-cpu.js";
import { createWebGpuBackend } from "./backend-webgpu.js";

export const MAX_TOKENS = 48;
/** Below this many prompts, GPU dispatch + readback cost more than the parallelism saves. */
export const GPU_MIN_BATCH = 32;

export function decodeWeights(spec, bytes) {
  const q = new Int8Array(bytes), w = new Float32Array(spec.n_weights);
  for (const { offset, shape, scale } of Object.values(spec.tensors)) {
    const size = shape.reduce((a, b) => a * b, 1);
    for (let i = 0; i < size; i++) w[offset + i] = q[offset + i] * scale;
  }
  return w;
}

function softmaxMax(logits, at, n) {
  let best = 0, mx = -Infinity, sum = 0;
  for (let k = 0; k < n; k++) if (logits[at + k] > mx) { mx = logits[at + k]; best = k; }
  for (let k = 0; k < n; k++) sum += Math.exp(logits[at + k] - mx);
  return { best, p: 1 / sum };
}

export function createParser(spec, lexicon, backends) {
  const prog = compileProgram(spec, SLOTS);

  async function tag(tokenLists, backend) {
    const B = tokenLists.length, T = Math.max(1, ...tokenLists.map((t) => t.length));
    const rows = new Uint32Array(B * T * SLOTS), lengths = new Uint32Array(B);
    tokenLists.forEach((toks, q) => {
      lengths[q] = toks.length;
      toks.forEach((t, i) => rows.set(featurize(t), (q * T + i) * SLOTS));
    });
    const logits = await backend.run(prog, rows, lengths, T);
    return tokenLists.map((toks, q) => toks.map((_, i) => {
      const { best, p } = softmaxMax(logits, (q * T + i) * prog.nRoles, prog.nRoles);
      return { role: spec.roles[best], p };
    }));
  }

  function finish(tokens, tagged, truncated) {
    const roles = tagged.map((t) => t.role);
    const out = compileRoles(tokens, roles, lexicon);
    if (truncated) out.diagnostics.unshift(`only the first ${MAX_TOKENS} words were read`);
    let regexp = null;
    if (out.source !== null) {
      // .source escapes "/" so the result can be pasted as a /literal/.
      try { regexp = new RegExp(out.source, out.flags); out.source = regexp.source; }
      catch (e) { out.diagnostics.push("compiled to an invalid regex: " + e.message); out.source = null; }
    }
    const confidence = tagged.length ? Math.min(...tagged.map((t) => t.p)) : 0;
    return { ...out, regexp, tokens, roles, confidence };
  }

  const pick = (choice, B) => {
    if (choice === "cpu" || !backends.webgpu) {
      if (choice === "webgpu") throw new Error("WebGPU isn't available here");
      return backends.cpu;
    }
    return choice === "webgpu" || B >= GPU_MIN_BATCH ? backends.webgpu : backends.cpu;
  };

  async function parseMany(texts, { backend = "auto" } = {}) {
    const all = texts.map(tokenize);
    const lists = all.map((t) => t.slice(0, MAX_TOKENS));
    const used = pick(backend, texts.length);
    const nonEmpty = lists.map((t, i) => (t.length ? i : -1)).filter((i) => i >= 0);
    const tagged = nonEmpty.length ? await tag(nonEmpty.map((i) => lists[i]), used) : [];
    const results = lists.map((toks, i) => finish(toks, [], all[i].length > MAX_TOKENS));
    nonEmpty.forEach((i, k) => { results[i] = finish(lists[i], tagged[k], all[i].length > MAX_TOKENS); });
    return results.map((r) => ({ ...r, backend: used.name }));
  }

  return {
    backends,
    parseMany,
    parse: async (text, opts) => (await parseMany([text], opts))[0],
    tag,
  };
}

/** gpu-query's trick: run a fixed batch through both backends and distrust the GPU if they disagree. */
async function selfCheck(parser) {
  const probe = ["three digits followed by a dash", "lines not containing error", "email", "starts with foo and ends with bar"]
    .map(tokenize);
  const [a, b] = await Promise.all([parser.tag(probe, parser.backends.cpu), parser.tag(probe, parser.backends.webgpu)]);
  return JSON.stringify(a.map((x) => x.map((t) => t.role))) === JSON.stringify(b.map((x) => x.map((t) => t.role)));
}

export async function load({ base = new URL(".", import.meta.url), adapterOptions, webgpu = true } = {}) {
  const [spec, lexicon, bytes] = await Promise.all([
    fetch(new URL("model.json", base)).then((r) => r.json()),
    fetch(new URL("lexicon.json", base)).then((r) => r.json()),
    fetch(new URL("model.bin", base)).then((r) => r.arrayBuffer()),
  ]);
  const weights = decodeWeights(spec, bytes);
  const backends = { cpu: createCpuBackend(weights), webgpu: null };
  let gpuError = null;
  if (webgpu) {
    try { backends.webgpu = await createWebGpuBackend(weights, { adapterOptions }); }
    catch (e) { gpuError = e.message; }
  }
  const parser = createParser(spec, lexicon, backends);
  if (backends.webgpu && !(await selfCheck(parser).catch(() => false))) {
    backends.webgpu = null;
    gpuError = "the GPU's answers disagreed with the CPU's, so it isn't being used";
  }
  return { ...parser, gpuError, modelBytes: bytes.byteLength };
}
