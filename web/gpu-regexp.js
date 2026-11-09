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

export const READINGS = 64;

/** Labellings in order of probability: each word's role is picked from its few most likely, best total first. */
export function* readings(tagged, roleNames, limit = READINGS) {
  const opts = tagged.map((t) => t.lp.map((lp, k) => ({ role: roleNames[k], lp })).sort((a, b) => b.lp - a.lp)
    .filter((o, j) => j === 0 || o.lp > -7).slice(0, 3));
  const score = (ranks) => ranks.reduce((a, r, i) => a + opts[i][r].lp, 0);
  const start = tagged.map(() => 0);
  const queue = [{ ranks: start, s: score(start) }], seen = new Set([start.join()]);
  for (let n = 0; n < limit && queue.length; n++) {
    let bi = 0;
    for (let i = 1; i < queue.length; i++) if (queue[i].s > queue[bi].s) bi = i;
    const { ranks, s } = queue.splice(bi, 1)[0];
    yield { roles: ranks.map((r, i) => opts[i][r].role), lp: s };
    for (let i = 0; i < ranks.length; i++) {
      if (ranks[i] + 1 >= opts[i].length) continue;
      const next = ranks.slice(); next[i]++;
      if (seen.has(next.join())) continue;
      seen.add(next.join());
      queue.push({ ranks: next, s: score(next) });
    }
  }
}

/** The first reading, as written or then anchored, whose regex passes `fits`. */
export function searchReadings(tokens, tagged, roleNames, lexicon, fits) {
  let rank = 0, tried = 0;
  const seen = new Set();
  for (const { roles } of readings(tagged, roleNames)) {
    const out = compileRoles(tokens, roles, lexicon);
    const s = out.source;
    if (s !== null && !seen.has(s)) {
      seen.add(s);
      const variants = [[s, null]];
      const starts = s.startsWith("^"), ends = s.endsWith("$") && !s.endsWith("\\$");
      const body = s.slice(starts ? 1 : 0, ends ? -1 : undefined);
      if (!starts || !ends) variants.push([`^(?:${body})$`, "anchored to the whole string, to fit the examples"]);
      if (!starts) variants.push([`^(?:${s})`, "anchored at the start, to fit the examples"]);
      if (!ends) variants.push([`(?:${s})$`, "anchored at the end, to fit the examples"]);
      for (const [src, wrap] of variants) {
        let re;
        try { re = new RegExp(src, out.flags); } catch { continue; }
        tried++;
        if (fits(re)) return { found: { roles, source: re.source, wrap, rank }, tried };
      }
    }
    rank++;
  }
  return { found: null, tried };
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
      const at = (q * T + i) * prog.nRoles;
      const { best, p } = softmaxMax(logits, at, prog.nRoles);
      // Every role's log-probability, so examples can pick among less likely readings without rerunning the model.
      const row = Array.from(logits.subarray(at, at + prog.nRoles));
      const mx = Math.max(...row), lse = mx + Math.log(row.reduce((a, x) => a + Math.exp(x - mx), 0));
      return { role: spec.roles[best], p, lp: row.map((x) => x - lse) };
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

  /**
   * Like parse, but with strings that should (yes) and shouldn't (no) match. Tries the most likely readings
   * of the prompt, each as written and anchored, and returns the most likely regex that gets every example right.
   * With no examples, or when nothing fits, it's the plain parse plus a note on which examples fail.
   */
  async function parseWithExamples(text, { yes = [], no = [] } = {}, opts = {}) {
    const [plain] = await parseMany([text], opts);
    yes = yes.filter((s) => s !== ""); no = no.filter((s) => s !== "");
    if (!yes.length && !no.length) return plain;
    const tokens = plain.tokens;
    const fits = (re) => yes.every((s) => re.test(s)) && no.every((s) => !re.test(s));
    if (plain.regexp && fits(plain.regexp)) return { ...plain, examples: { fit: true, tried: 1 } };
    const tagged = tokens.length ? (await tag([tokens], pick(opts.backend ?? "auto", 1)))[0] : [];
    const { found, tried } = searchReadings(tokens, tagged, spec.roles, lexicon, fits);
    if (!found) {
      const failing = [...yes.filter((s) => !plain.regexp?.test(s)), ...no.filter((s) => !!plain.regexp?.test(s))];
      return { ...plain, examples: { fit: false, tried, failing } };
    }
    found.tried = tried;
    const out = finish(tokens, found.roles.map((role, i) => ({ role, p: Math.exp(tagged[i].lp[spec.roles.indexOf(role)]) })), false);
    if (found.wrap) {
      out.source = found.source; out.regexp = new RegExp(found.source, out.flags);
      out.diagnostics.push(found.wrap);
    }
    if (found.rank) out.diagnostics.push(`picked reading ${found.rank + 1} of ${READINGS}, the most likely one that fits the examples`);
    return { ...out, backend: plain.backend, examples: { fit: true, tried: found.tried, rank: found.rank } };
  }

  return {
    backends,
    parseMany,
    parse: async (text, opts) => (await parseMany([text], opts))[0],
    parseWithExamples,
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
