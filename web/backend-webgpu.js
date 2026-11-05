// WebGPU executor: one dispatch per op. Dense ops run one invocation per token; scans run one
// invocation per prompt, so every prompt in a batch advances at once.
import { OP } from "./program.js";

const WG = 32;
const WGSL = /* wgsl */ `
struct P { B: u32, T: u32, a: u32, b: u32, out: u32, dIn: u32, dOut: u32, w: u32,
            bias: u32, act: u32, slots: u32, reverse: u32 };
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read_write> t: array<f32>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read> lengths: array<u32>;
@group(0) @binding(4) var<uniform> p: P;

fn R() -> u32 { return p.B * p.T; }

@compute @workgroup_size(${WG}) fn embed(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= R()) { return; }
  let q = g.x / p.T; let i = g.x % p.T; let dst = p.out * R() + g.x * p.dOut;
  for (var j = 0u; j < p.dOut; j++) { t[dst + j] = 0.0; }
  if (i >= lengths[q]) { return; }
  for (var s = 0u; s < p.slots; s++) {
    let row = p.w + rows[g.x * p.slots + s] * p.dOut;
    for (var j = 0u; j < p.dOut; j++) { t[dst + j] += W[row + j]; }
  }
}

@compute @workgroup_size(${WG}) fn linear(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= R()) { return; }
  let src = p.a * R() + g.x * p.dIn; let dst = p.out * R() + g.x * p.dOut;
  for (var k = 0u; k < p.dOut; k++) {
    var s = W[p.bias + k];
    let wr = p.w + k * p.dIn;
    for (var c = 0u; c < p.dIn; c++) { s += W[wr + c] * t[src + c]; }
    switch p.act {
      case 1u: { s = tanh(s); }
      case 2u: { s = 1.0 / (1.0 + exp(-s)); }
      case 3u: { s = max(s, 0.0); }
      default: {}
    }
    t[dst + k] = s;
  }
}

@compute @workgroup_size(${WG}) fn scan(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.B) { return; }
  let n = lengths[g.x];
  for (var i = 0u; i < p.T * p.dOut; i++) { t[p.out * R() + g.x * p.T * p.dOut + i] = 0.0; }
  for (var j = 0u; j < p.dOut; j++) {
    var h = 0.0;
    for (var i = 0u; i < n; i++) {
      let pos = select(i, n - 1u - i, p.reverse != 0u);
      let r = (g.x * p.T + pos) * p.dOut + j;
      let a = t[p.a * R() + r];
      h = a * h + (1.0 - a) * t[p.b * R() + r];
      t[p.out * R() + r] = h;
    }
  }
}

@compute @workgroup_size(${WG}) fn concat(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= R()) { return; }
  let half = p.dOut / 2u; let dst = p.out * R() + g.x * p.dOut;
  for (var j = 0u; j < half; j++) {
    t[dst + j] = t[p.a * R() + g.x * half + j];
    t[dst + half + j] = t[p.b * R() + g.x * half + j];
  }
}

@compute @workgroup_size(${WG}) fn add(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= R()) { return; }
  let live = g.x % p.T < lengths[g.x / p.T];
  for (var j = 0u; j < p.dOut; j++) {
    let r = g.x * p.dOut + j;
    t[p.out * R() + r] = select(0.0, t[p.a * R() + r] + t[p.b * R() + r], live);
  }
}

@compute @workgroup_size(${WG}) fn conv(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= R()) { return; }
  let q = g.x / p.T; let i = i32(g.x % p.T); let n = i32(lengths[q]);
  for (var j = 0u; j < p.dOut; j++) {
    var s = 0.0;
    if (i < n) {
      s = t[p.a * R() + g.x * p.dOut + j];
      for (var k = 0i; k < 5i; k++) {
        let at = i + k - 2i;
        if (at >= 0i && at < n) {
          s += W[p.w + u32(k) * p.dOut + j] * t[p.a * R() + (q * p.T + u32(at)) * p.dOut + j];
        }
      }
    }
    t[p.out * R() + g.x * p.dOut + j] = s;
  }
}

@compute @workgroup_size(${WG}) fn pool(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= R()) { return; }
  let q = g.x / p.T; let half = p.dOut / 2u; let n = lengths[q];
  let dst = p.out * R() + g.x * p.dOut;
  for (var j = 0u; j < half; j++) {
    var m = 0.0;
    for (var i = 0u; i < n; i++) { m += t[p.a * R() + (q * p.T + i) * half + j]; }
    t[dst + j] = t[p.a * R() + g.x * half + j];
    t[dst + half + j] = m / f32(max(n, 1u));
  }
}
`;

const ENTRY = { [OP.EMBED]: "embed", [OP.LINEAR]: "linear", [OP.SCAN]: "scan", [OP.CONCAT]: "concat", [OP.CONV]: "conv", [OP.ADD]: "add", [OP.POOL]: "pool" };
const FIELDS = ["B", "T", "a", "b", "out", "dIn", "dOut", "w", "bias", "act", "slots", "reverse"];

export async function createWebGpuBackend(weights, { adapterOptions } = {}) {
  if (!globalThis.navigator?.gpu) throw new Error("this browser has no WebGPU");
  const adapter = await navigator.gpu.requestAdapter(adapterOptions);
  if (!adapter) throw new Error("no WebGPU adapter available");
  const limit = adapter.limits.maxStorageBufferBindingSize;
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: limit, maxBufferSize: adapter.limits.maxBufferSize },
  });

  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: WGSL });
  const storage = (type) => ({ visibility: GPUShaderStage.COMPUTE, buffer: { type } });
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, ...storage("read-only-storage") },
      { binding: 1, ...storage("storage") },
      { binding: 2, ...storage("read-only-storage") },
      { binding: 3, ...storage("read-only-storage") },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const pipelines = {};
  for (const [op, entryPoint] of Object.entries(ENTRY))
    pipelines[op] = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint } });
  const err = await device.popErrorScope();
  if (err) throw new Error("WGSL: " + err.message);

  const wbuf = device.createBuffer({ size: weights.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(wbuf, 0, weights);
  const align = device.limits.minUniformBufferOffsetAlignment;

  return {
    name: "webgpu",
    adapter: adapter.info?.description || adapter.info?.architecture || adapter.info?.vendor || "unknown adapter",
    maxFloats: Math.floor(limit / 4),
    /** rows: Uint32Array [B, T, slots]; lengths: Uint32Array [B]. Returns logits [B, T, nRoles]. */
    async run(prog, rows, lengths, T) {
      const B = lengths.length, R = B * T;
      const tape = device.createBuffer({ size: prog.width * R * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const rbuf = device.createBuffer({ size: rows.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const lbuf = device.createBuffer({ size: lengths.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(rbuf, 0, rows);
      device.queue.writeBuffer(lbuf, 0, lengths);
      const u = new Uint32Array(prog.ops.length * (align / 4));
      prog.ops.forEach((o, i) => FIELDS.forEach((f, j) => {
        u[i * (align / 4) + j] = f === "B" ? B : f === "T" ? T : o[f] ?? 0;
      }));
      const ubuf = device.createBuffer({ size: u.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(ubuf, 0, u);
      const size = prog.nRoles * R * 4;
      const read = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

      const bg = device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: wbuf } },
          { binding: 1, resource: { buffer: tape } },
          { binding: 2, resource: { buffer: rbuf } },
          { binding: 3, resource: { buffer: lbuf } },
          { binding: 4, resource: { buffer: ubuf, size: FIELDS.length * 4 } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      prog.ops.forEach((o, i) => {
        pass.setPipeline(pipelines[o.op]);
        pass.setBindGroup(0, bg, [i * align]);
        pass.dispatchWorkgroups(Math.ceil((o.op === OP.SCAN ? B : R) / WG));
      });
      pass.end();
      enc.copyBufferToBuffer(tape, prog.logits * R * 4, read, 0, size);
      device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();
      for (const b of [tape, rbuf, lbuf, ubuf, read]) b.destroy();
      return out;
    },
  };
}
