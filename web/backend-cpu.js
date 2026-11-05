// Reference executor for the op list, and the default for small inputs: below a few dozen prompts,
// the GPU's dispatch + readback overhead outweighs its parallelism.
import { OP, ACT } from "./program.js";

const activate = [(v) => v, Math.tanh, (v) => 1 / (1 + Math.exp(-v)), (v) => (v > 0 ? v : 0)];

export function createCpuBackend(weights) {
  return {
    name: "cpu",
    /** rows: Uint32Array [B, T, slots]; lengths: Uint32Array [B]. Returns logits [B, T, nRoles]. */
    async run(prog, rows, lengths, T) {
      const B = lengths.length, t = new Float32Array(prog.width * B * T);
      for (const o of prog.ops) exec(o, t, weights, rows, lengths, T, B);
      return t.subarray(prog.logits * B * T, (prog.logits + prog.nRoles) * B * T);
    },
  };
}

function exec(o, tape, W, rows, lengths, T, B) {
  const R = B * T, base = (x) => x * R; // each op's tape region is width * R floats
  const { dIn, dOut } = o;
  const a = base(o.a ?? 0), b = base(o.b ?? 0), out = base(o.out);
  switch (o.op) {
    case OP.EMBED:
      for (let q = 0; q < B; q++)
        for (let i = 0; i < lengths[q]; i++) {
          const r = q * T + i, dst = out + r * dOut;
          for (let s = 0; s < o.slots; s++) {
            const row = o.w + rows[r * o.slots + s] * dOut;
            for (let j = 0; j < dOut; j++) tape[dst + j] += W[row + j];
          }
        }
      break;
    case OP.LINEAR: {
      const f = activate[o.act];
      for (let q = 0; q < B; q++)
        for (let i = 0; i < T; i++) {
          const r = q * T + i, src = a + r * dIn, dst = out + r * dOut;
          for (let k = 0; k < dOut; k++) {
            let s = W[o.bias + k];
            const wr = o.w + k * dIn;
            for (let c = 0; c < dIn; c++) s += W[wr + c] * tape[src + c];
            tape[dst + k] = o.act === ACT.NONE ? s : f(s);
          }
        }
      break;
    }
    case OP.SCAN:
      for (let q = 0; q < B; q++)
        for (let j = 0; j < dOut; j++) {
          let h = 0;
          const n = lengths[q];
          for (let i = 0; i < n; i++) {
            const pos = o.reverse ? n - 1 - i : i, r = (q * T + pos) * dOut + j;
            const g = tape[a + r];
            h = g * h + (1 - g) * tape[b + r];
            tape[out + r] = h;
          }
        }
      break;
    case OP.CONCAT:
      for (let r = 0; r < R; r++) {
        const half = dOut / 2, dst = out + r * dOut;
        for (let j = 0; j < half; j++) {
          tape[dst + j] = tape[a + r * half + j];
          tape[dst + half + j] = tape[b + r * half + j];
        }
      }
      break;
    case OP.ADD:
      // Padding rows stay zero: real rows only, so later ops see the same values as a shorter batch.
      for (let q = 0; q < B; q++)
        for (let i = 0; i < lengths[q] * dOut; i++) {
          const r = q * T * dOut + i;
          tape[out + r] = tape[a + r] + tape[b + r];
        }
      break;
    case OP.CONV:
      for (let q = 0; q < B; q++)
        for (let i = 0; i < lengths[q]; i++) {
          const dst = out + (q * T + i) * dOut, src = a + (q * T + i) * dOut;
          for (let j = 0; j < dOut; j++) {
            let s = tape[src + j];
            for (let k = 0; k < 5; k++) {
              const at = i + k - 2;
              if (at >= 0 && at < lengths[q]) s += W[o.w + k * dOut + j] * tape[a + (q * T + at) * dOut + j];
            }
            tape[dst + j] = s;
          }
        }
      break;
    case OP.POOL: {
      const half = dOut / 2;
      for (let q = 0; q < B; q++) {
        const mean = new Float32Array(half), n = Math.max(lengths[q], 1);
        for (let i = 0; i < lengths[q]; i++)
          for (let j = 0; j < half; j++) mean[j] += tape[a + (q * T + i) * half + j];
        for (let i = 0; i < T; i++) {
          const dst = out + (q * T + i) * dOut, src = a + (q * T + i) * half;
          for (let j = 0; j < half; j++) {
            tape[dst + j] = tape[src + j];
            tape[dst + half + j] = mean[j] / n;
          }
        }
      }
      break;
    }
  }
}
