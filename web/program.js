// Compiles the tagger into a flat op list over a "tape" of f32 activations, laid out [batch, T, width].
// The JS and WGSL backends execute exactly this list, so they can be tested against each other.
export const OP = { EMBED: 0, LINEAR: 1, SCAN: 2, CONCAT: 3, CONV: 4, ADD: 5, POOL: 6 };
export const ACT = { NONE: 0, TANH: 1, SIGMOID: 2, RELU: 3 };

export function compile(spec, slots) {
  const { d, layers, hidden, n_roles: nRoles } = spec.cfg;
  const W = (name) => spec.tensors[name].offset;
  const ops = [];
  let top = 0; // tape usage in units of "floats per token"; the backend scales by batch * T
  const alloc = (width) => { const at = top; top += width; return at; };

  const dense = (name, a, dIn, dOut, act = ACT.NONE) => {
    const out = alloc(dOut);
    ops.push({ op: OP.LINEAR, a, out, dIn, dOut, act, w: W(name + ".w"), bias: W(name + ".b") });
    return out;
  };

  let x = alloc(d);
  ops.push({ op: OP.EMBED, out: x, dOut: d, slots, w: W("emb") });

  for (let l = 0; l < layers; l++) {
    const scan = (dir, reverse) => {
      const u = dense(`scans.${l}.u${dir}`, x, d, d, ACT.TANH);
      const g = dense(`scans.${l}.a${dir}`, x, d, d, ACT.SIGMOID);
      const out = alloc(d);
      ops.push({ op: OP.SCAN, a: g, b: u, out, dOut: d, reverse: reverse ? 1 : 0 });
      return out;
    };
    const f = scan("f", false), b = scan("b", true);
    const both = alloc(2 * d);
    ops.push({ op: OP.CONCAT, a: f, b, out: both, dIn: d, dOut: 2 * d });
    const p = dense(`scans.${l}.proj`, both, 2 * d, d);
    const sum = alloc(d);
    ops.push({ op: OP.ADD, a: x, b: p, out: sum, dOut: d });
    x = sum;
  }

  const z = alloc(d);
  ops.push({ op: OP.CONV, a: x, out: z, dOut: d, w: W("conv") });
  const withPool = alloc(2 * d);
  ops.push({ op: OP.POOL, a: z, out: withPool, dIn: d, dOut: 2 * d });
  const h = dense("head1", withPool, 2 * d, hidden, ACT.RELU);
  const logits = dense("head2", h, hidden, nRoles);
  return { ops, width: top, logits, nRoles };
}
