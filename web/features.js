// Mechanical tokenizer + hashed sparse features. Port of train/features.py; eval/parity.mjs checks them.
const SPLIT = /"[^"]*"|`[^`]*`|(?<![\p{L}\p{N}])'[^']*'(?![\p{L}\p{N}])|\p{L}+|[0-9]+|\S/gu;

export const SIZES = { word: 768, skel: 192, pre: 192, suf: 192, shape: 28, length: 9 };
const OFFSETS = {};
let at = 0;
for (const [k, v] of Object.entries(SIZES)) { OFFSETS[k] = at; at += v; }
export const N_ROWS = at;
export const SLOTS = Object.keys(SIZES).length;

const encoder = new TextEncoder();
function fnv(s) {
  let h = 0x811c9dc5;
  for (const b of encoder.encode(s)) h = Math.imul(h ^ b, 0x01000193) >>> 0;
  return h;
}

export function tokenize(text) {
  const toks = [];
  for (const m of text.matchAll(SPLIT)) {
    const t = m[0];
    const quoted = t.length >= 2 && "\"`'".includes(t[0]) && t.at(-1) === t[0];
    toks.push({ text: quoted ? t.slice(1, -1) : t, start: m.index, end: m.index + t.length, quoted });
  }
  toks.forEach((t, i) => {
    t.glueL = i > 0 && toks[i - 1].end === t.start;
    t.glueR = i + 1 < toks.length && toks[i + 1].start === t.end;
  });
  return toks;
}

function caseShape(t) {
  if (t.quoted) return 6;
  const s = t.text;
  if (/^[0-9]+$/.test(s)) return 4;
  if (!/\p{L}/u.test(s)) return 5;
  if (s === s.toLowerCase()) return 0;
  if (s === s.toUpperCase()) return 1;
  const [first, ...rest] = s;
  if (first === first.toUpperCase() && rest.join("") === rest.join("").toLowerCase()) return 2;
  return 3;
}

export function featurize(t) {
  const shape = OFFSETS.shape + caseShape(t) * 4 + (t.glueL ? 2 : 0) + (t.glueR ? 1 : 0);
  const chars = [...t.text];
  const length = OFFSETS.length + Math.min(chars.length, SIZES.length - 1);
  if (t.quoted) return [OFFSETS.word, OFFSETS.skel, OFFSETS.pre, OFFSETS.suf, shape, length];
  const low = t.text.toLowerCase();
  const lc = [...low];
  const skel = low.replace(/[aeiou]/g, "") || low;
  return [
    OFFSETS.word + fnv("w" + low) % SIZES.word,
    OFFSETS.skel + fnv("s" + skel) % SIZES.skel,
    OFFSETS.pre + fnv("p" + lc.slice(0, 3).join("")) % SIZES.pre,
    OFFSETS.suf + fnv("x" + lc.slice(-3).join("")) % SIZES.suf,
    shape,
    length,
  ];
}
