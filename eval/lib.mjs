import { readFileSync } from "node:fs";
import { decodeWeights, createParser } from "../web/gpu-regexp.js";
import { createCpuBackend } from "../web/backend-cpu.js";

const root = new URL("../", import.meta.url);
export const readJson = (p) => JSON.parse(readFileSync(new URL(p, root)));

export function loadCpu() {
  const spec = readJson("web/model.json");
  const buf = readFileSync(new URL("web/model.bin", root));
  const weights = decodeWeights(spec, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const parser = createParser(spec, readJson("web/lexicon.json"), { cpu: createCpuBackend(weights), webgpu: null });
  return { spec, weights, parser };
}
