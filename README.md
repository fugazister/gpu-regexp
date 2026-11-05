# gpu-regexp

Describe a pattern in English and get a regex. A 44K-parameter model labels your words, on WebGPU or the CPU, and a deterministic compiler turns the labels into the regex. It runs entirely in the browser.

An experiment inspired by [gpu-lexer](https://shud.in/thoughts/ideas-behind-gpu-lexer) and the projects that followed it. **Check the regex before you ship it.**

```sh
python3 -m http.server -d web 8000   # then open http://localhost:8000
```

```js
import { load } from "./web/gpu-regexp.js";

const gr = await load();
const r = await gr.parse("starts with a capital letter and ends with a period");
r.source;      // "^[A-Z].*\\.$"
r.regexp;      // RegExp, or null if nothing could be compiled
r.spans;       // which words got which role
r.diagnostics; // what it had to guess at or couldn't read
r.confidence;  // softmax probability of the least certain word

await gr.parseMany(prompts); // batches; uses WebGPU from 32 prompts up
```

## How it works

```
"3 digits then a dash"
  tokenize, hash     3   digits  then  a  dash       no vocabulary, only spelling hashes and shape
  tag (model)        NUM ATOM    THEN  O  ATOM       28 roles
  compile            \d{3}-                          lexicon + rules, always valid syntax
```

1. **Tokenize and featurize (CPU).** Each token becomes six hashed features: word, consonant skeleton, prefix, suffix, shape and length. With no vocabulary, `digts` still lands near `digits`.
2. **Tag (WebGPU or CPU).** Summed embeddings, two layers of gated bidirectional scans, a 5-tap convolution, and a two-layer head. Weights are int6, trained with fake quantization.
3. **Compile (CPU).** `web/compile.js` holds all the regex semantics. A 385-form lexicon decides what an `ATOM` span means, tolerating small typos. The model never writes regex text, so its mistakes produce a wrong regex, not an invalid one.

Training data comes from a grammar (`train/grammar.py`) that emits labels and a gold regex built independently of the compiler.

## Results

Scored by behaviour: a prompt passes if its regex matches every string it should and none it shouldn't. Prompts are hand-written. `eval/test.json` was written before seeing any model output and was run once.

| | Tagger + compiler (this repo) | Seq2seq transformer (first attempt, not in this repo) |
|---|---|---|
| Parameters | 44,140 | 213,751 |
| Weights | 43 kB (29 kB gzipped) | 209 kB |
| Dev prompts (20) | 19 (95%) | 17 (85%) |
| Held-out test prompts (40) | 34 (85%) | 30 (75%) |
| Invalid regexes on the test set | 0 | 3 |

Read these with care. Forty prompts by one author is a small sample, and the gap between the two models is only four prompts. The checks are also lenient: `a word that has exactly five letters` passes with `\w*[a-zA-Z]{5}\w*`, which also matches longer words. These numbers do not establish accuracy on real users' phrasing.

### Speed

Tagging only, median of 7 warm runs, RTX 4060 through Dawn in Node. Prompts are 86 fixtures repeated to fill the batch. CPU is the JS backend on the same machine.

| Prompts | CPU | WebGPU | Speedup |
|---:|---:|---:|---:|
| 1 | 0.4 ms | 1.3 ms | 0.3× |
| 32 | 11.2 ms | 2.2 ms | 5.0× |
| 256 | 143 ms | 10 ms | 14× |
| 1,024 | 470 ms | 41 ms | 12× |
| 4,096 | 1,978 ms | 152 ms | 13× |
| 16,384 | 8,058 ms | 596 ms | 14× |

A single prompt is faster on the CPU, so the demo normally types on the CPU despite the name. This is one machine, not a cross-device benchmark. The page's **Compare CPU and GPU** button measures yours.

## Limitations

- **Coverage is the grammar and lexicon.** Consonants, thousands separators and group repetition (`(ab){3}`) aren't supported.
- **Words it doesn't know can land in the regex.** `numbers with commas as thousands separators` gives `\d+,+(?:thousands separators)+`, and `a filename with no spaces` gives `filename[^ ]*`.
- **Mistakes can be silent.** `either yes or no` drops the `no` and gives `/yes/`. `thre digits` loses the count and gives `\d+`. Gibberish like `asdf qwer zxcv` still produces a regex, though with low confidence. Diagnostics only cover what the compiler can detect.
- **Fuzzy matching only helps words tagged as atoms.** In `digts folowed by a dahs`, `dahs` is copied as literal text.
- **Presets are opinionated approximations**, such as email and URL. Generated patterns aren't checked for catastrophic backtracking, so don't run them on untrusted input without review.
- The tokenizers can differ on unusual non-ASCII text; parity is tested on the generated prompts and a few handwritten edge cases, which are mostly ASCII.

## Development

Python 3.12, `uv`, Node 22+.

```sh
uv venv --python 3.12 train/.venv
VIRTUAL_ENV=train/.venv uv pip install -r train/requirements.txt

train/.venv/bin/python train/train.py --steps 15000 --bs 512 --workers 4   # 7 min on an RTX 4060
train/.venv/bin/python train/export.py        # web/model.*, eval/fixtures.json
train/.venv/bin/python train/dump_oracle.py   # web/lexicon.json, eval/oracle.json

node eval/roundtrip.mjs            # compiler on gold labels: MUST be 100%
node eval/parity.mjs               # JS features and logits match torch; batching changes nothing
node eval/webgpu.mjs               # WGSL matches JS in headless Chrome (software adapter)
node eval/score.mjs eval/dev.json -v

npm i --no-save webgpu && node eval/native-gpu.mjs   # WGSL and timings on a real GPU
```

Tune against `eval/dev.json`. Using `eval/test.json` for tuning would make its score meaningless.

The architecture and evaluation habits come from [gpu-lexer](https://gpu-lexer.vercel.app), [gpu-time](https://github.com/arikchakma/gpu-time), [gpu-query](https://github.com/safzanpirani/gpu-query), [gpu-cron](https://github.com/manuschillerdev/gpu-cron) and [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).
