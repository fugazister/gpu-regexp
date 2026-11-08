# gpu-regexp

Describe a pattern in English and get a regex. A 44K-parameter model labels your words, on WebGPU or the CPU, and a deterministic compiler turns the labels into the regex. It runs entirely in the browser.

**[Try the demo](https://fugazister.github.io/gpu-regexp/)**

An experiment inspired by [gpu-lexer](https://shud.in/thoughts/ideas-behind-gpu-lexer) and the projects that followed it. **Check the regex before you ship it.**

To run it locally:

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
3. **Compile (CPU).** `web/compile.js` holds all the regex semantics. A 387-form lexicon decides what an `ATOM` span means, tolerating small typos. The model never writes regex text, so its mistakes produce a wrong regex, not an invalid one.

Training data comes from a grammar (`train/grammar.py`) that emits labels and a gold regex built independently of the compiler. It also includes 192 real descriptions from StructuredRegex, whose word labels were inferred by searching for roles that compile to a regex behaving like the dataset's own (`train/label_real.mjs`).

## Results

Scored by behaviour: a prompt passes if its regex matches every string it should and none it shouldn't.

| Prompts | Passed |
|---|---:|
| Test set (201) | 166 (82.6%, 95% CI 76.7–87.2) |
| Dev set (60) | 56 (93.3%) |
| `natural` test prompts only (45) | 22 (49%) |
| [StructuredRegex](https://github.com/xiye17/StructuredRegex), written by other people (996 across two test splits) | 93 (9.3%) |
| StructuredRegex, only regexes the grammar could express (159) | 41 (25.8%) |

| Input that isn't a request | Refused |
|---|---:|
| Hand-written: mash, chat, code, foreign text (60) | 56 (93%) |
| Real prose: licences and an article (284) | 167 (59%) |

Read these with care. One author wrote the first three sets and the grammar, so they aren't independent. StructuredRegex is independent, and much harder than typing a short prompt: its descriptions are long and formal, and most describe regexes this project can't express. The real answer on everyday phrasing is somewhere in between. The checks are also lenient: a prompt can pass with a regex that's broader than it should be.

**[EXPERIMENTS.md](EXPERIMENTS.md)** explains how better training data took the test score from 80% to about 84%, what didn't improve, why the independent score is so low, and how sure we can be.

An earlier comparison with a seq2seq transformer, on a smaller test set (40 prompts) that has since been retired:

| | Tagger + compiler (this repo) | Seq2seq transformer (first attempt, not in this repo) |
|---|---|---|
| Parameters | 44,140 | 213,751 |
| Weights | 43 kB (28 kB gzipped) | 209 kB |
| Held-out test prompts (40), at the time | 34 (85%) | 30 (75%) |
| Invalid regexes on that test set | 0 | 3 |

The gap between the two is only four prompts, so this is weak evidence on its own.

### Speed

Tagging only, median of 7 warm runs, RTX 4060 through Dawn in Node. Prompts are 86 fixtures repeated to fill the batch. CPU is the JS backend on the same machine.

| Prompts | CPU | WebGPU | Speedup |
|---:|---:|---:|---:|
| 1 | 0.3 ms | 1.3 ms | 0.2× |
| 32 | 10.8 ms | 2.1 ms | 5.2× |
| 256 | 102 ms | 10 ms | 10× |
| 1,024 | 411 ms | 54 ms | 7.7× |
| 4,096 | 1,641 ms | 131 ms | 13× |
| 16,384 | 6,608 ms | 632 ms | 10× |

A single prompt is faster on the CPU, so the demo normally types on the CPU despite the name. This is one machine, not a cross-device benchmark. The page's **Compare CPU and GPU** button measures yours.

## Limitations

- **Natural phrasing often fails.** Only 49% of the `natural` test prompts pass.
- **Coverage is the grammar and lexicon.** Thousands separators and group repetition (`(ab){3}`) aren't supported.
- **Words it doesn't know can land in the regex.** `numbers with commas as thousands separators` gives `\d+,+`, which drops the digits after the comma.
- **Refusal is partial, and confidence can't be trusted.** `the quick brown fox` and `asdf qwer zxcv` get no regex, but about 41% of real prose still gets one, more often for long sentences. Diagnostics only cover what the compiler can detect.
- **Mistakes can be silent.** `a string of exactly ten digits` loses the count and gives `\d+`.
- **Typos still cost some prompts.** 18 of 20 typo test prompts pass, but `digts folowed by a dahs` still drops the dash and gives `\d+`.
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
node eval/score.mjs eval/dev-v2.json -v

npm i --no-save webgpu && node eval/native-gpu.mjs   # WGSL and timings on a real GPU

scripts/run_experiment.sh <name> <commit> [seed]     # train and score one commit in its own worktree
```

Tune against `eval/dev-v2.json`. Tuning against `eval/test-v2.json` would make its score meaningless, so the tools only print its totals.

The architecture and evaluation habits come from [gpu-lexer](https://gpu-lexer.vercel.app), [gpu-time](https://github.com/arikchakma/gpu-time), [gpu-query](https://github.com/safzanpirani/gpu-query), [gpu-cron](https://github.com/manuschillerdev/gpu-cron) and [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

## Third-party data

`train/real_labels.jsonl` and `results/*.streg*.json` quote descriptions from [StructuredRegex](https://github.com/xiye17/StructuredRegex), used under [its MIT licence](licenses/StructuredRegex-MIT.txt).
