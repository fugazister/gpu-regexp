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

Training data comes from a grammar (`train/grammar.py`) that emits labels and a gold regex built independently of the compiler.

## Results

Scored by behaviour: a prompt passes if its regex matches every string it should and none it shouldn't.

| Prompts | Passed |
|---|---:|
| Test set (201) | 169 (84.1%, 95% CI 78.4–88.5) |
| Dev set (60) | 57 (95%) |
| `natural` test prompts only (45) | 20 (44%) |

Read these with care. One author wrote all the prompts and the grammar, so the test set isn't independent, and real users' phrasing would probably score lower. The `natural` row is the closest thing here to real use. The checks are also lenient: a prompt can pass with a regex that's broader than it should be.

**[EXPERIMENTS.md](EXPERIMENTS.md)** explains how better training data took the test score from 80% to 84%, what didn't improve, and how sure we can be.

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
| 1 | 0.2 ms | 1.2 ms | 0.2× |
| 32 | 12.6 ms | 2.2 ms | 5.7× |
| 256 | 133 ms | 8.2 ms | 16× |
| 1,024 | 530 ms | 69 ms | 7.7× |
| 4,096 | 2,113 ms | 145 ms | 15× |
| 16,384 | 8,724 ms | 649 ms | 13× |

A single prompt is faster on the CPU, so the demo normally types on the CPU despite the name. This is one machine, not a cross-device benchmark. The page's **Compare CPU and GPU** button measures yours.

## Limitations

- **Natural phrasing often fails.** Only 44% of the `natural` test prompts pass, and better training data didn't change that.
- **Coverage is the grammar and lexicon.** Thousands separators and group repetition (`(ab){3}`) aren't supported.
- **Words it doesn't know can land in the regex.** `numbers with commas as thousands separators` gives `\d+commas as thousands`, and `a filename with no spaces` gives `filename[^ ]*`.
- **It doesn't refuse, and confidence can't be trusted.** `the quick brown fox` gives `/brown/` at confidence .99, and `asdf qwer zxcv` gives `/asdfzxcv/` at .48. Diagnostics only cover what the compiler can detect.
- **Mistakes can be silent.** `a string of exactly ten digits` loses the count and gives `\d+`.
- **Fuzzy matching doesn't catch everything.** In `digts folowed by a dahs`, `dahs` is copied as literal text.
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

`eval/human.json` is for prompts written by people who haven't seen the grammar. It's the most honest test this project can have. Each entry looks like `{"c": "natural", "q": "the prompt", "yes": ["should match"], "no": ["shouldn't match"]}`.

The architecture and evaluation habits come from [gpu-lexer](https://gpu-lexer.vercel.app), [gpu-time](https://github.com/arikchakma/gpu-time), [gpu-query](https://github.com/safzanpirani/gpu-query), [gpu-cron](https://github.com/manuschillerdev/gpu-cron) and [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).
