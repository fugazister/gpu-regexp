# Experiments

## Status

- **Done:** Round 1, data fixes. The E5-s2 model is shipped.
- **Now:** Nothing is running.
- **Next:** Stop sentence periods becoming literals, using `dev` from StructuredRegex to measure it, then Round 2 (see [Next](#next)).

## Round 1: Can better training data beat the limits?

**Short answer:** Yes for phrasing close to the grammar, and no for natural phrasing.

| Test set (201 prompts, mean of 2 seeds) | Before (E0) | After (E5) |
|---|---:|---:|
| Everything except `natural` (156) | 140.5 (90%) | 149 (96%) |
| `natural` (45) | 20 (44%) | 20 (44%) |
| **Total** | **160.5 (80%)** | **169 (84%)** |

The model improved without any change to its architecture or size. Most of the gain came from one category, and the category that looks most like real users didn't move.

## The method

Each fix is its own commit, and `scripts/run_experiment.sh` trains that commit in its own git worktree. A result can therefore always be traced to the exact code that produced it. Each run takes about 7 minutes on an RTX 4060.

There are three measurements, because each can fool you in a different way:

| Measurement | What it answers | How it can mislead |
|---|---|---|
| Generated data | Did it learn the grammar? | It's near perfect whether or not the model is useful |
| Dev set (60 prompts, read freely) | What should I fix next? | Tuning against it inflates its score |
| Test set (201 prompts, totals only) | How good is it now? | Reading its failures turns it into a dev set |

A prompt passes if its regex matches every string it should and none it shouldn't. That tests behaviour, so any regex that behaves correctly passes, not just one exact answer.

## The runs

| Run | Change | ML problem it targets | Generated (sequence) | Dev /60 | Test /201 |
|---|---|---|---:|---:|---:|
| E0-s1 | Baseline | | 99.22% | 54 | 161 |
| E0-s2 | Same code, different seed | Measuring noise | 99.14% | 53 | 160 |
| E1 | Removed `only`, `just`, `has` and others from the filler words | Label noise | 99.46% | 55 | |
| E2 | Common words appear as bare literals | Sampling bias | 99.38% | 56 | |
| E3 | Typos on spelled-out numbers, plus fuzzy number words | Augmentation gap | 99.38% | 55 | |
| E4 | Added `like 1.2.3` and `in hh:mm format` phrasing | Distribution shift | 99.32% | 56 | |
| E5-s1 | Added consonants, dropped stranded counts | Lexicon and compiler gap | 99.30% | 56 | 169 |
| E5-s2 | Same code, different seed | Measuring noise | 99.36% | 57 | 169 |

The test set was scored only for E0 and E5, so the intermediate runs couldn't influence decisions through it.

## What the data shows

### 1. Training accuracy barely moved while real accuracy did

Generated-data accuracy stays around 99.3% across all runs, while dev moves from 54 to 57. On `natural` test prompts the pass rate is 44%. These aren't the same metric (one compares labels, the other behaviour), but the gap is too big to explain away.

This is **distribution shift**: the model is excellent on the distribution it was trained on, and that says little about any other one. The generated-data score couldn't have told us about any of these fixes, so a "99% accurate" model can mean very little.

### 2. Two models with the same score aren't the same model

Changing only the random seed gives nearly identical totals, but different prompts flip:

| Same code, seed 1 vs seed 2 | Totals | Prompts that flipped |
|---|---:|---:|
| E0 on test | 161 vs 160 | 9 |
| E5 on test | 169 vs 169 | 8 |

So about 4% of prompts depend on the seed alone. This is **seed variance**, and it's why a one-seed comparison can't credit a fix with a single prompt.

E1 to E4 each ran on one seed, so their step-by-step dev changes are not evidence. Here's what happened to two prompts that none of those fixes targeted:

- `a filename with no spaces` passed at E1, then failed again at E5.
- `numbers with commas as thousands separators` passed at E2, then failed at E3.

Neither was a fix working or breaking. They were noise.

### 3. Probabilities show what pass/fail hides

A passing prompt can still be close to failing. `train/probe.py` tracks the probability the model gives the correct role on the exact word each fix targets:

| Word → right role | Fix | E0-s1 | E0-s2 | E1 | E2 | E3 | E4 | E5-s1 | E5-s2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `only` → ONLY | E1 | .647 | .947 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `no` in `yes or no` → LIT | E2 | .003 | .984 | .243 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| `thre` → NUM | E3 | .000 | .017 | .000 | .009 | **.985** | .998 | .990 | .982 |
| `format` → O | E4 | .000 | .000 | .000 | .007 | .001 | **1.000** | 1.000 | 1.000 |
| `consonants` → ATOM | E5 | .000 | .739 | .008 | .990 | .998 | .670 | **1.000** | 1.000 |

Bold marks the run that made the targeted change. Three lessons:

- **The fixes did what they were meant to.** `thre` and `format` jump from about 0 to about 1 at exactly their own step. That's much stronger evidence than the pass/fail table.
- **One baseline seed would have misled us.** `either yes or no` looked like a clear bug with E0-s1 (.003), but E0-s2 already had it right (.984). The real improvement is from a coin flip between seeds to 1.000 on both.
- **Side effects happen.** `consonants` reached .998 at E3 without being targeted, then dropped to .670 at E4, whose change was about something else. Training data interacts across the whole model.

The `like` probe from `version number like 1.2.3` was already 1.000 everywhere, so it isn't in the table. The failure there was elsewhere in the prompt.

### 4. Model and system errors are different

The tagger is only half of this system. To split the credit, this ablation mixes old and new weights with old and new compiler and lexicon, on dev:

| Dev /60 (s1, s2) | Old compiler + lexicon | New compiler + lexicon |
|---|---:|---:|
| Old weights | 54, 53 | 55, 55 |
| New weights | 55, 56 | **56, 57** |

Both contribute, and neither alone reaches the final score. `a vowel followed by two consonants` passed with the new compiler and lexicon even on old weights, so retraining wasn't what fixed it. When a model seems wrong, check the rest of the system first.

### 5. Where the test gain came from

Mean of two seeds, E0 → E5:

| Category | E0 | E5 |
|---|---:|---:|
| basic /25 | 23.5 | 23.5 |
| anchors /25 | 24.5 | 24.5 |
| negation /16 | 14 | 14.5 |
| **alternation /18** | **11.5** | **17.5** |
| presets /42 | 40 | 41 |
| misc /10 | 9 | 9 |
| typos /20 | 18 | 19 |
| **natural /45** | **20** | **20** |

Alternation gained 6 of the 8.5 prompts. E2 (sampling bias) is the likely cause, because on dev it stopped `no` being read as a negation in `either yes or no`. That's a hypothesis, since test wasn't scored at E2.

`natural` stayed flat, even though E4 added natural-sounding phrasing. Hand-adding patterns one at a time is not closing that gap.

## How sure can we be?

The paired sign test only looks at prompts where the two runs disagree:

| Test set, E0 → E5 | Gained | Lost | Two-sided p |
|---|---:|---:|---:|
| Seed 1 | 13 | 5 | 0.096 |
| Seed 2 | 13 | 4 | 0.049 |

The final score is 169/201, **84.1% (95% CI 78.4–88.5)**. This is suggestive, not proof: only one of the two seed comparisons reaches p < 0.05, and they aren't independent because they share a test set. The case for a real effect is that the gain (+8 and +9) is much larger than the seed gap (0 to 1), that both seeds agree, and that the probes moved as predicted.

**The shipping decision:** E5 went live because it gained in both seeds and also has the compiler and lexicon its weights were trained for. E5-s2 was picked over E5-s1 on dev (57 vs 56), which is within noise; they tie on test. In effect the choice was arbitrary.

### Why this result could be too optimistic

- **The test set isn't independent.** The same author wrote the grammar and all 201 prompts, knowing what the grammar supports. Real users would probably score lower, which is why the next step is a public dataset.
- **Possible contamination:** Some E4 phrasing resembles test prompts (such as `version numbers like 2.0.1`). The fixes came from dev and retired-test failures, but the author had seen the test prompts while writing them. `natural` stayed flat, so there's no sign this inflated the score, but it can't be ruled out.
- **The test has now been used.** Its totals and category scores have been seen, and they point at `natural`. Any further tuning aimed at that score will gradually spoil it, so the next rounds need new prompts.
- **Possible regression:** `a string of exactly ten digits` passes at E0-s1 and E1 to E3, but fails at E0-s2, at E4 and on both E5 seeds, which includes the shipped model. It may be seed noise or a side effect of E4. It hasn't been investigated.

## The ideas, in one line each

- **Label noise:** Training labels contradict each other. `only` was both a real role and filler, so the model learned to hedge (.647).
- **Sampling bias:** The training data has patterns the real world doesn't. `no` nearly always meant negation, so bare `no` couldn't be a literal.
- **Augmentation gap:** The model only tolerates variation it was shown. It saw typos on most words but not on numbers.
- **Distribution shift:** Test data doesn't look like training data, which is the main remaining problem here.
- **Seed variance:** The same code and data give a different model on each run, so measure it before crediting a change.
- **Ablation:** Remove or swap one part at a time to see what each one contributes.
- **Data leakage (contamination):** Information from the test set reaches your decisions, so its score stops predicting new data.
- **Overconfidence:** High probability doesn't mean correct. `the quick brown fox` gives `/brown/` at confidence .99.

## Independent check: StructuredRegex

**Short answer:** On descriptions written by other people, the model passes about 2%, and about 6% of the ones its grammar could express. The 84% above doesn't carry over to this data.

[StructuredRegex](https://github.com/xiye17/StructuredRegex) has descriptions from crowdworkers who never saw this project, each with strings that should and shouldn't match. Run `eval/streg.mjs` to score it. The model wasn't retrained for this.

| Split | Descriptions (regexes) | Passed | In grammar scope | Passed in scope |
|---|---:|---:|---:|---:|
| `dev` (read freely) | 351 (120) | 11 (3.1%, CI 0.9–5.2) | 64 | 5 (7.8%, CI 1.5–15.6) |
| `teste` (totals only) | 367 (340) | 9 (2.5%, CI 1.1–4.3) | 57 | 3 (5.3%, CI 0.0–12.3) |
| `testi` (totals only) | 629 (336) | 11 (1.7%, CI 0.8–3.0) | 102 | 6 (5.9%, CI 1.0–10.8) |

Rules, set before scoring the two test splits:

- **Scope is judged from the gold regex alone,** never from the model's output: a sequence of single quantified characters or classes, optionally inside starts-with, ends-with or contains, or a top-level choice between atoms. Only about 16% of descriptions qualify, because the rest use nested `and`, `not` or repeated groups.
- **Confidence intervals resample by regex,** since descriptions of one regex share their examples and aren't independent.
- **Whole-string scoring barely helps.** Counting a regex that passes once anchored adds 0 to 3 prompts per split, so missing anchors aren't the problem.

### Why it fails, from reading `dev`

- **Sentence periods become literals.** `A string that contains a '*'.` gives `^.*\*\..*$`, which demands a real period. Removing sentence-ending periods from `dev` raised passes from 11 to 18, and in scope from 5 to 9. It's the cheapest fix available, because the training data never pairs periods with prose.
- **Long multi-sentence prose is outside the training distribution.** The median description is about 36 tokens, and 41 of the 351 on `dev` exceed the 48-token limit. The grammar's prompts are short.
- **Unseen structure.** `A list of four strings. The first ...` has no counterpart in the grammar, so words like `first` and `second` leak into the regex.
- **Some outputs are confidently wrong.** `The string consists of 1 or more capital letters, followed by ...` was read as an unordered "contains all of these".

### How far to trust this

- **The harness had three bugs, all caught by checks.** Gold regexes had to pass their own examples, and that exposed nested `and`/`not` handling and CRLF line endings. Quoted descriptions showed up when reading `dev`. After the fixes, gold passes on all 1,347 descriptions.
- **The test splits were scored more than once** because of those bugs. Only totals were printed, and all the fixes came from `dev`, so `teste` and `testi` are still usable.
- **This dataset is harder than typing into the demo.** Its descriptions are long, formal and aimed at complex regexes, so 2% doesn't mean the demo fails 98% of the time on short prompts.
- **In-scope doesn't mean supported phrasing.** Scope only says the regex could be built, not that the grammar taught that way of describing it.
- **These aren't comparable to the dataset paper's results,** which come from models trained on this data and scored by regex equivalence.

The lesson is the one from `natural`, at larger scale: a model scored on prompts written by its own author can look far better than it is.

## What's still broken

Prompts below are from the shipped model, on dev or made up by hand:

| Prompt | Output | Problem |
|---|---|---|
| `the quick brown fox` | `/brown/`, confidence .99 | Out-of-distribution input, overconfident |
| `asdf qwer zxcv` | `/asdfzxcv/`, confidence .48 | Should refuse |
| `numbers with commas as thousands separators` | `/\d+commas as thousands/` | Not in the grammar |
| `a filename with no spaces` | `/filename[^ ]*/` | Not in the grammar |
| `a string of exactly ten digits` | `/\d+/` | Loses the count, cause unknown |
| `(ab) three times` | `/(?:\(ab\)){3}/` | No group repetition (the parentheses are literal) |

## Next

**First: fix sentence periods.** Teach the grammar that a period closing a sentence is filler, retrain, and measure on StructuredRegex `dev`. [KB13 and NL-RX-Turk](https://github.com/nicholaslocascio/deep-regex) remain available as further independent tests, once match strings are generated for them.

**Round 2: teach it to refuse, then measure calibration.** Add nonsense and off-topic negatives to training, then check whether a stated 90% confidence is right 90% of the time. If it works, `the quick brown fox` gets "I can't read this" instead of `/brown/`. It's the fix users would notice most, and it teaches calibration, abstention and precision/recall trade-offs.

**After that:**

1. **Examples as a safety net.** Generate the top few candidate regexes and test them against strings the user says should or shouldn't match. This can fix wrong answers without improving the model, and it's the best showcase in this list.
2. **Model experiments.** Ablation of each layer, learning curves (does more data still help?), a CRF output layer, and more seeds.
3. **LLM-paraphrased training data, verified by the compiler.** This is the most direct attack on `natural`.

## Reproduce

```sh
scripts/run_experiment.sh E5-s2 d4eaada 2       # on a GPU machine; SCORE_TEST=1 also scores test
node eval/compare.mjs results/E0-s2.dev.json results/E5-s2.dev.json --show
train/.venv/bin/python train/probe.py results/E0-s1.pt results/E5-s2.pt
```

Score files are in `results/`. Checkpoints (`*.pt`) aren't committed, so the `probe.py` line needs them retrained first.
