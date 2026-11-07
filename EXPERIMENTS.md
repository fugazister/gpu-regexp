# Experiments

## Status

- **Done:** Round 1 data fixes, an independent score, the sentence period fix, and Round 2 (refusal). The E7-s2 model is shipped.
- **Now:** Nothing is running.
- **Next:** Train on StructuredRegex's real descriptions and upgrade the compiler (see [Next](#next)).

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
- **Overconfidence:** High probability doesn't mean correct. `asdf qwer zxcv` gives `/zxcv/` at confidence 1.00.

## Independent check: StructuredRegex

**Short answer:** On descriptions written by other people, the first score was about 2%, and about 6% of the ones the grammar could express. The 84% above doesn't carry over to this data. The sentence period fix (E6) raised it to 4.6% and 10%.

[StructuredRegex](https://github.com/xiye17/StructuredRegex) has descriptions from crowdworkers who never saw this project, each with strings that should and shouldn't match. Run `eval/streg.mjs` to score it. The table below is E5-s2, which wasn't retrained for this.

| Split | Descriptions (regexes) | Passed | In grammar scope | Passed in scope |
|---|---:|---:|---:|---:|
| `dev` (read freely) | 351 (120) | 11 (3.1%, CI 0.9–5.2) | 64 | 5 (7.8%, CI 1.5–15.6) |
| `teste` (totals only) | 367 (340) | 9 (2.5%, CI 1.1–4.3) | 57 | 3 (5.3%, CI 0.0–12.3) |
| `testi` (totals only) | 629 (336) | 11 (1.7%, CI 0.8–3.0) | 102 | 6 (5.9%, CI 1.0–10.8) |

Rules, set before scoring the two test splits:

- **Scope is judged from the gold regex alone,** never from the model's output: a sequence of single quantified characters or classes, optionally inside starts-with, ends-with or contains, or a top-level choice between atoms. Only about 17% of descriptions qualify, because the rest use nested `and`, `not` or repeated groups.
- **Confidence intervals resample by regex,** since descriptions of one regex share their examples and aren't independent.
- **Whole-string scoring barely helps.** Counting a regex that passes once anchored adds 0 to 3 prompts per split, so missing anchors aren't the problem.

### Why it fails, from reading `dev`

- **Sentence periods become literals.** `A string that contains a '*'.` gives `^.*\*\..*$`, which demands a real period. Removing sentence-ending periods from `dev` raised passes from 11 to 18, and in scope from 5 to 9. It's the cheapest fix available, and the training data explains it: across 20,000 generated prompts, a period ended the prompt only 16 times and was never labelled as filler.
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

### E6: Sentence periods are filler

**Hypothesis:** A period closing a sentence is filler, and the model only needs to see that. **Change:** The grammar now ends 30% of prompts with a period and sometimes puts one between clauses. Architecture and size are unchanged.

| Passed | E5-s1 | E5-s2 | E6-s1 (shipped) | E6-s2 |
|---|---:|---:|---:|---:|
| StructuredRegex `dev` /351 | 7 | 11 | 22 | 20 |
| StructuredRegex `teste` + `testi` /996 | 16 | 20 | **46** | 39 |
| In scope, `teste` + `testi` /159 | 6 | 9 | **16** | 15 |
| Own test set /201 | 169 | 169 | 167 | 168 |
| Own dev set /60 | 56 | 57 | 57 | 57 |

- **The gain held on data it never saw.** The fix came from reading `dev`, so `dev` alone would flatter it. E5 → E6 on `teste` was 9 gained, 1 lost (seed 1) and 4 gained, 1 lost (seed 2). On `testi` it was 23 gained, 1 lost and 19 gained, 3 lost. The p-values treat descriptions as independent, so they're somewhat too small; three of the four are below 0.05, and seed 2 on `teste` isn't (p=0.375).
- **It costs little on the old set.** The own test set moved by 1 to 2 prompts (p=0.69 and p=1.0), which is within noise, though it's in the wrong direction.
- **E6-s1 was picked before seeing the frozen splits,** because it scored higher on `dev` (22 vs 20). Averaging the seeds, 42.5/996 (4.3%) is the fairer estimate.
- **Fixing the training data can make overconfidence worse.** `asdf qwer zxcv` went from `/asdfzxcv/` at .48 to `/zxcv/` at 1.00. One possible cause, not tested: more filler in training could make the model drop unknown words more readily.

**A mistake and what it showed:** A shell bug ran the first two "seeds" both with seed 1. A true seed 2 replaced it, and the accident became a useful measurement. **The same seed doesn't reproduce exactly:** the repeat flipped 2 prompts on StructuredRegex `dev` and 3 on the own test set, probably because training workers feed batches in an unpredictable order, though GPU nondeterminism is also possible. Either way, part of what these logs call seed variance is really this. The repeat's scores are in `results/E6-s1-repeat.*`.

### How far could the score go?

How much of StructuredRegex `train` + `dev` (2,524 descriptions) becomes expressible with each compiler upgrade. This is a ceiling, since it says nothing about reading the description correctly:

| Added support | Expressible |
|---|---:|
| Today | 17% |
| Class choices in a sequence (`digit or letter`) | 30% |
| Repeated groups and "not this class" | 49% |
| Top-level `and` / `not` constraints | 77% |
| Nested `and` / `not` inside a sequence | 100% |

## Round 2: Teaching it to refuse

**Short answer:** Yes for short input, and only partly for real prose. Refusal went from 5% to 98% on the hand-written test, but real prose shows 66%, and that's the fairer number.

**Hypothesis:** The model never saw text that isn't a request, so it had no way to learn "none of this is a pattern". **Change (E7):** 12% of training samples are now non-requests (keyboard mash, word salad, questions, chat, code, foreign text, symbols), with every word labelled filler. If nothing is tagged, the compiler returns no regex. A lone word still counts as a literal. Architecture and size are unchanged.

| | E6-s1 | E6-s1, threshold .98 | E7-s1 | E7-s2 (shipped) |
|---|---:|---:|---:|---:|
| Hand-written non-requests refused /60 (frozen) | 3 | 55 | 59 | **59** |
| Real prose refused /284 | 3 | | 146 | **188** |
| Own test set passed /201 | 167 | 147 | 166 | **167** |
| Own dev set passed /60 | 57 | 53 | 57 | **58** |
| `natural` test prompts passed /45 | 20 | | 23 | **21** |
| StructuredRegex `teste` + `testi` passed /996 | 46 | 13 | 57 | **43** |
| StructuredRegex in scope passed /159 | 16 | | 25 | **18** |

In the threshold column, a prompt passes only if the model also answers it. That threshold was chosen on dev, where it refused about as much as E7 does. Blank cells weren't measured.

- **Training on negatives beat a confidence threshold.** A threshold on E6 reached 55/60 refusals, but it lost 20 good answers on the test set. E7 reached 59/60 without losing a net answer: 4 gained and 4 lost against E6-s1 (p=1.0). A threshold can only filter what the model already believes, while negatives change what it believes.
- **The threshold is 0.** The sweep on dev showed that a higher threshold on E7 only loses good answers, so the product needed no change. Seed 2 was chosen on dev (29/30 refused, 58/60 passed) before any frozen set was scored.
- **Real requests held up, as far as these sets can tell.** StructuredRegex went from 46 to 43 (`teste` 1 gained and 6 lost, p=0.125; `testi` 9 gained and 7 lost, p=0.80). The `teste` split leans the wrong way, so it's worth watching. Averaged over seeds, E7 gets 50/996 and E6 gets 42.5/996.

### The frozen test was too easy

The test set was frozen before the negatives were written, but one author wrote both, in the same seven categories. So it mostly measures whether the model learned my idea of a non-request. After scoring it, I added a check with text nobody wrote for this project: 284 sentences from the GPL-3.0, Apache-2.0 and MPL-2.0 licences and [the article that inspired this project](https://shud.in/thoughts/ideas-behind-gpu-lexer). None of them asks for a pattern.

| Refused, real prose | E6-s1 | E7-s1 | E7-s2 |
|---|---:|---:|---:|
| All /284 | 3 (1%) | 146 (51%) | **188 (66%, 95% CI about 60–71)** |
| 4–8 words /65 | 1 | 46 (71%) | 51 (78%) |
| 9–15 words /86 | 1 | 53 (62%) | 66 (77%) |
| 16–30 words /133 | 1 | 47 (35%) | 71 (53%) |

- **The same data can give 98% or 66%.** The gap between those two numbers is distribution shift, and it's the same lesson as `natural` in Round 1. A test written by the author of the training data flatters the model.
- **Long input refuses less.** One tagged word is enough to produce a regex, so more words may mean more chances to tag one. That's a hypothesis, not a tested cause.
- **Seeds differ a lot here.** The same code gives 51% or 66% on prose, which the 59/60 on both seeds completely hides.
- **Limits of this check:** It was added after the frozen results, the sentences are mostly legal text, and it counts any regex as wrong even where a sentence could loosely be read as a request. The CI treats sentences as independent, so it's too narrow.

### Confidence is still not trustworthy

On real requests (own dev and StructuredRegex `dev`, E7-s2), answers with confidence from .80 to .95 are right 21.6% of the time, and .95 to .99 only 14.3%. Above .99 they're right 82.7%. Confidence is low on many wrong prose answers: at threshold .6, E7-s2 would refuse 225/284 prose sentences and lose no good answers on own dev, but StructuredRegex `dev` would fall from 20 to 17. This wasn't applied, because it was noticed after the frozen results were seen.

## What's still broken

Prompts below are from the shipped model, on dev or made up by hand:

| Prompt | Output | Problem |
|---|---|---|
| `what time is it` | `/(?:[01]\d\|2[0-3]):[0-5]\d(?::[0-5]\d)?/`, confidence .95 | Should refuse: a question that contains a pattern word |
| `Gray is plain text, including the spaces.` | `/ +/`, confidence .90 | Should refuse: real prose often gets through |
| `numbers with commas as thousands separators` | `/\d+,+/` | Not in the grammar, though it passes the lenient dev check |
| `a string of exactly ten digits` | `/\d+/` | Loses the count, cause unknown |
| `a filename with no spaces` | `/[^ ]*/` | New in E7-s2: fails a dev check that E6-s1 passed |
| `(ab) three times` | `/(?:\(ab\)){3}/` | No group repetition (the parentheses are literal) |

## Next

**Round 3 (recommended): make real descriptions readable.** Train on StructuredRegex `train` and upgrade the compiler, items 3 and 4 below. Refusal is good enough for short input, and accuracy on real descriptions is still the biggest gap.

**Parked: better refusal on prose (E8).** Add real prose from a new source to the negatives, then test a threshold around .6. The licences and article have now been read, so they count as dev data, and measuring this honestly needs prose that hasn't been looked at.

**After that:**

1. **Examples as a safety net.** Generate the top few candidate regexes and test them against strings the user says should or shouldn't match. This can fix wrong answers without improving the model, and it's the best showcase in this list.
2. **Model experiments.** Ablation of each layer, learning curves (does more data still help?), a CRF output layer, and more seeds.
3. **Train on real descriptions.** StructuredRegex `train` has 2,173, but no word-by-word labels, so labels would have to be inferred and kept only if they compile to an equivalent regex. This is the biggest lever on in-scope failures. It would use only `train`, keeping `dev`, `teste` and `testi` for measuring.
4. **Upgrade the compiler,** working down the expressibility table from the top.
5. **LLM-paraphrased training data, verified by the compiler.** Another direct attack on `natural`. [KB13 and NL-RX-Turk](https://github.com/nicholaslocascio/deep-regex) are also available as further independent tests, once match strings are generated for them.

## Reproduce

```sh
scripts/run_experiment.sh E5-s2 d4eaada 2       # on a GPU machine; SCORE_TEST=1 also scores test
node eval/compare.mjs results/E0-s2.dev.json results/E5-s2.dev.json --show
train/.venv/bin/python train/probe.py results/E0-s1.pt results/E5-s2.pt
SCORE_TEST=1 scripts/run_experiment.sh E7-s2 19c3299 2
node eval/refuse.mjs test --threshold 0         # frozen refusal test, with the model in web/
node eval/refuse-real.mjs /usr/share/licenses/spdx/GPL-3.0-only.txt   # any text files; the article text isn't committed
```

Score files are in `results/`. Checkpoints (`*.pt`) aren't committed, so the `probe.py` line needs them retrained first.
