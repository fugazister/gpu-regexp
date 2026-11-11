# Experiments

## Status

- **Done:** Round 1 data fixes, an independent score, the sentence period fix, Round 2 (refusal), Round 3 (real descriptions), Round 4 (typos), Round 5 (relabelling), Round 6 (a compiler fix; E12 not shipped), Round 7 (examples as a safety net), Round 8 (refusing real prose), and Round 9 (model experiments; nothing shipped). The E13-s2 model is live.
- **Now:** Nothing is running.
- **Next:** Data that looks like real users (see [Next](#next)).

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

## Round 3: Training on real descriptions

**Short answer:** Yes. The score on descriptions the model never saw more than doubled, from 43 to 93 of 996, while the own test set held within noise. It cost typo handling.

**The problem:** StructuredRegex has 2,173 training descriptions, but only a regex and example strings for each, not the word-by-word roles this model trains on. Labelling them by hand wasn't an option.

**The method: learning from denotations.** `train/label_real.mjs` starts from the current model's roles and runs a beam search that changes as few words as it can, until the compiled regex behaves like the gold one. A label is kept only if it agrees with the gold regex on the dataset's examples and on about 120 strings generated from the gold regex.

**The known trap is spurious labels.** They give the right regex for the wrong reason. Reading samples found `starts/START with/END`, `first/ONLY`, `string/Q_PLUS` and `times/NEG`, all passing every check. Training on those would add the kind of label noise that E1 removed, so two filters were added:

- A role with no regex text of its own must sit on a phrase the grammar really uses for that role. Otherwise the word goes back to filler, and if the regex then breaks, the description is dropped.
- Labels that would teach something false about common phrases are rejected, such as "ends with" labelled as filler.

| From 2,173 training descriptions | Count |
|---|---:|
| Share a regex with a held-out split, so excluded | 14 |
| Compiler can reach the regex | 725 |
| Search found a labelling | 284 |
| After the filters | **192, across 127 regexes** |

Reading 30 of the final labels by hand found 3 with a small error, such as `3 to 4` with `to` left as filler, so some noise remains. These 192 make up 15% of each batch, with their counts randomly swapped so they're harder to memorise.

**Two changes, measured separately:**

- **E8: the compiler and grammar.** "Starts with a and ends with b" leaves a gap. With three or more parts in order, there's now no gap, because that describes the whole string.
- **E9: E8 plus the 192 real descriptions.** Seed 2 was chosen on dev before any frozen set was scored.

| | E7-s1 | E7-s2 | E8-s1 | E8-s2 | E9-s1 | E9-s2 (shipped) |
|---|---:|---:|---:|---:|---:|---:|
| StructuredRegex `dev` /351 | 26 | 20 | 28 | 28 | 47 | **53** |
| StructuredRegex `teste` + `testi` /996 | 57 | 43 | 59 | 52 | 90 | **93** |
| In scope, `teste` + `testi` /159 | 25 | 18 | 28 | 20 | 40 | **40** |
| Same, forgiving a missing anchor /996 | 62 | 53 | 68 | 62 | 124 | **122** |
| Own test set /201 | 166 | 167 | 161 | 164 | 161 | **165** |
| `natural` /45 | 23 | 21 | 20 | 20 | 22 | **24** |
| `typos` /20 | 15 | 18 | 15 | 16 | 13 | **11** |
| Own dev set /60 | 57 | 58 | 57 | 57 | 57 | **58** |
| Non-requests refused /60 | 59 | 59 | 58 | 60 | 57 | **58** |
| Real prose refused /284 | 146 | 188 | 176 | 185 | 165 | **169** |

- **The gain held on unseen data.** E7-s2 → E9-s2 was 19 gained and 4 lost on `teste` (p=0.003), and 40 gained and 5 lost on `testi` (p<0.0001). Both E9 seeds agree. The p-values treat descriptions as independent, so they're somewhat too small.
- **The own test set held.** E7-s2 → E9-s2 was 7 gained and 9 lost (p=0.80). On E9-s1 it was 4 gained and 10 lost (p=0.18), which is within noise but leans the wrong way.
- **The compiler rule alone isn't proven.** E8 looked good on dev (20 → 28), but on the frozen splits E8-s2 was 3 gained and 2 lost, then 16 gained and 8 lost (p=0.15). It went live with E7 weights before those splits were scored, which was an unplanned side effect of pushing the code.
- **Small data went a long way, for a reason worth checking.** 192 sentences beat about 7.7 million synthetic samples on this measure. A likely reason is that they cover phrasing the grammar never produces, but that's a hypothesis.

### What it cost

- **Typos got worse, and this looks real.** `typos` fell from 18 to 11 on the shipped seed, and both E9 seeds are below both E7 seeds. `digts folowed by a dahs` now gives `/dahs/`. The dev sets didn't show this, so it was only found on the test set, and any fix needs its own dev prompts. One possible cause, not tested: real descriptions replace 15% of each batch and have few typos.
- **Prose refusal didn't really move.** The shipped seed went from 188 to 169 of 284, but that comparison flatters E7-s2. Averaged over seeds, E7 and E9 both refuse 167.
- **Confidence is still unreliable.** Answers above .99 are right 78.9% of the time on real requests, and those from .95 to .99 only 37.9%.

### What's left

- **Most reachable descriptions still aren't used.** 192 of 725 is 26%. Relabelling with E9 could find more, because a better model starts closer to the answer.
- **29 more correct answers only lack an anchor.** Forgiving a missing anchor gives 122 instead of 93, so the model reads those descriptions correctly and the regex just isn't tied to the whole string.

## Round 4: Typos

**Short answer:** Typos were weak all along, not only after E9, and E10 fixes most of the gap. It may cost a little elsewhere, which can't be told apart from seed noise yet.

### The dev set showed an old gap

A new dev set (`eval/typos-dev.json`, 39 prompts) gives each misspelt prompt a correctly spelt twin (`eval/typos-dev-clean.json`). Missing a twin means the model can't do that prompt at all, and missing only the typo version means the typo cost it.

| Model | Typo prompts /39 | Clean twins /39 |
|---|---:|---:|
| E7-s1, E7-s2 | 23, 22 | 35, 35 |
| E9-s1, E9-s2 | 22, 18 | 36, 35 |
| **E10-s1, E10-s2** | **30, 29** | 36, 35 |

E7 and E9 lose about 14 prompts to typos, so most of the 18 → 11 drop on the test set was seed noise on top of a gap that was already there. Reading the roles showed the cause: the tagger labels the misspelt word as filler (`numbrs/O`, `begins/START wiht/O abc/O`, `a tiem like 12:30` all `O`). When the role is right, the lexicon already fixes the spelling (`leters` → letters).

The grammar explained it. Only 10% of samples had a typo, always in one word of five letters or more, and only from three kinds of edit. So `with`, `two`, `and`, `date` and `time` were never misspelt, and keyboard slips or words run together (`endswith`) never appeared.

### E10: more kinds of typos

E10 (`ede75ca`) puts typos in 25% of samples, with one or two words, three letters or more. The new edits are keyboard-neighbour substitutions, inserted letters, swaps that include the first letter, and words run together. Nothing else changed.

| | E9-s2 (was live) | E10-s2 (shipped) |
|---|---:|---:|
| `typos` /20 (frozen) | 11 | **18** |
| Own test /201 (frozen) | 165 | 167 (11 gained, 9 lost, p=0.82) |
| `natural` /45 (frozen) | 24 | 22 |
| StructuredRegex `teste`+`testi` /996 (frozen) | 93 | 84 (`teste` 5/9, p=0.42; `testi` 7/12, p=0.36) |
| Non-requests refused /60 (frozen) | 58 | 55 |
| Real prose refused /284 | 169 | 164 |

- **The seed was chosen on dev.** E10-s2 scored 29 against 30 on typo dev, 51 against 47 on StructuredRegex dev, and refused 29/30 against 28/30, so it was picked before the frozen sets were scored.
- **The typo gain is clear.** Both seeds gain about 9 on typo dev with no change on the clean twins, and the frozen `typos` went from 11 to 18. `digts folowed by a dahs` now gives `/\d+/` (the dash is still lost).
- **The losses aren't significant, and they may be real.** Averaged over seeds on dev, E9 and E10 are within one prompt on StructuredRegex (50 vs 49), own dev (57.5 vs 56.5) and refusal. E9-s2 was also the luckier of its seeds. The frozen StructuredRegex drop leans the wrong way, though, and a third seed would show whether it's noise.
- **Prose refusal drops in both seeds.** E9 averaged 167 of 284 and E10 averages 155 (146 and 164). One possible cause, not tested: ordinary prose with misspellings now looks more like keywords.

## Round 5: Relabelling with a better model

**Short answer:** Starting the label search from E10 found 45 more labels, but reading them showed many were right only by luck. With stricter checks, E11 wins back most of what E10 lost on StructuredRegex and keeps the typo gain.

### More labels, and more luck

Starting from E10, the search found every old label again with the same roles, plus 45 new ones. Many of the new ones passed only by luck:

- `[r/NUM] lowercase letters`: a letter tagged as a count.
- `more than [3] [numbers]` gives `(?:\d+){3}`: the gold regex is just as sloppy, so the label passes while teaching that "more than" is filler.
- `2 or more digits . However they can [not] be [Capital Letters]`: "digits" left as filler, and `[^A-Z]` happens to match.
- `any combination of letters, special characters, or digits`: "digits" left as filler, and the `.` covers it.

Three checks now drop such labels: a tag of NUM on a one- or two-letter word, a quantity phrase left as filler ("more than", "or more", "at least"), and a class word left as filler with no literal, class word or negation next to it. The first version also dropped "4 more letters" (meaning another four), "characters that are not letters", "a number of" and "letters in lowercase", which cut 21 good old labels. After narrowing, 5 old labels go, all for good reason. The result is 226 labels over 139 regexes (was 192 over 127).

### E11

E11 (`5074d3b`) changes only the labels. The seeds tied on dev (`typos-dev` 28 vs 31, own dev 58 vs 56, StructuredRegex dev 53 vs 51, refusal 27 vs 28 of 30), so real prose (149 vs 167 of 284) picked E11-s2 before the frozen sets were scored.

| Frozen sets, scored once | E9-s2 | E10-s2 | **E11-s2 (shipped)** |
|---|---:|---:|---:|
| StructuredRegex `teste`+`testi` /996 | 93 | 84 | **91** |
| StructuredRegex in scope /159 | 40 | 36 | **41** |
| `typos` /20 | 11 | 18 | **18** |
| Own test /201 | 165 | 167 | 166 |
| Non-requests refused /60 | 58 | 55 | 56 |
| Real prose refused /284 | 169 | 164 | 167 |

- **Not significant on its own.** Against E10-s2, `teste` is 4 gained and 4 lost, `testi` 18 gained and 11 lost (p=0.26), and own test 4 gained and 5 lost (p=1.0). The direction matches dev, where the E11 seeds average 52 on StructuredRegex and E10 averages 49.
- **Averaged over seeds, E10's possible cost is mostly gone.** E11 matches E10 on typo dev (29.5) and own dev (57 vs 56.5), with prose at 158 vs 155 and refusal dev at 27.5 vs 28.5.
- **Something new to watch:** 2 test prompts get no regex, where E10-s2 had none, and `negation` fell from 14 to 12. Confidence also fell on several prompts in the table below.
- **Checks:** round trip 100%, CPU parity OK, real GPU 0 role differences. A game was running on the GPU machine during that 5-second check, against the rule of not loading it then: the command printed the check instead of stopping on it. GPU commands now stop first if it's running.

## Round 6: The compiler, and "consists of"

**Short answer:** A one-line compiler fix shipped. E12 taught "the string consists of" as a whole-string match, and it worked on those prompts, but it didn't beat E11 overall, so it wasn't shipped.

### Counts on "numbers"

"number" and "numbers" are a preset meaning `\d+`, so a count wrapped the whole run: "3 numbers" gave `(?:\d+){3}`, which means three or more digits. A count on `\d+` now applies to the digit, so "3 numbers" gives `\d{3}` and "3 or more numbers" gives `\d{3,}` (`b197785`, shipped as `e0be2b8`).

With the E11-s2 weights, dev didn't change except one more StructuredRegex dev answer on E11-s1. The fix changed 39 of 1,197 frozen outputs and none on the own test. Scored once: StructuredRegex 91 → 93 of 996 (2 gained, 0 lost), with own test and refusal unchanged.

### E12: "the string consists of"

About 20% of StructuredRegex descriptions say "the string consists of …", which means a whole-string match, and the grammar never used the phrase. E12 (`1e058d7`, branch `exp/e12`) adds "consists of", "made up of", "composed of" and similar to the grammar's whole-string leads. When a real label passes only as a whole-string match and its description uses one of those phrases, the labeller now tags the phrase as whole-string. That gave 253 labels (was 226), 37 of them with the whole-string role (was 6). Both seeds were trained on the laptop, because the GPU machine was in use.

| Dev, two seeds each | E11 (new compiler) | E12 |
|---|---:|---:|
| StructuredRegex dev /351 | 54, 51 | 50, 59 |
| `typos-dev` /39 | 28, 31 | 30, 27 |
| Own dev /60 | 58, 56 | 56, 56 |
| Non-requests refused /30 | 27, 28 | 28, 26 |
| Real prose refused /284 | 149, 167 | 154, 151 |

- **It learned the phrase.** "The string consists of 3 letters followed by 2 digits" gives `^[a-zA-Z]{3}\d{2}$`, where E11 gave no anchors.
- **It targeted too little.** Only 2 dev descriptions were waiting on this phrase. The 29 answers that lacked only an anchor mostly use other wording.
- **Not shipped.** The StructuredRegex gain comes from one seed (+8, against −1 for the other), and both seeds lose a little on typos and refusal. The frozen sets weren't scored for E12.

## Round 7: Examples as a safety net

**Short answer:** Letting the user give a few strings that should and shouldn't match helps more than any training change so far, and the model doesn't change at all.

### How it works

The tagger already gives every role's probability for each word. `parseWithExamples` lists the 64 most likely labellings in order (each word picks from its three most likely roles). It compiles each one and tries it as written, anchored to the whole string, anchored at the start, then at the end. It returns the first regex that gets every example right. Without examples, or when nothing fits, it returns the plain parse and lists the examples that fail.

### Measuring it honestly

The examples must not grade themselves. `eval/examples.mjs` splits each StructuredRegex description's strings: the first 1 or 3 of each kind go to the picker, and only the last 3 decide pass or fail. The plain parse is scored on the same last 3, so its score here (57 on dev) is higher than the usual 51, which checks every string.

| StructuredRegex, scored on held-out strings | dev /351 | `teste`+`testi` /996 (frozen, scored once) |
|---|---:|---:|
| Description only | 57 | 125 |
| Plus 1 + 1 examples | 77 (20 gained, 0 lost) | 209 (93 gained, 9 lost) |
| **Plus 3 + 3 examples** | **86** (29 gained, 0 lost) | **224** (100 gained, 1 lost) |

- **What it fixes, from reading dev:** missing anchors; less likely roles the tagger nearly chose (`characters that aren't capital letters` → `[^A-Z]{4}` instead of `(?:aren){4}`); and typos (`digts folowed by a dahs` → `\d+-`, reading 2).
- **Where it can hurt:** with 1 + 1 examples, 9 frozen answers that were right became wrong, because a less likely reading fits the two examples but not the held-out strings. With 3 + 3 that falls to 1.
- **64 readings is enough.** On dev, 16 readings gave 81 and 256 gave 87, against 86 for 64.
- **It can't fix what no reading can express.** When nothing fits, the page says so and shows which examples fail.
- **Overfitting to examples is real.** Some fixes pass the held-out strings while still reading the description loosely, for example `(?:[^\w\s]|[a-z])5` for "special characters, lowercase letters, or the number 5".

## Round 8: Refusing real prose

**Short answer:** Training on real book sentences fixed refusal. On two books nobody read, refusal went from 62% to 96%, at no cost to real requests. A confidence threshold couldn't do that.

### Why prose got through

The made-up negatives (word salad, chat, code, foreign text) deliberately left out words the grammar gives a role to, such as "with", "and" and "letters", after E1 showed that contradictory labels hurt. Real prose is full of those words, so the model had never seen them in a sentence that isn't a request.

### E13

E13 (`7198d09`) adds 6,000 sentences from six Project Gutenberg books (Pride and Prejudice, Sherlock Holmes, Origin of Species, Frankenstein, Tom Sawyer, A Tale of Two Cities) as a new kind of negative, about a fifth of all negatives. `train/prose.py` picks them with the same rules `eval/refuse-real.mjs` uses. Two other books, Moby Dick and The Art of War, give a frozen test of 400 sentences (`eval/refuse-prose-test.txt`). It was extracted without being printed, and scored once at the end.

The alternative was a confidence threshold. On dev with E11-s2, a threshold of 0.6 took prose refusal from 167 to 237 of 284, but StructuredRegex dev fell from 51 right to 43.

| Dev | E11-s1, s2 | E13-s1, s2 |
|---|---:|---:|
| Real prose refused /284 (licences and an article) | 149, 167 | **262, 269** |
| Non-requests refused /30 | 27, 28 | 29, 29 |
| StructuredRegex dev right /351 | 54, 51 | 53, 54 |
| Own dev /60 | 58, 56 | 56, 57 |
| `typos-dev` /39 | 28, 31 | 30, 28 |

E13-s2 won on three of the four dev measures, and was picked before the frozen sets were scored.

| Frozen, scored once | E11-s2 (with the Round 6 compiler) | **E13-s2 (shipped)** |
|---|---:|---:|
| Book sentences refused /300 | 187 (62%) | **289 (96%)** |
| Of those, 16–30 words /129 | 66 | 127 |
| Non-requests refused /60 | 56 | 58 |
| StructuredRegex `teste`+`testi` /996 | 93 | 97 (13 gained, 9 lost) |
| Own test /201 | 166 | 167 (3 gained, 2 lost) |
| `typos` /20 | 18 | 16 |

- **Training beat filtering again,** as in Round 2: the threshold only filters what the model believes, and training changed the belief.
- **No cost to real requests could be detected.** StructuredRegex, own test and dev all held or rose. Examples mode on dev gives 88 with 3 + 3 (was 86).
- **Typos fell by 2, probably noise.** On `typos-dev` the two models average the same (29.5 against 29).
- **Long sentences were the problem, and are now the best part:** 127 of 129 refused, against 66 before.
- **Not every refusal case is solved.** `Gray is plain text, including the spaces.` still gives `/ +/`, because it's about spaces.
- **Checks:** round trip 100%, CPU parity OK, real GPU 0 role differences (checked that the game wasn't running first).

## Round 9: What each part of the model is worth

**Short answer:** The model is not what holds this project back. Making it bigger, smaller, or training it longer barely moves the scores on real text. The scan layers and the spelling features matter, and the convolution does nothing.

### The method

Every variant was trained twice with the shipped data and code (`1579663`), and scored on dev only. A new `--ablate` option holds a part at zero during training, so the model keeps its shape and the browser code scores it unchanged. The shipped setup got four seeds to measure noise: StructuredRegex dev ranges from 50 to 54, so only a gap beyond about ±3 means anything. The numbers below are seed averages; every run is in `results/R9-summary.tsv`.

| Variant | Params | Generated | `typos-dev` /39 | StructuredRegex dev /351 | Real prose refused /284 |
|---|---:|---:|---:|---:|---:|
| **Shipped (4 seeds)** | 44,140 | 97.4% | 29.0 | 52.8 | 262 |
| No scan layers | 36,988 | 93.3% | 29.0 | **46.0** | **236** |
| 1 scan layer | 40,564 | 96.5% | 30.5 | 50.0 | 250 |
| 3 scan layers | 47,716 | 98.0% | 30.5 | 50.5 | 268 |
| No convolution | 44,140 | 97.3% | 29.5 | 54.0 | 261 |
| No sentence average | 44,140 | 97.3% | 27.0 | 53.0 | **246** |
| No whole-word hash | 44,140 | 95.9% | 29.5 | **46.5** | 259 |
| Whole-word hash only (no skeleton, prefix, suffix) | 44,140 | 93.2% | **21.0** | 47.5 | 265 |
| Half width (d=16) | 27,388 | 94.8% | 30.0 | 50.0 | 260 |
| Wider (d=32) | 62,940 | 98.3% | 30.5 | 51.5 | 264 |
| 3,000 steps | 44,140 | 93.3% | 27.0 | 53.0 | 245 |
| 7,500 steps | 44,140 | 96.2% | 28.0 | 52.5 | 259 |
| 30,000 steps (shipped: 15,000) | 44,140 | 98.0% | 30.5 | 54.5 | 268 |

Own dev (55.5 to 58 of 60) and hand-written refusal (28.5 to 30 of 30) barely moved for any variant, so they're left out.

### What it shows

- **Distribution shift, measured directly.** From 3,000 to 30,000 steps, accuracy on generated prompts went from 93% to 98%, while StructuredRegex dev stayed between 52.5 and 54.5. More training teaches the grammar better, not real phrasing. The same holds for width: 98.3% generated at d=32, with no gain on real text.
- **Capacity isn't the limit.** Half the width (38% fewer parameters) and 50% more width both land within noise of the shipped model. What would help is data that looks like real users.
- **The scan layers carry context.** Without them, StructuredRegex drops by about 7 and prose refusal by 26. One layer is in between, and a third adds nothing.
- **The convolution does nothing.** Removing it changes no measure. It's only 120 weights, but it could go.
- **The sentence average is what refusal uses.** Without it, prose refusal drops by 16 and the other measures hold. Deciding whether a whole sentence is a request is the one job that needs a view of the whole sentence.
- **Each feature group has its own job.** Without the spelling features (skeleton, prefix, suffix), typo dev falls from 29 to 21, which is the job they were designed for. Without the whole-word hash, StructuredRegex drops by 6, because most words are spelt correctly and the exact word is the strongest clue.
- **Longer training helps refusal a little.** Prose refusal grows from 245 at 3,000 steps to 268 at 30,000. 30,000 steps also gives the best StructuredRegex average (54.5), but within noise, so nothing was shipped.

## What's still broken

Prompts below are from the shipped model, on dev or made up by hand:

| Prompt | Output | Problem |
|---|---|---|
| `digts folowed by a dahs` | `/\d+/`, confidence .76 | Better since E10 (was `/dahs/`), but a misspelt atom at the end still drops out |
| `a filename with no spaces` | `/[^ ]*/`, confidence .56 | Loses the anchors: should be `^[^ ]*$` |
| `Gray is plain text, including the spaces.` | `/ +/`, confidence .94 | Should refuse: real prose often gets through |
| `numbers with commas as thousands separators` | `/\d+,+/` | Not in the grammar, though it passes the lenient dev check |
| `a string of exactly ten digits` | `/\d+/`, confidence .40 | Loses the count, cause unknown |
| `(ab) three times` | `/(?:\(ab\)){3}/` | No group repetition (the parentheses are literal) |

## Next

**Round 10 (recommended): data that looks like real users.** Round 9 showed the model isn't the limit, and the grammar is already learned to 98%. Paraphrase grammar prompts with an LLM, keep only paraphrases whose compiled regex still behaves like the original, and train on them. This is a direct attack on `natural`, still the weakest category.

**After that:**

1. **Remove the convolution** and train 30,000 steps, as a tidy-up with a frozen check.
2. **More independent tests.** [KB13 and NL-RX-Turk](https://github.com/nicholaslocascio/deep-regex) can be used once match strings are generated for them.

## Reproduce

```sh
scripts/run_experiment.sh E5-s2 b0162f6 2       # on a GPU machine; SCORE_TEST=1 also scores test
node eval/compare.mjs results/E0-s2.dev.json results/E5-s2.dev.json --show
train/.venv/bin/python train/probe.py results/E0-s1.pt results/E5-s2.pt
SCORE_TEST=1 scripts/run_experiment.sh E7-s2 aa3b9a5 2
node eval/refuse.mjs test --threshold 0         # frozen refusal test, with the model in web/
SCORE_TEST=1 scripts/run_experiment.sh E9-s2 593157a 2
SCORE_TEST=1 scripts/run_experiment.sh E10-s2 ede75ca 2
node eval/score.mjs eval/typos-dev.json; node eval/score.mjs eval/typos-dev-clean.json
SCORE_TEST=1 scripts/run_experiment.sh E11-s2 5074d3b 2
node eval/examples.mjs dev -v                    # examples as a safety net, scored on held-out strings
SCORE_TEST=1 scripts/run_experiment.sh E13-s2 7198d09 2
node eval/refuse-real.mjs eval/refuse-prose-test.txt   # frozen book prose
STEPS=15000 TRAIN_ARGS="--ablate conv" scripts/run_experiment.sh R9-noconv-s1 1579663 1   # Round 9 variants: --layers, --d, --hidden, --ablate
node train/label_real.mjs train/real_labels.jsonl   # needs StructuredRegex train.tsv in eval/external/streg/
node eval/refuse-real.mjs /usr/share/licenses/spdx/GPL-3.0-only.txt   # any text files; the article text isn't committed
```

Score files are in `results/`. Checkpoints (`*.pt`) aren't committed, so the `probe.py` line needs them retrained first.
