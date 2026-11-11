#!/usr/bin/env bash
# Trains and scores one commit in its own worktree, so results always match the code that made them.
#   scripts/run_experiment.sh <name> <commit> [seed]
# PY is the Python to train with (default: train/.venv in this checkout).
# TRAIN_ARGS passes extra options to train.py, such as "--layers 1" or "--ablate conv".
# Test sets are only scored when SCORE_TEST=1: use that for the baseline and the final model, not each run.
set -euo pipefail
name=$1; commit=$2; seed=${3:-1}
repo=$(cd "$(dirname "$0")/.." && pwd)
results=$repo/results
tree=$repo/../gpu-regexp-exp/$name
mkdir -p "$results"
rm -rf "$tree"
git -C "$repo" worktree prune
git -C "$repo" worktree add -q --detach "$tree" "$commit"
cd "$tree"
py=${PY:-$repo/train/.venv/bin/python}

"$py" train/train.py --steps "${STEPS:-15000}" --bs 512 --workers "${WORKERS:-4}" --threads 2 --seed "$seed" --out train/out ${TRAIN_ARGS:-} > train/train.log 2>&1
"$py" train/export.py train/out/tagger.pt
"$py" train/dump_oracle.py
node eval/roundtrip.mjs 3
node eval/parity.mjs
"$py" train/evaluate.py train/out/tagger.pt "$results/$name.generated.json"
node eval/score.mjs eval/dev-v2.json --json "$results/$name.dev.json"
if [ "${SCORE_TEST:-0}" = 1 ]; then node eval/score.mjs eval/test-v2.json --json "$results/$name.test.json"; fi
if [ -d "$repo/eval/external" ] && [ -f eval/streg.mjs ]; then
  cp -r "$repo/eval/external" eval/
  node eval/streg.mjs dev --json "$results/$name.streg-dev.json"
  if [ -f eval/refuse.mjs ]; then node eval/refuse.mjs dev --json "$results/$name.refuse-dev.json"; fi
fi
cp train/out/tagger.pt "$results/$name.pt"
tail -1 train/train.log
echo "experiment $name done at $(git rev-parse --short HEAD), seed $seed"
