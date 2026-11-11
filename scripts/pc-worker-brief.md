# Task: time gpu-regexp training on the PC's CPU (E13 on CPU, seeds 1 and 2)

You are a worker on Max's PC. Your worktree is a fresh checkout of `gutlab/gpu-regexp` at `origin/main`. Read `HANDOFF.md` ("How to run an experiment") for background. Max approved this run (forum topic t/169, post 12).

**Goal:** train the live model's setup (commit `7198d09`) on the CPU with seeds 1 and 2, one after the other, and report how long each seed takes and its dev scores. The coordinator compares these with the earlier GPU runs E13-s1 and E13-s2.

## Rules

- **Don't change code.** You only run the experiment and commit result JSONs.
- **Never push to `main`, never force-push, and never print `$FORGEJO_AGENT_TOKEN`.** Push only to the branch `exp/e13-cpu`.
- **Never score test sets.** Don't set `SCORE_TEST`, and never print or read `eval/refuse-prose-test.txt`.
- **Keep CPU use low.** The 12 cores are shared with other workers. Use `WORKERS=2` and the script's 2 torch threads, and export `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- **Max can take the PC back at any time.** Commit and push after each seed.

## Steps

1. **Check the venv.** `cat /home/orca/.venvs/gpu-regexp/.pins` must print `torch==2.14.1 numpy==2.5.4`. If the file is missing, wait up to 15 minutes (re-check every minute), because the venv may still be building. If it's still missing, stop and report `worker_done` as failed: "venv missing".
   Also run `/home/orca/.venvs/gpu-regexp/bin/python -c "import torch,numpy; print(torch.__version__, numpy.__version__, torch.cuda.is_available())"` and `node --version`, and keep the output for the report.
2. **Fetch StructuredRegex dev** (MIT licence, not in git) into `eval/external/streg/`, pinned to commit `7d49aa2a176677fb5a553b05a6ad7d8fc24bc37d`:
   ```sh
   mkdir -p eval/external/streg
   for s in dev teste testi; do
     curl -fsSL "https://raw.githubusercontent.com/xiye17/StructuredRegex/7d49aa2a176677fb5a553b05a6ad7d8fc24bc37d/data/raw/$s.tsv" -o eval/external/streg/$s.tsv
   done
   wc -l eval/external/streg/*.tsv
   ```
   `eval/external/` is gitignored, so never commit it. If the download fails, keep going: the run then skips StructuredRegex dev, and you say so in the report.
3. **Run seed 1, then seed 2**, from the worktree root, each with a 2-hour limit:
   ```sh
   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   for seed in 1 2; do
     start=$(date +%s)
     PY=/home/orca/.venvs/gpu-regexp/bin/python WORKERS=2 timeout 2h scripts/run_experiment.sh E13-cpu-s$seed 7198d09 $seed > /tmp/E13-cpu-s$seed.log 2>&1
     echo "seed $seed exit $? seconds $(( $(date +%s) - start ))"
   done
   ```
   - `run_experiment.sh` makes its own worktree at `../gpu-regexp-exp/E13-cpu-s<seed>` and writes `results/E13-cpu-s<seed>.*.json` here.
   - If a seed exits non-zero or hits the timeout (exit 124), don't start the other seed. Report the last 40 lines of its log (`/tmp/E13-cpu-s<seed>.log`, plus `../gpu-regexp-exp/E13-cpu-s<seed>/train/train.log`).
   - After each seed that finishes, commit its JSON files (not `*.pt`) on branch `exp/e13-cpu`, and push: `git push origin HEAD:refs/heads/exp/e13-cpu`.
     Commit message: `results: E13-cpu-s<seed>, E13 trained on the PC's CPU`, with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
4. **Report** with `worker_done`. Include:
   - versions (torch, numpy, CUDA available, node) and the StructuredRegex fetch result
   - for each seed: exit code, wall-clock seconds, and the last line of `train.log` (it has the generated-data accuracy)
   - for each seed, the `pass` field of `results/E13-cpu-s<seed>.dev.json` and `.streg-dev.json`, and the round trip and parity output from its log (the coordinator reads the other JSONs from the branch)
   - the pushed commit hash on `exp/e13-cpu`

**Done means:** both seeds finished and their results are pushed. A failure (missing venv, crash, timeout) is a valid outcome if the report says which step failed and quotes the error.
