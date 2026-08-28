# Third-party benchmark code & data

## Vendored in this repo (`third_party/`)

These directories are shipped directly (code + benchmark inputs). Run/eval
artifacts from our experiments were stripped.

| dir | what | stripped from the working copy |
|---|---|---|
| `arena-hard-auto/` | Arena-Hard-Auto harness (`gen_answer.py`/`gen_judgment.py`/`show_result.py`) + `data/arena-hard-v2.0/question.jsonl` + the pairwise baseline answers (`model_answer/gemini-2.0-flash-001.jsonl` for creative_writing, `model_answer/o3-mini-2025-01-31.jsonl` for hard_prompt) | all `model_judgment/*` and non-baseline `model_answer/*` (run artifacts), `data/arena-hard-v0.1/`, upstream extras unused by the suite (leaderboard browser/builder, bedrock configs) |
| `creative-writing-bench/` | EQ-Bench Creative Writing v3 harness + `data/` | `results/` (HTML reports), per-run `creative_bench_runs*.json` / `elo_results*.json` (see PRISTINE note below), and `elo_results.zip` (the ELO ladder — unused, the suite runs `--no-elo`) |
| `instruction_following_eval/` | official IFEval scoring lib (`evaluation_lib.py`) | upstream CLI/tests/sample responses unused by `score_ifeval.py` |

`arena-hard-auto/config/api_config.yaml` ships with only local `127.0.0.1`
endpoints (the `qwen36-judge`/`gemma4-judge` judges plus prior-run model
slugs); the orchestrator force-writes the evaluated checkpoint's endpoint
entry and the judge ports on every run.

**These three config files are tracked, so running the eval leaves `git status`
dirty** (`api_config.yaml`, `arena-hard-v2.0.yaml`, `gen_answer_config.yaml`).
`api_config.yaml` also keeps one entry per evaluated slug, so entries accumulate
across runs. Check out the config directory before committing, or you will
publish machine-local ports and internal run slugs.

### PRISTINE baseline state for EQ-Bench (one-time setup)

The orchestrator resets the EQ-Bench working state from
`creative_bench_runs.PRISTINE.json` before every run. That file is byte-identical
to the shipped zip — recreate it once. The harness also needs a `.env`; copy the
documented local-only `.env_example` that sits beside it:

```bash
cd third_party/creative-writing-bench
unzip -p creative_bench_runs.zip    > creative_bench_runs.PRISTINE.json
cp .env_example .env
```

`run_eval.sh` performs that same copy automatically when `.env` is missing, so
it is a convenience rather than a prerequisite — but the file must never be
*deleted*: with no `.env` the harness silently falls back to
`https://api.openai.com`. `.env_example` documents every field and notes that
`run_eval.sh` rewrites both endpoint ports per run, so it needs no hand-editing.

Upstream's `elo_results.zip` is **not** shipped: this suite scores EQ-Bench by
rubric only and runs the harness with `--no-elo`, so the ELO ladder is never
loaded. (Upstream's own README still documents the ELO flow; it does not apply
here.)

### Second working copies for 2-lane eval

Lane B of an 8-GPU 2-lane sweep (`PORT_BASE=9500` + `ARENA_DIR`/`BENCH_DIR`,
see `benchmarks/README.md`) uses working copies suffixed `-lane2` so two
concurrent lanes never share benchmark working files:

```bash
cp -r third_party/arena-hard-auto        third_party/arena-hard-auto-lane2
cp -r third_party/creative-writing-bench third_party/creative-writing-bench-lane2
```

## Cloned at setup time (NOT shipped)

| repo | pinned commit | clone into | local patch |
|---|---|---|---|
| https://github.com/tatsu-lab/alpaca_eval | `cd543a1` | `third_party/alpaca_eval` | `patches/alpaca_eval-pairwise_evaluator.patch` (dtype fix that pandas ≥ 3.0 enforces; a no-op on the pandas 2.x pinned in the genbench env — apply with `git apply`) |
| https://github.com/suzgunmirac/BIG-Bench-Hard | `9ee07bd` | `third_party/BIG-Bench-Hard` | none |

```bash
cd third_party
git clone https://github.com/tatsu-lab/alpaca_eval && git -C alpaca_eval checkout cd543a1
git -C alpaca_eval apply ../patches/alpaca_eval-pairwise_evaluator.patch
git clone https://github.com/suzgunmirac/BIG-Bench-Hard && git -C BIG-Bench-Hard checkout 9ee07bd
```

After cloning, install alpaca_eval editable into the genbench env and copy the
local-judge annotator config into it (back in `benchmarks/`, not `third_party/`):

```bash
cd ..    # the block above left you in benchmarks/third_party
uv pip install --python <genbench-venv>/bin/python -e third_party/alpaca_eval
cp -r scripts/eval_gen/alpaca_annotator \
  third_party/alpaca_eval/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_local_qwen36
```
