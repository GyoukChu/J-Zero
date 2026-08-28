# R-Zero baseline — paper runs

This directory is the R-Zero baseline as run for the paper: the `recipe_R-Zero`
pipeline (runs inside a `verl` clone as `verl/recipe_R-Zero/`) plus the exact
launcher scripts used for the three baseline runs (Qwen3-4B-Base,
Qwen3-8B-Base, Llama-3.2-3B-Instruct).
`README.md` (from the recipe) documents the pipeline itself; this file maps the
paper runs to their launchers and environments.

## Environment

- Training env: `requirements/requirements-train-vllm011.txt` (Python 3.12, vLLM **0.11**, torch 2.8, transformers 4.56).
- `verl` must be installed editable into that env:
  `uv pip install --python <venv>/bin/python -e <verl_dir> --no-deps`
  (a `uv venv` has no in-venv `pip`), and this directory must sit at
  `<verl_dir>/recipe_R-Zero/` (scripts `cd` relative to it).

## Paper runs → launchers → results

| Run (base model) | Launcher (`launchers/`) |
|---|---|
| Qwen3-4B-Base | `launch_rzero_4b.sh` |
| Qwen3-8B-Base | `launch_rzero_8b.sh` |
| Llama-3.2-3B-Instruct | `launch_rzero_llama3b.sh` |

Every launcher sets env knobs (`BASE_MODEL`, `OUT_ROOT`, `CKPT_ROOT`, `RZERO_NITER`,
GPU layout `RZERO_*_GPUS`) and `exec`s `run_rzero.sh`, which loops
challenger GRPO (5 steps) → question gen (`NUM_Q=8000`) → filter (m=10
majority-vote, band [0.25, 0.75]) → solver GRPO (15 steps) per iteration.
No reward model in this baseline. All three runs used **3 iterations**
(`RZERO_NITER=3`, the launcher default) and `RZERO_SKIP_PER_ITER_EVAL=1`.

> Launchers take machine-specific paths from the environment. Export all of these
> before running:
>
> | env var | meaning |
> |---|---|
> | `RZERO_ENV_FILE` | secrets file with `HF_TOKEN` / `WANDB_API_KEY` (or set `WANDB_MODE=offline`) |
> | `RZERO_VENV` | venv **directory** built from `requirements/requirements-train-vllm011.txt` (the scripts `source $RZERO_VENV/bin/activate`) |
> | `OUT_ROOT` | run artifacts (generated questions, filtered data, logs, evals) |
> | `CKPT_ROOT` | checkpoint root |
> | `DATA_DIR` | **required once this recipe is installed as `<verl>/recipe_R-Zero`** — the 3 verl-format val parquets, shipped at `<repo>/benchmarks/data`. `solver_train.sh` / `evaluate/run_solver_eval.sh` fall back to `$PROJECT_ROOT/data` and to a repo-relative path, but neither exists when the recipe has been copied out of the repo into a verl checkout, so set it explicitly: `export DATA_DIR=<repo>/benchmarks/data` |

## Per-iteration evaluation

`run_rzero.sh` can also run the 7-benchmark solver eval at the end of every
iteration. It is **OFF by default** (the launchers pin it off, and the paper
numbers come from the post-hoc sweep below). Turn it on with
`RZERO_SKIP_PER_ITER_EVAL=0` — `RZERO_SKIP_ITER_EVAL` is an accepted alias for
the same switch, and `pipeline/run_jzero.sh` uses the identical contract.
Turning it on also requires the recheck-judge env vars `JUDGE_VENV_PY` and
`NOTHINK_JINJA` (see `evaluate/serve_recheck_judge.sh`).

Evaluate each iteration's solver checkpoint with the 14-benchmark suite in
`../../benchmarks/`:

```bash
for it in 1 2 3; do
  GPU_SET=0,1,2,3 bash ../../benchmarks/scripts/eval_standard/run_eval.sh \
    "$CKPT_ROOT/iter${it}_solver_full/global_step_15/actor/huggingface" "iter${it}"
done
```
