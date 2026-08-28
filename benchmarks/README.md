# Evaluation Suite (14 benchmarks)

One command evaluates one checkpoint on all 14 benchmarks on 4 GPUs,
with only local judges — **no external API calls anywhere**.

```bash
# requires the one-time setup + env vars below (see "Path portability")
GPU_SET=0,1,2,3 bash scripts/eval_standard/run_eval.sh <model_path> <slug>
# results -> outputs/eval/<slug>/RESULTS.txt
```

| Group | Benchmarks | Protocol |
|---|---|---|
| EM (3) | MMLU-Pro, SuperGPQA, BBH | R-Zero `\boxed{}`, greedy, micro-avg |
| Math (7) | MATH500, MinervaMath, OlympiadBench, GSM8K, AMC23, AIME24, AIME25 | verl math_verify + local Qwen3.6-27B recheck |
| IFEval (1) | google IFEval | official 4 metrics |
| Generation (3) | AlpacaEval 2.0 (LC winrate), Arena-Hard-v2.0, EQ-Bench Creative v3 | local judges: Qwen3.6-27B no-think (AlpacaEval + Arena **hard_prompt**), gemma-4-31b-it (Arena **creative_writing** + EQ) |

Arena-Hard-v2.0 uses **per-category judges** (`arena-hard-auto/utils/judge_utils.py`
→ `JUDGE_SETTINGS`): the `hard_prompt` category (500 qs, style-controlled, vs
baseline `o3-mini-2025-01-31`) is judged by **Qwen3.6-27B (no-thinking)**, and the
`creative_writing` category (250 qs, vs baseline `gemini-2.0-flash-001`) is judged
by **gemma-4-31b-it**. `gen_judgment.py` routes each question to its category's
judge and writes under `model_judgment/<judge>/`; `show_result.py` scores each
category from that judge's dir automatically.

AlpacaEval 2.0 uses the official GPT-4-Turbo anchor (`reference_gpt4turbo.json`,
shipped) with annotator `weighted_alpaca_eval_local_qwen36` (official
`alpaca_eval_clf_gpt4_turbo` prompt, judge swapped to a local no-thinking
Qwen3.6-27B endpoint). Judges are local, so scores are not leaderboard-comparable
but are internally consistent across checkpoints (same judge, protocol, seed).

### Judge context budget & score coverage

Judges are served with `--max-model-len 32768` (Qwen3.6) and 49152 (gemma-4;
a degenerate piece can re-tokenize ~2x larger in gemma's tokenizer) via
`serve_judge_vllm.sh`/`MAXLEN`, sized so that every judge prompt this suite
can produce — EQ-Bench's rubric + a full `max_tokens=12000` piece, Arena's
question + both answers — fits together with the judge's 4096-token
completion budget. The test endpoint stays at 16384 (its largest request is
EQ-Bench's 730-token prompt + 12000-token completion). As a tripwire, the score files record judge
coverage: `eqbench_score.json` carries `tasks_scored`/`tasks_total`,
`arena_score.json` carries per-category `*_judged`/`*_total` — anything below
full coverage there means judgments were lost (endpoint failures, unparseable
judge output) and the generation score averages the judged subset.

## Layout

```
scripts/eval_standard/    run_eval.sh (the orchestrator), run_em_dp4.sh,
                          per-bench gen/score helpers, collect_results.py
scripts/eval_gen/         judge serving (serve_judge_vllm.sh, qwen36_nothink.jinja),
                          alpaca annotator config
em_rzero/                 R-Zero-protocol EM evals
                          (eval_mmlupro/supergpqa/bbh.py) — from the official
                          R-Zero repo's evaluation/
                          (modified for robust evaluation)
third_party/              vendored benchmark harnesses + clone/patch instructions
                          (see THIRD_PARTY.md)
data/                     the 7 math parquets
outputs/eval_standard/_prepared/   IFEval prompt jsonl (built in one-time setup step 4)
outputs/eval_gen/alpaca/  AlpacaEval prompts + official GPT-4-Turbo reference
```

## Environments (see the repo `requirements/` directory)

| env | requirements | role |
|---|---|---|
| judge serving (vLLM 0.23) | `../requirements/requirements-eval.txt` | all judge/test endpoints (Qwen3.6-27B, gemma-4-31b-it) |
| genbench (Py 3.10) | `../requirements/requirements-eval-genbench.txt` | alpaca_eval CLI, EQ-Bench orchestration |
| arena (Py 3.12) | `../requirements/requirements-eval-arena.txt` | arena-hard-auto harness |

## One-time setup

1. Create the environments above; clone + patch the external benchmark repos
   (THIRD_PARTY.md), install alpaca_eval editable and copy the annotator config in.
2. Recreate the EQ-Bench PRISTINE state from the shipped zip (THIRD_PARTY.md),
   and seed the harness `.env` from the documented example next to it:
   `cp third_party/creative-writing-bench/.env_example third_party/creative-writing-bench/.env`
   (the orchestrator rewrites the ports per run, so it needs no editing).
   `run_eval.sh` makes this copy for you when `.env` is missing — but never
   delete the file: with no `.env` the EQ-Bench harness silently falls back to
   `https://api.openai.com`.
3. Pre-download the judge models into your HF cache: `Qwen/Qwen3.6-27B` and
   `google/gemma-4-31b-it`. Judge/test serving is the one hard-offline part of
   the suite (`scripts/eval_gen/serve_judge_vllm.sh` pins `HF_HUB_OFFLINE=1`),
   so an uncached judge model fails at weight load. Everything else runs with
   the Hub **online** (`HF_HUB_OFFLINE=0`, exported by `run_eval.sh` and
   `run_em_dp4.sh`), so the EM datasets `TIGER-Lab/MMLU-Pro` and
   `m-a-p/SuperGPQA` are fetched on first use and need no setup step —
   evaluation does need network access.

   To evaluate fully offline instead, export `HF_HUB_OFFLINE=1` (both scripts
   honour a caller-set value) and seed the **datasets** cache once while
   online:
   ```bash
   python -c "import datasets; datasets.load_dataset('TIGER-Lab/MMLU-Pro'); datasets.load_dataset('m-a-p/SuperGPQA')"
   ```
   `hf download --repo-type dataset` is **not** a substitute: it fills
   `~/.cache/huggingface/hub/`, and offline `load_dataset()` reads only
   `~/.cache/huggingface/datasets/`.
4. `python scripts/eval_standard/prep_ifeval.py --out outputs/eval_standard/_prepared/ifeval.jsonl`
   (REQUIRED — builds the IFEval prompt file `run_eval.sh` reads).
5. Download the NLTK tokenizer data into the env that scores IFEval — that is
   `RZERO_GEN_VENV`, the training env, since `run_eval.sh` runs
   `score_ifeval.py` with it:
   ```bash
   python -m nltk.downloader punkt_tab
   ```
   REQUIRED. IFEval is the suite's only live NLTK consumer:
   `third_party/instruction_following_eval/instructions_util.py` loads
   `nltk:tokenizers/punkt/english.pickle` and `instructions.py` calls
   `nltk.word_tokenize`. On the pinned nltk 3.9.x **both resolve to `punkt_tab`**
   — the legacy `punkt` package alone is not enough — and without it PHASE 2 dies
   with `LookupError: Resource 'punkt_tab' not found`. Neither
   `requirements-train-vllm011.txt` nor plain `pip install nltk` fetches corpus
   data, and NLTK data is a separate channel from the HF cache, so nothing in
   step 3 covers it — download it explicitly.

   EQ-Bench Creative needs **no** NLTK data: its `core/metrics.py` (which would
   want `cmudict`/`stopwords`, plus a `wordfreq` package that no requirements
   file installs) is not on the harness' import path —
   `creative_writing_bench.py` reaches only `core.{benchmark,conversation,scoring,elo,…}`.

## Path portability

Scripts are functionally identical to the versions that produced the paper
numbers; machine-specific absolute paths have been replaced by repo-relative
anchoring (scripts locate this `benchmarks/` directory from their own path)
plus the environment variables below. Export the required ones before running:

| env var | meaning | consumed by |
|---|---|---|
| `RECIPE_DIR` (required) | R-Zero recipe dir (math eval) — install `../baselines/r-zero` as `<verl>/recipe_R-Zero` | `run_eval.sh`, `run_em_dp4.sh` |
| `RZERO_GEN_VENV` (required) | generation venv (vLLM; `../requirements/requirements-train-vllm011.txt`) | `run_eval.sh` |
| `GENBENCH_PY` (required) | genbench venv python | `run_eval.sh` |
| `ARENA_PY` (required) | arena venv python | `run_eval.sh` |
| `ALPACA_EVAL_BIN` (required) | `alpaca_eval` CLI from the genbench venv | `run_eval.sh` |
| `JUDGE_VENV` (required) | vLLM 0.23 judge-serving venv (`../requirements/requirements-eval.txt`) | `serve_judge_vllm.sh` |
| `PORT_BASE` (optional) | lane port band, default 7400 (phase-3 judges +30..32, phase-4 judge +20, test +51) | `run_eval.sh` |
| `ARENA_DIR` / `BENCH_DIR` (optional) | arena / EQ-Bench working copies (lane B of a 2-lane sweep) | `run_eval.sh` |
| `EM_GEN_VENV` (optional) | generation venv; only needed to run `run_em_dp4.sh` standalone, where it is mandatory — `run_eval.sh` passes it automatically | `run_em_dp4.sh` |
| `DATA_DIR` (optional) | math-7 eval parquets; defaults to the shipped `benchmarks/data/` | `run_eval.sh` → `run_solver_eval.sh` |
| `EM_VLLM_BASE` / `EM_MASTER_BASE` (optional) | vLLM/torch-dist internal port bases for the EM/IFEval embedded generators (defaults 52000/56000, offset by GPU index) | `run_eval.sh`, `run_em_dp4.sh` |
| `BENCH_ROOT` (optional) | override the auto-detected `benchmarks/` root | all orchestrators |
| `HF_HOME` (optional) | model cache, defaults to `~/.cache/huggingface` | all serving/gen |

## Running

```bash
# once per shell: point the scripts at your environments (see Path portability)
export RECIPE_DIR=<verl>/recipe_R-Zero JUDGE_VENV=<judge-venv> \
       RZERO_GEN_VENV=<gen-venv> GENBENCH_PY=<genbench-venv>/bin/python \
       ARENA_PY=<arena-venv>/bin/python ALPACA_EVAL_BIN=<genbench-venv>/bin/alpaca_eval

# one checkpoint, 4 GPUs
GPU_SET=0,1,2,3 bash scripts/eval_standard/run_eval.sh <ckpt> <slug>

# resume from a phase after failure (1=EM, 2=math+IFEval, 3=Arena‖EQ, 4=Alpaca)
PHASE_START=4 GPU_SET=0,1,2,3 bash scripts/eval_standard/run_eval.sh <ckpt> <slug>

# all iterations of a finished run, sequentially:
for it in 1 2 3 4 5 6; do
  GPU_SET=0,1,2,3 bash scripts/eval_standard/run_eval.sh \
    <ckpt_root>/iter${it}_solver/global_step_15/actor/huggingface iter${it}
done

# two concurrent lanes on an 8-GPU host (needs the -lane2 working copies, THIRD_PARTY.md):
GPU_SET=0,1,2,3 PORT_BASE=7400 bash scripts/eval_standard/run_eval.sh <ckptA> <slugA> &
GPU_SET=4,5,6,7 PORT_BASE=9500 \
  ARENA_DIR=$PWD/third_party/arena-hard-auto-lane2 \
  BENCH_DIR=$PWD/third_party/creative-writing-bench-lane2 \
  bash scripts/eval_standard/run_eval.sh <ckptB> <slugB> &
```

GPU semantics: `GPU_SET` lists exactly 4 GPUs — the first 3 serve judges, the 4th
serves the persistent test/gen endpoint; math (Phase 2) uses all 4. The script
never touches GPUs outside `GPU_SET` (scope-safe `kill_port`, no broad pkill).

Collect results for one checkpoint (also printed automatically at the end of a run):

```bash
python scripts/eval_standard/collect_results.py outputs/eval/<slug>
```
