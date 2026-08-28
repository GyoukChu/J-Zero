#!/bin/bash
# Core 4-GPU launcher for the J-Zero self-play pipeline.
#
# The RM (Stage 3) is trained on the UNION of two preference sources:
#   (A) challenger-solver pairs  (chosen=solver / rejected=challenger-as-solver)
#         RZERO_RM_N_CHAL pairs   [default 256]
#   (B) IDA subtask-amplification pairs (chosen=challenger's composition of the
#         solver's sub-answers, rejected=one-shot solver answer)
#         RZERO_RM_N_SUBTASK pairs  [default 256]
#   -> merged to 512 pairs, BT-trained for RZERO_RM_STEPS=8 (batch 64).
# The orchestrator run_jzero.sh calls rm_train.sh at Stage 3.
#
# Usage: called by run_jzero_4b.sh / run_jzero_8b.sh (the released-run
# wrappers), or directly:
#   nohup bash launch/launch_jzero_4gpu.sh > run.log 2>&1 &

set -euo pipefail

# ---------------------------------------------------------------------------
# Caller-supplied roots
# ---------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:?export PROJECT_ROOT=... (directory holding .env, data/ and outputs/)}"
export PROJECT_ROOT
VERL_REPO="${VERL_REPO:?export VERL_REPO=/path/to/verl checkout (pipeline/ installed as \$VERL_REPO/recipe_J-Zero)}"
RECIPE_DIR="${RECIPE_DIR:-$VERL_REPO/recipe_J-Zero}"
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3-4B-Base"}
RM_INIT=${RZERO_RM_INIT_PATH:-"Skywork/Skywork-Reward-V2-Llama-3.1-8B"}

RM_TRAIN_ENABLE=${RM_TRAIN_ENABLE:-1}

OUT_ROOT_DEFAULT="$PROJECT_ROOT/outputs/run_jzero"
CKPT_ROOT_DEFAULT="$VERL_REPO/checkpoints/J-Zero"
if [[ "${SMOKE:-0}" == "1" ]]; then
    TAG="jzero_smoke"                 # must match run_jzero.sh SMOKE TAG
else
    TAG="${RZERO_TAG:-jzero_full}"    # must match run_jzero.sh non-SMOKE TAG
fi

OUT_ROOT=${OUT_ROOT:-"$OUT_ROOT_DEFAULT"}
CKPT_ROOT=${CKPT_ROOT:-"$CKPT_ROOT_DEFAULT"}
mkdir -p "$OUT_ROOT/logs" "$CKPT_ROOT"

# ---------------------------------------------------------------------------
# Secrets: HF_TOKEN + WANDB_API_KEY from .env (no OPENAI key needed — all rewards are local).
# ---------------------------------------------------------------------------
ENV_FILE="$PROJECT_ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[launch_jzero] FATAL: .env not found at $ENV_FILE" >&2; exit 1
fi
{ set +x; } 2>/dev/null
set -a; source "$ENV_FILE"; set +a
if [[ -z "${HF_TOKEN:-}" ]]; then echo "[launch_jzero] FATAL: HF_TOKEN unset" >&2; exit 1; fi
if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "[launch_jzero] FATAL: WANDB_API_KEY unset (or set WANDB_MODE=offline/disabled)" >&2; exit 1
fi
echo "[launch_jzero] HF_TOKEN len=${#HF_TOKEN}"
set -x

# ---------------------------------------------------------------------------
# Python env + recipe prelude.
# ---------------------------------------------------------------------------
{ set +x; } 2>/dev/null
# shellcheck disable=SC1090
source "${RZERO_VENV_ACTIVATE:?export RZERO_VENV_ACTIVATE=/path/to/venv/bin/activate (env from requirements/requirements-train-vllm011.txt + verl installed editable)}"
set -x
source "$RECIPE_DIR/env_prelude.sh"

# ---------------------------------------------------------------------------
# GPU SCOPE: 0,1,2,3.
# ---------------------------------------------------------------------------
export RZERO_GPUS="0,1,2,3"

# Verified env overrides (trainable defaults of the released runs).
export JZERO_CHALLENGER_MEM_UTIL=${JZERO_CHALLENGER_MEM_UTIL:-0.55}
export JZERO_SOLVER_MEM_UTIL=${JZERO_SOLVER_MEM_UTIL:-0.55}
export RZERO_REWARD_MEM_UTIL=${RZERO_REWARD_MEM_UTIL:-0.7}
export RZERO_SETTLE_MS=5000
export RZERO_CHUNK_MAX=64
export RZERO_REWARD_CHUNK_SIZE=128

# vLLM port base for the 0-3 family.
export VLLM_PORT=31000
export VERL_MASTER_PORT_RANGE=30000-40000
export PYTHONFAULTHANDLER=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-J-Zero}"

# RZERO ports + RM knobs.
export RZERO_BASE_PORT=${RZERO_BASE_PORT:-18000}
export RZERO_RM_BASE_PORT=${RZERO_RM_BASE_PORT:-19000}
export RZERO_RM_NUM_SERVERS_CHAL=${RZERO_RM_NUM_SERVERS_CHAL:-1}
export RZERO_RM_NUM_SERVERS_SOL=${RZERO_RM_NUM_SERVERS_SOL:-2}
export RZERO_RM_NUM_SERVERS_FILT=${RZERO_RM_NUM_SERVERS_FILT:-4}
export RZERO_RM_MAX_LEN=${RZERO_RM_MAX_LEN:-8192}
export RZERO_RM_LR=${RZERO_RM_LR:-5e-7}

# *** Stage-3 pair builder: IDA subtask amplification. ***
export RZERO_RM_SUBTASK_MAXSUB=${RZERO_RM_SUBTASK_MAXSUB:-5}

# RM training data: chalneg(256) + subtask(256) = 512. These SMOKE/non-SMOKE
# defaults are duplicated in pipeline/rm_train.sh and rzero_rm_steps() in
# pipeline/env_prelude.sh — change one, change both. SMOKE=1 must scale the RM
# stage down (with full-size defaults the ~100 smoke questions all land in the
# solver parquet, leaving the pair builders an empty pool and Stage 3 fails).
if [[ "${SMOKE:-0}" == "1" ]]; then
    export RZERO_RM_BATCH=${RZERO_RM_BATCH:-8}
    export RZERO_RM_N_CHAL=${RZERO_RM_N_CHAL:-8}
    export RZERO_RM_N_SUBTASK=${RZERO_RM_N_SUBTASK:-8}
    export RZERO_RM_STEPS=${RZERO_RM_STEPS:-2}       # 8 batch * 2 steps = 16 pairs
else
    export RZERO_RM_BATCH=${RZERO_RM_BATCH:-64}
    export RZERO_RM_N_CHAL=${RZERO_RM_N_CHAL:-256}       # challenger-solver pairs (coverage, both domains)
    export RZERO_RM_N_SUBTASK=${RZERO_RM_N_SUBTASK:-256} # subtask-amplification pairs (above-solver signal, both domains)
    export RZERO_RM_STEPS=${RZERO_RM_STEPS:-8}           # 64 batch * 8 steps = 512 pairs
fi

# Solver-filter band OFF -> [0.0,1.0] passes every Q. Defaults duplicated in
# pipeline/solver_train.sh, question_filter/run_parallel_rm.sh and filter_rm.py.
export RZERO_RM_FILTER_LO=${RZERO_RM_FILTER_LO:-0.0}
export RZERO_RM_FILTER_HI=${RZERO_RM_FILTER_HI:-1.0}
if [[ "${SMOKE:-0}" == "1" ]]; then
    # Keep part of the smoke question pool OUT of the solver parquet so the
    # Stage-3 pair builders have a non-empty exclusion-surviving pool.
    export RZERO_RM_FILTER_TOPN=${RZERO_RM_FILTER_TOPN:-48}
else
    export RZERO_RM_FILTER_TOPN=${RZERO_RM_FILTER_TOPN:-1920}
fi
export RZERO_RM_MICRO_BATCH=${RZERO_RM_MICRO_BATCH:-4}
export RZERO_RM_GRAD_CKPT=${RZERO_RM_GRAD_CKPT:-1}

export RZERO_NITER=${RZERO_NITER:-10}

# Per-iter eval: default OFF. Set to 0 and point RZERO_EVAL_SCRIPT at an eval
# hook to evaluate each iteration in-loop (alias of RZERO_SKIP_PER_ITER_EVAL;
# run_jzero.sh resolves the pair). SMOKE=1 skips per-iter eval unconditionally.
export RZERO_SKIP_ITER_EVAL=${RZERO_SKIP_ITER_EVAL:-1}

# Inputs to run_jzero.sh.
export OUT_ROOT CKPT_ROOT BASE_MODEL
export RZERO_RM_INIT_PATH="$RM_INIT"
export RZERO_RM_TRAIN_ENABLE="$RM_TRAIN_ENABLE"

# ---------------------------------------------------------------------------
# Pre-flight cleanup (GPU 0-3 scope).
# ---------------------------------------------------------------------------
bash "$RECIPE_DIR/cleanup_between_phases.sh" "pre-jzero-launch"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[launch_jzero] starting run_jzero.sh (NITER=$RZERO_NITER, GPU $RZERO_GPUS)"
RZERO_LOG="$OUT_ROOT/logs/run_jzero.log"

set +e
( cd "$RECIPE_DIR"; bash ./run_jzero.sh ) 2>&1 | tee -a "$RZERO_LOG"
RZERO_RC=${PIPESTATUS[0]}
set -e
if [[ $RZERO_RC -ne 0 ]]; then
    echo "[launch_jzero] FATAL: run_jzero.sh rc=$RZERO_RC (resumable — rerun)" >&2
    exit "$RZERO_RC"
fi
echo "[launch_jzero] done. solver ckpts under $CKPT_ROOT/iter{1..$RZERO_NITER}_solver_${TAG}/"
