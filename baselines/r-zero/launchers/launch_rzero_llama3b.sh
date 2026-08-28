#!/bin/bash
# Base R-Zero on meta-llama/Llama-3.2-3B-Instruct, 4 GPU (0-3).
#   challenger: 2-GPU FSDP trainer (GPU 0,1, rollout colocated) + 2 Flask reward
#               servers (GPU 2,3, ports 18000-18001).
#   solver:     gen/filter on all 4 GPU + 4-GPU FSDP trainer.
# Set SMOKE=1 for a quick end-to-end layout check (1-step, 100 q, 1 iter).
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR/.."

# RZERO_ENV_FILE: env file exporting secrets (HF_TOKEN, WANDB_API_KEY).
RZERO_ENV_FILE="${RZERO_ENV_FILE:?export RZERO_ENV_FILE=/path/to/.env with HF_TOKEN and WANDB_API_KEY}"
set -a
source "$RZERO_ENV_FILE"
set +a

# RZERO_VENV: venv built from requirements/requirements-train-vllm011.txt.
export RZERO_VENV="${RZERO_VENV:?export RZERO_VENV=/path/to/venv built from requirements/requirements-train-vllm011.txt}"
# Offline runs resolve repo ids (e.g. meta-llama/Llama-3.2-3B-Instruct in the
# reward server) against the HF cache, so HF_HOME must point at where the
# models actually live.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# --- GPU layout (4 GPU: 0-3) ---
export RZERO_GPUS="0,1,2,3"                # cleanup_between_phases scope
export RZERO_CHAL_TRAIN_GPUS="0,1"         # challenger FSDP trainer (2 GPU)
export RZERO_REWARD_GPUS="2 3"             # Flask reward servers (2)
export RZERO_NUM_SERVERS=2                 # must == #reward GPUs
export RZERO_SOLVER_TRAIN_GPUS="0,1,2,3"   # solver FSDP trainer (4 GPU)
export RZERO_GEN_GPUS="0 1 2 3"            # gen/filter parallel (4 GPU)

# --- Run isolation (see env_prelude.sh) ---
# cleanup_between_phases.sh kills ray by SUBSTRING match on $RAY_TMPDIR, and the
# J-Zero runs use env_prelude.sh's /tmp/ray_rzero default — keep this path free
# of the substring "ray_rzero" or the isolation is only one-way.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_rzb_llama3b}"
export RZERO_CACHE_ROOT="${RZERO_CACHE_ROOT:-$HOME/.cache/r_zero_baseline}"

# --- run config ---
# OUT_ROOT: run artifacts (questions/logs/evals). CKPT_ROOT: checkpoint root.
export OUT_ROOT="${OUT_ROOT:?export OUT_ROOT=/path/to/output root for this run}"
export CKPT_ROOT="${CKPT_ROOT:?export CKPT_ROOT=/path/to/checkpoint root}"
export BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export RZERO_NITER=${RZERO_NITER:-3}
export RZERO_START_ITER=${RZERO_START_ITER:-1}
export RZERO_SKIP_PER_ITER_EVAL=${RZERO_SKIP_PER_ITER_EVAL:-1}
# SMOKE is passed through from the environment if set.

# SMOKE=1 makes run_rzero.sh use RZERO_SMOKE_NITER (default 1), not RZERO_NITER.
if [[ "${SMOKE:-0}" == "1" ]]; then EFF_NITER="${RZERO_SMOKE_NITER:-1}"; else EFF_NITER="$RZERO_NITER"; fi
echo "=== [launch_rzero_llama3b] $(date '+%F %T') SMOKE=${SMOKE:-0} NITER=$EFF_NITER START=$RZERO_START_ITER ==="
echo "    chal_trainer=$RZERO_CHAL_TRAIN_GPUS reward=$RZERO_REWARD_GPUS(x$RZERO_NUM_SERVERS) solver=$RZERO_SOLVER_TRAIN_GPUS"
exec bash run_rzero.sh
