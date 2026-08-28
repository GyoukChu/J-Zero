#!/usr/bin/env bash
# J-Zero released run: Qwen3-8B-Base, rounds 1-10.
#
# Identical to run_jzero_4b.sh except the model, round count, and the
# 8B memory-fit knobs (CPU offload + smaller vLLM footprint) needed to fit
# the colocated vLLM rollout + FSDP actor on 4 GPUs.
# Resumable — rerunning continues from the first unfinished stage.
#
# Run tag chalneg_ida8b_full matches the released checkpoints' folder naming.
set -uo pipefail
CW="${PROJECT_ROOT:?export PROJECT_ROOT=... (directory holding .env, data/ and outputs/)}"
VERL="${VERL_REPO:?export VERL_REPO=/path/to/verl checkout (pipeline/ installed as \$VERL_REPO/recipe_J-Zero)}"
export PROJECT_ROOT VERL_REPO
LAUNCH_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B-Base}
export RZERO_TAG=chalneg_ida8b_full
export RZERO_NITER=${RZERO_NITER:-10}                 # paper setting: 10 rounds
export CKPT_ROOT=${CKPT_ROOT:-$VERL/checkpoints/J-Zero-8B}
export OUT_ROOT=${OUT_ROOT:-$CW/outputs/run_jzero_8b}

# --- 8B memory fit (colocated vLLM rollout + FSDP actor on 4 GPUs) ---
export JZERO_ACTOR_OPT_OFFLOAD=True     # Adam states -> CPU (biggest saving)
export JZERO_ACTOR_PARAM_OFFLOAD=True   # param shard -> CPU when idle
export JZERO_REF_PARAM_OFFLOAD=True
export JZERO_FREE_CACHE=True            # free vLLM KV cache during the actor update
export JZERO_SOLVER_MEM_UTIL=0.45       # smaller vLLM footprint
export JZERO_CHALLENGER_MEM_UTIL=0.45
# NOTE: leave PYTORCH_CUDA_ALLOC_CONF unset (expandable_segments is
# incompatible with verl's colocated rollout).

mkdir -p "$OUT_ROOT/logs"

# Effective values: SMOKE=1 overrides round count and tag inside run_jzero.sh;
# keep these two in sync with its SMOKE block.
if [[ "${SMOKE:-0}" == "1" ]]; then
  EFF_NITER="${RZERO_SMOKE_NITER:-1}"; EFF_TAG="jzero_smoke"
else
  EFF_NITER="$RZERO_NITER"; EFF_TAG="$RZERO_TAG"
fi
EFF_START="${RZERO_START_ITER:-1}"

echo "[run_jzero_8b] rounds $EFF_START-$EFF_NITER starting (tag=$EFF_TAG)"
if bash "$LAUNCH_DIR/launch_jzero_4gpu.sh" >>"$OUT_ROOT/logs/run_jzero_8b.log" 2>&1; then
  echo "[run_jzero_8b] rounds $EFF_START-$EFF_NITER finished (tag=$EFF_TAG, ckpts under $CKPT_ROOT)"
else
  echo "[run_jzero_8b] exited non-zero (see $OUT_ROOT/logs/run_jzero_8b.log). Resumable — rerun this script." >&2
  exit 1
fi
