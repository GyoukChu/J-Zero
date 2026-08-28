#!/usr/bin/env bash
# J-Zero run: meta-llama/Llama-3.2-3B-Instruct, rounds 1-15.
#
# Identical to run_jzero_4b.sh except the base model and ckpt/output/tag
# identity (no memory-fit overrides needed at 3B). Resumable — rerunning
# continues from the first unfinished stage.
#
# NOTE: Llama-3.2 is a gated hub repo — export an HF_TOKEN with access, or
# pre-download it into the HF cache. It is an *Instruct* checkpoint, unlike
# the Qwen3-*-Base runs.
set -uo pipefail
CW="${PROJECT_ROOT:?export PROJECT_ROOT=... (directory holding .env, data/ and outputs/)}"
VERL="${VERL_REPO:?export VERL_REPO=/path/to/verl checkout (pipeline/ installed as \$VERL_REPO/recipe_J-Zero)}"
export PROJECT_ROOT VERL_REPO
LAUNCH_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}
export RZERO_TAG=chalneg_ida_llama3b
export RZERO_NITER=${RZERO_NITER:-15}                 # same round count as the 4B run
export CKPT_ROOT=${CKPT_ROOT:-$VERL/checkpoints/J-Zero-Llama3.2-3B}
export OUT_ROOT=${OUT_ROOT:-$CW/outputs/run_jzero_llama3b}

mkdir -p "$OUT_ROOT/logs"

# Effective values: SMOKE=1 overrides round count and tag inside run_jzero.sh;
# keep these two in sync with its SMOKE block.
if [[ "${SMOKE:-0}" == "1" ]]; then
  EFF_NITER="${RZERO_SMOKE_NITER:-1}"; EFF_TAG="jzero_smoke"
else
  EFF_NITER="$RZERO_NITER"; EFF_TAG="$RZERO_TAG"
fi
EFF_START="${RZERO_START_ITER:-1}"

echo "[run_jzero_llama3b] rounds $EFF_START-$EFF_NITER starting (tag=$EFF_TAG)"
if bash "$LAUNCH_DIR/launch_jzero_4gpu.sh" >>"$OUT_ROOT/logs/run_jzero_llama3b.log" 2>&1; then
  echo "[run_jzero_llama3b] rounds $EFF_START-$EFF_NITER finished (tag=$EFF_TAG, ckpts under $CKPT_ROOT)"
else
  echo "[run_jzero_llama3b] exited non-zero (see $OUT_ROOT/logs/run_jzero_llama3b.log). Resumable — rerun this script." >&2
  exit 1
fi
