#!/usr/bin/env bash
# J-Zero released run: Qwen3-4B-Base, rounds 1-15.
#
# Wraps launch_jzero_4gpu.sh (which carries every pipeline knob: filter band
# OFF, 256+256 preference pairs, RM 8x64 BT steps, IDA subtask builder).
# Resumable: finished challenger/solver/RM checkpoints are detected and
# skipped, so rerunning this script continues from the first unfinished stage.
#
# Run tag chalneg_ida_full matches the released checkpoints' folder naming
# (iterN_{challenger,solver,rm}_chalneg_ida_full/...).
set -uo pipefail
CW="${PROJECT_ROOT:?export PROJECT_ROOT=... (directory holding .env, data/ and outputs/)}"
VERL="${VERL_REPO:?export VERL_REPO=/path/to/verl checkout (pipeline/ installed as \$VERL_REPO/recipe_J-Zero)}"
export PROJECT_ROOT VERL_REPO
LAUNCH_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B-Base}
export RZERO_TAG=chalneg_ida_full
export RZERO_NITER=${RZERO_NITER:-15}                 # paper setting: 15 rounds for the 4B run
export CKPT_ROOT=${CKPT_ROOT:-$VERL/checkpoints/J-Zero-4B}
export OUT_ROOT=${OUT_ROOT:-$CW/outputs/run_jzero_4b}

mkdir -p "$OUT_ROOT/logs"

# Effective values: SMOKE=1 overrides round count and tag inside run_jzero.sh;
# keep these two in sync with its SMOKE block.
if [[ "${SMOKE:-0}" == "1" ]]; then
  EFF_NITER="${RZERO_SMOKE_NITER:-1}"; EFF_TAG="jzero_smoke"
else
  EFF_NITER="$RZERO_NITER"; EFF_TAG="$RZERO_TAG"
fi
EFF_START="${RZERO_START_ITER:-1}"

echo "[run_jzero_4b] rounds $EFF_START-$EFF_NITER starting (tag=$EFF_TAG)"
if bash "$LAUNCH_DIR/launch_jzero_4gpu.sh" >>"$OUT_ROOT/logs/run_jzero_4b.log" 2>&1; then
  echo "[run_jzero_4b] rounds $EFF_START-$EFF_NITER finished (tag=$EFF_TAG, ckpts under $CKPT_ROOT)"
else
  echo "[run_jzero_4b] exited non-zero (see $OUT_ROOT/logs/run_jzero_4b.log). Resumable — rerun this script." >&2
  exit 1
fi
