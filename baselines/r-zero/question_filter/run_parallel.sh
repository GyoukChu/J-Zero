#!/bin/bash
# 4 parallel processes on GPU 0,1,2,3 — each runs the solver m=10 times per
# generated question of its shard.
#
# Usage: bash run_parallel.sh <solver_model> <save_name> <in_dir> <out_dir> [num_samples]

set -euo pipefail

RZERO_VENV="${RZERO_VENV:?export RZERO_VENV=/path/to/venv built from requirements/requirements-train-vllm011.txt}"
source "$RZERO_VENV/bin/activate"

MODEL=${1:?"solver_model required"}
SAVE_NAME=${2:?"save_name required"}
IN_DIR=${3:?"in_dir required"}
OUT_DIR=${4:?"out_dir required"}
NUM_SAMPLES=${5:-10}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_SKIP_DEEP_GEMM_WARMUP=1

GPUS=(${RZERO_GEN_GPUS:-0 1 2 3})
PIDS=()
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    # Unique per GPU: the torch.distributed default port would race.
    MASTER_PORT=$((39500 + GPU * 7))
    CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$MASTER_PORT python "$SCRIPT_DIR/evaluate.py" \
        --model "$MODEL" \
        --num_samples "$NUM_SAMPLES" \
        --seed "$i" \
        --save_name "$SAVE_NAME" \
        --in_dir "$IN_DIR" \
        --out_dir "$OUT_DIR" \
        > "$LOG_DIR/eval_gpu${GPU}.log" 2>&1 &
    PIDS[$i]=$!
    sleep 3
done

echo "[run_parallel.sh] eval pids: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[run_parallel.sh] pid $pid failed"
        FAIL=1
    fi
done
exit $FAIL
