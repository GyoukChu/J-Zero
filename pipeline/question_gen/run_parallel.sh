#!/bin/bash
# Run 4 parallel question_generate.py processes on GPU 0,1,2,3.
#
# Usage: bash run_parallel.sh <challenger_model> <save_name> <out_dir> <per_gpu_samples>

set -euo pipefail

# shellcheck disable=SC1090
source "${RZERO_VENV_ACTIVATE:?export RZERO_VENV_ACTIVATE=/path/to/venv/bin/activate (env from requirements/requirements-train-vllm011.txt + verl installed editable)}"

MODEL=${1:?"challenger_model required"}
SAVE_NAME=${2:?"save_name required"}
OUT_DIR=${3:?"out_dir required"}
PER_GPU=${4:?"per_gpu_samples required"}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_SKIP_DEEP_GEMM_WARMUP=1

GPUS=(0 1 2 3)
PIDS=()
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    # Unique MASTER_PORT per GPU so concurrent vLLM workers don't race for
    # torch.distributed's default rendezvous port (29500).
    MASTER_PORT=$((39000 + GPU * 7))
    # Unique VLLM_PORT per worker too: vllm 0.11 binds VLLM_PORT verbatim (no
    # walk), so a shared value means EADDRINUSE on the c10d TCPStore port. Same
    # port-slot scheme as question_filter/run_parallel_rm.sh, disjoint band.
    WORKER_VLLM_PORT=$(( ${VLLM_PORT:-31000} + 300 + i * 20 ))
    CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$MASTER_PORT VLLM_PORT=$WORKER_VLLM_PORT python "$SCRIPT_DIR/question_generate.py" \
        --model "$MODEL" \
        --num_samples "$PER_GPU" \
        --seed "$i" \
        --save_name "$SAVE_NAME" \
        --out_dir "$OUT_DIR" \
        > "$LOG_DIR/gen_gpu${GPU}.log" 2>&1 &
    PIDS[$i]=$!
    sleep 3
done

echo "[run_parallel.sh] generation pids: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[run_parallel.sh] pid $pid failed"
        FAIL=1
    fi
done
exit $FAIL
