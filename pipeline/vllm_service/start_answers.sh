#!/bin/bash
# Launch a single answers-only vLLM Flask server (no majority vote): returns
# raw n=10 answers per question, RM scoring happens downstream. Used by the
# challenger stage on GPU 2 of the 4-GPU layout.
#
# Usage: bash start_answers.sh <solver_model_path> <log_dir> <gpu> [port]

set -euo pipefail

# shellcheck disable=SC1090
source "${RZERO_VENV_ACTIVATE:?export RZERO_VENV_ACTIVATE=/path/to/venv/bin/activate (env from requirements/requirements-train-vllm011.txt + verl installed editable)}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../env_prelude.sh"

MODEL_PATH=${1:?"solver_model_path required"}
LOG_DIR=${2:?"log_dir required"}
GPU=${3:?"gpu index required (e.g. 2)"}
PORT=${4:-${RZERO_BASE_PORT:-18000}}

mkdir -p "$LOG_DIR"

# Distinct torch.distributed master port: the challenger trainer on GPU 0,1
# calls init_process_group too, and must not share our rendezvous port.
DIST_PORT=$((PORT + 100))

CUDA_VISIBLE_DEVICES=$GPU VLLM_DP_MASTER_PORT=$DIST_PORT MASTER_PORT=$DIST_PORT \
    python "$SCRIPT_DIR/start_vllm_server_answers.py" \
    --port "$PORT" --model_path "$MODEL_PATH" \
    > "$LOG_DIR/vllm_answers_${PORT}.log" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/vllm_answers_${PORT}.pid"
echo "[start_answers.sh] launched answers vLLM on GPU=$GPU port=$PORT pid=$PID"

WAIT_SECS=${VLLM_WAIT_SECS:-1200}
POLL=5
ITERS=$(( WAIT_SECS / POLL ))
for i in $(seq 1 $ITERS); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[start_answers.sh] FATAL: vLLM pid $PID died" >&2
        tail -40 "$LOG_DIR/vllm_answers_${PORT}.log" >&2 || true
        exit 1
    fi
    if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[start_answers.sh] port $PORT is up"
        exit 0
    fi
    sleep $POLL
done

echo "[start_answers.sh] FATAL: port $PORT did not respond within ${WAIT_SECS}s" >&2
tail -40 "$LOG_DIR/vllm_answers_${PORT}.log" >&2 || true
kill -TERM "$PID" 2>/dev/null || true
exit 1
