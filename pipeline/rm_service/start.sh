#!/bin/bash
# Launch N Skywork-reward HTTP servers, one per assigned GPU.
#
# Usage: bash start.sh <model_path> <log_dir> <gpu_csv> [base_port] [tag]
#
#   model_path : RM checkpoint (HF path or local dir). Iter 1 uses
#                Skywork/Skywork-Reward-V2-Llama-3.1-8B; iter>=2 uses RM_t-1.
#   log_dir    : where to write per-server stdout/stderr + pidfiles.
#   gpu_csv    : comma-separated GPU indices, one per server (e.g. "2,3").
#   base_port  : first port (default ${RZERO_RM_BASE_PORT:-19000}). Server i
#                (0-based) binds base_port + i.
#   tag        : optional short tag for log filenames.

set -euo pipefail

# shellcheck disable=SC1090
source "${RZERO_VENV_ACTIVATE:?export RZERO_VENV_ACTIVATE=/path/to/venv/bin/activate (env from requirements/requirements-train-vllm011.txt + verl installed editable)}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../env_prelude.sh"

MODEL_PATH=${1:?"model_path required"}
LOG_DIR=${2:?"log_dir required"}
GPU_CSV=${3:?"gpu_csv required (e.g. '2,3')"}
BASE_PORT=${4:-${RZERO_RM_BASE_PORT:-19000}}
TAG=${5:-rm}

mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
N=${#GPUS[@]}
if [[ "$N" -lt 1 ]]; then
    echo "[rm_service/start.sh] FATAL: gpu_csv parsed to 0 GPUs" >&2
    exit 1
fi

PORTS=()
for i in "${!GPUS[@]}"; do
    PORTS+=("$((BASE_PORT + i))")
done

echo "[rm_service/start.sh] launching $N RM servers (model=$MODEL_PATH)"
echo "[rm_service/start.sh]   GPUs: ${GPUS[*]}"
echo "[rm_service/start.sh]   ports: ${PORTS[*]}"

PIDS=()
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    PORT=${PORTS[$i]}
    # Stagger by 8s so model loads don't fight for disk + HF cache lock.
    if [[ "$i" -gt 0 ]]; then
        sleep 8
    fi
    CUDA_VISIBLE_DEVICES=$GPU \
        python "$SCRIPT_DIR/start_rm_server.py" \
        --port "$PORT" \
        --model_path "$MODEL_PATH" \
        > "$LOG_DIR/${TAG}_${PORT}.log" 2>&1 &
    PID=$!
    PIDS+=("$PID")
    echo "$PID" > "$LOG_DIR/${TAG}_${PORT}.pid"
    echo "[rm_service/start.sh]   started GPU=$GPU port=$PORT pid=$PID"
done

# Wait until every /health responds OK. Fail-fast if any server dies.
WAIT_SECS=${RZERO_RM_WAIT_SECS:-1800}   # default 30 min: RM is 8B + HF cache cold
POLL_INTERVAL=5
ITERS=$(( WAIT_SECS / POLL_INTERVAL ))

for j in "${!PORTS[@]}"; do
    PORT=${PORTS[$j]}
    PID=${PIDS[$j]}
    echo "[rm_service/start.sh] waiting for port $PORT (pid=$PID, up to ${WAIT_SECS}s)"
    ok=0
    for i in $(seq 1 $ITERS); do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "[rm_service/start.sh] FATAL: RM server pid $PID (port $PORT) died." >&2
            echo "[rm_service/start.sh]   tail of $LOG_DIR/${TAG}_${PORT}.log:" >&2
            tail -40 "$LOG_DIR/${TAG}_${PORT}.log" >&2 || true
            # Reap siblings so they don't keep holding GPU memory.
            for other in "${PIDS[@]}"; do
                if [[ "$other" != "$PID" ]] && kill -0 "$other" 2>/dev/null; then
                    kill -TERM "$other" 2>/dev/null || true
                fi
            done
            exit 1
        fi
        if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "[rm_service/start.sh] port $PORT is up"
            ok=1
            break
        fi
        sleep $POLL_INTERVAL
    done
    if [[ "$ok" != "1" ]]; then
        echo "[rm_service/start.sh] FATAL: port $PORT did not respond in ${WAIT_SECS}s" >&2
        echo "[rm_service/start.sh]   tail of $LOG_DIR/${TAG}_${PORT}.log:" >&2
        tail -40 "$LOG_DIR/${TAG}_${PORT}.log" >&2 || true
        for p in "${PIDS[@]}"; do
            kill -TERM "$p" 2>/dev/null || true
        done
        exit 1
    fi
done

echo "[rm_service/start.sh] all $N RM servers ready"
