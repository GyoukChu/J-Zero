#!/bin/bash
# Launch N vLLM Flask reward servers on the GPUs in RZERO_REWARD_GPUS (default
# "2 3") for the challenger training step. Ports are BASE_PORT + i (server index).
# challenger_reward.py queries the same ports for i in 0..RZERO_NUM_SERVERS-1, so
# keep RZERO_REWARD_GPUS count == RZERO_NUM_SERVERS.
#
# Usage: bash start.sh <solver_model_path> <log_dir>

set -euo pipefail

RZERO_VENV="${RZERO_VENV:?export RZERO_VENV=/path/to/venv built from requirements/requirements-train-vllm011.txt}"
source "$RZERO_VENV/bin/activate"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../env_prelude.sh"

MODEL_PATH=${1:?"solver_model_path required"}
LOG_DIR=${2:?"log_dir required"}

BASE_PORT=${RZERO_BASE_PORT:-18000}
REWARD_GPUS=(${RZERO_REWARD_GPUS:-2 3})
NSRV=${#REWARD_GPUS[@]}

mkdir -p "$LOG_DIR"

# Each server's vLLM v1 EngineCore opens a torch.distributed master port, so
# they need distinct ones to avoid EADDRINUSE races.
PORTS=()
for i in "${!REWARD_GPUS[@]}"; do
    GPU=${REWARD_GPUS[$i]}
    PORT=$((BASE_PORT + i))
    DIST=$((BASE_PORT + 100 + i))
    PORTS[$i]=$PORT
    CUDA_VISIBLE_DEVICES=$GPU VLLM_DP_MASTER_PORT=$DIST MASTER_PORT=$DIST \
        python "$SCRIPT_DIR/start_vllm_server.py" \
        --port "$PORT" --model_path "$MODEL_PATH" \
        > "$LOG_DIR/vllm_${PORT}.log" 2>&1 &
    echo $! > "$LOG_DIR/vllm_${PORT}.pid"
    # Stagger past multiproc init: shared compile-cache contention + port races.
    sleep 8
done
echo "[start.sh] Launched $NSRV vLLM servers on GPUs ${REWARD_GPUS[*]} ports ${PORTS[*]}"

reap_all() {
    for p in $(cat "$LOG_DIR"/vllm_*.pid 2>/dev/null); do
        kill -TERM "$p" 2>/dev/null || true
    done
}

# Wait for every /health endpoint; fail fast if a server dies or times out.
WAIT_SECS=${VLLM_WAIT_SECS:-1200}      # default 20 min: vLLM compile + warmup
POLL_INTERVAL=5
ITERS=$(( WAIT_SECS / POLL_INTERVAL ))

for port in "${PORTS[@]}"; do
    pid=$(cat "$LOG_DIR/vllm_${port}.pid")
    echo "[start.sh] waiting for port $port (pid=$pid, up to ${WAIT_SECS}s) ..."
    ok=0
    for i in $(seq 1 $ITERS); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[start.sh] FATAL: vllm server pid $pid (port $port) died." >&2
            echo "[start.sh]   tail of log $LOG_DIR/vllm_${port}.log:" >&2
            tail -40 "$LOG_DIR/vllm_${port}.log" >&2 || true
            reap_all
            exit 1
        fi
        if curl -fs "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "[start.sh] port $port is up"
            ok=1
            break
        fi
        sleep $POLL_INTERVAL
    done
    if [[ "$ok" != "1" ]]; then
        echo "[start.sh] FATAL: port $port did not respond within ${WAIT_SECS}s." >&2
        echo "[start.sh]   tail of log $LOG_DIR/vllm_${port}.log:" >&2
        tail -40 "$LOG_DIR/vllm_${port}.log" >&2 || true
        reap_all
        exit 1
    fi
done

echo "[start.sh] all $NSRV servers ready"
