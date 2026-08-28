#!/bin/bash
# Kill the vLLM Flask servers started by start.sh.
#
# Usage: bash stop.sh <log_dir>

set -u

LOG_DIR=${1:?"log_dir required"}

BASE_PORT=${RZERO_BASE_PORT:-18000}
# Iterating start.sh's pid files scopes the kill to exactly this run's servers,
# for any N = RZERO_NUM_SERVERS.
shopt -s nullglob
for PID_FILE in "$LOG_DIR"/vllm_*.pid; do
    port=$(basename "$PID_FILE" .pid); port=${port#vllm_}
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        # vLLM's multiprocessing children (EngineCore, resource_tracker) survive
        # parent death and HOLD GPU memory, so kill the tree before the parent.
        pkill -KILL -P "$PID" 2>/dev/null || true
        kill -KILL "$PID" 2>/dev/null || true
        echo "[stop.sh] killed pid $PID + children (port $port)"
    fi
    rm -f "$PID_FILE"
done
shopt -u nullglob

# NEVER add a host-wide `pkill -f` for start_vllm_server.py / VLLM::EngineCore /
# multiprocessing.resource_tracker here: those match every such process on the
# host regardless of GPU, killing unrelated concurrent runs. Orphans on our own
# GPUs are handled by the GPU-scoped cleanup_between_phases.sh that every caller
# runs immediately after this script.

# Give CUDA a moment to actually release device memory.
sleep 3
echo "[stop.sh] done"
