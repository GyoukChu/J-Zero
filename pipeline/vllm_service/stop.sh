#!/bin/bash
# Kill the vLLM Flask servers started by start_answers.sh.
#
# Usage: bash stop.sh <log_dir>

set -u

LOG_DIR=${1:?"log_dir required"}

BASE_PORT=${RZERO_BASE_PORT:-18000}
# Kill exactly the servers this run started, via the pid files start_answers.sh
# wrote — scoped to THIS run's LOG_DIR, any number of servers.
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

# NEVER add a host-wide pattern kill here (`pkill -f "VLLM::EngineCore"`,
# "start_vllm_server.py", "multiprocessing.resource_tracker", ...): those match
# every vLLM process on the host regardless of GPU, including unrelated runs.
# Orphans on our own GPUs are picked up by the GPU-scoped
# `cleanup_between_phases.sh "post-..."` that every caller runs right after.

# Give CUDA a moment to actually release device memory.
sleep 3
echo "[stop.sh] done"
