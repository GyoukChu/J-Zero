#!/bin/bash
# Kill all Skywork RM HTTP servers started by start.sh.
#
# Usage: bash stop.sh <log_dir> [base_port] [num_servers] [tag]

set -u

LOG_DIR=${1:?"log_dir required"}
BASE_PORT=${2:-${RZERO_RM_BASE_PORT:-19000}}
NUM=${3:-4}
TAG=${4:-rm}

# Stop any server in [BASE_PORT, BASE_PORT+NUM) AND a few extras (safety).
for off in 0 1 2 3 4 5 6 7; do
    PORT=$((BASE_PORT + off))
    PID_FILE="$LOG_DIR/${TAG}_${PORT}.pid"
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            pkill -KILL -P "$PID" 2>/dev/null || true
            kill -KILL "$PID" 2>/dev/null || true
            echo "[rm_service/stop.sh] killed pid $PID + children (port $PORT)"
        fi
        rm -f "$PID_FILE"
    fi
done

# Never add a host-wide `pkill -f start_rm_server.py` here — it would violate
# the GPU-scope policy. Survivors are picked up by the GPU-scoped
# `cleanup_between_phases.sh "post-..."` that every caller runs right after.

sleep 3
echo "[rm_service/stop.sh] done"
