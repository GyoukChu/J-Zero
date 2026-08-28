#!/bin/bash
# RM-based question-filter orchestrator.
#
#   Phase A: 4 parallel solver vLLM processes on GPU 0,1,2,3. Each runs
#            gen_answers_rm.py on its shard of challenger-generated questions.
#   Phase B: up to 4 RM HTTP servers on GPU 0..3 (Skywork). filter_rm.py fans
#            out HTTP calls across the servers.
#   Phase C: upload_to_parquet_rm.py converts the filtered JSON to verl parquet.
#
# Phases A and B share GPUs 0-3, so A must fully release them before B starts.
#
# Usage:
#   bash run_parallel_rm.sh <solver_model> <rm_model> <save_name> \
#                                  <in_dir> <out_dir> <parquet_path>
#                                  [num_samples=10]

set -euo pipefail

# shellcheck disable=SC1090
source "${RZERO_VENV_ACTIVATE:?export RZERO_VENV_ACTIVATE=/path/to/venv/bin/activate (env from requirements/requirements-train-vllm011.txt + verl installed editable)}"

SOLVER_MODEL=${1:?"solver_model required"}
RM_MODEL=${2:?"rm_model required"}
SAVE_NAME=${3:?"save_name required"}
IN_DIR=${4:?"in_dir required"}
OUT_DIR=${5:?"out_dir required"}
PARQUET=${6:?"parquet_path required"}
NUM_SAMPLES=${7:-10}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
RECIPE_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )
source "$RECIPE_DIR/env_prelude.sh"

mkdir -p "$OUT_DIR" "$OUT_DIR/_logs"
LOG_DIR="$OUT_DIR/_logs"

export RZERO_RM_BASE_PORT=${RZERO_RM_BASE_PORT:-19000}
export RZERO_RM_NUM_SERVERS_FILT=${RZERO_RM_NUM_SERVERS_FILT:-4}
# Band OFF ([0.0, 1.0] = every question passes) is the released J-Zero setting:
# selection is the std top-N alone. Export a narrower LO/HI to re-enable it.
# DUPLICATED DEFAULTS — same pair in pipeline/solver_train.sh and in
# filter_rm.py's --min_avg/--max_avg.
export RZERO_RM_FILTER_LO=${RZERO_RM_FILTER_LO:-0.0}
export RZERO_RM_FILTER_HI=${RZERO_RM_FILTER_HI:-1.0}
export RZERO_RM_FILTER_TOPN=${RZERO_RM_FILTER_TOPN:-1920}

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_SKIP_DEEP_GEMM_WARMUP=1

# --- Phase A: 4-way parallel vLLM answer generation (GPU 0-3) -------------
echo "[run_parallel_rm] Phase A: spawning 4 vLLM solvers on GPU 0-3"
GPUS=(0 1 2 3)
PIDS=()
# vLLM binds its internal ports deterministically from VLLM_PORT, so every
# worker needs a DISJOINT 20-port slot or they collide with each other and with
# the preceding question-gen phase ("EADDRINUSE port: <base>+1"). The +500
# offset keeps this band clear of question-gen's; the wait below lets TIME_WAIT
# from a crashed prior attempt drain.
VLLM_PORT_BASE=${VLLM_PORT:-31000}
FILTER_VLLM_BASE=$((VLLM_PORT_BASE + 500))
for _w in $(seq 1 40); do
    _busy=$(ss -ltn 2>/dev/null | awk -v b="$FILTER_VLLM_BASE" '{p=$4; sub(/.*:/,"",p); if(p+0>=b && p+0<b+80) print p}' | head -1)
    [ -z "$_busy" ] && break
    echo "[run_parallel_rm] waiting for filter vLLM port band ${FILTER_VLLM_BASE}+ to clear (busy=$_busy, ${_w}/40)"
    sleep 3
done
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    MASTER_PORT=$((39500 + GPU * 7))
    WORKER_VLLM_PORT=$((FILTER_VLLM_BASE + i * 20))
    CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$MASTER_PORT VLLM_PORT=$WORKER_VLLM_PORT \
        python "$SCRIPT_DIR/gen_answers_rm.py" \
        --model "$SOLVER_MODEL" \
        --num_samples "$NUM_SAMPLES" \
        --seed "$i" \
        --save_name "$SAVE_NAME" \
        --in_dir "$IN_DIR" \
        --out_dir "$OUT_DIR" \
        > "$LOG_DIR/gen_gpu${GPU}.log" 2>&1 &
    PIDS[$i]=$!
    sleep 3
done
echo "[run_parallel_rm]   gen pids: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[run_parallel_rm] pid $pid failed"
        FAIL=1
    fi
done
if [[ "$FAIL" -ne 0 ]]; then
    echo "[run_parallel_rm] FATAL: at least one Phase A worker failed" >&2
    bash "$RECIPE_DIR/cleanup_between_phases.sh" "filter-phaseA-fail" || true
    exit 1
fi

# Required between A and B: the RM servers come up on the same GPUs, so vLLM's
# TCPStore and GPU memory must be released first.
bash "$RECIPE_DIR/cleanup_between_phases.sh" "pre-rm-phaseB" || true

# --- Phase B: up to 4 RM HTTP servers on GPU 0..3 + filter ----------------
echo "[run_parallel_rm] Phase B: launching $RZERO_RM_NUM_SERVERS_FILT RM servers"
case "$RZERO_RM_NUM_SERVERS_FILT" in
    1) GPU_CSV="0" ;;
    2) GPU_CSV="0,1" ;;
    3) GPU_CSV="0,1,2" ;;
    4) GPU_CSV="0,1,2,3" ;;
    *) echo "[run_parallel_rm] FATAL: RZERO_RM_NUM_SERVERS_FILT must be 1..4" >&2; exit 1 ;;
esac

bash "$RECIPE_DIR/rm_service/start.sh" "$RM_MODEL" "$LOG_DIR" "$GPU_CSV" \
    "$RZERO_RM_BASE_PORT" "filt"

FILTERED_JSON="$OUT_DIR/${SAVE_NAME}_filtered.json"
set +e
python "$SCRIPT_DIR/filter_rm.py" \
    --in_dir "$OUT_DIR" \
    --save_name "$SAVE_NAME" \
    --out_path "$FILTERED_JSON" \
    --rm_base_port "$RZERO_RM_BASE_PORT" \
    --num_servers "$RZERO_RM_NUM_SERVERS_FILT" \
    --min_avg "$RZERO_RM_FILTER_LO" \
    --max_avg "$RZERO_RM_FILTER_HI" \
    --top_k "$RZERO_RM_FILTER_TOPN" \
    2>&1 | tee "$LOG_DIR/filter.log"
FILTER_RC=${PIPESTATUS[0]}
set -e

# Always tear down RM servers regardless of filter outcome.
bash "$RECIPE_DIR/rm_service/stop.sh" "$LOG_DIR" \
    "$RZERO_RM_BASE_PORT" "$RZERO_RM_NUM_SERVERS_FILT" "filt" || true
bash "$RECIPE_DIR/cleanup_between_phases.sh" "post-rm-phaseB" || true

if [[ "$FILTER_RC" -ne 0 ]]; then
    echo "[run_parallel_rm] FATAL: filter_rm.py exited $FILTER_RC" >&2
    exit "$FILTER_RC"
fi

# --- Phase C: parquet upload ----------------------------------------------
python "$SCRIPT_DIR/upload_to_parquet_rm.py" \
    --in_path "$FILTERED_JSON" \
    --save_name "$SAVE_NAME" \
    --out_path "$PARQUET" \
    2>&1 | tee "$LOG_DIR/upload.log"

echo "[run_parallel_rm] done. parquet=$PARQUET"
