#!/bin/bash
# Per-iter solver evaluation on 7 benchmarks (4 GPUs in parallel) + LLM recheck.
#
# Usage: bash run_solver_eval.sh <solver_model_path> <out_dir>
#
# Stages:
#   1. gen_eval_responses.py on 4 GPUs, DP-by-question — every GPU runs all 7
#      datasets over the questions where orig_idx % 4 == its shard id, writing
#      <out_dir>/responses/<ds>.shard{i}of4.jsonl, merged afterwards.
#   2. llm_recheck.py (verl_mv base grader) against a self-hosted vLLM judge.
#   3. <out_dir>/summary.json: final_pct, baseline_pct, gap_vs_baseline.

set -euo pipefail

RZERO_VENV="${RZERO_VENV:?export RZERO_VENV=/path/to/venv built from requirements/requirements-train-vllm011.txt}"
source "$RZERO_VENV/bin/activate"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../env_prelude.sh"

MODEL=${1:?"solver model path required"}
OUT_DIR=${2:?"out_dir required"}

# DATA_DIR: dir holding the 7 verl-format eval parquets (math500 etc.). Tried in
# order: caller-set $DATA_DIR, $PROJECT_ROOT/data, the in-repo relative path
# (valid only while this recipe sits at <repo>/baselines/r-zero/), then any
# benchmarks/data above SCRIPT_DIR — the README installs it as <verl>/recipe_R-Zero.
resolve_data_dir() {
    local c d
    # An explicitly-set DATA_DIR is authoritative: falling through would silently
    # evaluate a different copy of the parquets than the caller asked for.
    if [[ -n "${DATA_DIR:-}" ]]; then
        [[ -f "$DATA_DIR/math500.parquet" ]] && { echo "$DATA_DIR"; return 0; }
        echo "ERROR: DATA_DIR=$DATA_DIR is set but contains no math500.parquet." >&2
        return 1
    fi
    # Guarded: with PROJECT_ROOT unset this candidate would be the literal /data.
    [[ -n "${PROJECT_ROOT:-}" && -f "$PROJECT_ROOT/data/math500.parquet" ]] \
        && { echo "$PROJECT_ROOT/data"; return 0; }
    c="$SCRIPT_DIR/../../../benchmarks/data"
    [[ -f "$c/math500.parquet" ]] && { echo "$c"; return 0; }
    d="$SCRIPT_DIR"
    while [[ "$d" != "/" ]]; do
        [[ -f "$d/benchmarks/data/math500.parquet" ]] && { echo "$d/benchmarks/data"; return 0; }
        d=$(dirname "$d")
    done
    return 1
}
if ! DATA_DIR=$(resolve_data_dir); then
    echo "ERROR: could not resolve the verl-format eval parquets (math500.parquet etc.)." >&2
    echo "  Searched: \$DATA_DIR if set — otherwise \$PROJECT_ROOT/data," >&2
    echo "            $SCRIPT_DIR/../../../benchmarks/data, and every benchmarks/data above $SCRIPT_DIR." >&2
    echo "  Fix: export DATA_DIR=<repo>/benchmarks/data" >&2
    exit 1
fi
export DATA_DIR
RESP_DIR="$OUT_DIR/responses"
RECHECK_DIR="$OUT_DIR/recheck"
LOG_DIR="$RESP_DIR/logs"
mkdir -p "$RESP_DIR" "$RECHECK_DIR" "$LOG_DIR"

echo "[run_solver_eval] model=$MODEL"
echo "[run_solver_eval] out_dir=$OUT_DIR"

# ---------------------------------------------------------------------------
# Stage 1: DP-by-question parallel generation — one vLLM instance (tp=1) per
# GPU, each covering ~1/4 of every dataset; shards concatenated afterwards.
# ---------------------------------------------------------------------------
DATASETS="math500 minervamath olympiadbench gsm8k amc23 aime24 aime25"

# EVAL_GPUS: exactly 4 GPU indices for the 4-shard split (default 0,1,2,3).
# Mapping is positional (first listed GPU = shard 0), so any 4-GPU set does
# identical work. Ports below are keyed off the absolute GPU index rather than
# the shard, so a peer eval on a disjoint GPU set cannot collide with ours.
EVAL_GPUS=${EVAL_GPUS:-"0,1,2,3"}
IFS=',' read -ra EVAL_GPU_ARR <<< "$EVAL_GPUS"
if [[ ${#EVAL_GPU_ARR[@]} -ne 4 ]]; then
    echo "[run_solver_eval] FATAL: EVAL_GPUS must list exactly 4 indices, got ${#EVAL_GPU_ARR[@]} ($EVAL_GPUS)" >&2
    exit 1
fi
# SOLV_DP_TOTAL (default 4 = full datasets). Only shards 0..3 ever run, so a
# value >4 is the eval-smoke knob: it evaluates a 4/SOLV_DP_TOTAL subsample of
# every dataset (e.g. 100 -> ~4% of each benchmark).
DPT=${SOLV_DP_TOTAL:-4}
echo "[run_solver_eval] EVAL_GPUS=$EVAL_GPUS dp_total=$DPT"

PIDS=()
T0=$(date +%s)
SHARD=0
for GPU in "${EVAL_GPU_ARR[@]}"; do
    LOG="$LOG_DIR/gpu${GPU}.log"
    # Unique per GPU: 4 vLLM instances initialising at once otherwise race on
    # torch.distributed's ephemeral port.
    GPU_MASTER_PORT=$((${SOLV_MASTER_BASE:-55000} + GPU * 100))
    # Unique per GPU so vllm 0.11.0's _get_open_port() (TOCTOU race across
    # parallel processes) walks from distinct bases.
    GPU_VLLM_PORT=$((${SOLV_VLLM_BASE:-51000} + GPU * 100))
    echo "[run_solver_eval] launch GPU $GPU shard $SHARD/4 (MASTER_PORT=$GPU_MASTER_PORT VLLM_PORT=$GPU_VLLM_PORT)"
    (
        CUDA_VISIBLE_DEVICES=$GPU \
        MASTER_PORT=$GPU_MASTER_PORT \
        MASTER_ADDR=127.0.0.1 \
        VLLM_PORT=$GPU_VLLM_PORT \
        python "$SCRIPT_DIR/gen_eval_responses.py" \
            --model "$MODEL" \
            --data_dir "$DATA_DIR" \
            --out_dir "$RESP_DIR" \
            --datasets $DATASETS \
            --dp_shard $SHARD \
            --dp_total $DPT \
            >"$LOG" 2>&1
        # Copy the per-shard gen_summary into LOG_DIR for visibility.
        SHARD_SUM="$RESP_DIR/gen_summary.shard${SHARD}of${DPT}.json"
        if [[ -f "$SHARD_SUM" ]]; then
            cp "$SHARD_SUM" "$LOG_DIR/gen_summary_gpu${GPU}.json"
        fi
    ) &
    PIDS+=($!)
    SHARD=$((SHARD + 1))
    sleep 5  # stagger vLLM init to further reduce port/ray contention
done

# Wait for all four, fail loudly if any died
FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "[run_solver_eval] ERROR: pid $PID exited non-zero" >&2
        FAIL=1
    fi
done
T1=$(date +%s)
echo "[run_solver_eval] generation done in $((T1-T0))s"
if [[ $FAIL -ne 0 ]]; then
    echo "[run_solver_eval] FATAL: at least one GPU generation failed; see $LOG_DIR/gpu*.log" >&2
    exit 1
fi

# Merge shard outputs into per-dataset <ds>.jsonl (llm_recheck reads this).
echo "[run_solver_eval] merging shards"
for DS in $DATASETS; do
    MERGED="$RESP_DIR/${DS}.jsonl"
    if [[ -f "$MERGED" ]]; then
        echo "[run_solver_eval]   $DS: merged file already exists, skip"
        continue
    fi
    SHARDS=( "$RESP_DIR/${DS}".shard*of${DPT}.jsonl )
    if [[ ! -f "${SHARDS[0]}" ]]; then
        echo "[run_solver_eval]   WARN: no shards for $DS"
        continue
    fi
    cat "${SHARDS[@]}" > "$MERGED"
    N=$(wc -l < "$MERGED")
    echo "[run_solver_eval]   $DS: merged $N records -> $MERGED"
    rm -f "${SHARDS[@]}"
done
# Marker for orchestrators (benchmarks/scripts/eval_standard/run_eval.sh):
# generation has finished on every EVAL_GPU, so all but the judge GPU are free.
touch "$OUT_DIR/.gen_done"

# ---------------------------------------------------------------------------
# Stage 2: LLM recheck (verl_mv base + LLM judge on the < 0.5 samples). The
# judge is served locally on one now-free eval GPU and torn down before the
# next iteration reclaims it. No external API.
# ---------------------------------------------------------------------------
echo "[run_solver_eval] running llm_recheck (local judge)..."

    JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.6-27B}
    JUDGE_SERVED=${JUDGE_SERVED:-qwen36-judge}
    # Generation has finished, so every EVAL_GPU is free now. Use the first.
    JUDGE_GPU=${JUDGE_GPU:-${EVAL_GPU_ARR[0]}}
    JUDGE_PORT=${JUDGE_PORT:-$((8400 + JUDGE_GPU))}
    JUDGE_BASE="http://127.0.0.1:${JUDGE_PORT}/v1"
    JUDGE_LOG="$LOG_DIR/judge_serve.log"

    echo "[run_solver_eval] launching vLLM judge $JUDGE_MODEL on GPU $JUDGE_GPU (port $JUDGE_PORT)"
    GPU="$JUDGE_GPU" PORT="$JUDGE_PORT" \
        bash "$SCRIPT_DIR/serve_recheck_judge.sh" "$JUDGE_MODEL" "$JUDGE_SERVED" \
        > "$JUDGE_LOG" 2>&1 &
    JUDGE_PID=$!

    # Guarantee teardown on ANY exit so the next iter's training reclaims the GPU.
    cleanup_judge() {
        if [[ -n "${JUDGE_PID:-}" ]] && kill -0 "$JUDGE_PID" 2>/dev/null; then
            echo "[run_solver_eval] stopping judge (pid $JUDGE_PID)"
            kill "$JUDGE_PID" 2>/dev/null || true
            for _ in $(seq 1 30); do kill -0 "$JUDGE_PID" 2>/dev/null || break; sleep 1; done
            kill -9 "$JUDGE_PID" 2>/dev/null || true
        fi
    }
    trap cleanup_judge EXIT

    # Wait for the server (27B load on B200 takes ~1-3 min).
    READY=0
    for _ in $(seq 1 150); do
        if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
            echo "[run_solver_eval] FATAL: judge serve process died; see $JUDGE_LOG" >&2
            tail -40 "$JUDGE_LOG" >&2 || true
            exit 1
        fi
        if curl -sf "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; then
            READY=1; break
        fi
        sleep 5
    done
    if [[ $READY -ne 1 ]]; then
        echo "[run_solver_eval] FATAL: judge not ready after timeout; see $JUDGE_LOG" >&2
        exit 1
    fi
    echo "[run_solver_eval] judge ready at $JUDGE_BASE"

    python "$SCRIPT_DIR/llm_recheck.py" \
        --in_dir "$RESP_DIR" \
        --out_dir "$RECHECK_DIR" \
        --datasets math500 minervamath olympiadbench gsm8k amc23 aime24 aime25 \
        --base_grader verl_mv \
        --local_base_url "$JUDGE_BASE" \
        --judge_model "$JUDGE_SERVED" \
        --prompt_style new \
        --max_workers "${RECHECK_MAX_WORKERS:-8}" \
        --recheck_stall_secs "${RECHECK_STALL_SECS:-150}" \
        2>&1 | tee "$LOG_DIR/recheck.log"

    # Success path: free the GPU now, before Stage 3, and disarm the trap.
    cleanup_judge
    trap - EXIT

# ---------------------------------------------------------------------------
# Stage 3: summary.json with gap-vs-baseline
# ---------------------------------------------------------------------------
python - <<PYEOF
import json, os
out_dir   = "${OUT_DIR}"
recheck   = os.path.join(out_dir, "recheck", "recheck_summary.json")
out_path  = os.path.join(out_dir, "summary.json")

# Qwen3-4B-Base baseline (combo 4) reference measurement
BASELINE = {
    "math500":       74.00,
    "minervamath":   46.32,
    "olympiadbench": 40.95,
    "gsm8k":         86.05,
    "amc23":         27.19,
    "aime24":         6.25,
    "aime25":         3.65,
}

with open(recheck) as fh:
    rs = json.load(fh)

ds_order = ["math500","minervamath","olympiadbench","gsm8k","amc23","aime24","aime25"]
rows = {}
for ds in ds_order:
    s = rs["datasets"].get(ds)
    if s is None:
        rows[ds] = {"final_pct": None, "baseline_pct": BASELINE[ds], "gap_vs_baseline": None}
        continue
    fp = s["final_pct"]
    rows[ds] = {
        "final_pct":       fp,
        "baseline_pct":    BASELINE[ds],
        "gap_vs_baseline": fp - BASELINE[ds],
        "n":               s["n"],
        "base_correct":    s["base_correct"],
        "upgraded":        s["upgraded"],
    }

_judge = rs.get("model", "qwen36-judge")
summary = {
    "model":   "${MODEL}",
    "grader":  f"verl_mv + LOCAL {_judge} (enable_thinking=False, prompt_style=new)",
    "datasets": rows,
}
with open(out_path, "w") as fh:
    json.dump(summary, fh, indent=2)

# Pretty print
print()
print("=" * 78)
print(f"  Solver eval summary: ${MODEL}")
print("=" * 78)
print(f"  {'dataset':<16}{'final_pct':>12}{'baseline':>12}{'gap':>10}")
print("  " + "-" * 50)
for ds in ds_order:
    r = rows[ds]
    fp = r["final_pct"]
    if fp is None:
        print(f"  {ds:<16}{'(missing)':>12}{r['baseline_pct']:>12.2f}{'-':>10}")
    else:
        gap = r["gap_vs_baseline"]
        sign = "+" if gap >= 0 else ""
        print(f"  {ds:<16}{fp:>11.2f}%{r['baseline_pct']:>11.2f}%{sign}{gap:>9.2f}")
print("=" * 78)
print(f"  written: {out_path}")
PYEOF

echo "[run_solver_eval] done."
