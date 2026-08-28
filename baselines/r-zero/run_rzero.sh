#!/bin/bash
# Orchestrate 3 iterations of R-Zero self-play.
#
# env vars:
#   SMOKE=1   -> 1-step challenger, 100 total questions, 1-step solver, 1 iteration.
#                Each of those is overridable via RZERO_SMOKE_{CHAL_STEPS,CHAL_SAVE,
#                SOLVER_STEPS,SOLVER_SAVE,NUM_Q,NITER} (same contract as
#                recipe_J-Zero/run_jzero.sh).
#
# Usage: bash run_rzero.sh

set -euo pipefail
set -x

RZERO_VENV="${RZERO_VENV:?export RZERO_VENV=/path/to/venv built from requirements/requirements-train-vllm011.txt}"
source "$RZERO_VENV/bin/activate"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Cache redirects away from a noexec /tmp, offline flags, vLLM backend pins.
source "$SCRIPT_DIR/env_prelude.sh"
VERL_REPO=$( cd "$SCRIPT_DIR/.." && pwd )

# OUT_ROOT: run artifacts (generated questions, filtered data, logs, evals).
OUT_ROOT="${OUT_ROOT:?export OUT_ROOT=/path/to/output root for this run}"
CKPT_ROOT=${CKPT_ROOT:-"$VERL_REPO/checkpoints/R-Zero-Repro"}
mkdir -p "$OUT_ROOT/logs" "$CKPT_ROOT"

BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3-4B-Base"}

if [[ "${SMOKE:-0}" == "1" ]]; then
    # Every smoke knob is overridable, matching pipeline/run_jzero.sh so both
    # pipelines share one contract (e.g. RZERO_SMOKE_NITER=2 to exercise the
    # iteration hand-off rather than a single round).
    CHAL_STEPS=${RZERO_SMOKE_CHAL_STEPS:-1}
    CHAL_SAVE=${RZERO_SMOKE_CHAL_SAVE:-$CHAL_STEPS}
    SOLVER_STEPS=${RZERO_SMOKE_SOLVER_STEPS:-1}
    SOLVER_SAVE=${RZERO_SMOKE_SOLVER_SAVE:-$SOLVER_STEPS}
    NUM_Q=${RZERO_SMOKE_NUM_Q:-100}
    NITER=${RZERO_SMOKE_NITER:-1}
    TAG="smoke"
    export SMOKE=1   # propagate to solver_train.sh
else
    CHAL_STEPS=5
    CHAL_SAVE=5
    SOLVER_STEPS=15
    SOLVER_SAVE=15
    NUM_Q=8000
    # 3 = what all three paper baseline runs used (see RUNS.md); the R-Zero
    # paper's own schedule is 10 (export RZERO_NITER=10).
    NITER=${RZERO_NITER:-3}
    TAG="full"
fi

# Per-iter eval switch: RZERO_SKIP_PER_ITER_EVAL is canonical and
# RZERO_SKIP_ITER_EVAL its alias, so setting either to 0 turns per-iter eval on
# (an explicit RZERO_SKIP_PER_ITER_EVAL wins). Default 1 = OFF, matching the
# launchers/ scripts and pipeline/run_jzero.sh.
SKIP_PER_ITER_EVAL=${RZERO_SKIP_PER_ITER_EVAL:-${RZERO_SKIP_ITER_EVAL:-1}}

echo "[run_rzero] BASE=$BASE_MODEL TAG=$TAG NITER=$NITER"

# Optional resume: when RZERO_START_ITER > 1, skip iter 1..(START-1) and bootstrap
# PREV_SOLVER / PREV_CHALLENGER from iter(START-1) checkpoints under CKPT_ROOT.
START_ITER=${RZERO_START_ITER:-1}
echo "[run_rzero] START_ITER=$START_ITER"

latest_hf_ckpt() {
    # <ckpt_root>/<exp>/global_step_X/actor/huggingface for the largest X.
    local exp_dir="$1"
    local latest
    latest=$(ls -1 "$exp_dir" 2>/dev/null | grep '^global_step_[0-9]\+$' | sort -t_ -k3 -n | tail -1 || true)
    if [[ -z "$latest" ]]; then
        echo "ERROR: no checkpoint in $exp_dir" >&2
        exit 1
    fi
    echo "$exp_dir/$latest/actor/huggingface"
}

if [[ "$START_ITER" -gt 1 ]]; then
    PREV_ITER=$((START_ITER - 1))
    PREV_SOLVER=$(latest_hf_ckpt "$CKPT_ROOT/iter${PREV_ITER}_solver_${TAG}")
    PREV_CHALLENGER=$(latest_hf_ckpt "$CKPT_ROOT/iter${PREV_ITER}_challenger_${TAG}")
    echo "[run_rzero] resume: PREV_SOLVER=$PREV_SOLVER"
    echo "[run_rzero] resume: PREV_CHALLENGER=$PREV_CHALLENGER"
else
    PREV_SOLVER="$BASE_MODEL"
    PREV_CHALLENGER="$BASE_MODEL"
fi

for ITER in $(seq "$START_ITER" $NITER); do
    ITER_DIR="$OUT_ROOT/iter${ITER}_${TAG}"
    mkdir -p "$ITER_DIR/logs"

    # --- (a) Train Challenger ---------------------------------------------
    # Stages whose HF export already exists are skipped, so rerunning after a
    # crash continues from the first unfinished stage (as in the J-Zero pipeline).
    CHAL_EXP="iter${ITER}_challenger_${TAG}"
    if compgen -G "$CKPT_ROOT/$CHAL_EXP/global_step_*/actor/huggingface/model*.safetensors*" >/dev/null; then
        echo "[run_rzero] iter $ITER challenger already exported — skipping stage"
    else
        bash "$SCRIPT_DIR/challenger_train.sh" \
            "$PREV_SOLVER" \
            "$PREV_CHALLENGER" \
            "$CHAL_EXP" \
            "$ITER_DIR/logs" \
            "$CKPT_ROOT" \
            "$CHAL_STEPS" "$CHAL_SAVE"
    fi

    NEW_CHALLENGER=$(latest_hf_ckpt "$CKPT_ROOT/$CHAL_EXP")

    # --- (b) Train Solver (gen -> filter -> train) ------------------------
    SOL_EXP="iter${ITER}_solver_${TAG}"
    if compgen -G "$CKPT_ROOT/$SOL_EXP/global_step_*/actor/huggingface/model*.safetensors*" >/dev/null; then
        echo "[run_rzero] iter $ITER solver already exported — skipping stage"
    else
        bash "$SCRIPT_DIR/solver_train.sh" \
            "$PREV_SOLVER" \
            "$NEW_CHALLENGER" \
            "$SOL_EXP" \
            "$ITER_DIR" \
            "$CKPT_ROOT" \
            "$NUM_Q" "$SOLVER_STEPS" "$SOLVER_SAVE"
    fi

    NEW_SOLVER=$(latest_hf_ckpt "$CKPT_ROOT/$SOL_EXP")
    PREV_SOLVER="$NEW_SOLVER"
    PREV_CHALLENGER="$NEW_CHALLENGER"

    echo "[run_rzero] iter $ITER done. solver=$NEW_SOLVER"

    # --- (c) Per-iter 7-benchmark eval (verl_mv + LLM recheck, local judge) ---
    # OFF by default: the paper runs evaluate all iters post-hoc with the
    # 14-bench sweep. See SKIP_PER_ITER_EVAL above.
    if [[ "${SMOKE:-0}" != "1" && "$SKIP_PER_ITER_EVAL" != "1" ]]; then
        EVAL_ROOT="$OUT_ROOT/eval_${TAG}"
        mkdir -p "$EVAL_ROOT"
        echo "[run_rzero] iter $ITER eval starting"
        bash "$SCRIPT_DIR/evaluate/run_solver_eval.sh" "$NEW_SOLVER" "$EVAL_ROOT/iter${ITER}" \
            || echo "[run_rzero] WARN: iter $ITER eval failed (training continues)"
        echo "[run_rzero] iter $ITER eval done"
    fi
done

echo "[run_rzero] all training iters done."
