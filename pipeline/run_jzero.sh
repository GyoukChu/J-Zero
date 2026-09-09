#!/bin/bash
# J-Zero orchestrator. NITER iterations of RM-in-the-loop self-play on
# GPU 0-3: challenger GRPO -> question gen + RM filter (band OFF) -> solver
# GRPO -> RM Bradley-Terry on the UNION of challenger-negative pairs +
# IDA subtask-amplification pairs (Stage 3 = rm_train.sh).
#
# Install contract: this pipeline/ directory must live directly inside a verl
# checkout (any subdirectory name, e.g. <verl>/recipe_J-Zero) — VERL_REPO is derived as the parent directory of this
# script (used for checkpoints and `python -m verl.trainer.main_ppo`).
set -euo pipefail
set -x
# shellcheck disable=SC1090
source "${RZERO_VENV_ACTIVATE:?export RZERO_VENV_ACTIVATE=/path/to/venv/bin/activate (env from requirements/requirements-train-vllm011.txt + verl installed editable)}"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# The custom reward modules import repo-relative packages (rm_service.*,
# reward.*), which resolve via sys.path[0]=cwd — so run every stage from here.
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/env_prelude.sh"
VERL_REPO=$( cd "$SCRIPT_DIR/.." && pwd )
PROJECT_ROOT="${PROJECT_ROOT:?export PROJECT_ROOT=... (directory holding .env, data/ and outputs/)}"
export PROJECT_ROOT

OUT_ROOT=${OUT_ROOT:-"$PROJECT_ROOT/outputs/run_jzero"}
CKPT_ROOT=${CKPT_ROOT:-"$VERL_REPO/checkpoints/J-Zero"}
mkdir -p "$OUT_ROOT/logs" "$CKPT_ROOT"
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3-4B-Base"}
RZERO_RM_INIT_PATH=${RZERO_RM_INIT_PATH:-"Skywork/Skywork-Reward-V2-Llama-3.1-8B"}
RZERO_RM_TRAIN_ENABLE=${RZERO_RM_TRAIN_ENABLE:-1}

if [[ "${SMOKE:-0}" == "1" ]]; then
    CHAL_STEPS=${RZERO_SMOKE_CHAL_STEPS:-1}; CHAL_SAVE=${RZERO_SMOKE_CHAL_SAVE:-$CHAL_STEPS}
    SOLVER_STEPS=${RZERO_SMOKE_SOLVER_STEPS:-1}; SOLVER_SAVE=${RZERO_SMOKE_SOLVER_SAVE:-$SOLVER_STEPS}
    NUM_Q=${RZERO_SMOKE_NUM_Q:-100}; NITER=${RZERO_SMOKE_NITER:-1}; TAG="jzero_smoke"; export SMOKE=1
else
    CHAL_STEPS=5; CHAL_SAVE=5; SOLVER_STEPS=15; SOLVER_SAVE=15; NUM_Q=8000
    NITER=${RZERO_NITER:-10}; TAG="${RZERO_TAG:-jzero_full}"
fi

# Per-iter eval switch (default 1 = OFF). RZERO_SKIP_ITER_EVAL is an alias of
# RZERO_SKIP_PER_ITER_EVAL (the canonical name wins); same resolution as
# baselines/r-zero/run_rzero.sh.
SKIP_PER_ITER_EVAL=${RZERO_SKIP_PER_ITER_EVAL:-${RZERO_SKIP_ITER_EVAL:-1}}

echo "[run_jzero] BASE=$BASE_MODEL TAG=$TAG NITER=$NITER RM_INIT=$RZERO_RM_INIT_PATH"
echo "[run_jzero] RM pairs: chalneg=${RZERO_RM_N_CHAL:-256} subtask=${RZERO_RM_N_SUBTASK:-256}"
START_ITER=${RZERO_START_ITER:-1}

latest_hf_ckpt() {
    local exp_dir="$1" latest
    latest=$(ls -1 "$exp_dir" 2>/dev/null | grep '^global_step_[0-9]\+$' | sort -t_ -k3 -n | tail -1 || true)
    [[ -z "$latest" ]] && { echo "ERROR: no checkpoint in $exp_dir" >&2; exit 1; }
    echo "$exp_dir/$latest/actor/huggingface"
}

if [[ "$START_ITER" -gt 1 ]]; then
    PREV_ITER=$((START_ITER - 1))
    PREV_SOLVER=$(latest_hf_ckpt "$CKPT_ROOT/iter${PREV_ITER}_solver_${TAG}")
    PREV_CHALLENGER=$(latest_hf_ckpt "$CKPT_ROOT/iter${PREV_ITER}_challenger_${TAG}")
    if [[ "$RZERO_RM_TRAIN_ENABLE" == "1" && -d "$CKPT_ROOT/iter${PREV_ITER}_rm_${TAG}" ]]; then
        PREV_RM=$(latest_hf_ckpt "$CKPT_ROOT/iter${PREV_ITER}_rm_${TAG}")
    else
        PREV_RM="$RZERO_RM_INIT_PATH"
    fi
    echo "[run_jzero] resume: PREV_RM=$PREV_RM"
else
    PREV_SOLVER="$BASE_MODEL"; PREV_CHALLENGER="$BASE_MODEL"; PREV_RM="$RZERO_RM_INIT_PATH"
fi

for ITER in $(seq "$START_ITER" $NITER); do
    ITER_DIR="$OUT_ROOT/iter${ITER}_${TAG}"; mkdir -p "$ITER_DIR/logs"
    CHAL_EXP="iter${ITER}_challenger_${TAG}"; SOL_EXP="iter${ITER}_solver_${TAG}"

    EXPECTED_CHAL_CKPT="$CKPT_ROOT/$CHAL_EXP/global_step_${CHAL_STEPS}/actor/huggingface"
    if [[ "${RZERO_REUSE_CHALLENGER_CKPT:-1}" == "1" && -f "$EXPECTED_CHAL_CKPT/config.json" ]]; then
        NEW_CHALLENGER="$EXPECTED_CHAL_CKPT"
    else
        bash "$SCRIPT_DIR/challenger_train.sh" "$PREV_SOLVER" "$PREV_CHALLENGER" "$PREV_RM" \
            "$CHAL_EXP" "$ITER_DIR/logs" "$CKPT_ROOT" "$CHAL_STEPS" "$CHAL_SAVE"
        NEW_CHALLENGER=$(latest_hf_ckpt "$CKPT_ROOT/$CHAL_EXP")
    fi

    EXPECTED_SOL_CKPT="$CKPT_ROOT/$SOL_EXP/global_step_${SOLVER_STEPS}/actor/huggingface"
    if [[ "${RZERO_REUSE_SOLVER_CKPT:-1}" == "1" && -f "$EXPECTED_SOL_CKPT/config.json" ]]; then
        NEW_SOLVER="$EXPECTED_SOL_CKPT"
    else
        bash "$SCRIPT_DIR/solver_train.sh" "$PREV_SOLVER" "$NEW_CHALLENGER" "$PREV_RM" \
            "$SOL_EXP" "$ITER_DIR" "$CKPT_ROOT" "$NUM_Q" "$SOLVER_STEPS" "$SOLVER_SAVE"
        NEW_SOLVER=$(latest_hf_ckpt "$CKPT_ROOT/$SOL_EXP")
    fi

    NEW_RM="$PREV_RM"
    if [[ "$RZERO_RM_TRAIN_ENABLE" == "1" ]]; then
        RM_EXP="iter${ITER}_rm_${TAG}"
        rzero_rm_pair_mix "$ITER"
        echo "[run_jzero] iter $ITER RM pairs: chalneg=$RZERO_RM_N_CHAL subtask=$RZERO_RM_N_SUBTASK"
        # Must use rzero_rm_steps() (env_prelude.sh) — the same derivation
        # rm_train.sh uses for its SAVE_DIR — or the resume check misses.
        EXPECTED_RM_CKPT="$CKPT_ROOT/$RM_EXP/global_step_$(rzero_rm_steps)/actor/huggingface"
        if [[ "${RZERO_REUSE_RM_CKPT:-1}" == "1" && -f "$EXPECTED_RM_CKPT/config.json" ]]; then
            NEW_RM="$EXPECTED_RM_CKPT"
            echo "[run_jzero] iter $ITER RM checkpoint exists, skipping -> $NEW_RM"
        else
            bash "$SCRIPT_DIR/rm_train.sh" "$NEW_SOLVER" "$NEW_CHALLENGER" "$PREV_RM" \
                "$RM_EXP" "$ITER_DIR" "$CKPT_ROOT" "$SOL_EXP" "$ITER_DIR/filtered" \
                "$ITER_DIR/parquet/${SOL_EXP}.parquet"
            NEW_RM=$(latest_hf_ckpt "$CKPT_ROOT/$RM_EXP")
            echo "[run_jzero] iter $ITER RM trained -> $NEW_RM"
        fi
    fi

    PREV_SOLVER="$NEW_SOLVER"; PREV_CHALLENGER="$NEW_CHALLENGER"; PREV_RM="$NEW_RM"
    echo "[run_jzero] iter $ITER done. solver=$NEW_SOLVER"

    if [[ "${SMOKE:-0}" != "1" && "$SKIP_PER_ITER_EVAL" != "1" ]]; then
        EVAL_ROOT="$OUT_ROOT/eval_${TAG}"; mkdir -p "$EVAL_ROOT"
        if [[ ! -f "$EVAL_ROOT/iter${ITER}/summary.json" ]]; then
            bash "${RZERO_EVAL_SCRIPT:?per-iter eval enabled but RZERO_EVAL_SCRIPT unset — point it at baselines/r-zero/evaluate/run_solver_eval.sh}" "$NEW_SOLVER" "$EVAL_ROOT/iter${ITER}" \
                || echo "[run_jzero] WARN: iter $ITER eval failed (training continues)"
        fi
    fi
done
echo "[run_jzero] all training iters done."
