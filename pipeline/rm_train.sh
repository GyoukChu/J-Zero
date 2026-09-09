#!/bin/bash
# Challenger-negative + IDA subtask-amplification RM stage (Stage 3). Fully
# internal, NO external supervision; works on verifiable and unverifiable
# questions alike.
#   (A) chalneg  [N_CHAL, default 256] build_pairs_challenger_neg.py
#         chosen=solver answer / rejected=challenger-as-solver answer
#   (B) subtask  [N_SUB, default 256] build_pairs_subtask.py
#         chosen=solver's composition of its own sub-answers / rejected=one-shot
# No RM servers are started: pair labels never come from the Judge. The builders
# read the K=10 solver answer pool from the filter stage, or regenerate it (see
# RZERO_RM_FRESH_SOLVER below).
#
# Usage: bash rm_train.sh <solver_model> <challenger_model> <rm_init> <exp_name> \
#                         <iter_root> <ckpt_root> <solver_save_name> \
#                         <solver_filtered_dir> <solver_train_parquet>
set -euo pipefail
set -x
# shellcheck disable=SC1090
source "${RZERO_VENV_ACTIVATE:?export RZERO_VENV_ACTIVATE=/path/to/venv/bin/activate (env from requirements/requirements-train-vllm011.txt + verl installed editable)}"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/env_prelude.sh"

SOLVER_MODEL=${1:?}; CHALLENGER_MODEL=${2:?}; RM_INIT=${3:?}; EXP_NAME=${4:?}
ITER_ROOT=${5:?}; CKPT_ROOT=${6:?}; SOLVER_SAVE_NAME=${7:?}
SOLVER_FILTERED_DIR=${8:?}; SOLVER_TRAIN_PARQUET=${9:?}
LOG_DIR="$ITER_ROOT/logs_rm"; GEN_DIR="$ITER_ROOT/rm_gen"; mkdir -p "$LOG_DIR" "$GEN_DIR"

# DUPLICATED DEFAULTS — keep in sync with launch/launch_jzero_4gpu.sh and with
# rzero_rm_steps() in env_prelude.sh. The launcher normally exports all of
# these; the values below only apply when rm_train.sh is driven without it.
if [[ "${SMOKE:-0}" == "1" ]]; then RM_BATCH=${RZERO_RM_BATCH:-8}; N_CHAL=${RZERO_RM_N_CHAL:-8}; N_SUB=${RZERO_RM_N_SUBTASK:-8}
else RM_BATCH=${RZERO_RM_BATCH:-64}; N_CHAL=${RZERO_RM_N_CHAL:-256}; N_SUB=${RZERO_RM_N_SUBTASK:-256}; fi
# Normalized smoke flag for ${VAR:+} expansions below: a bare ${SMOKE:+} would
# treat an explicit SMOKE=0 as smoke and silently relax the full run's
# expected-pair hard-fails.
SMOKE_ON=""; [[ "${SMOKE:-0}" == "1" ]] && SMOKE_ON=1
TOTAL_PAIRS=$(( N_CHAL + N_SUB ))
# Must stay the single derivation shared with run_jzero.sh's RM resume check —
# never re-derive it here (see rzero_rm_steps() in env_prelude.sh).
RM_STEPS=$(rzero_rm_steps)
RM_LR=${RZERO_RM_LR:-5e-7}; RM_MAX_LEN=${RZERO_RM_MAX_LEN:-8192}
RM_MICRO_BATCH=${RZERO_RM_MICRO_BATCH:-4}; RM_GRAD_CKPT=${RZERO_RM_GRAD_CKPT:-1}
# Pair generation uses one independent vLLM worker per GPU.
GEN_GPUS=${RZERO_RM_GEN_GPUS:-${RZERO_CHALNEG_GEN_GPU:-0,1,2,3}}
GEN_MEM_UTIL=${RZERO_CHALNEG_GEN_MEM_UTIL:-0.85}
OVERSAMPLE=${RZERO_CHALNEG_OVERSAMPLE:-1.5}
# Fresh-solver pair building (default ON): regenerate the solver-side answers
# (chalneg chosen / subtask one-shot rejected) with the just-trained solver_t.
# Set RZERO_RM_FRESH_SOLVER=0 to reuse the pre-training filter-stage pool.
FRESH_SOLVER=${RZERO_RM_FRESH_SOLVER:-1}
CHALNEG_FRESH_ARGS=(); SUBTASK_FRESH_ARGS=()
if [[ "$FRESH_SOLVER" == "1" ]]; then
    CHALNEG_FRESH_ARGS=(--solver_model "$SOLVER_MODEL")
    SUBTASK_FRESH_ARGS=(--fresh_rejected)
fi

{ set +x; } 2>/dev/null
: "${HF_TOKEN:?HF_TOKEN must be exported by the caller}"
set -x
rm_cleanup(){ bash "$SCRIPT_DIR/cleanup_between_phases.sh" "post-rm-train" || true; }
trap rm_cleanup EXIT
bash "$SCRIPT_DIR/cleanup_between_phases.sh" "pre-rm-train" || true

CHAL_PAIRS="$GEN_DIR/${EXP_NAME}_chal_pairs.json"
SUB_PAIRS="$GEN_DIR/${EXP_NAME}_subtask_pairs.json"
PAIRS_JSON="$GEN_DIR/${EXP_NAME}_pairs.json"

if [[ -f "$PAIRS_JSON" ]]; then
    echo "[rm_train] reusing merged pairs $PAIRS_JSON"
else
    # --- (A) chalneg ---------------------------------------------------------
    if [[ ! -f "$CHAL_PAIRS" ]]; then
        echo "[rm_train] (A) chalneg pair build (N=$N_CHAL)"
        CUDA_VISIBLE_DEVICES=$GEN_GPUS python "$SCRIPT_DIR/rm_data/build_pairs_challenger_neg.py" \
            --in_dir "$SOLVER_FILTERED_DIR" --save_name "$SOLVER_SAVE_NAME" \
            --exclude_parquet "$SOLVER_TRAIN_PARQUET" --out_path "$CHAL_PAIRS" \
            --challenger_model "$CHALLENGER_MODEL" --target_pairs "$N_CHAL" \
            --oversample "$OVERSAMPLE" --gpu_mem_util "$GEN_MEM_UTIL" \
            "${CHALNEG_FRESH_ARGS[@]}" 2>&1 | tee "$LOG_DIR/chalneg_build_pairs.log"
        bash "$SCRIPT_DIR/cleanup_between_phases.sh" "post-chalneg-build" || true
    fi
    # --- (B0) chalneg-only ablation: N_SUB=0 -> skip the stage-B builder -----
    # The empty pair list makes the (B) block below a no-op and lets
    # merge_pairs.py proceed on chalneg alone (expected==N_CHAL).
    if [[ "$N_SUB" -eq 0 && ! -f "$SUB_PAIRS" ]]; then
        echo "[rm_train] N_SUB=0 -> chalneg-only ablation; writing empty subtask pairs"
        echo '[]' > "$SUB_PAIRS"
    fi
    # --- (B) above-solver positive: IDA subtask amplification -----------------
    MAXSUB=${RZERO_RM_SUBTASK_MAXSUB:-5}
    if [[ ! -f "$SUB_PAIRS" ]]; then
        echo "[rm_train] (B) IDA subtask-amplification build (N=$N_SUB, max_subtasks=$MAXSUB)"
        CUDA_VISIBLE_DEVICES=$GEN_GPUS python "$SCRIPT_DIR/rm_data/build_pairs_subtask.py" \
            --model "$SOLVER_MODEL" \
            --in_dir "$SOLVER_FILTERED_DIR" --save_name "$SOLVER_SAVE_NAME" \
            --exclude_parquet "$SOLVER_TRAIN_PARQUET" --out_path "$SUB_PAIRS" \
            --target_pairs "$N_SUB" --max_subtasks "$MAXSUB" --gpu_mem_util "$GEN_MEM_UTIL" \
            "${SUBTASK_FRESH_ARGS[@]}" 2>&1 | tee "$LOG_DIR/subtask_build_pairs.log"
        bash "$SCRIPT_DIR/cleanup_between_phases.sh" "post-subtask-build" || true
    fi
    # --- merge ---------------------------------------------------------------
    echo "[rm_train] merge chalneg + subtask -> $PAIRS_JSON (expected=$TOTAL_PAIRS)"
    python "$SCRIPT_DIR/rm_data/merge_pairs.py" \
        --pairs_a "$CHAL_PAIRS" --pairs_b "$SUB_PAIRS" \
        --name_a chalneg --name_b subtask \
        --out_path "$PAIRS_JSON" --expected "$TOTAL_PAIRS" ${SMOKE_ON:+--allow_short} 2>&1 | tee "$LOG_DIR/merge_pairs.log"
fi
[[ -f "$PAIRS_JSON" ]] || { echo "[rm_train] FATAL: merged pairs missing" >&2; exit 1; }

# --- (C) FSDP Bradley-Terry training on GPU 0-3 ---------------------------
# The RM is NOT a verl actor — rm_train_loop.py is a plain torchrun FSDP loop.
# It must keep mimicking verl's global_step_N/actor/huggingface layout so
# run_jzero.sh's latest_hf_ckpt() and RM resume check work unchanged here.
SAVE_DIR="$CKPT_ROOT/$EXP_NAME/global_step_${RM_STEPS}/actor/huggingface"; mkdir -p "$SAVE_DIR"
echo "[rm_train] (C) BT train (batch=$RM_BATCH steps=$RM_STEPS total=$TOTAL_PAIRS)"
CUDA_VISIBLE_DEVICES=0,1,2,3 RZERO_RM_ALLOW_SHORT="${RZERO_RM_ALLOW_SHORT:-${SMOKE_ON:+1}}" torchrun --nproc_per_node=4 \
    --master_port="${RZERO_RM_MASTER_PORT:-$((20000 + RANDOM % 10000))}" "$SCRIPT_DIR/rm_train_loop.py" \
    --pairs_json "$PAIRS_JSON" --init_model "$RM_INIT" --save_dir "$SAVE_DIR" \
    --batch_size "$RM_BATCH" --num_steps "$RM_STEPS" --lr "$RM_LR" --max_len "$RM_MAX_LEN" \
    --micro_batch "$RM_MICRO_BATCH" --gradient_checkpointing "$RM_GRAD_CKPT" --exp_name "$EXP_NAME" \
    2>&1 | tee "$LOG_DIR/rm_train.log"
{ set +e; } 2>/dev/null
# rm_train_loop.py pins its wandb dir to the recipe directory (dir=HERE)
# regardless of WANDB_DIR, so always sync from there — following WANDB_DIR
# here would sync a directory the RM run never writes to.
WANDB_RUN_DIR="$SCRIPT_DIR/wandb"
if ! command -v wandb >/dev/null 2>&1; then
    echo "[rm_train] wandb-flush: SKIPPED, no wandb CLI on PATH (offline runs will not be synced)"
elif [[ ! -d "$WANDB_RUN_DIR" ]]; then
    echo "[rm_train] wandb-flush: WARN no wandb run dir at $WANDB_RUN_DIR — nothing synced. If wandb writes elsewhere, export WANDB_DIR to point here."
else
    timeout 120 wandb sync --include-offline --sync-all "$WANDB_RUN_DIR" 2>&1 | tail -10 || true
fi
{ set -e; } 2>/dev/null
[[ -f "$SAVE_DIR/config.json" ]] || { echo "[rm_train] FATAL: no config.json in $SAVE_DIR" >&2; exit 1; }
echo "[rm_train] done. new RM at $SAVE_DIR"
