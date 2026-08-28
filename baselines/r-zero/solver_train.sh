#!/bin/bash
# Solver iteration:
#   1) generate N questions with the (just-trained) challenger (4 GPU parallel)
#   2) for each question, generate m=10 solver answers, majority-vote, filter
#   3) build verl parquet
#   4) GRPO train solver on it (4 GPU FSDP)
#
# Usage: bash solver_train.sh <solver_init_model> <challenger_model> <exp_name> <iter_root> <ckpt_root> [num_questions_total] [max_steps] [save_freq]

set -euo pipefail
set -x

RZERO_VENV="${RZERO_VENV:?export RZERO_VENV=/path/to/venv built from requirements/requirements-train-vllm011.txt}"
source "$RZERO_VENV/bin/activate"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/env_prelude.sh"

SOLVER_INIT=${1:?"solver_init_model required"}
CHALLENGER_MODEL=${2:?"challenger_model required"}
EXP_NAME=${3:?"exp_name required"}
ITER_ROOT=${4:?"iter_root required (output dir for this iter)"}
CKPT_ROOT=${5:?"ckpt_root required"}
NUM_Q_TOTAL=${6:-8000}
MAX_STEPS=${7:-15}
SAVE_FREQ=${8:-15}

# --- 0. Resolve the val parquets UP FRONT ------------------------------------
# verl needs resolvable data.val_files even with in-training validation off (see
# the NOTE below). This must run at entry, not just before the trainer: question
# generation + m=10 filtering come first and take hours at full scale.
#
# Candidates in order: caller-set $DATA_DIR, $PROJECT_ROOT/data (the root
# README's layout), $SCRIPT_DIR/../../benchmarks/data (valid only while this
# recipe sits at <repo>/baselines/r-zero/), then any benchmarks/data above
# $SCRIPT_DIR — which covers the README's <verl>/recipe_R-Zero install layout.
resolve_data_dir() {
    local c d
    # An explicitly-set DATA_DIR is authoritative: falling through to another
    # candidate would silently train against a different copy of the parquets.
    if [[ -n "${DATA_DIR:-}" ]]; then
        [[ -f "$DATA_DIR/math500.parquet" ]] && { echo "$DATA_DIR"; return 0; }
        echo "ERROR: DATA_DIR=$DATA_DIR is set but contains no math500.parquet." >&2
        return 1
    fi
    # Guarded: with PROJECT_ROOT unset this would be the literal /data, a real
    # shared mount on some clusters.
    [[ -n "${PROJECT_ROOT:-}" && -f "$PROJECT_ROOT/data/math500.parquet" ]] \
        && { echo "$PROJECT_ROOT/data"; return 0; }
    c="$SCRIPT_DIR/../../benchmarks/data"
    [[ -f "$c/math500.parquet" ]] && { echo "$c"; return 0; }
    d="$SCRIPT_DIR"
    while [[ "$d" != "/" ]]; do
        [[ -f "$d/benchmarks/data/math500.parquet" ]] && { echo "$d/benchmarks/data"; return 0; }
        d=$(dirname "$d")
    done
    return 1
}
if ! DATA_DIR=$(resolve_data_dir); then
    echo "ERROR: could not resolve the verl-format eval parquets" >&2
    echo "  (math500/minervamath/olympiadbench.parquet)." >&2
    echo "  Searched: \$DATA_DIR if set — otherwise \$PROJECT_ROOT/data," >&2
    echo "            $SCRIPT_DIR/../../benchmarks/data, and every benchmarks/data above $SCRIPT_DIR." >&2
    echo "  Fix: export DATA_DIR=<repo>/benchmarks/data" >&2
    exit 1
fi
export DATA_DIR
echo "[solver_train.sh] DATA_DIR=$DATA_DIR"

# NOTE: in-training validation is disabled across the board. verl's training-time
# val routes through solver_reward.py, a bare rule-based grader that under-scores
# answers vs. paper-reported numbers. The authoritative eval is
# evaluate/run_solver_eval.sh, which run_rzero.sh calls after each iter.
if [[ "${SMOKE:-0}" == "1" ]]; then
    VAL_BEFORE_TRAIN=False
    TEST_FREQ=-1
    MIN_ROWS_REQUIRED=8
    TRAIN_BATCH=8     # smoke: a handful of questions survives filter, 1 step
    PPO_MINI_BATCH=4
else
    VAL_BEFORE_TRAIN=False
    TEST_FREQ=-1
    MIN_ROWS_REQUIRED=16
    TRAIN_BATCH=128
    PPO_MINI_BATCH=32
fi

LOG_DIR="$ITER_ROOT/logs"
mkdir -p "$LOG_DIR"
GEN_DIR="$ITER_ROOT/generated"
FILT_DIR="$ITER_ROOT/filtered"
PARQUET_DIR="$ITER_ROOT/parquet"
mkdir -p "$GEN_DIR" "$FILT_DIR" "$PARQUET_DIR"

# Secrets must come from the caller's env; trace is off across the check so the
# values are never echoed. WANDB_API_KEY is optional iff WANDB_MODE=disabled.
{ set +x; } 2>/dev/null
: "${HF_TOKEN:?HF_TOKEN must be exported by the caller (export HF_TOKEN=hf_...)}"
if [[ "${WANDB_MODE:-}" != "disabled" && "${WANDB_MODE:-}" != "offline" ]]; then
    : "${WANDB_API_KEY:?WANDB_API_KEY must be exported by the caller (or set WANDB_MODE=disabled)}"
fi
set -x
export VLLM_USE_DEEP_GEMM=0
export VLLM_SKIP_DEEP_GEMM_WARMUP=1
export VLLM_DISABLE_COMPILE_CACHE=1
export HYDRA_FULL_ERROR=1

# Redundant with challenger_train.sh's EXIT trap, but needed when this script is
# launched standalone or that trap was bypassed: no stale ray cluster / verl
# trainer / vLLM worker may be alive when we ray.init() below.
bash "$SCRIPT_DIR/cleanup_between_phases.sh" "solver-entry" || true

# Cleanup also on exit, so the *next* iter's challenger starts clean.
solver_cleanup() {
    bash "$SCRIPT_DIR/cleanup_between_phases.sh" "post-solver" || true
}
trap solver_cleanup EXIT

# gen/filter run one process per GPU in RZERO_GEN_GPUS, so the per-GPU sample
# count is divided by the GPU count to hold NUM_Q_TOTAL constant.
NUM_GEN_GPUS=$(echo ${RZERO_GEN_GPUS:-0 1 2 3} | wc -w)
PER_GPU=$((NUM_Q_TOTAL / NUM_GEN_GPUS))
echo "[solver_train.sh] generating $NUM_Q_TOTAL questions ($PER_GPU per GPU x $NUM_GEN_GPUS GPUs)"

# --- 1. Generate questions (4 GPU parallel) ---------------------------------
bash "$SCRIPT_DIR/question_gen/run_parallel.sh" \
    "$CHALLENGER_MODEL" "$EXP_NAME" "$GEN_DIR" "$PER_GPU" \
    2>&1 | tee "$LOG_DIR/gen.log"

# --- 2. Filter via solver (4 GPU parallel) ----------------------------------
bash "$SCRIPT_DIR/question_filter/run_parallel.sh" \
    "$SOLVER_INIT" "$EXP_NAME" "$GEN_DIR" "$FILT_DIR" 10 \
    2>&1 | tee "$LOG_DIR/filter.log"

# --- 3. Build training parquet ---------------------------------------------
TRAIN_PARQUET="$PARQUET_DIR/${EXP_NAME}.parquet"
python "$SCRIPT_DIR/question_filter/upload_to_parquet.py" \
    --in_dir "$FILT_DIR" \
    --save_name "$EXP_NAME" \
    --out_path "$TRAIN_PARQUET" \
    --min_score 0.25 --max_score 0.75 \
    --data_source "rzero_self_play" \
    2>&1 | tee "$LOG_DIR/upload.log"

if [[ ! -s "$TRAIN_PARQUET" ]]; then
    echo "[solver_train.sh] ERROR: empty train parquet" >&2
    exit 1
fi

ROW_COUNT=$(python -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_PARQUET')))")
echo "[solver_train.sh] train parquet rows: $ROW_COUNT"
if [[ "$ROW_COUNT" -lt "$MIN_ROWS_REQUIRED" ]]; then
    echo "[solver_train.sh] ERROR: too few rows ($ROW_COUNT < $MIN_ROWS_REQUIRED) for training" >&2
    exit 1
fi
# verl needs train_batch_size <= dataset rows or the dataloader comes back
# empty. Full-scale runs keep >>128 rows; this guards small ones.
while [[ "$ROW_COUNT" -lt "$TRAIN_BATCH" && "$TRAIN_BATCH" -gt "$MIN_ROWS_REQUIRED" ]]; do
    TRAIN_BATCH=$((TRAIN_BATCH / 2))
    PPO_MINI_BATCH=$(( PPO_MINI_BATCH > TRAIN_BATCH ? TRAIN_BATCH : PPO_MINI_BATCH ))
    echo "[solver_train.sh] adaptive batch: rows=$ROW_COUNT -> train_batch=$TRAIN_BATCH mini=$PPO_MINI_BATCH"
done

# --- 3.5 cleanup before verl trainer ---------------------------------------
# The gen + filter vLLM workers leave their torch.distributed TCPStores holding
# the rendezvous port, breaking the trainer's process_group init.
bash "$SCRIPT_DIR/cleanup_between_phases.sh" "pre-solver-trainer" || true

# --- 4. GRPO train solver (FSDP) -------------------------------------------
TRAIN_GPUS=${RZERO_SOLVER_TRAIN_GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)
NUM_NODES=1

CKPT_DIR="$CKPT_ROOT/$EXP_NAME"
mkdir -p "$CKPT_DIR"

# Resolved and validated at entry (step 0). Unused at runtime — validation is
# off — but verl still requires the paths to resolve.
VAL_FILES="['$DATA_DIR/math500.parquet','$DATA_DIR/minervamath.parquet','$DATA_DIR/olympiadbench.parquet']"

PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python3 -u -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2560 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="$SCRIPT_DIR/reward/solver_reward.py" \
    reward.custom_reward_function.name=compute_score \
    data.shuffle=True \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$SOLVER_INIT" \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.use_remove_padding=${RZERO_USE_RMPAD:-True} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${RZERO_ULYSSES_SP:-2} \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${RZERO_ACTOR_PARAM_OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${RZERO_ACTOR_OPT_OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl+ \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=${RZERO_REF_PARAM_OFFLOAD:-False} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=${RZERO_FREE_CACHE:-False} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${RZERO_SOLVER_MEM_UTIL:-0.55} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    trainer.balance_batch=False \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.logger=['console','wandb'] \
    trainer.project_name="R-Zero-Repro" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=$MAX_STEPS \
    trainer.rollout_data_dir="$LOG_DIR/rollouts_solver" \
    ${RZERO_EXTRA_HYDRA:-} \
    2> >(tee "$LOG_DIR/trainer.stderr" >&2) \
    | tee "$LOG_DIR/${EXP_NAME}.log"

# --- 5. Convert ckpt to HF --------------------------------------------------
LATEST=$(ls -1 "$CKPT_DIR" 2>/dev/null | grep '^global_step_[0-9]\+$' | sort -t_ -k3 -n | tail -1 || true)
if [[ -z "$LATEST" ]]; then
    echo "[solver_train.sh] ERROR: no checkpoint in $CKPT_DIR" >&2
    exit 1
fi
CKPT_ACTOR="$CKPT_DIR/$LATEST/actor"
HF_DIR="$CKPT_DIR/$LATEST/actor/huggingface"
if [[ -d "$CKPT_ACTOR" ]]; then
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$CKPT_ACTOR" \
        --target_dir "$HF_DIR"
    echo "[solver_train.sh] HF weights written to $HF_DIR"
else
    echo "[solver_train.sh] WARNING: no actor dir at $CKPT_ACTOR" >&2
fi
