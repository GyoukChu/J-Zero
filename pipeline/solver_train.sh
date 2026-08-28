#!/bin/bash
# J-Zero solver stage (Stage 2). Phases run in order on the same 4 GPUs:
#   1) Generate NUM_Q_TOTAL questions with the just-trained challenger (4-GPU vLLM, GPU 0-3).
#   2) RM-based filter: 4-way vLLM answer-gen on GPU 0-3, then 4 RM servers (GPU 0-3) score and
#      filter (avg ∈ [LO, HI] + top-N by std). Produces a verl parquet.
#   3) GRPO train solver (FSDP on GPU 0,1) with 2 RM HTTP servers (GPU 2,3)
#      providing per-rollout reward = sigmoid(raw_RM).
#
# Usage: bash solver_train.sh <solver_init> <challenger_model> <rm_model> \
#                                    <exp_name> <iter_root> <ckpt_root> \
#                                    [num_questions_total=8000] [max_steps=15] [save_freq=15]

set -euo pipefail
set -x

# shellcheck disable=SC1090
source "${RZERO_VENV_ACTIVATE:?export RZERO_VENV_ACTIVATE=/path/to/venv/bin/activate (env from requirements/requirements-train-vllm011.txt + verl installed editable)}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/env_prelude.sh"

SOLVER_INIT=${1:?"solver_init required"}
CHALLENGER_MODEL=${2:?"challenger_model required"}
RM_MODEL=${3:?"rm_model required"}
EXP_NAME=${4:?"exp_name required"}
ITER_ROOT=${5:?"iter_root required"}
CKPT_ROOT=${6:?"ckpt_root required"}
NUM_Q_TOTAL=${7:-8000}
MAX_STEPS=${8:-15}
SAVE_FREQ=${9:-15}

# --- Smoke mode ----------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
    VAL_BEFORE_TRAIN=False
    TEST_FREQ=-1
    MIN_ROWS_REQUIRED=8
    TRAIN_BATCH=8
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

# --- Secrets check --------------------------------------------------------
{ set +x; } 2>/dev/null
: "${HF_TOKEN:?HF_TOKEN must be exported by the caller (export HF_TOKEN=hf_...)}"
if [[ "${WANDB_MODE:-}" != "disabled" && "${WANDB_MODE:-}" != "offline" ]]; then
    : "${WANDB_API_KEY:?WANDB_API_KEY must be exported (or set WANDB_MODE=disabled)}"
fi
set -x

export VLLM_USE_DEEP_GEMM=0
export VLLM_SKIP_DEEP_GEMM_WARMUP=1
export VLLM_DISABLE_COMPILE_CACHE=1
export HYDRA_FULL_ERROR=1

export RZERO_RM_BASE_PORT=${RZERO_RM_BASE_PORT:-19000}
export RZERO_RM_NUM_SERVERS_SOL=${RZERO_RM_NUM_SERVERS_SOL:-2}
export RZERO_RM_NUM_SERVERS_FILT=${RZERO_RM_NUM_SERVERS_FILT:-4}
# Filter band OFF by default (= the released J-Zero run): avg_rm is a mean of
# sigmoids, so [0.0, 1.0] passes every question and selection is the std top-N
# alone. Export a narrower LO/HI to re-enable the band.
# DUPLICATED DEFAULTS — same pair in question_filter/run_parallel_rm.sh and
# question_filter/filter_rm.py's --min_avg/--max_avg.
export RZERO_RM_FILTER_LO=${RZERO_RM_FILTER_LO:-0.0}
export RZERO_RM_FILTER_HI=${RZERO_RM_FILTER_HI:-1.0}
export RZERO_RM_FILTER_TOPN=${RZERO_RM_FILTER_TOPN:-1920}

# Variant knobs (defaults = released Ours behavior).
#   RZERO_SOL_REWARD_PATH      : custom reward file (default solver_reward_rm.py)
#   RZERO_SOL_FAIL_EXTRA_KEYS  -> rzero_batch_manager failure-dict key set
#   (RZERO_FILTER_STD_MODE reaches filter_rm.py via the environment.)
SOL_REWARD_PATH=${RZERO_SOL_REWARD_PATH:-}
export RZERO_FAIL_EXTRA_KEYS="${RZERO_SOL_FAIL_EXTRA_KEYS:-}"
# Reward-chunk overrides; unset falls through to the global values.
# Group-relative reward variants need the WHOLE step in one chunk: set
# RZERO_SOL_CHUNK_MAX/EXPECT to train_batch*rollout_n and raise
# RZERO_SOL_MAX_WAIT_MS so the leader can wait for all registrations.
export RZERO_CHUNK_MAX="${RZERO_SOL_CHUNK_MAX:-${RZERO_CHUNK_MAX:-64}}"
export RZERO_CHUNK_EXPECT="${RZERO_SOL_CHUNK_EXPECT:-${RZERO_CHUNK_EXPECT:-0}}"
export RZERO_MAX_WAIT_MS="${RZERO_SOL_MAX_WAIT_MS:-${RZERO_MAX_WAIT_MS:-30000}}"

bash "$SCRIPT_DIR/cleanup_between_phases.sh" "solver-entry" || true
solver_cleanup() {
    bash "$SCRIPT_DIR/rm_service/stop.sh" "$LOG_DIR" \
        "$RZERO_RM_BASE_PORT" "$RZERO_RM_NUM_SERVERS_SOL" "sol" || true
    bash "$SCRIPT_DIR/cleanup_between_phases.sh" "post-solver" || true
}
trap solver_cleanup EXIT

PER_GPU=$((NUM_Q_TOTAL / 4))
echo "[solver_train.sh] generating $NUM_Q_TOTAL questions ($PER_GPU per GPU)"

# --- 1. Generate questions with challenger_t (4-GPU vLLM on 0-3) ----------
bash "$SCRIPT_DIR/question_gen/run_parallel.sh" \
    "$CHALLENGER_MODEL" "$EXP_NAME" "$GEN_DIR" "$PER_GPU" \
    2>&1 | tee "$LOG_DIR/gen.log"

# --- 2. RM-based filter (gen answers + RM score + parquet) ----------------
TRAIN_PARQUET="$PARQUET_DIR/${EXP_NAME}.parquet"
bash "$SCRIPT_DIR/question_filter/run_parallel_rm.sh" \
    "$SOLVER_INIT" "$RM_MODEL" "$EXP_NAME" "$GEN_DIR" "$FILT_DIR" \
    "$TRAIN_PARQUET" 10 \
    2>&1 | tee "$LOG_DIR/filter_rm.log"

if [[ ! -s "$TRAIN_PARQUET" ]]; then
    echo "[solver_train.sh] ERROR: empty train parquet" >&2
    exit 1
fi

ROW_COUNT=$(python -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_PARQUET')))")
echo "[solver_train.sh] train parquet rows: $ROW_COUNT"
if [[ "$ROW_COUNT" -lt "$MIN_ROWS_REQUIRED" ]]; then
    echo "[solver_train.sh] ERROR: too few rows ($ROW_COUNT < $MIN_ROWS_REQUIRED)" >&2
    exit 1
fi

# --- Adaptive train_batch safety net ---------------------------------------
# Fewer survivors than train_batch makes verl hard-crash ("Train dataloader is
# empty!", drop_last=True). Only fires in that collapse regime — normal runs
# keep TRAIN_BATCH=128, SMOKE keeps its own tiny batch — so a degenerate iter
# still yields a measurable point.
if [[ "${SMOKE:-0}" != "1" && "$ROW_COUNT" -lt "$TRAIN_BATCH" ]]; then
    echo "[solver_train.sh] WARN: survivors ($ROW_COUNT) < train_batch ($TRAIN_BATCH); adaptive fallback to avoid empty-dataloader crash"
    if   [[ "$ROW_COUNT" -ge 64 ]]; then TRAIN_BATCH=64;  PPO_MINI_BATCH=32
    elif [[ "$ROW_COUNT" -ge 32 ]]; then TRAIN_BATCH=32;  PPO_MINI_BATCH=16
    else                                  TRAIN_BATCH=16;  PPO_MINI_BATCH=8
    fi
    echo "[solver_train.sh] adaptive: TRAIN_BATCH=$TRAIN_BATCH PPO_MINI_BATCH=$PPO_MINI_BATCH"
fi

# Cleanup before verl trainer + RM-server bring-up.
bash "$SCRIPT_DIR/cleanup_between_phases.sh" "pre-solver-trainer" || true

# --- 3a. Launch 2 RM HTTP servers on GPU 2,3 ------------------------------
bash "$SCRIPT_DIR/rm_service/start.sh" "$RM_MODEL" "$LOG_DIR" "2,3" \
    "$RZERO_RM_BASE_PORT" "sol"

# --- 3b. GRPO solver trainer (FSDP on GPU 0,1) ----------------------------
TRAIN_GPUS="0,1"
NUM_GPUS=2
NUM_NODES=1

CKPT_DIR="$CKPT_ROOT/$EXP_NAME"
mkdir -p "$CKPT_DIR"

# Eval val files (in-training val disabled by default; kept for symmetry).
DATA_DIR="${PROJECT_ROOT:?export PROJECT_ROOT=... (directory holding .env, data/ with the val parquets, and outputs/)}/data"
VAL_FILES="['$DATA_DIR/math500.parquet','$DATA_DIR/minervamath.parquet','$DATA_DIR/olympiadbench.parquet']"

PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
    python3 -u -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2560 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=RZeroBatchManager \
    reward.reward_manager.module.path="file://$SCRIPT_DIR/reward/rzero_batch_manager.py" \
    reward.reward_manager.module.name=RZeroBatchManager \
    reward.num_workers=1 \
    reward.custom_reward_function.path="${SOL_REWARD_PATH:-$SCRIPT_DIR/reward/solver_reward_rm.py}" \
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
    actor_rollout_ref.model.use_remove_padding=${JZERO_USE_RMPAD:-True} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${JZERO_ULYSSES_SP:-2} \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${JZERO_ACTOR_PARAM_OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${JZERO_ACTOR_OPT_OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl+ \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=${JZERO_REF_PARAM_OFFLOAD:-False} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=${JZERO_FREE_CACHE:-False} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${JZERO_SOLVER_MEM_UTIL:-0.55} \
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
    trainer.project_name="${WANDB_PROJECT:-J-Zero}" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=$MAX_STEPS \
    trainer.rollout_data_dir="$LOG_DIR/rollouts_solver" \
    ${RZERO_EXTRA_HYDRA:-} \
    2> >(tee "$LOG_DIR/trainer.stderr" >&2) \
    | tee "$LOG_DIR/${EXP_NAME}.log" || true
PYTHON_RC=${PIPESTATUS[0]}

# --- 3c. Wandb flush safety net (post-trainer, pre-cleanup) ----------------
{ set +e; } 2>/dev/null
# See challenger_train.sh: wandb writes to $WANDB_DIR when set, else ./wandb
# (= $SCRIPT_DIR here), and wandb 0.27.0 nests offline runs one level down in
# $WANDB_DIR/wandb/offline-run-*, so syncing $WANDB_DIR itself finds nothing.
WANDB_RUN_DIR="${WANDB_DIR:+$WANDB_DIR/wandb}"
WANDB_RUN_DIR="${WANDB_RUN_DIR:-$SCRIPT_DIR/wandb}"
if ! command -v wandb >/dev/null 2>&1; then
    echo "[solver_train.sh] wandb-flush: SKIPPED, no wandb CLI on PATH (offline runs will not be synced)"
elif [[ ! -d "$WANDB_RUN_DIR" ]]; then
    echo "[solver_train.sh] wandb-flush: WARN no wandb run dir at $WANDB_RUN_DIR — nothing synced. If wandb writes elsewhere, export WANDB_DIR to point here."
else
    echo "[solver_train.sh] wandb-flush: 'wandb sync --include-offline --sync-all' on $WANDB_RUN_DIR (rc_trainer=$PYTHON_RC)"
    timeout 120 wandb sync --include-offline --sync-all "$WANDB_RUN_DIR" 2>&1 | tail -10 || true
    sleep 5
fi
{ set -e; } 2>/dev/null

# --- 4. FSDP -> HF ---------------------------------------------------------
LATEST=$(ls -1 "$CKPT_DIR" 2>/dev/null | grep '^global_step_[0-9]\+$' | sort -t_ -k3 -n | tail -1)
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
