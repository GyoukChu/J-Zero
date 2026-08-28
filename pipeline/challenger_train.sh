#!/bin/bash
# J-Zero challenger stage (Stage 1).
#   GPUs 0,1 -> FSDP2 trainer (verl GRPO over batch=16 × n=4 = 64 problems).
#   GPU 2    -> answers-only vLLM solver server (returns n=10 raw answers/Q).
#   GPU 3    -> 1 Skywork RM HTTP server (scores all 640 (Q,A) pairs).
#
# Prompts: data/build_dummy.py (general-domain CHALLENGER_INSTRUCTION).
# Reward:  reward/challenger_reward_rm.py.
#
# Usage: bash challenger_train.sh <solver_model> <challenger_init> <rm_model> \
#                                        <exp_name> <log_dir> <ckpt_root> \
#                                        [max_steps] [save_freq]

set -euo pipefail
set -x

# shellcheck disable=SC1090
source "${RZERO_VENV_ACTIVATE:?export RZERO_VENV_ACTIVATE=/path/to/venv/bin/activate (env from requirements/requirements-train-vllm011.txt + verl installed editable)}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/env_prelude.sh"

SOLVER_MODEL=${1:?"solver_model required"}
INIT_MODEL=${2:?"challenger_init required"}
RM_MODEL=${3:?"rm_model required"}
EXP_NAME=${4:?"exp_name required"}
LOG_DIR=${5:?"log_dir required"}
CKPT_ROOT=${6:?"ckpt_root required"}
MAX_STEPS=${7:-5}
SAVE_FREQ=${8:-5}

# --- Secrets check ----------------------------------------------------------
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

# Single answers-only vLLM on GPU 2 (port = RZERO_BASE_PORT).
export RZERO_NUM_SERVERS=1
export RZERO_BASE_PORT=${RZERO_BASE_PORT:-18000}

# Single RM HTTP server on GPU 3 (port = RZERO_RM_BASE_PORT).
export RZERO_RM_NUM_SERVERS_CHAL=${RZERO_RM_NUM_SERVERS_CHAL:-1}
export RZERO_RM_BASE_PORT=${RZERO_RM_BASE_PORT:-19000}

# Variant knobs (defaults = released Ours behavior).
#   RZERO_CHAL_REWARD_PATH   : custom reward file (default challenger_reward_rm.py)
#   RZERO_CHAL_EXTRA_HYDRA   : challenger-only hydra overrides
#   RZERO_CHAL_FAIL_EXTRA_KEYS -> rzero_batch_manager failure-dict key set
CHAL_REWARD_PATH=${RZERO_CHAL_REWARD_PATH:-}
export RZERO_FAIL_EXTRA_KEYS="${RZERO_CHAL_FAIL_EXTRA_KEYS:-}"
# Reward-chunk overrides; unset falls through to the global values.
export RZERO_CHUNK_MAX="${RZERO_CHAL_CHUNK_MAX:-${RZERO_CHUNK_MAX:-64}}"
export RZERO_CHUNK_EXPECT="${RZERO_CHAL_CHUNK_EXPECT:-${RZERO_CHUNK_EXPECT:-0}}"
export RZERO_MAX_WAIT_MS="${RZERO_CHAL_MAX_WAIT_MS:-${RZERO_MAX_WAIT_MS:-30000}}"

# BLEU cluster + reward dispatch knobs.
export RZERO_BLEU_TAU=0.5
export RZERO_TEMP_DIR="$LOG_DIR/_reward_tmp"
mkdir -p "$LOG_DIR" "$CKPT_ROOT" "$RZERO_TEMP_DIR"

JZERO_CHALLENGER_MEM_UTIL=${JZERO_CHALLENGER_MEM_UTIL:-0.7}
RZERO_EXTRA_HYDRA=${RZERO_EXTRA_HYDRA:-}

# --- 1. Launch answers-only solver vLLM on GPU 2 ---------------------------
bash "$SCRIPT_DIR/vllm_service/start_answers.sh" \
    "$SOLVER_MODEL" "$LOG_DIR" 2 "$RZERO_BASE_PORT"

# --- 2. Launch 1 RM HTTP server on GPU 3 -----------------------------------
bash "$SCRIPT_DIR/rm_service/start.sh" "$RM_MODEL" "$LOG_DIR" "3" \
    "$RZERO_RM_BASE_PORT" "chal"

cleanup() {
    bash "$SCRIPT_DIR/rm_service/stop.sh" "$LOG_DIR" \
        "$RZERO_RM_BASE_PORT" 1 "chal" || true
    bash "$SCRIPT_DIR/vllm_service/stop.sh" "$LOG_DIR" || true
    # Never an unscoped `pkill -f` here — the host may run other users' jobs.
    # cleanup_between_phases.sh honours RZERO_GPUS=0,1,2,3 and catches whatever
    # vllm_service/stop.sh missed.
    bash "$SCRIPT_DIR/cleanup_between_phases.sh" "post-challenger" || true
}
trap cleanup EXIT

# --- 3. Build dummy training parquet (J-Zero CHALLENGER_INSTRUCTION) -------
DUMMY_PARQUET="$SCRIPT_DIR/data/dummy_train.parquet"
rm -f "$DUMMY_PARQUET"
python "$SCRIPT_DIR/data/build_dummy.py" --out_path "$DUMMY_PARQUET" --n 2048

# --- 4. GRPO trainer (FSDP on GPU 0,1) -------------------------------------
TRAIN_GPUS="0,1"
NUM_GPUS=2
NUM_NODES=1

CKPT_DIR="$CKPT_ROOT/$EXP_NAME"
mkdir -p "$CKPT_DIR"

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
    reward.custom_reward_function.path="${CHAL_REWARD_PATH:-$SCRIPT_DIR/reward/challenger_reward_rm.py}" \
    reward.custom_reward_function.name=compute_score \
    data.shuffle=False \
    data.train_files="$DUMMY_PARQUET" \
    data.val_files="[$DUMMY_PARQUET]" \
    data.train_batch_size=${RZERO_CHAL_TRAIN_BSZ:-16} \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$INIT_MODEL" \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=$JZERO_CHALLENGER_MEM_UTIL \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${WANDB_PROJECT:-J-Zero}" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=$MAX_STEPS \
    trainer.rollout_data_dir="$LOG_DIR/rollouts_challenger" \
    $RZERO_EXTRA_HYDRA ${RZERO_CHAL_EXTRA_HYDRA:-} \
    2> >(tee "$LOG_DIR/trainer.stderr" >&2) \
    | tee "$LOG_DIR/${EXP_NAME}.log" || true
PYTHON_RC=${PIPESTATUS[0]}

# --- 4b. Wandb flush safety net (post-trainer, pre-cleanup) ----------------
{ set +e; } 2>/dev/null
# wandb writes to $WANDB_DIR when set, else ./wandb — and every stage runs from
# $SCRIPT_DIR, hence the default. Hardcoding the path instead silently skips the
# sync for anyone who moved WANDB_DIR.
# wandb 0.27.0 nests runs one level down: with WANDB_DIR set, offline runs land
# in $WANDB_DIR/wandb/offline-run-*, so syncing $WANDB_DIR itself finds nothing.
WANDB_RUN_DIR="${WANDB_DIR:+$WANDB_DIR/wandb}"
WANDB_RUN_DIR="${WANDB_RUN_DIR:-$SCRIPT_DIR/wandb}"
if ! command -v wandb >/dev/null 2>&1; then
    echo "[challenger_train.sh] wandb-flush: SKIPPED, no wandb CLI on PATH (offline runs will not be synced)"
elif [[ ! -d "$WANDB_RUN_DIR" ]]; then
    echo "[challenger_train.sh] wandb-flush: WARN no wandb run dir at $WANDB_RUN_DIR — nothing synced. If wandb writes elsewhere, export WANDB_DIR to point here."
else
    echo "[challenger_train.sh] wandb-flush: 'wandb sync --include-offline --sync-all' on $WANDB_RUN_DIR (rc_trainer=$PYTHON_RC)"
    timeout 120 wandb sync --include-offline --sync-all "$WANDB_RUN_DIR" 2>&1 | tail -10 || true
    sleep 5
fi
{ set -e; } 2>/dev/null

# --- 5. Convert FSDP ckpt -> HF --------------------------------------------
LATEST=$(ls -1 "$CKPT_DIR" 2>/dev/null | grep '^global_step_[0-9]\+$' | sort -t_ -k3 -n | tail -1)
if [[ -z "$LATEST" ]]; then
    echo "[challenger_train.sh] ERROR: no checkpoint in $CKPT_DIR" >&2
    exit 1
fi
CKPT_ACTOR="$CKPT_DIR/$LATEST/actor"
HF_DIR="$CKPT_DIR/$LATEST/actor/huggingface"
if [[ -d "$CKPT_ACTOR" ]]; then
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$CKPT_ACTOR" \
        --target_dir "$HF_DIR"
    echo "[challenger_train.sh] HF weights written to $HF_DIR"
else
    echo "[challenger_train.sh] WARNING: no actor dir at $CKPT_ACTOR" >&2
fi
