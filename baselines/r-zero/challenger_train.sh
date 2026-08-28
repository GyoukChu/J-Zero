#!/bin/bash
# Train the challenger with verl GRPO.
#   GPUs 0,1: FSDP trainer (rollout n=4, per R-Zero questioner_train_penalty.sh).
#   GPUs 2,3: Flask vLLM servers wrapping the *current* solver, queried per batch
#             by reward/challenger_reward.py for each problem's p_hat.
#
# Usage: bash challenger_train.sh <solver_model_path> <challenger_init_model_path> <exp_name> <log_dir> <ckpt_root> [max_steps] [save_freq]

set -euo pipefail
set -x

RZERO_VENV="${RZERO_VENV:?export RZERO_VENV=/path/to/venv built from requirements/requirements-train-vllm011.txt}"
source "$RZERO_VENV/bin/activate"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/env_prelude.sh"

SOLVER_MODEL=${1:?"solver_model_path required"}
INIT_MODEL=${2:?"challenger_init_model_path required"}
EXP_NAME=${3:?"exp_name required"}
LOG_DIR=${4:?"log_dir required"}
CKPT_ROOT=${5:?"ckpt_root required"}
MAX_STEPS=${6:-5}
SAVE_FREQ=${7:-5}

VERL_REPO=$( cd "$SCRIPT_DIR/.." && pwd )

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
export RZERO_NUM_SERVERS=${RZERO_NUM_SERVERS:-2}
export RZERO_BASE_PORT=${RZERO_BASE_PORT:-18000}
export RZERO_BLEU_TAU=0.5
export RZERO_TEMP_DIR="$LOG_DIR/_reward_tmp"
# Debug overrides: RZERO_CHALLENGER_MEM_UTIL sets rollout.gpu_memory_utilization;
# RZERO_EXTRA_HYDRA is newline-separated extra hydra overrides (one per line)
# appended to the trainer command.
RZERO_CHALLENGER_MEM_UTIL=${RZERO_CHALLENGER_MEM_UTIL:-0.7}
RZERO_EXTRA_HYDRA=${RZERO_EXTRA_HYDRA:-}
mkdir -p "$LOG_DIR" "$CKPT_ROOT" "$RZERO_TEMP_DIR"

# --- 1. Launch 2 vLLM reward servers on GPU 2, 3 ---------------------------
bash "$SCRIPT_DIR/vllm_service/start.sh" "$SOLVER_MODEL" "$LOG_DIR"
cleanup() {
    bash "$SCRIPT_DIR/vllm_service/stop.sh" "$LOG_DIR" || true
    # Required so solver_train.sh can ray.init() a fresh cluster: stale ray head
    # daemons hold the TCPStore ports the next phase needs.
    bash "$SCRIPT_DIR/cleanup_between_phases.sh" "post-challenger" || true
}
trap cleanup EXIT

# --- 2. Build dummy training parquet ----------------------------------------
# Force-rebuild: a stale parquet from an older recipe version would silently
# re-use its schema / row count.
DUMMY_PARQUET="$SCRIPT_DIR/data/dummy_train.parquet"
rm -f "$DUMMY_PARQUET"
python "$SCRIPT_DIR/data/build_dummy.py" --out_path "$DUMMY_PARQUET" --n 2048

# --- 3. GRPO trainer (FSDP) ------------------------------------------------
TRAIN_GPUS=${RZERO_CHAL_TRAIN_GPUS:-"0,1"}
NUM_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)
NUM_NODES=1

CKPT_DIR="$CKPT_ROOT/$EXP_NAME"
mkdir -p "$CKPT_DIR"

PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python3 -u -m verl.trainer.main_ppo \
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
    reward.custom_reward_function.path="$SCRIPT_DIR/reward/challenger_reward.py" \
    reward.custom_reward_function.name=compute_score \
    data.shuffle=False \
    data.train_files="$DUMMY_PARQUET" \
    data.val_files="[$DUMMY_PARQUET]" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$INIT_MODEL" \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=${RZERO_REF_PARAM_OFFLOAD:-False} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=${RZERO_FREE_CACHE:-False} \
    actor_rollout_ref.rollout.gpu_memory_utilization=$RZERO_CHALLENGER_MEM_UTIL \
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
    trainer.project_name="R-Zero-Repro" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=$MAX_STEPS \
    trainer.rollout_data_dir="$LOG_DIR/rollouts_challenger" \
    $RZERO_EXTRA_HYDRA \
    2> >(tee "$LOG_DIR/trainer.stderr" >&2) \
    | tee "$LOG_DIR/${EXP_NAME}.log"

# --- 4. Convert FSDP ckpt -> HF ---------------------------------------------
LATEST=$(ls -1 "$CKPT_DIR" 2>/dev/null | grep '^global_step_[0-9]\+$' | sort -t_ -k3 -n | tail -1 || true)
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
