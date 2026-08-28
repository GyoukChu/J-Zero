#!/usr/bin/env bash
# G-Zero paper run: Llama-3.2-3B-Instruct (tag=lora_llama3b_paper), rounds 1-10.
# Every hyperparameter comes from the paper-faithful config.py defaults; the
# llama3 renderer picks up Llama-3.2-Instruct's chat template and <|eot_id|> eos.
# Resumable per round via runs/lora_llama3b_paper/resume_state.json — rerun to
# continue after a crash. SMOKE=1 runs a tiny pre-flight instead.
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Reduce CUDA allocator fragmentation (fine here: plain vLLM, no colocated rollout).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# With the model pre-downloaded, offline mode avoids HF-token-expiry 401 mid-run.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
# If the host has no C compiler, force torch.compile to eager.
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}

# Python with torch+vllm+peft (requirements/requirements-train-vllm011.txt or
# scripts/recreate_verl_env.sh); override G_ZERO_PY to point at your env.
PY="${G_ZERO_PY:-python}"

# SMOKE=1 -> tiny end-to-end pre-flight: 100 questions, 1 Challenger GRPO step,
# 1 DPO step, 1 round, short generations, under a separate _smoke tag so it never
# resumes from the paper run's resume_state.json. Knobs override via G_ZERO_SMOKE_*.
if [[ "${SMOKE:-0}" == "1" ]]; then
  TAG="${G_ZERO_SMOKE_TAG:-lora_llama3b_smoke}"
  NUM_ROUNDS="${G_ZERO_SMOKE_NROUNDS:-1}"
  EXTRA_ARGS=(
    --num_questions         "${G_ZERO_SMOKE_NUM_Q:-100}"
    --challenger_steps      "${G_ZERO_SMOKE_CHAL_STEPS:-1}"
    --challenger_batch_size "${G_ZERO_SMOKE_CHAL_BATCH:-8}"
    --challenger_group_size "${G_ZERO_SMOKE_CHAL_GROUP:-4}"
    --phase1_max_tokens     "${G_ZERO_SMOKE_PHASE1_MAX_TOKENS:-512}"
    --challenger_max_tokens "${G_ZERO_SMOKE_CHAL_MAX_TOKENS:-512}"
    --solver_max_tokens     "${G_ZERO_SMOKE_SOLVER_MAX_TOKENS:-1024}"
    --dpo_max_steps         "${G_ZERO_SMOKE_DPO_STEPS:-1}"
    --vllm_max_model_len    "${G_ZERO_SMOKE_MAX_MODEL_LEN:-4096}"
  )
else
  TAG=lora_llama3b_paper
  NUM_ROUNDS="${G_ZERO_NROUNDS:-10}"
  EXTRA_ARGS=()
fi
LOG="run_${TAG}.log"

"$PY" -m g_zero.multi_round \
  --num_rounds "$NUM_ROUNDS" --run_phase1 true \
  --tag "$TAG" --model_name meta-llama/Llama-3.2-3B-Instruct --renderer_name llama3 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  >> "$LOG" 2>&1
