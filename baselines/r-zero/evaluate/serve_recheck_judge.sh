#!/bin/bash
# Serve the per-iter MATH-recheck judge for run_solver_eval.sh as a LOCAL
# OpenAI-compatible vLLM endpoint on ONE GPU (tp=1) — no external API.
#
# Required env: JUDGE_VENV_PY (a dedicated vllm >= 0.23 venv, separate from the
# training env) and NOTHINK_JINJA (chat template forcing enable_thinking=False).
#
# Usage (caller backgrounds this, polls http://127.0.0.1:$PORT/health, then kills):
#   GPU=0 PORT=8400 bash serve_recheck_judge.sh [model] [served_name]
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
VENV_PY="${JUDGE_VENV_PY:?export JUDGE_VENV_PY=/path/to/judge venv python (vllm >= 0.23)}"
NOTHINK_JINJA="${NOTHINK_JINJA:?export NOTHINK_JINJA=/path/to/qwen36 no-think chat template (.jinja)}"

# Cache dirs + /tmp-noexec workarounds.
source "$SCRIPT_DIR/../env_prelude.sh" 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MODEL=${1:-Qwen/Qwen3.6-27B}
SERVED=${2:-qwen36-judge}
GPU=${GPU:?"GPU index required"}
PORT=${PORT:?"PORT required"}
MEM_FRACTION=${MEM_FRACTION:-0.85}
# Must fit prompt (question + full solution, up to ~5k tok) + max_tokens (32768).
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}

if [[ ! -f "$NOTHINK_JINJA" ]]; then
    echo "[serve_recheck_judge] FATAL: nothink chat template not found: $NOTHINK_JINJA" >&2
    exit 1
fi

echo "[serve_recheck_judge] model=$MODEL served=$SERVED GPU=$GPU port=$PORT max_model_len=$MAX_MODEL_LEN"
export CUDA_VISIBLE_DEVICES="$GPU"
# Keyed off the GPU index so a peer server on another GPU cannot collide.
export VLLM_PORT=$((38000 + GPU))
exec "$VENV_PY" -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$MEM_FRACTION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "${RECHECK_JUDGE_MAX_SEQS:-16}" \
    --dtype bfloat16 \
    --enforce-eager \
    --trust-remote-code \
    --chat-template "$NOTHINK_JINJA" \
    --limit-mm-per-prompt '{"image":0,"video":0}'
