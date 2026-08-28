#!/bin/bash
# Serve a judge model via vLLM 0.23 (JUDGE_VENV, built from
# requirements/requirements-eval.txt) as ONE OpenAI-compatible endpoint,
# data-parallel across DP GPUs. All judge serving is unified on vLLM (sglang
# --dp-size dies at init on Qwen3.6-27B's hybrid mamba arch). NO external API.
#
# Usage: GPUS=4,5,6 PORT=8430 DP=3 [CHAT_TEMPLATE=...] bash serve_judge_vllm.sh <model> <served_name>
set -euo pipefail
RJ="${JUDGE_VENV:?export JUDGE_VENV=/path/to/venv built from requirements-eval.txt}/bin/python"
# HF_HUB_OFFLINE is pinned to 1 here, not defaulted, while the rest of the
# suite runs online (run_eval.sh) for the EM datasets. Judge/test weights are
# always pre-cached (benchmarks/README.md one-time setup step 3), so serving
# has nothing to download and a stale HF token cannot turn a Hub round-trip
# during weight load into a hard 401 hours into a run.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Disable Blackwell TRTLLM-gen attention — engines can wedge resuming from idle
# with it on. Needs patches/vllm023-disable-trtllm-attn.patch in the serving
# venv; stock vLLM 0.23 ignores the flag, so it is harmless if unpatched.
export VLLM_DISABLE_TRTLLM_ATTN=1

MODEL=${1:?"model"}
SERVED=${2:?"served_name"}
GPUS=${GPUS:?"GPUS"}
PORT=${PORT:?"PORT"}
DP=${DP:-1}
MEM=${MEM:-0.85}
MAXLEN=${MAXLEN:-16384}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-}
EXTRA=()
[[ -n "$CHAT_TEMPLATE" ]] && EXTRA+=(--chat-template "$CHAT_TEMPLATE")
# Unique DP RPC port per server so two concurrent vLLM-DP servers don't collide.
DPPORT=$(( 29500 + PORT % 1000 ))
# Unique internal-port BASE per server: vLLM's get_open_port() (engine
# distributed init / TCPStore) is host-wide and NOT isolated by
# CUDA_VISIBLE_DEVICES, so two judges started close in time can race-pick the
# same port and fail engine init with EADDRINUSE. Lane A (PORT_BASE=7400) lands
# on 404xx, lane B (9500) on 405xx — disjoint from DPPORT (299xx/300xx), the
# API ports (74xx/95xx), and the recheck-judge VLLM_PORT band (380xx).
export VLLM_PORT=$(( 40000 + PORT % 1000 ))

# Per-instance compile caches OUTSIDE /tmp: with a noexec /tmp the default
# /tmp/torchinductor_* cache makes the first request-time triton JIT fail at
# dlopen and the engine dies (every request 500s). Port-keyed so concurrent
# judges don't race one shared cache.
CACHE_ROOT="${JUDGE_CACHE_ROOT:-$HOME/.cache/judge_vllm}/$PORT"
mkdir -p "$CACHE_ROOT/triton" "$CACHE_ROOT/inductor" "$CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$CACHE_ROOT/inductor"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"

echo "[serve_judge_vllm] model=$MODEL served=$SERVED GPUS=$GPUS dp=$DP port=$PORT maxlen=$MAXLEN dpport=$DPPORT vllm_port=$VLLM_PORT"
export CUDA_VISIBLE_DEVICES="$GPUS"
exec "$RJ" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$SERVED" \
  --host 127.0.0.1 --port "$PORT" \
  --data-parallel-size "$DP" --tensor-parallel-size 1 \
  --data-parallel-rpc-port "$DPPORT" \
  --gpu-memory-utilization "$MEM" --max-model-len "$MAXLEN" \
  --dtype bfloat16 --enforce-eager --trust-remote-code \
  "${EXTRA[@]}" \
  --limit-mm-per-prompt '{"image":0,"video":0}'