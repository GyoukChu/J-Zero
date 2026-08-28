#!/bin/bash
# Common environment overrides for all R-Zero recipe scripts.
#
# Sourced (not executed) by run_rzero.sh, challenger_train.sh, solver_train.sh,
# vllm_service/start.sh, evaluate/run_solver_eval.sh and
# evaluate/serve_recheck_judge.sh. question_gen/run_parallel.sh and
# question_filter/run_parallel.sh inherit these as children of solver_train.sh;
# run standalone they lose the cache redirect and the offline flags.
#
# The redirect matters because on hosts with /tmp mounted noexec, triton writes
# .so files under /tmp/torchinductor_$USER/ and dlopen()s them, which fails with
# "failed to map segment from shared object" at vLLM warmup.

# ${VAR:-default} lets a sibling run on disjoint GPUs point these at per-run
# paths; concurrent writes to a shared inductor/triton cache can kill vLLM
# EngineCore workers.
RZERO_CACHE_ROOT="${RZERO_CACHE_ROOT:-$HOME/.cache/rzero}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RZERO_CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RZERO_CACHE_ROOT/torchinductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RZERO_CACHE_ROOT/vllm_compile}"
export TMPDIR="${TMPDIR:-$RZERO_CACHE_ROOT/tmp}"
# triton also stages the compiled .so through TMPDIR before moving it.
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-$RZERO_CACHE_ROOT/torchinductor}"

# Ray's AF_UNIX socket paths (<tmpdir>/ray/session_<long>/sockets/...) can exceed
# Linux's 107-byte limit under a long $HOME, so keep them on a short /tmp path —
# no .so execution happens there, so noexec is fine. A caller-set RAY_TMPDIR is
# respected: cleanup_between_phases.sh scopes its ray kill to this path, so
# concurrent runs need distinct values to avoid killing each other's daemons.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_rzero}"
mkdir -p "$RAY_TMPDIR"

mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TMPDIR"

# vLLM compile / kernel knobs
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_SKIP_DEEP_GEMM_WARMUP=${VLLM_SKIP_DEEP_GEMM_WARMUP:-1}

# Attention backend — must match pipeline/env_prelude.sh (J-Zero) and
# g_zero/local_backend.py so all three pipelines run the same kernels. Both
# reasons are load-bearing:
#  1. Without a CUDA toolkit (no nvcc), vLLM's FlashInfer paths JIT-compile via
#     nvcc and crash with "FileNotFoundError: 'nvcc'".
#  2. On SM100 (B200-class), an unpinned backend auto-selects FlashInfer, which
#     auto-enables TRTLLM-gen attention; a colocated rollout EngineCore can then
#     futex-deadlock on idle/resume. (vLLM 0.11's own switch is
#     VLLM_USE_TRTLLM_ATTENTION=0 — VLLM_DISABLE_TRTLLM_ATTN is read only by
#     patches/vllm023-disable-trtllm-attn.patch.)
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_USE_FLASHINFER=${VLLM_USE_FLASHINFER:-0}
export VLLM_DISABLE_FLASHINFER_PREFILL=${VLLM_DISABLE_FLASHINFER_PREFILL:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

# Reward HTTP timeout for challenger_reward.py's per-step solver query: GBS(16)
# * rollout.n(4) * solver_m(10) = 640 generations over 2 Flask servers. The 2h
# ceiling only has to exceed the slowest step, not match it.
export RZERO_HTTP_TIMEOUT=${RZERO_HTTP_TIMEOUT:-7200}

# Force fully offline tokenizer/model loads by default: transformers >= 5 calls
# list_repo_templates() inside AutoTokenizer.from_pretrained, which hits the Hub
# even when every file is cached, so a stale token turns into a hard fail at
# tokenizer load. TRANSFORMERS_OFFLINE covers older code paths. Caller-set values
# win (export HF_HUB_OFFLINE=0 to go online).
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# Bound wandb's end-of-stage flush: without it an unreachable backend hangs every
# stage at teardown with the GPUs idle. Unsent data stays on disk for the
# `wandb sync` the stage scripts already run. The verl patch adds a coarser outer
# bound (JZERO_WANDB_SHUTDOWN_TIMEOUT, default 120s).
export WANDB_FINISH_TIMEOUT=${WANDB_FINISH_TIMEOUT:-120}
