#!/bin/bash
# Common environment overrides (plus the shared rzero_rm_steps derivation at
# the bottom) for the J-Zero pipeline — sourced, not executed, by every stage.
#
# Hosts that mount /tmp noexec break triton, which writes .so files into /tmp
# and dlopen()s them ("failed to map segment from shared object" at vLLM
# warmup). All such caches go to an exec-allowed dir instead.

# noexec-free caches
RZERO_CACHE_ROOT="${RZERO_CACHE_ROOT:-$HOME/.cache/rzero}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RZERO_CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RZERO_CACHE_ROOT/torchinductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RZERO_CACHE_ROOT/vllm_compile}"
export TMPDIR="${TMPDIR:-$RZERO_CACHE_ROOT/tmp}"
# triton also stages the compiled .so in TMPDIR before moving it.
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-$RZERO_CACHE_ROOT/torchinductor}"

# Ray's AF_UNIX socket paths (<tmpdir>/ray/session_<long>/sockets/...) exceed
# Linux's 107-byte limit when TMPDIR is a long $HOME path, so ray's runtime
# files get a short /tmp path (nothing is executed there, so noexec is fine).
# A caller-set RAY_TMPDIR is respected: concurrent runs on disjoint GPU sets
# need cleanup's `pkill -f "$RAY_TMPDIR"` scoped to their OWN ray daemons.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_rzero}"
mkdir -p "$RAY_TMPDIR"

mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TMPDIR"

# vLLM compile / kernel knobs
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_SKIP_DEEP_GEMM_WARMUP=${VLLM_SKIP_DEEP_GEMM_WARMUP:-1}

# On hosts without a CUDA toolkit (no nvcc), vLLM's FlashInfer paths JIT-compile
# via nvcc and crash with "FileNotFoundError: 'nvcc'". Default to the native
# PyTorch sampler + FlashAttention backend; RZERO_CC/RZERO_CXX can point any
# remaining triton JIT at a caller-supplied compiler.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_USE_FLASHINFER=${VLLM_USE_FLASHINFER:-0}
export VLLM_DISABLE_FLASHINFER_PREFILL=${VLLM_DISABLE_FLASHINFER_PREFILL:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
if [[ -n "${RZERO_CC:-}" && -x "${RZERO_CC}" ]]; then
    export CC="${CC:-$RZERO_CC}"
    export TRITON_CC="${TRITON_CC:-$RZERO_CC}"
    if [[ -n "${RZERO_CXX:-}" ]]; then
        export CXX="${CXX:-$RZERO_CXX}"
    fi
fi

# Reward HTTP timeout for challenger_reward_rm.py's per-step solver query:
# GBS(16) * rollout.n(4) * solver_m(10) = 640 generations per step, all on the
# single answers-only vLLM server — hence the 2h ceiling.
export RZERO_HTTP_TIMEOUT=${RZERO_HTTP_TIMEOUT:-7200}

# Offline tokenizer/model loads by default: transformers >=5.x calls
# list_repo_templates() inside AutoTokenizer.from_pretrained, which hits the Hub
# even for fully cached files and hard-fails on a 401 (e.g. expired HF token).
# TRANSFORMERS_OFFLINE is the legacy flag, kept for older code paths.
# Set HF_HUB_OFFLINE=0 to go back online and pull new models.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# ---------------------------------------------------------------------------
# Stage-3 (RM) step count — the ONE derivation, called by both scripts that
# must agree on it:
#   rm_train.sh  saves to  <ckpt>/<exp>/global_step_$(rzero_rm_steps)/actor/...
#   run_jzero.sh looks for exactly that directory to decide "RM already done".
# Deriving it in two places let a finished RM stage be silently retrained on
# resume. SMOKE=1 scales the pair budget down; caller-set values win in both
# modes and RZERO_RM_STEPS overrides the derivation entirely.
# DUPLICATED DEFAULTS — same numbers in launch/launch_jzero_4gpu.sh.
rzero_rm_steps() {
    local batch n_chal n_sub total
    if [[ "${SMOKE:-0}" == "1" ]]; then
        batch=${RZERO_RM_BATCH:-8}; n_chal=${RZERO_RM_N_CHAL:-8}; n_sub=${RZERO_RM_N_SUBTASK:-8}
    else
        batch=${RZERO_RM_BATCH:-64}; n_chal=${RZERO_RM_N_CHAL:-256}; n_sub=${RZERO_RM_N_SUBTASK:-256}
    fi
    total=$(( n_chal + n_sub ))
    echo "${RZERO_RM_STEPS:-$(( (total + batch - 1) / batch ))}"
}

# Stage-3 pair-mix schedule (off by default). RZERO_RM_PAIR_SCHEDULE=linear:
# chalneg share = RZERO_RM_PAIR_R0 (0.8) at iter 1 -> 0.5 at RZERO_RM_PAIR_ITER_END (7),
# linear in between, 0.5 after; total pairs (and rzero_rm_steps) unchanged.
rzero_rm_pair_mix() {  # $1 = iteration
    local default_pairs=256
    [[ "${SMOKE:-0}" == "1" ]] && default_pairs=8
    export RZERO_RM_N_CHAL=${RZERO_RM_N_CHAL:-$default_pairs}
    export RZERO_RM_N_SUBTASK=${RZERO_RM_N_SUBTASK:-$default_pairs}
    [[ "${RZERO_RM_PAIR_SCHEDULE:-off}" == "linear" ]] || return 0
    local it="$1" end="${RZERO_RM_PAIR_ITER_END:-7}" r0 r total n_chal
    [[ "$end" =~ ^[1-9][0-9]*$ && "$end" -ge 2 ]] || {
        echo "[rm_pair_mix] RZERO_RM_PAIR_ITER_END must be an integer >= 2" >&2; return 1;
    }
    r0=$(awk -v r="${RZERO_RM_PAIR_R0:-0.8}" 'BEGIN {
        if (r !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ || r <= 0 || r > 1) exit 1
        printf "%d", r*1000
    }') || { echo "[rm_pair_mix] RZERO_RM_PAIR_R0 must be a number in (0, 1]" >&2; return 1; }
    total=$(( RZERO_RM_N_CHAL + RZERO_RM_N_SUBTASK ))
    if [[ -z "${RZERO_RM_PAIR_TOTAL:-}" ]]; then export RZERO_RM_PAIR_TOTAL=$total; else total=$RZERO_RM_PAIR_TOTAL; fi
    if [[ "$it" -ge "$end" ]]; then r=500; else r=$(( r0 - (r0 - 500) * (it - 1) / (end - 1) )); fi
    n_chal=$(( (total * r + 500) / 1000 ))
    [[ "$n_chal" -gt 0 ]] || { echo "[rm_pair_mix] mix must yield at least one challenger-negative pair" >&2; return 1; }
    export RZERO_RM_N_CHAL=$n_chal RZERO_RM_N_SUBTASK=$(( total - n_chal ))
}

# Bound wandb's end-of-stage flush: without it an unreachable backend hangs
# every stage at teardown with the GPUs idle. Unsent data stays on disk for the
# `wandb sync` the stage scripts run. The verl patch adds a coarser outer bound
# (JZERO_WANDB_SHUTDOWN_TIMEOUT, default 120s).
export WANDB_FINISH_TIMEOUT=${WANDB_FINISH_TIMEOUT:-120}
