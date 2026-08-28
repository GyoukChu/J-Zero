#!/bin/bash
set -e

# =============================================================================
# verl install script (conda/venv), kept in sync with upstream verl's *stable*
# Docker images docker/Dockerfile.stable.{vllm,sglang} (paths in the verl repo,
# not this one). Both target CUDA 13.0 and pin torch==2.11.0; we install only
# *prebuilt wheels*, so this works on Ampere (sm_80/86), Hopper (sm_90) and
# Blackwell (sm_100/120) without a local CUDA-13 toolkit.
#
# Backends are MUTUALLY EXCLUSIVE (each stable image ships exactly one):
#   USE_SGLANG=0 (default) -> vLLM;  USE_SGLANG=1 -> SGLang
# Default USE_MEGATRON=0 (FSDP), matching the conda quick-start.
#
# Intentional deviations from the Docker images:
#   * vLLM comes from the official prebuilt wheel, so verl's two unmerged
#     weight-sync patches (PR #44483, #45589) are NOT applied; they only affect
#     sleep-mode / FlashInfer-TRTLLM-MoE resharding.
#   * flash-attn 2.8.3 has no cu13+torch2.11 wheel; the nearest published cu13
#     wheel (built against torch2.10) is used, so training may need to fall
#     back to attn_implementation="sdpa" on an ABI mismatch.
#   * USE_MEGATRON=1 builds TransformerEngine/Megatron from source and REQUIRES
#     a full CUDA 13 toolkit (nvcc); apex is not built here.
#   * DeepEP is NOT installed (no wheel, needs CUDA 13 nvcc). Build it manually
#     for large MoE expert-parallel rollouts; the SGLang base image ships it.
# =============================================================================

USE_MEGATRON=${USE_MEGATRON:-0}
USE_SGLANG=${USE_SGLANG:-0}

export MAX_JOBS=${MAX_JOBS:-32}

# ---- Pinned versions (mirror upstream verl's docker/Dockerfile.stable.{vllm,sglang}) --------
CUDA_INDEX="https://download.pytorch.org/whl/cu130"
TORCH_VERSION="2.11.0"
TORCH_VISION_VERSION="0.26.0"
TORCH_AUDIO_VERSION="2.11.0"
VLLM_VERSION="0.23.0"
SGLANG_VERSION="0.5.12"
# transformers pin is backend-specific: the vLLM image force-installs 5.3.0
# (overriding vLLM's own 'transformers!=5.3.*' marker), the SGLang image keeps
# sglang 0.5.12's exact 5.6.0 pin.
if [ "$USE_SGLANG" -eq 1 ]; then
    TRANSFORMERS_VERSION="5.6.0"
else
    # 5.12.1, not the image's 5.3.0: the official vLLM wheel breaks with 5.3.x
    # (vLLM declares transformers!=5.3.*; the image papers over it with source
    # patches), and gemma4 (google/gemma-4-E4B-it) needs >=5.12 to be detected.
    TRANSFORMERS_VERSION="5.12.1"
fi
TRL_VERSION="0.27.0"
FLASH_ATTN_VERSION="2.8.3"
# flash-attn prebuilt wheel target (nearest cu13 build to torch ${TORCH_VERSION})
FLASH_ATTN_TORCH_TAG="cu13torch2.10cxx11abiTRUE"
TRANSFORMER_ENGINE_VERSION="v2.15"
MCORE_VERSION="core_v0.16.1"
SGL_KERNEL_WHL="sglang_kernel-0.4.2.post2+cu130-cp310-abi3-manylinux2014_x86_64.whl"
SGL_KERNEL_URL="https://github.com/sgl-project/whl/releases/download/v0.4.2.post2/${SGL_KERNEL_WHL}"

ARCH="$(uname -m)"

echo "0. install PyTorch ${TORCH_VERSION} (CUDA 13.0 wheels)"
pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" "torchvision==${TORCH_VISION_VERSION}" "torchaudio==${TORCH_AUDIO_VERSION}" \
    --index-url "${CUDA_INDEX}"

echo "1. install inference framework (backend = $([ "$USE_SGLANG" -eq 1 ] && echo SGLang || echo vLLM))"
if [ "$USE_SGLANG" -eq 1 ]; then
    # Mirrors Dockerfile.stable.sglang, whose base image ships sglang/sgl-kernel/
    # DeepEP; we pull them via pip. --extra-index-url keeps torch==${TORCH_VERSION}
    # resolvable (it exists only on the cu130 index).
    pip install --no-cache-dir "sglang[all]==${SGLANG_VERSION}" --extra-index-url "${CUDA_INDEX}"
    pip install --no-cache-dir torch-memory-saver
    # Force the cu130 sgl-kernel build, matching the Docker image (x86_64 only).
    if [ "$ARCH" = "x86_64" ]; then
        pip install --no-cache-dir --force-reinstall "${SGL_KERNEL_URL}" \
            || echo "WARN: cu130 sgl-kernel force-reinstall failed; the pip-resolved sgl-kernel will be used"
    fi
else
    # Mirrors Dockerfile.stable.vllm, but uses the official prebuilt wheel rather
    # than a patched source build (see header). vLLM 0.23.0 pulls torch==2.11.0 +
    # flashinfer-python/cubin==0.6.12 as deps.
    pip install --no-cache-dir "vllm==${VLLM_VERSION}" --extra-index-url "${CUDA_INDEX}"
fi

echo "2. install verl runtime dependencies"
pip install --no-cache-dir \
    accelerate datasets peft hf-transfer \
    "numpy>=2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    "ray[default]" codetiming hydra-core pylatexenc "qwen-vl-utils==0.0.14" wandb dill pybind11 \
    liger-kernel mathruler cachetools pytest-asyncio nvtx matplotlib \
    pytest py-spy pre-commit ruff tensorboard \
    "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "pyext is lack of maintenance and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

# Pinned AFTER the inference framework, matching the Docker images: for vLLM this
# overrides its 'transformers!=5.3.*' marker (pip warns, exit 0), for SGLang it
# reasserts sglang's own 5.6.0 pin.
echo "3. pin transformers==${TRANSFORMERS_VERSION} and trl==${TRL_VERSION} (verl stack)"
pip install --no-cache-dir "transformers[hf_xet]==${TRANSFORMERS_VERSION}"
if [ "$USE_SGLANG" -eq 1 ]; then
    # sglang depends on 'kernels' with no upper bound, but transformers ${TRANSFORMERS_VERSION}
    # needs >=0.12,<0.13; a newer one makes transformers' hub_kernels raise at
    # import time and breaks `import sglang`.
    pip install --no-cache-dir "kernels>=0.12.0,<0.13"
fi
pip install --no-cache-dir --no-deps "trl==${TRL_VERSION}"

echo "4. install FlashAttention ${FLASH_ATTN_VERSION} (prebuilt cu13 wheel)"
# No cu13+torch2.11 wheel is published; install the nearest cu13 build. Release wheels
# bundle SM 8.0/9.0/10.0/12.0, so one x86_64 wheel covers Ampere/Hopper/Blackwell.
if [ "$ARCH" = "x86_64" ]; then
    FA_WHL="flash_attn-${FLASH_ATTN_VERSION}+${FLASH_ATTN_TORCH_TAG}-cp312-cp312-linux_x86_64.whl"
    pip install --no-cache-dir \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${FA_WHL}" \
        || echo "WARN: flash-attn wheel install failed; use attn_implementation=sdpa or build from source against CUDA 13"
else
    echo "WARN: no prebuilt flash-attn wheel selected for ${ARCH}; install manually or use sdpa"
fi
python -c "import flash_attn; print('flash_attn OK', flash_attn.__version__)" \
    || echo "WARN: flash_attn import failed (likely torch-ABI mismatch vs ${TORCH_VERSION}); use attn_implementation=sdpa or build flash-attn from source against CUDA 13"

echo "5. install torchcodec (CUDA 13 build; optional, needs system ffmpeg libs)"
pip install --no-cache-dir torchcodec --index-url "${CUDA_INDEX}" \
    || echo "WARN: torchcodec install failed (optional video decoding); continuing"

echo "6. fix opencv (best-effort)"
pip install --no-cache-dir opencv-python || echo "WARN: opencv-python install failed; continuing"
pip install --no-cache-dir opencv-fixer && python -c "from opencv_fixer import AutoFix; AutoFix()" \
    || echo "WARN: opencv-fixer/AutoFix failed (network or headless display); continuing"

if [ "$USE_MEGATRON" -eq 1 ]; then
    echo "7. install TransformerEngine ${TRANSFORMER_ENGINE_VERSION} and Megatron ${MCORE_VERSION}"
    echo "   NOTE: builds from source and REQUIRES a full CUDA 13 toolkit (nvcc) + cmake on this machine."
    echo "   TransformerEngine can take a very long time to build; please be patient."
    pip install --no-cache-dir ninja onnxscript
    NVTE_FRAMEWORK=pytorch pip install --no-cache-dir --no-deps --no-build-isolation \
        "git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_VERSION}"
    pip install --no-cache-dir --no-deps "git+https://github.com/NVIDIA/Megatron-LM.git@${MCORE_VERSION}"
    pip install --no-cache-dir -U "git+https://github.com/ISEEKYAN/mbridge.git@main"
fi

# Runs LAST (mirroring both Dockerfiles) so the Megatron block's `mbridge -U` cannot
# clobber this pin: verl weight resharding needs ncclCommSuspend/ncclCommResume from
# NCCL >= 2.29.7 (RFC: https://github.com/verl-project/verl/issues/6266).
echo "8. override NCCL (>=2.29.7) for verl weight resharding"
pip install --no-cache-dir --no-deps --upgrade "nvidia-nccl-cu13>=2.29.7,<3.0"

# torch's cu130 wheel already bundles nvidia-cudnn-cu13; ensure the python package is present
# (no -U: keep the cuDNN version torch 2.11.0 was built against).
echo "9. ensure cuDNN (cu13) python package"
pip install --no-cache-dir nvidia-cudnn-cu13 \
    || echo "WARN: nvidia-cudnn-cu13 install skipped; relying on torch-bundled cuDNN"

# ---- J-Zero / R-Zero / G-Zero pipeline extras -------------------------------
# Utility deps imported by the training pipelines but not by verl itself
# (question filtering / BLEU dedup, RM reward HTTP service, eval timeout guards,
# answer parsing), validated on the B200 test env.
# math-verify backs verl's math_verify grader for the R-Zero solver reward:
# without it every solver reward silently scores 0 (verl swallows the
# ImportError per call). langdetect + immutabledict back the vendored IFEval
# scoring lib (benchmarks/third_party/instruction_following_eval).
echo "10. install pipeline extras (nltk / scikit-learn / flask / stopit / regex / math-verify)"
pip install --no-cache-dir nltk scikit-learn flask stopit regex \
    "math-verify==0.9.0" "latex2sympy2-extended==1.11.0" \
    langdetect immutabledict

# NLTK tokenizer data, needed only by benchmark IFEval scoring (which runs in this
# env on nltk 3.9.x); training's only nltk use is sentence_bleu on pre-split token
# lists. Downloads to $NLTK_DATA if set, else ~/nltk_data. The vllm011 env has no
# equivalent step — run `python -m nltk.downloader punkt_tab` there by hand.
echo "11. download NLTK punkt_tab data (needed by benchmark IFEval scoring, not by training)"
python -m nltk.downloader punkt_tab \
    || echo "WARN: NLTK punkt_tab download failed (no network?) — harmless for training; required before running benchmarks/ IFEval"

echo "Successfully installed all packages"
