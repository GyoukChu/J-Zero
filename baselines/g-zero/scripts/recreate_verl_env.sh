#!/usr/bin/env bash
# Recreate the 'verl' conda env (python 3.12 + torch 2.8 / vllm 0.11 / peft 0.19)
# that the G-Zero local pipeline runs on. Idempotent-ish: removes any partial
# verl env first.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f /opt/conda/etc/profile.d/conda.sh ] && source /opt/conda/etc/profile.d/conda.sh || true  # conda optional; any python 3.12 env manager works

echo "=== removing any existing verl env ==="
conda env remove -y -n verl 2>/dev/null || true

echo "=== creating verl (python 3.12) ==="
conda create -y -n verl python=3.12

conda activate verl
python --version
pip install --upgrade pip

# NB: transformers MUST stay on the 4.x line. vllm 0.11.0 declares only
# `transformers>=4.55.2`, so a loose pin pulls 5.x, which drops
# `all_special_tokens_extended` and crashes vllm's get_cached_tokenizer().
# 4.56.1 matches requirements/requirements-train-vllm011.txt; keep the two in sync.
echo "=== installing torch/vllm/peft stack (transformers pinned to 4.56.1 for vllm 0.11) ==="
pip install \
  vllm==0.11.0 \
  peft==0.19.0 \
  datasets==4.8.0 \
  accelerate \
  nltk \
  scikit-learn \
  regex \
  transformers==4.56.1 \
  huggingface_hub==0.36.2 \
  numpy

echo "=== installing flash-attn wheel (torch2.8 cp312, prebuilt — no compiler needed) ==="
pip install --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

echo "=== import smoke test ==="
python - <<'PY'
import torch, vllm, peft, transformers, datasets
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), torch.version.cuda)
print("vllm", vllm.__version__, "| peft", peft.__version__, "| transformers", transformers.__version__)
import flash_attn
print("flash_attn", flash_attn.__version__)
PY

echo "=== g_zero import smoke test ==="
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "import g_zero, g_zero.multi_round, g_zero.phase1, g_zero.phase3, g_zero.local_backend; print('g_zero imports OK')"

echo "=== DONE: verl env ready ==="
