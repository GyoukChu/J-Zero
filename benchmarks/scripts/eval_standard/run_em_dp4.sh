#!/bin/bash
# Run ONE EM bench data-parallel across N GPUs (default 4): each GPU runs one
# model instance (tp=1) over a 1/N question shard, then we sum the per-shard
# .score sidecars for the Overall Accuracy. Mirrors the math eval's DP-by-question.
#
# Usage: GPUS=0,1,2,3 bash run_em_dp4.sh <bench> <model_path> <out_dir>
#   bench in: mmlupro | bbh | supergpqa
set -uo pipefail
source "${EM_GEN_VENV:?export EM_GEN_VENV=/path/to/generation-venv (requirements/requirements-train-vllm011.txt)}/bin/activate"
# Hub ONLINE by default, set BEFORE the prelude below (which forces
# HF_HUB_OFFLINE=1 but honours a caller value): offline load_dataset() reads
# only the datasets cache — a populated hub cache is not enough — so the EM
# benches died with OfflineModeIsEnabled. run_eval.sh exports the same default.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
source "${RECIPE_DIR:?export RECIPE_DIR=<verl>/recipe_R-Zero (install baselines/r-zero there)}/env_prelude.sh"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_DISABLE_COMPILE_CACHE=1

BENCH=${1:?bench}; MODEL=${2:?model_path}; OUT=${3:?out_dir}
GPUS=${GPUS:-0,1,2,3}
IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
mkdir -p "$OUT"

ROOT="${BENCH_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
RZERO=$ROOT/em_rzero
case "$BENCH" in
  mmlupro)   SCRIPT="$RZERO/eval_mmlupro.py";   EXTRA="";  SEED_DS="TIGER-Lab/MMLU-Pro" ;;
  supergpqa) SCRIPT="$RZERO/eval_supergpqa.py"; EXTRA="";  SEED_DS="m-a-p/SuperGPQA" ;;
  bbh)       SCRIPT="$RZERO/eval_bbh.py";       EXTRA="--results_file $OUT/${BENCH}_final_results.jsonl"; SEED_DS="" ;;
  *) echo "unknown bench $BENCH" >&2; exit 1 ;;
esac

# Seed the datasets cache ONCE before spawning shards. N concurrent cold-cache
# load_dataset() calls race on the builder filelock, and a CDN connection
# dropped during that race leaves every shard blocked forever (CLOSE-WAIT
# sockets, no read timeout) with the engines already holding all N GPUs. With
# a warm cache the per-shard load_dataset() is a pure local read. Already
# cached -> quick no-op; a failure here (timeout guards the same dropped-CDN
# hang, and offline-with-cold-cache dies here too) aborts before any GPU work.
if [ -n "$SEED_DS" ]; then
    echo "[dp$N $BENCH] seeding datasets cache: $SEED_DS"
    timeout "${EM_SEED_TIMEOUT:-1800}" python - "$SEED_DS" <<'PY' \
        || { echo "[dp$N $BENCH] FATAL: dataset cache seed failed for $SEED_DS" >&2; exit 1; }
import sys
import datasets
datasets.load_dataset(sys.argv[1])
PY
fi

echo "[dp$N $BENCH] model=$MODEL GPUS=$GPUS"
T0=$(date +%s)
PIDS=()
for i in "${!GPU_ARR[@]}"; do
    G=${GPU_ARR[$i]}
    # Per-shard cache dirs so concurrent vLLM inits don't race on shared
    # triton/inductor cache writes (corruption freezes every shard mid-run).
    SHARD_CACHE_ROOT=$HOME/.cache/em_dp4_gpu${G}
    mkdir -p "$SHARD_CACHE_ROOT/triton" "$SHARD_CACHE_ROOT/inductor" "$SHARD_CACHE_ROOT/vllm"
    ( CUDA_VISIBLE_DEVICES=$G VLLM_PORT=$((${EM_VLLM_BASE:-52000} + G*100)) MASTER_PORT=$((${EM_MASTER_BASE:-56000} + G*100)) MASTER_ADDR=127.0.0.1 \
      TRITON_CACHE_DIR="$SHARD_CACHE_ROOT/triton" \
      TORCHINDUCTOR_CACHE_DIR="$SHARD_CACHE_ROOT/inductor" \
      PYTORCH_KERNEL_CACHE_PATH="$SHARD_CACHE_ROOT/inductor" \
      VLLM_CACHE_ROOT="$SHARD_CACHE_ROOT/vllm" \
      python "$SCRIPT" --model_path "$MODEL" --output_file "$OUT/${BENCH}.shard${i}of${N}.json" \
        --tp 1 --dp_shard "$i" --dp_total "$N" $EXTRA > "$OUT/${BENCH}.shard${i}.log" 2>&1 ) &
    PIDS+=($!)
    sleep 4   # stagger vLLM init
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
T1=$(date +%s)
echo "[dp$N $BENCH] generation+score done in $((T1-T0))s (fail=$FAIL)"

# A crashed shard must fail the bench: merging a subset of the questions as if
# it were complete is a silently-wrong score.
if [ "$FAIL" = 1 ]; then
    echo "[dp$N $BENCH] FATAL: at least one shard failed — not merging (see $OUT/${BENCH}.shard*.log)"
    exit 1
fi

# sum per-shard .score sidecars
python - "$OUT" "$BENCH" "$N" <<'PY'
import json, sys, glob, os
out, bench, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = f = 0
percat = {}
got = 0
for sc in sorted(glob.glob(os.path.join(out, f"{bench}.shard*of{n}.json.score"))):
    d = json.load(open(sc)); got += 1
    s += d["success"]; f += d["fail"]
    for c, (cs, cf) in d.get("per_category", {}).items():
        pc = percat.setdefault(c, [0, 0]); pc[0] += cs; pc[1] += cf
tot = s + f
micro = s / tot * 100 if tot else 0.0
valid = [c for c, (cs, cf) in percat.items() if cs + cf > 0]
macro = sum(percat[c][0]/(percat[c][0]+percat[c][1]) for c in valid)/len(valid)*100 if valid else 0.0
if got != n:
    raise SystemExit(f"[dp{n} {bench}] FATAL: only {got}/{n} shard score files present — not merging")
print(f"[dp{n} {bench}] shards merged: {got}/{n}  n={tot}  success={s}")
print(f"Overall Accuracy: {micro/100:.4f}")
print(f"Micro Average Accuracy: {micro:.2f}%")
print(f"Macro Average Accuracy: {macro:.2f}%")
json.dump({"dataset": bench, "n": tot, "success": s, "micro": round(micro,2), "macro": round(macro,2)},
          open(os.path.join(out, f"{bench}.merged_score.json"), "w"), indent=2)
PY