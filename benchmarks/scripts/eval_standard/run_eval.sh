#!/bin/bash
# =============================================================================
# Run the FULL 14-benchmark evaluation suite on ONE checkpoint — pure vLLM,
# 4 GPUs, local judges only (no external API calls anywhere).
#
#   Phase 1  EM 3 (MMLU-Pro/SuperGPQA/BBH)  : DP=4 question-sharding (run_em_dp4.sh)
#   Phase 2  math 7 (MATH500/Minerva/Olympiad/
#            GSM8K/AMC23/AIME24/AIME25)     : run_solver_eval.sh + IFEval (1 GPU)
#   Phase 3  Arena-Hard-v2 || EQ-Bench CW   : parallel; per-category judges:
#            hard_prompt (style ctrl) -> 2x Qwen3.6-27B no-think,
#            creative_writing + EQ-Bench -> 1x gemma-4-31b-it
#   Phase 4  AlpacaEval 2.0                 : Qwen3.6-27B judge,
#            GPT-4-Turbo baseline, weighted_alpaca_eval_local_qwen36 annotator
#
# Judges are single tp=1 vLLM 0.23 instances (JUDGE_VENV). The test/gen model
# is served PERSISTENTLY on :$P_TEST across Phases 3+4. Startup is sequential
# and health-checked; cleanup is port-scoped (kill_port), never a broad pkill.
#
# Two concurrent lanes on one 8-GPU host: give each lane 4 GPUs and a disjoint
# port band + its own arena/eqbench working copies (see THIRD_PARTY.md):
#   GPU_SET=0,1,2,3 PORT_BASE=7400 bash run_eval.sh <model> <slugA> &
#   GPU_SET=4,5,6,7 PORT_BASE=9500 ARENA_DIR=.../arena-hard-auto-lane2 \
#     BENCH_DIR=.../creative-writing-bench-lane2 bash run_eval.sh <model> <slugB> &
#
# Usage: GPU_SET=0,1,2,3 bash run_eval.sh <model_path> <slug> [out_dir]
# =============================================================================
set -uo pipefail
MODEL=${1:?"model_path"}
SLUG=${2:?"slug"}
ROOT="${BENCH_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
OUT=${3:-$ROOT/outputs/eval/$SLUG}
SELF=$ROOT/scripts/eval_standard
GEN=$ROOT/scripts/eval_gen
# Generation venv: the vLLM 0.11 train env. For models saved by
# transformers-5/vllm-0.23, point RZERO_GEN_VENV at a vLLM 0.23 env instead —
# vLLM 0.11 mis-detokenizes those (doubled closing brackets), corrupting EM/math.
GEN_VENV="${RZERO_GEN_VENV:?export RZERO_GEN_VENV=/path/to/generation-venv (requirements/requirements-train-vllm011.txt)}"
VERL_PY="$GEN_VENV/bin/python"
GB_PY="${GENBENCH_PY:?export GENBENCH_PY=<genbench-venv>/bin/python (requirements-eval-genbench.txt)}"
AR_PY="${ARENA_PY:?export ARENA_PY=<arena-venv>/bin/python (requirements-eval-arena.txt)}"
AE_BIN="${ALPACA_EVAL_BIN:?export ALPACA_EVAL_BIN=<genbench-venv>/bin/alpaca_eval}"
NOTHINK=$GEN/qwen36_nothink.jinja
# Judge/test serving venv (vLLM 0.23) — serve_judge_vllm.sh (phases 3/4) and
# the math recheck judge (phase 2).
JUDGE_VENV="${JUDGE_VENV:?export JUDGE_VENV=/path/to/venv built from requirements-eval.txt}"
# The EM phase's load_dataset() calls need the Hub ONLINE (a populated hub
# cache does NOT satisfy offline load_dataset()); export here overrides the
# HF_HUB_OFFLINE=1 the training prelude would otherwise force. Judge serving
# stays offline (serve_judge_vllm.sh sets HF_HUB_OFFLINE=1 itself). To run
# fully offline, set HF_HUB_OFFLINE=1 and seed the datasets cache first
# (benchmarks/README.md one-time setup step 3).
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
mkdir -p "$OUT"
PHASE_START=${PHASE_START:-1}   # resume from this phase (1=EM, 2=math+IFEval, 3=Arena/EQ-Bench, 4=AlpacaEval)
# R-Zero recipe, used only by Phase 2. Checked up front so Phase 1 (hours of
# GPU work) can't die afterwards on a missing env var; not required for
# PHASE_START>=3 reruns, which never reach the recipe.
RECIPE=""
if [ "$PHASE_START" -le 2 ]; then
    RECIPE="${RECIPE_DIR:?export RECIPE_DIR=<verl>/recipe_R-Zero (install baselines/r-zero there)}"
fi
TIMING="$OUT/timing.log"; [ "$PHASE_START" -le 1 ] && : > "$TIMING"
mark(){ flock "$TIMING" -c "echo '$1' '$2' >> '$TIMING'"; }
RUN0=$(date +%s)

GPU_SET=${GPU_SET:-0,1,2,3}
IFS=',' read -ra G <<< "$GPU_SET"
[ ${#G[@]} -eq 4 ] || { echo "GPU_SET must list exactly 4 GPUs"; exit 1; }
JG="${G[0]},${G[1]},${G[2]}"   # 3 GPUs for the 3 single-vLLM judges
TG="${G[3]}"                   # 1 GPU for the persistent test/gen model

# Served-model name (phases 3+4); distinct from the slug so the arena
# api_config key and the vLLM served-model-name never collide.
SERVED_NAME="eval-$(echo "$SLUG" | sed 's/[^a-zA-Z0-9]/-/g')"

# Disjoint per-lane port band. PORT_BASE+30..32 = phase-3 per-category judges
# (2x Qwen + 1x gemma), +20 = phase-4 qwen judge, +51 = persistent test/gen
# endpoint.
PB=${PORT_BASE:-7400}
P_G1=$((PB+30)); P_G2=$((PB+31)); P_G3=$((PB+32))
P_Q1=$((PB+20))
P_TEST=$((PB+51))

# Kill processes listening on the named TCP port(s), plus children and
# grandchildren (vLLM workers/engine cores). Scope-safe: never a broad pkill,
# and never `kill -PGID` — nohup'd children share PGID with this script, so a
# process-group signal would take the script down too.
kill_port(){
  local p pid kids grandkids
  for p in "$@"; do
    pid=$(ss -tlnp 2>/dev/null | grep -E ":${p}\s" | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -z "$pid" ] && continue
    kids=$(pgrep -P "$pid" 2>/dev/null | tr '\n' ' ')
    grandkids=""
    for k in $kids; do
      grandkids="$grandkids $(pgrep -P "$k" 2>/dev/null | tr '\n' ' ')"
    done
    kill -TERM $pid $kids $grandkids 2>/dev/null || true
    sleep 3
    kill -KILL $pid $kids $grandkids 2>/dev/null || true
  done
  sleep 2
}

# Wait for an HTTP endpoint to return 200. Default 20-minute deadline so a
# cold vLLM weight load (Qwen3.6-27B, ~50 GB) has room.
wait_health(){
  local url=$1 max=${2:-240}
  local i
  for i in $(seq 1 $max); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

# Clear any stale listener on our ports (a zombie from a crashed run makes a
# new vLLM fail "Engine core initialization failed").
preflight_cleanup(){
  kill_port $P_Q1 $P_G1 $P_G2 $P_G3 $P_TEST
}

echo "[eval] model=$MODEL slug=$SLUG GPU_SET=$GPU_SET (judge=$JG, test=$TG) out=$OUT"
preflight_cleanup

# ---------------------------------------------------------------------------
# PHASE 1: EM 3 (DP=4) sequential on the 4 GPUs
# ---------------------------------------------------------------------------
if [ "$PHASE_START" -le 1 ]; then
echo "[eval] PHASE 1: EM DP=4 (sequential)"
for B in supergpqa mmlupro bbh; do
  t=$(date +%s)
  GPUS="$GPU_SET" EM_GEN_VENV="$GEN_VENV" bash "$SELF/run_em_dp4.sh" "$B" "$MODEL" "$OUT/em/$B" > "$OUT/em_${B}.log" 2>&1
  mark "$B" $(( $(date +%s) - t ))
done
fi

# ---------------------------------------------------------------------------
# PHASE 2: math 7 (EVAL_GPUS=$GPU_SET) then IFEval (1 GPU)
# ---------------------------------------------------------------------------
if [ "$PHASE_START" -le 2 ]; then
echo "[eval] PHASE 2: math 7 + IFEval"
t=$(date +%s)
EVAL_GPUS="$GPU_SET" JUDGE_GPU="${G[0]}" RZERO_VENV="$GEN_VENV" \
  DATA_DIR="${DATA_DIR:-$ROOT/data}" \
  JUDGE_VENV_PY="$JUDGE_VENV/bin/python" \
  NOTHINK_JINJA="$NOTHINK" \
  bash "$RECIPE/evaluate/run_solver_eval.sh" "$MODEL" "$OUT/math_recheck" > "$OUT/math7.log" 2>&1
mark math7 $(( $(date +%s) - t ))
t=$(date +%s)
IFE="$OUT/ifeval"; mkdir -p "$IFE"; PREP="$ROOT/outputs/eval_standard/_prepared/ifeval.jsonl"
CUDA_VISIBLE_DEVICES="${G[0]}" VLLM_PORT=$((${EM_VLLM_BASE:-52000}+${G[0]}*100)) MASTER_PORT=$((${EM_MASTER_BASE:-56000}+${G[0]}*100)) \
  $VERL_PY "$SELF/gen_vllm.py" --model "$MODEL" --in "$PREP" --out "$IFE/responses.jsonl" \
    --dp_shard 0 --dp_total 1 --max_tokens 2048 --max_model_len 8192 > "$OUT/ifeval.log" 2>&1 \
  && $VERL_PY "$SELF/score_ifeval.py" --prepared "$PREP" --responses "$IFE/responses.jsonl" --out "$IFE/score.json" >> "$OUT/ifeval.log" 2>&1
mark ifeval $(( $(date +%s) - t ))
fi

# ---------------------------------------------------------------------------
# PHASE 3: per-category judges x3 (2x Qwen + 1x gemma, on $JG) + persistent
#          test model (on $TG)
#          -> Arena || EQ-Bench (parallel, share judges + test endpoint)
# ---------------------------------------------------------------------------
if [ "$PHASE_START" -le 3 ]; then
echo "[eval] PHASE 3: Qwen3.6 x2 (hard_prompt) + gemma x1 (creative_writing/EQ-Bench) + persistent test -> Arena || EQ-Bench"
kill_port $P_Q1 $P_G1 $P_G2 $P_G3 $P_TEST

# Judges on the 3 judge GPUs:
#   $P_G1,$P_G2 = Qwen3.6-27B (no-thinking) -> Arena hard_prompt (style control)
#   $P_G3       = gemma-4-31b-it            -> Arena creative_writing + EQ-Bench
# Sequential health-checked startup: parallel cold-loads race and one engine
# init failure cascades to the whole phase.
i=0
for spec in "qwen:$P_G1" "qwen:$P_G2" "gemma:$P_G3"; do
  kind=${spec%%:*}; p=${spec##*:}
  if [ "$kind" = qwen ]; then
    GPUS="${G[$i]}" PORT=$p DP=1 MAXLEN=32768 CHAT_TEMPLATE="$NOTHINK" \
      nohup bash "$GEN/serve_judge_vllm.sh" Qwen/Qwen3.6-27B qwen36-judge \
      > "$OUT/judge_qwen_$p.log" 2>&1 &
  else
    # gemma judges EQ-Bench pieces embedded whole: a degenerate 12000-token
    # piece re-tokenizes ~2x larger in gemma's tokenizer (24k measured), so
    # 32768 leaves <10% headroom over prompt+4096 output. 49152 makes the
    # judge prompt structurally unable to overflow.
    GPUS="${G[$i]}" PORT=$p DP=1 MAXLEN=49152 \
      nohup bash "$GEN/serve_judge_vllm.sh" google/gemma-4-31b-it gemma4-judge \
      > "$OUT/judge_gemma_$p.log" 2>&1 &
  fi
  i=$((i+1))
  wait_health "http://127.0.0.1:$p/health" 300 \
    || { echo "[eval] FATAL judge $p ($kind) failed to start; see $OUT/judge_${kind}_$p.log"; \
         kill_port $P_G1 $P_G2 $P_G3; exit 1; }
done
# Persistent test/gen vLLM at :$P_TEST — used by Phases 3 AND 4, not killed
# between them.
GPUS="$TG" PORT=$P_TEST DP=1 MAXLEN=16384 \
  nohup bash "$GEN/serve_judge_vllm.sh" "$MODEL" "$SERVED_NAME" \
  > "$OUT/serve_test.log" 2>&1 &
wait_health "http://127.0.0.1:$P_TEST/health" 240 \
  || { echo "[eval] FATAL test model failed to start; see $OUT/serve_test.log"; \
       kill_port $P_G1 $P_G2 $P_G3 $P_TEST; exit 1; }

ARENA=${ARENA_DIR:-$ROOT/third_party/arena-hard-auto}
BENCH=${BENCH_DIR:-$ROOT/third_party/creative-writing-bench}

# EQ-Bench .env is mandatory: without one the harness silently falls back to
# https://api.openai.com — seed it from the local-only example beside it.
# BENCH_DIR lanes get their own copy; .env_example ships in each working copy.
[ -f "$BENCH/.env" ] || cp "$BENCH/.env_example" "$BENCH/.env"
[ -f "$BENCH/creative_bench_runs.PRISTINE.json" ] \
  || { echo "[eval] FATAL: $BENCH/creative_bench_runs.PRISTINE.json missing — run the one-time unzip in THIRD_PARTY.md"; exit 1; }

# --- Arena-Hard v2 (gen on test endpoint, per-category judges) ---
ARENA_BLOCK(){
  local t=$(date +%s) rc=0
  # Drop any stale arena_score.json first: if this run dies, collect_results.py
  # must report "(missing)", not an earlier run's number.
  rm -f "$OUT/arena/arena_score.json"
  ( cd "$ARENA"
    printf 'bench_name: arena-hard-v2.0\nmodel_list:\n  - %s\n' "$SLUG" > config/gen_answer_config.yaml
    # Force-write this slug's endpoint and the judge ports on EVERY run: a
    # stale entry would silently point gen_answer at a dead port.
    "$AR_PY" - "$SLUG" "$SERVED_NAME" "$P_TEST" "$P_G1" "$P_G2" "$P_G3" <<'PY'
import re,sys,yaml
slug, eqname, p_test = sys.argv[1], sys.argv[2], sys.argv[3]
jports = sys.argv[4:7]
p="config/arena-hard-v2.0.yaml"; s=open(p).read()
s=re.sub(r"model_list:\n(?:\s*-\s*\S+\n?)*", f"model_list:\n  - {slug}\n", s)
open(p,"w").write(s)
cfg=yaml.safe_load(open("config/api_config.yaml"))
cfg[slug]={"model": eqname,
           "endpoints":[{"api_base":f"http://127.0.0.1:{p_test}/v1","api_key":"-"}],
           "api_type":"openai","parallel":256,"max_tokens":4096,"temperature":0.0}
def judge_entry(ports):
    return {"endpoints":[{"api_base":f"http://127.0.0.1:{q}/v1","api_key":"-"} for q in ports],
            "api_type":"openai","parallel":256,"max_tokens":4096,"temperature":0.0}
# Per-category judges (names MUST match JUDGE_SETTINGS[cat]["judge"]):
#   hard_prompt -> qwen36-judge on the first 2 judge ports (P_G1,P_G2)
#   creative_writing / EQ-Bench -> gemma4-judge on the 3rd port (P_G3)
cfg["qwen36-judge"]={"model":"qwen36-judge", **judge_entry(jports[:2])}
cfg["gemma4-judge"]={"model":"gemma4-judge", **judge_entry(jports[2:3])}
# safe_dump drops YAML comments, so re-emit the header it would otherwise erase.
with open("config/api_config.yaml","w") as f:
    f.write("# Arena-Hard-v2.0 judges. REWRITTEN BY run_eval.sh ON EVERY RUN:\n"
            "#   qwen36-judge (hard_prompt)      -> the 2 Qwen3.6-27B ports\n"
            "#   gemma4-judge (creative_writing) -> the 1 gemma-4-31b-it port\n"
            "# plus one entry per evaluated slug, appended and kept. Edits and\n"
            "# comments below this header do not survive a run.\n")
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
    # Abort on a failed gen stage: otherwise show_result.py quietly writes a
    # null score. No arena_score.json -> "(missing)" in the results table.
    $AR_PY gen_answer.py --config-file config/gen_answer_config.yaml --endpoint-file config/api_config.yaml > "$OUT/arena_gen.log" 2>&1 \
      || { echo "[eval] FATAL arena gen_answer.py rc=$? — see $OUT/arena_gen.log"; exit 1; }
    rm -f "data/arena-hard-v2.0/model_judgment/qwen36-judge/${SLUG}.jsonl" \
          "data/arena-hard-v2.0/model_judgment/gemma4-judge/${SLUG}.jsonl"
    $AR_PY gen_judgment.py --setting-file config/arena-hard-v2.0.yaml --endpoint-file config/api_config.yaml > "$OUT/arena_judge.log" 2>&1 \
      || { echo "[eval] FATAL arena gen_judgment.py rc=$? — see $OUT/arena_judge.log"; exit 1; }
    mkdir -p "$OUT/arena"
    # Score each category from its own judge dir; hard_prompt is
    # style-controlled (--control-features markdown length), creative_writing raw.
    score_of(){ # $1 category  $2 style(0/1)  $3 judge  -> prints SLUG's score float
      local cat=$1 sc=$2 judge=$3 args=""
      [ "$sc" = "1" ] && args="--control-features markdown length"
      $AR_PY show_result.py -j "$judge" -b arena-hard-v2.0 -c "$cat" $args 2>/dev/null \
        | $GB_PY -c "import sys,re
slug=sys.argv[1]; v=''
for l in sys.stdin:
    if re.search(r'(^|\s)'+re.escape(slug)+r'(\s|\$)', l):
        m=re.search(r'(-?[0-9]+\.[0-9]+)', l)
        if m: v=m.group(1); break
print(v)" "$SLUG"
    }
    HP_SC=$(score_of hard_prompt 1 qwen36-judge)
    HP_RAW=$(score_of hard_prompt 0 qwen36-judge)
    CW=$(score_of creative_writing 0 gemma4-judge)
    # Coverage alongside the scores: a question whose judgment has missing/None
    # games (judge-context 400s, unparseable judge output) never reaches
    # show_result's table, so the score silently averages the judged subset —
    # see "Judge context budget & score coverage" in benchmarks/README.md.
    $GB_PY - "$SLUG" "$OUT/arena/arena_score.json" "${HP_SC:-}" "${HP_RAW:-}" "${CW:-}" <<'PY' \
      > "$OUT/arena/show_result.log" 2>&1 || true
import json, sys
slug, out = sys.argv[1], sys.argv[2]
hp_sc, hp_raw, cw = (float(v) if v else None for v in sys.argv[3:6])
qcat = {}
for line in open("data/arena-hard-v2.0/question.jsonl"):
    q = json.loads(line); qcat[q["uid"]] = q.get("category")
def judged(judge):
    n = 0
    try:
        for line in open(f"data/arena-hard-v2.0/model_judgment/{judge}/{slug}.jsonl"):
            games = [g for g in (json.loads(line).get("games") or []) if isinstance(g, dict)]
            if games and all(g.get("score") is not None for g in games):
                n += 1
    except FileNotFoundError:
        pass
    return n
d={'hard_prompt': hp_sc, 'hard_prompt_raw': hp_raw, 'creative_writing': cw,
   'hard_prompt_judge':'qwen36-judge','hard_prompt_baseline':'o3-mini-2025-01-31','hard_prompt_style_control':['markdown','length'],
   'creative_writing_judge':'gemma4-judge','creative_writing_baseline':'gemini-2.0-flash-001',
   'hard_prompt_judged': judged('qwen36-judge'),
   'hard_prompt_total': sum(1 for c in qcat.values() if c == 'hard_prompt'),
   'creative_writing_judged': judged('gemma4-judge'),
   'creative_writing_total': sum(1 for c in qcat.values() if c == 'creative_writing')}
json.dump(d, open(out, 'w'), indent=1)
print('arena hard_prompt(SC)=%s raw=%s (judged %s/%s) creative_writing=%s (judged %s/%s)' % (
    d['hard_prompt'], d['hard_prompt_raw'], d['hard_prompt_judged'], d['hard_prompt_total'],
    d['creative_writing'], d['creative_writing_judged'], d['creative_writing_total']))
PY
  )
  rc=$?
  mark arena-hard $(( $(date +%s) - t ))
  return $rc
}

# --- EQ-Bench CW v3 (rubric only, --no-elo; test+gemma concurrent) ---
# Per-slug runs-file so two concurrent eval lanes can't overwrite each other's
# creative_bench_runs.json.
EQBENCH_BLOCK(){
  local t=$(date +%s)
  local RUNS_FILE="creative_bench_runs_${SLUG}.json"
  sed -i "s#TEST_API_URL=.*#TEST_API_URL=http://127.0.0.1:$P_TEST/v1/chat/completions#" "$BENCH/.env"
  sed -i "s#JUDGE_API_URL=.*#JUDGE_API_URL=http://127.0.0.1:$P_G3/v1/chat/completions#" "$BENCH/.env"
  cp -f "$BENCH/creative_bench_runs.PRISTINE.json" "$BENCH/$RUNS_FILE"
  rm -f "$BENCH"/*.lock 2>/dev/null
  ( cd "$BENCH" && $GB_PY creative_writing_bench.py --test-model "$SERVED_NAME" --judge-model gemma4-judge \
      --runs-file "$RUNS_FILE" --run-id "$SLUG" --iterations 3 --threads 96 --verbosity INFO --no-elo ) > "$OUT/eqbench.log" 2>&1
  local rc=$?
  mkdir -p "$OUT/eqbench"
  cp -f "$BENCH/$RUNS_FILE" "$OUT/eqbench/runs.json" 2>/dev/null
  rm -f "$BENCH/$RUNS_FILE" 2>/dev/null
  $GB_PY - "$OUT/eqbench/runs.json" "$SERVED_NAME" "$OUT/eqbench/eqbench_score.json" <<'PY'
import json,sys
runs=json.load(open(sys.argv[1])); served=sys.argv[2]; out=sys.argv[3]
sc={}
for k,v in runs.items():
    if v.get("test_model")==served and v.get("status")=="completed":
        br=v.get("results",{}).get("benchmark_results",{})
        # Coverage: tasks with no judge_scores (judge-context 400s at the gemma
        # endpoint) are silently excluded from the rubric average upstream —
        # see "Judge context budget & score coverage" in benchmarks/README.md.
        total=scored=0
        for prompts in v.get("creative_tasks",{}).values():
            for t in prompts.values():
                total+=1
                if any(b.get("judge_scores") for b in t.get("results_by_modifier",{}).values()):
                    scored+=1
        sc={"rubric_eqbench_creative_score":br.get("eqbench_creative_score"),
            "rubric_creative_score_0_20":br.get("creative_score_0_20"),
            "tasks_scored":scored,"tasks_total":total}
        break
json.dump(sc, open(out,"w"), indent=2)
PY
  mark eqbench-cw $(( $(date +%s) - t ))
  return $rc
}

ARENA_BLOCK > "$OUT/arena_runner.log" 2>&1 &
APID=$!
EQBENCH_BLOCK > "$OUT/eqbench_runner.log" 2>&1 &
EPID=$!
wait $APID; ARC=$?
wait $EPID; EQC=$?
echo "[eval] phase3: arena exit=$ARC, eqbench exit=$EQC"
if [ "$ARC" -ne 0 ]; then
  echo "[eval] phase3: arena-hard FAILED (see $OUT/arena_runner.log); it will show as (missing) in RESULTS.txt"
fi
if [ "$EQC" -ne 0 ]; then
  echo "[eval] phase3: eqbench FAILED (see $OUT/eqbench_runner.log); it will show as (missing) in RESULTS.txt"
fi

# Phase 3 -> 4 transition: KILL the phase-3 judges only, KEEP :$P_TEST test alive.
kill_port $P_G1 $P_G2 $P_G3
fi

# ---------------------------------------------------------------------------
# PHASE 4: Qwen3.6-27B judge (on ${G[0]}) + persistent test :$P_TEST
#          -> AlpacaEval (gen via HTTP to the test endpoint)
# ---------------------------------------------------------------------------
if [ "$PHASE_START" -le 4 ]; then
echo "[eval] PHASE 4: Qwen3.6 judge + reuse test endpoint -> AlpacaEval"

# Re-validate the persistent test endpoint (in case running phase4 standalone).
if ! curl -sf "http://127.0.0.1:$P_TEST/health" >/dev/null 2>&1; then
  echo "[eval] PHASE 4: :$P_TEST not up (standalone phase4 run?), bringing it up on \$TG=$TG"
  kill_port $P_TEST
  GPUS="$TG" PORT=$P_TEST DP=1 MAXLEN=16384 \
    nohup bash "$GEN/serve_judge_vllm.sh" "$MODEL" "$SERVED_NAME" \
    > "$OUT/serve_test.log" 2>&1 &
  wait_health "http://127.0.0.1:$P_TEST/health" 240 \
    || { echo "[eval] FATAL test :$P_TEST not healthy"; exit 1; }
fi

# Qwen3.6-27B is hybrid-mamba — DP=N dies at vLLM init, so serve a single
# tp=1 instance.
kill_port $P_Q1
GPUS="${G[0]}" PORT=$P_Q1 DP=1 MAXLEN=32768 CHAT_TEMPLATE="$NOTHINK" \
  nohup bash "$GEN/serve_judge_vllm.sh" Qwen/Qwen3.6-27B qwen36-judge \
  > "$OUT/judge_qwen_$P_Q1.log" 2>&1 &
wait_health "http://127.0.0.1:$P_Q1/health" 240 \
  || { echo "[eval] FATAL qwen36 $P_Q1 failed to start; see $OUT/judge_qwen_$P_Q1.log"; \
       kill_port $P_Q1 $P_TEST; exit 1; }
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# --- AlpacaEval 2.0 (GPT-4-Turbo baseline, Qwen3.6-27B no-think judge) ---
ALPACA_BLOCK(){
  local t=$(date +%s)
  local AOUT="$OUT/alpaca"; mkdir -p "$AOUT"
  local PROMPTS=$ROOT/outputs/eval_gen/alpaca/_prep/prompts.jsonl
  local REF=$ROOT/outputs/eval_gen/alpaca/_prep/reference_gpt4turbo.json
  $VERL_PY "$SELF/gen_http.py" --base_url http://127.0.0.1:$P_TEST/v1 \
    --model_name "$SERVED_NAME" --tokenizer "$MODEL" --in "$PROMPTS" \
    --out "$AOUT/responses.jsonl" --max_tokens 4096 --workers 96 \
    > "$OUT/alpaca_gen.log" 2>&1
  $GB_PY "$SELF/to_alpaca_outputs.py" --prompts "$PROMPTS" --responses "$AOUT/responses.jsonl" \
    --generator "$SLUG" --out "$AOUT/model_outputs.json" >> "$OUT/alpaca_gen.log" 2>&1
  ( cd "$ROOT" && OPENAI_API_BASE=http://127.0.0.1:$P_Q1/v1 OPENAI_API_KEY=dummy IS_ALPACA_EVAL_2=True \
    "$AE_BIN" evaluate --model_outputs "$AOUT/model_outputs.json" --reference_outputs "$REF" \
    --annotators_config weighted_alpaca_eval_local_qwen36 --output_path "$AOUT/" --caching_path None \
  ) > "$OUT/alpaca_judge.log" 2>&1
  mark alpacaeval $(( $(date +%s) - t ))
}

ALPACA_BLOCK > "$OUT/alpaca_runner.log" 2>&1
echo "[eval] phase4: alpaca exit=$?"
fi

kill_port $P_Q1 $P_G1 $P_G2 $P_G3 $P_TEST
mark TOTAL $(( $(date +%s) - RUN0 ))
echo "[eval] DONE. Collecting results..."
$GB_PY "$SELF/collect_results.py" "$OUT" | tee "$OUT/RESULTS.txt"