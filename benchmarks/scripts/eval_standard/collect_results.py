#!/usr/bin/env python
"""Collect the 14-benchmark results + per-bench timings into one table.

Reads the score artifacts written by run_eval.sh and prints:
  (1) results table (score per bench)
  (2) timing table (wall-clock per bench, from <out>/timing.log lines "<bench> <sec>")
Usage: collect_results.py <out_dir>
"""
import json, os, sys, glob, csv

out = sys.argv[1]
def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

rows = []  # (bench, score_str)

# --- EM DP=4 merged scores ---
for b in ["mmlupro", "supergpqa", "bbh"]:
    d = jload(os.path.join(out, "em", b, f"{b}.merged_score.json"))
    if d:
        rows.append((b, f"micro {d['micro']:.2f}% / macro {d.get('macro','?')}% (n={d['n']})"))
    else:
        rows.append((b, "(missing)"))

# --- IFEval ---
ife = jload(os.path.join(out, "ifeval", "score.json"))
ifs = (ife or {}).get("summary", ife or {})
if ifs.get("prompt_level_strict_acc") is not None:
    rows.append(("ifeval", f"prompt-strict {ifs['prompt_level_strict_acc']:.2f}% / inst-strict {ifs.get('instruction_level_strict_acc', float('nan')):.2f}%"))
else:
    rows.append(("ifeval", "(missing)"))

# --- math 7 ---
m = jload(os.path.join(out, "math_recheck", "summary.json"))
if m and "datasets" in m:
    for ds in ["math500", "minervamath", "olympiadbench", "gsm8k", "amc23", "aime24", "aime25"]:
        v = m["datasets"].get(ds, {})
        fp = v.get("final_pct")
        rows.append((ds, f"{fp:.2f}%" if fp is not None else "(missing)"))
else:
    for ds in ["math500", "minervamath", "olympiadbench", "gsm8k", "amc23", "aime24", "aime25"]:
        rows.append((ds, "(missing)"))

# --- AlpacaEval (LC win-rate, GPT-4-Turbo baseline) ---
lb = None
for p in glob.glob(os.path.join(out, "alpaca", "**", "leaderboard.csv"), recursive=True):
    lb = p; break
if lb:
    r = list(csv.DictReader(open(lb)))
    if r:
        rows.append(("alpacaeval", f"LC {float(r[0]['length_controlled_winrate']):.2f}% / raw {float(r[0]['win_rate']):.2f}%"))
    else:
        rows.append(("alpacaeval", "(empty)"))
else:
    rows.append(("alpacaeval", "(missing)"))

# --- Arena (hard_prompt win-rate vs o3-mini baseline) ---
# No arena_score.json at all = the run died before scoring (run_eval.sh clears
# stale files first), so report it like any failed bench. The file present with
# a null score is the narrower case where only show_result/parsing failed.
ar = jload(os.path.join(out, "arena", "arena_score.json"))
if ar is None:
    rows.append(("arena-hard", "(missing)"))
elif ar.get('hard_prompt') is not None:
    rows.append(("arena-hard", f"hard_prompt {ar.get('hard_prompt')}% (SC, vs o3-mini) | creative_writing {ar.get('creative_writing')}% (vs gemini-2.0-flash)"))
else:
    rows.append(("arena-hard", "(see show_result log)"))

# --- EQ-Bench: rubric (the ELO stage is disabled; see run_eval.sh EQBENCH_BLOCK) ---
eq = jload(os.path.join(out, "eqbench", "eqbench_score.json"))
if eq:
    rub = eq.get("rubric_eqbench_creative_score")
    rows.append(("eqbench-cw", f"rubric {rub}"))
else:
    rows.append(("eqbench-cw", "(see runs json)"))

# --- timing ---
timing = {}
tp = os.path.join(out, "timing.log")
if os.path.exists(tp):
    for line in open(tp):
        parts = line.split()
        if len(parts) >= 2:
            try:
                timing[parts[0]] = int(float(parts[1]))
            except ValueError:
                pass

print("\n================ EVAL RESULTS (14 benchmarks) ================")
print(f"{'benchmark':16} | {'score':45} | {'time':>9}")
print("-" * 78)
order = ["mmlupro","supergpqa","bbh","ifeval",
         "math500","minervamath","olympiadbench","gsm8k","amc23","aime24","aime25",
         "alpacaeval","arena-hard","eqbench-cw"]
sd = dict(rows)
for b in order:
    t = timing.get(b)
    ts = f"{t//60}m{t%60:02d}s" if t is not None else "-"
    print(f"{b:16} | {sd.get(b,'(missing)'):45} | {ts:>9}")
tot = timing.get("TOTAL")
if tot:
    print("-" * 78)
    print(f"{'TOTAL wall-clock':16} | {'':45} | {tot//60}m{tot%60:02d}s")

# Non-zero exit when any benchmark is missing/failed, so callers (run_eval.sh
# runs this as its last command under pipefail) can gate "done" on rc==0.
bad = [b for b in order if "(missing)" in sd.get(b, "(missing)") or "(empty)" in sd.get(b, "")
       or "(see " in sd.get(b, "")]
if bad:
    print(f"INCOMPLETE: {', '.join(bad)}")
    sys.exit(1)