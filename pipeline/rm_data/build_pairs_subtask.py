#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IDA-style (Iterated Amplification) subtask-decomposition RM pair builder.
  Christiano, Shlegeris & Amodei, "Supervising strong learners by amplifying
  weak experts" (2018).

Decompose the task into easier SUBTASKS, have the weak-expert model solve each,
then compose the sub-answers into the final whole. The gain is real because the
weak expert is genuinely better at the easy subtasks than at the hard whole.

Per prompt x:
  Step1  y_S0  = a one-shot Solver response from the pool          (REJECTED)
  Step2  q_1..q_m = decompose(x)        -- the few essential subtasks (challenger)
  Step3  a_i     = solve(q_i | x)       -- weak expert answers each subtask
  Step4  y_amp   = compose(x, {q_i,a_i})-- stitched whole answer    (CHOSEN)
  Step5  pair (y^-, y^+) = (y_S0, y_amp)

Three model passes: decompose / solve-subtasks (batched) / compose. The weak
expert defaults to --model, per the paper's "weak experts are copies of the
current agent"; --solver_model overrides it with a separate policy. Optional
--verify keeps a pair only if the amplified answer is judged an improvement.

Emits {"question","chosen","rejected"} — the rm_train_loop schema.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from typing import Dict, List

os.environ.setdefault("VLLM_USE_FLASHINFER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

# ---- Pass 1: decompose the task into the few essential subtasks --------------
DECOMPOSE_SYS = (
    "You are an expert at breaking a hard task into simpler subtasks. Given a task, "
    "output the FEW essential sub-questions/subtasks whose answers, once combined, "
    "fully solve the task. Rules:\n"
    "- Output the 3-5 most essential subtasks only — never more than 5.\n"
    "- Each subtask must be SIMPLER than the whole task and genuinely needed.\n"
    "- Order them so earlier subtasks are useful for later ones.\n"
    "- Make them specific to THIS task; avoid vague meta-steps like 'understand the "
    "problem' or 'write the answer'.\n"
    "Output a numbered list of the subtasks only, nothing else."
)


def _decompose_user(x: str) -> str:
    return (f"Task:\n{x}\n\nBreak this into the 3-5 essential, simpler subtasks that "
            "together solve it, ordered so earlier ones help later ones.")


# ---- Pass 2: weak expert solves each subtask (in the context of the task) ----
SOLVE_SUB_SYS = (
    "You are solving ONE subtask that is part of a larger task. Answer the subtask "
    "fully and correctly, using the larger task only as context. Be concise and "
    "self-contained — output only the answer to this subtask."
)


def _solve_user(x: str, q: str) -> str:
    return f"Larger task (context):\n{x}\n\nSubtask to solve now:\n{q}\n\nAnswer this subtask."


# ---- Pass 3: compose the sub-answers into the final whole answer -------------
COMPOSE_SYS = (
    "You are an expert problem solver. You are given a task, and a set of subtasks "
    "that have each already been solved. Combine the subtask solutions into ONE "
    "coherent, complete final answer to the original task: integrate them, resolve "
    "any contradictions, fix obvious subtask errors, and do not be unnecessarily "
    "verbose.\n"
    "Output ONLY the final answer to the original task — do NOT restate the subtasks, "
    "their solutions, or your procedure. If the task has a closed-form answer, put it "
    "within \\boxed{}."
)


def _compose_user(x: str, qa: List[Dict]) -> str:
    body = "\n\n".join(f"[Subtask {i+1}] {d['q']}\n[Solution {i+1}] {d['a']}" for i, d in enumerate(qa))
    return (f"Original task:\n{x}\n\nSolved subtasks:\n{body}\n\n"
            "Combine these into the single best final answer to the original task. "
            "Output ONLY the final answer.")


VERIFY_SYS = (
    "You are a strict evaluator. Given a task, an ORIGINAL answer and a REVISED "
    "answer, decide whether the REVISED answer is better overall. Reply with one "
    "line 'BETTER: yes' or 'BETTER: no'."
)


def _load_pool_rows(in_dir: str, save_name: str, exclude_parquet: str) -> List[Dict]:
    files = sorted(glob.glob(os.path.join(in_dir, f"{save_name}_*_answers.json")))
    if not files:
        raise SystemExit(f"[subtask] FATAL: no pool shards in {in_dir}/{save_name}_*_answers.json")
    rows: List[Dict] = []
    for f in files:
        rows.extend(json.load(open(f)))
    seen = set(); ded = []
    for r in rows:
        q = str(r.get("question", "")).strip()
        if q and q not in seen:
            seen.add(q); ded.append(r)
    train_qs = set()
    if exclude_parquet and os.path.exists(exclude_parquet):
        import pandas as pd
        df = pd.read_parquet(exclude_parquet)
        if "extra_info" in df.columns:
            for ei in df["extra_info"]:
                if isinstance(ei, dict) and "question" in ei:
                    train_qs.add(str(ei["question"]).strip())
    return [r for r in ded if str(r.get("question", "")).strip() not in train_qs]


def _nonempty(s) -> bool:
    return isinstance(s, str) and len(s.strip()) > 0


def _coherence(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    printable = sum(1 for c in s if (32 <= ord(c) < 127) or c in "\n\t")
    ascii_ratio = printable / len(s)
    if len(s) < 8:
        distinct = 1.0
    else:
        grams = [s[i:i + 4] for i in range(len(s) - 3)]
        distinct = len(set(grams)) / len(grams)
    return ascii_ratio * distinct


def _subtasks(text: str, k: int) -> List[str]:
    items = re.findall(r'(?m)^\s*[-*]?\s*\d+[\.\)]\s*(.+)$', text)
    if not items:
        items = re.findall(r'(?m)^\s*[-*•]\s*(.+)$', text)
    items = [it.strip() for it in items if it.strip()]
    return items[:k]


def _vllm(model_path, sys_prompt, users, *, seed, max_tokens, temperature, top_p, mem, n=1):
    import vllm
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    llm = vllm.LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=mem,
                   seed=int(seed), enforce_eager=True, trust_remote_code=True)
    sp = vllm.SamplingParams(n=n, max_tokens=max_tokens, temperature=temperature,
                             top_p=top_p, stop_token_ids=[tok.eos_token_id])
    chats = [([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + [{"role": "user", "content": u}] for u in users]
    if tok.chat_template:
        prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                           add_special_tokens=True) for c in chats]
    else:
        prompts = ["\n".join(f"{m['role']}: {m['content']}" for m in c) for c in chats]
    res = llm.generate(prompts, sampling_params=sp, use_tqdm=True)
    if n == 1:
        out = [r.outputs[0].text if r.outputs else "" for r in res]
    else:
        out = [[o.text for o in r.outputs] for r in res]
    del llm
    return out


# MUST match question_filter/gen_answers_rm.py:SOLVER_SYS, so a --fresh_rejected
# y_S0 comes from the same answer distribution family as the original pool.
SOLVER_SYS = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def main(a: argparse.Namespace) -> None:
    rng = random.Random(a.seed)
    rows = _load_pool_rows(a.in_dir, a.save_name, a.exclude_parquet)
    print(f"[subtask] {len(rows)} source Qs", flush=True)
    rng.shuffle(rows)
    target = int(a.target_pairs)
    solver_model = a.solver_model or a.model

    # One-shot solver answer (rejected): taken from the pool, or regenerated by
    # the trained solver under --fresh_rejected so y_S0 is not a stale
    # solver_{t-1} response. Selection (coherence-ranked top half, random pick)
    # is identical either way.
    items = []
    for r in rows:
        ans = [s for s in (r.get("answers", []) or []) if _nonempty(s)]
        if not ans:
            continue
        items.append({"question": r["question"], "_pool_answers": ans})
        if len(items) >= int(target * a.oversample):
            break
    print(f"[subtask] {len(items)} candidate Qs", flush=True)

    if a.fresh_rejected:
        print(f"[subtask] FRESH rejected mode: regenerating n={a.n_solver_samples} "
              f"one-shot answers per Q with trained solver {solver_model}", flush=True)
        fresh = _vllm(solver_model, SOLVER_SYS, [it["question"] for it in items],
                      seed=a.seed, max_tokens=a.max_tokens, temperature=1.0,
                      top_p=0.99, mem=a.gpu_mem_util, n=int(a.n_solver_samples))
        for it, ans in zip(items, fresh):
            it["_pool_answers"] = [s for s in ans if _nonempty(s)]
        n_before = len(items)
        items = [it for it in items if it["_pool_answers"]]
        print(f"[subtask] fresh one-shot answers ready ({len(items)}/{n_before} "
              f"Qs kept)", flush=True)

    for it in items:
        ranked = sorted(it.pop("_pool_answers"), key=_coherence, reverse=True)
        it["y_s0"] = rng.choice(ranked[:max(3, len(ranked) // 2)])

    # Pass 1: decompose into subtasks (challenger)
    dec = _vllm(a.model, DECOMPOSE_SYS, [_decompose_user(it["question"]) for it in items],
                seed=a.seed, max_tokens=320, temperature=a.temperature, top_p=a.top_p, mem=a.gpu_mem_util)
    for it, d in zip(items, dec):
        it["subtasks"] = _subtasks(d.strip(), int(a.max_subtasks))
    items = [it for it in items if it["subtasks"]]
    print(f"[subtask] {len(items)} decomposed (>=1 subtask)", flush=True)

    # Pass 2: weak expert solves each subtask (flatten all (task, subtask) pairs)
    flat_users, flat_idx = [], []
    for i, it in enumerate(items):
        for q in it["subtasks"]:
            flat_users.append(_solve_user(it["question"], q)); flat_idx.append((i, q))
    sub_ans = _vllm(solver_model, SOLVE_SUB_SYS, flat_users,
                    seed=a.seed, max_tokens=a.sub_max_tokens, temperature=a.temperature,
                    top_p=a.top_p, mem=a.gpu_mem_util)
    qa_by_item = {i: [] for i in range(len(items))}
    for (i, q), ans in zip(flat_idx, sub_ans):
        qa_by_item[i].append({"q": q, "a": ans.strip()})
    for i, it in enumerate(items):
        it["qa"] = qa_by_item[i]

    # Pass 3: compose sub-answers into the final whole answer (challenger)
    comp = _vllm(a.model, COMPOSE_SYS, [_compose_user(it["question"], it["qa"]) for it in items],
                 seed=a.seed, max_tokens=a.max_tokens, temperature=a.temperature,
                 top_p=a.top_p, mem=a.gpu_mem_util)
    for it, y in zip(items, comp):
        it["y_amp"] = y.strip()

    cand = [it for it in items if _nonempty(it.get("y_amp")) and it["y_amp"].strip() != it["y_s0"].strip()]
    print(f"[subtask] {len(cand)} amplified (after dedup)", flush=True)

    if a.verify:
        def _vuser(it):
            return (f"Task:\n{it['question']}\n\nORIGINAL answer:\n{it['y_s0']}\n\n"
                    f"REVISED answer:\n{it['y_amp']}\n\nIs REVISED better overall? Reply 'BETTER: yes/no'.")
        verdicts = _vllm(a.model, VERIFY_SYS, [_vuser(it) for it in cand],
                         seed=a.seed, max_tokens=64, temperature=0.0, top_p=1.0, mem=a.gpu_mem_util)
        kept = [it for it, v in zip(cand, verdicts) if re.search(r"BETTER:\s*yes", v, re.I)]
        print(f"[subtask] verify filter: {len(kept)}/{len(cand)} kept", flush=True)
        cand = kept

    kept = cand[:target]
    pairs = [{"question": it["question"], "chosen": it["y_amp"], "rejected": it["y_s0"]} for it in kept]
    if len(pairs) < target:
        raise ValueError(f"[subtask] FATAL: only {len(pairs)} < target {target}. "
                         "Raise --oversample, or relax --verify.")
    os.makedirs(os.path.dirname(a.out_path) or ".", exist_ok=True)
    json.dump(pairs, open(a.out_path, "w"))
    json.dump([{"question": it["question"],
                "subtasks": [d["q"] for d in it["qa"]]} for it in kept],
              open(a.out_path + ".subtasks.json", "w"), indent=2)
    nsub = [len(it["qa"]) for it in kept]
    json.dump({"policy": "IDA subtask-amplification (decompose->solve subtasks->compose)",
               "n_pairs": len(pairs), "max_subtasks": a.max_subtasks, "verify": bool(a.verify),
               "subtasks_per_task_mean": (sum(nsub) / len(nsub)) if nsub else 0,
               "chosen_len_mean": sum(len(p["chosen"]) for p in pairs) / len(pairs),
               "rejected_len_mean": sum(len(p["rejected"]) for p in pairs) / len(pairs),
               "samples": [{"q": it["question"][:200],
                            "subtasks": [d["q"][:120] for d in it["qa"]],
                            "chosen_head": it["y_amp"][:220],
                            "rejected_head": it["y_s0"][:160]} for it in kept[:6]]},
              open(a.out_path + ".meta.json", "w"), indent=2)
    print(f"[subtask] OK wrote {len(pairs)} -> {a.out_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="challenger HF path (decomposer/composer)")
    p.add_argument("--solver_model", default="", help="weak-expert HF path for subtask solving (default: --model)")
    p.add_argument("--fresh_rejected", action="store_true",
                   help="regenerate the one-shot rejected answer y_S0 with "
                        "--solver_model (the trained solver) instead of "
                        "reusing the pre-training pool answer")
    p.add_argument("--n_solver_samples", type=int, default=10,
                   help="fresh one-shot answers per Q in --fresh_rejected mode")
    p.add_argument("--in_dir", required=True)
    p.add_argument("--save_name", required=True)
    p.add_argument("--exclude_parquet", default="")
    p.add_argument("--out_path", required=True)
    p.add_argument("--target_pairs", type=int, default=256)
    p.add_argument("--max_subtasks", type=int, default=5, help="cap subtasks per task")
    p.add_argument("--oversample", type=float, default=1.8)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=4096, help="compose pass budget")
    p.add_argument("--sub_max_tokens", type=int, default=1024, help="per-subtask answer budget")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    main(p.parse_args())
