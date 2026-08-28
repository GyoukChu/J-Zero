#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Challenger-negative RM pair builder.

Pairing policy:
  - chosen   = one random solver answer from the TOP-QUARTILE, ranked by an
               RM-independent coherence score, of that Q's answers. A weak
               solver emits language-drift / repetition garbage that can be
               worse than the challenger's answer, so picking from ALL answers
               would flip the intended label on many pairs.
  - rejected = one answer from the *challenger* model asked to SOLVE the same
               question, fed the same SOLVER_SYS prompt the solver gets. The
               challenger is trained to generate questions, not answer them,
               so its answer is off-distribution and reliably weaker.

The RM is deliberately not used to select pairs: Skywork-Reward-V2 saturates
(it rewards largely-wrong answers highly), so any selection needing the RM to
already separate two solver answers collapses as the loop progresses. A
guaranteed-bad negative keeps the signal independent of the RM's calibration.

NO RM scoring, NO RM HTTP servers. Emits exactly `--target_pairs` pairs of
{"question", "chosen", "rejected"} — the schema rm_train_loop.py consumes.

Pipeline
--------
  0. Load pool (<in_dir>/<save_name>_*_answers.json), dedup by question text.
  1. Exclude Qs that trained solver_t this iter
     (<exclude_parquet>.extra_info.question), so RM and solver never train on
     the same Qs.
  2. Seeded shuffle; take ceil(target * oversample) candidate Qs having >=1
     solver answer.
  3. vLLM-generate 1 challenger answer per candidate Q (SOLVER_SYS prompt,
     temperature=1.0 / top_p=0.99 — same sampling family as gen_answers_rm.py).
  4. Pair them, skipping empty or degenerate (chosen == rejected) sides, and
     keep the first `target` survivors.
  5. Hard-fail below `target`, so a short pool never yields an under-sized RM
     update.

Writes a sidecar `<out_path>.meta.json` with counts and a few truncated
samples, since the challenger-as-solver output is off-distribution and worth
eyeballing.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from typing import Dict, List

# MUST match question_filter/gen_answers_rm.py:SOLVER_SYS byte for byte, so the
# challenger is fed the identical "as if asking the solver" context.
SOLVER_SYS = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


# ---------------------------------------------------------------------------
# Pool loader + training-Qs excluder.
# ---------------------------------------------------------------------------

def _load_pool(in_dir: str, save_name: str) -> List[Dict]:
    pattern = os.path.join(in_dir, f"{save_name}_*_answers.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[chalneg_pairs] FATAL: no shards matching {pattern}")
    rows: List[Dict] = []
    for f in files:
        with open(f) as fh:
            rows.extend(json.load(fh))
    return rows


def _load_train_qs(parquet_path: str) -> set:
    """Set of stripped question strings that trained solver_t this iter, read
    from the final parquet's extra_info column."""
    if not parquet_path or not os.path.exists(parquet_path):
        print(f"[chalneg_pairs] WARN parquet not found at {parquet_path}; "
              "no exclusion applied", flush=True)
        return set()
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit(
            f"[chalneg_pairs] FATAL: pandas required to read parquet: {e}")
    df = pd.read_parquet(parquet_path)
    qs: set = set()
    if "extra_info" in df.columns:
        for ei in df["extra_info"]:
            if isinstance(ei, dict) and "question" in ei:
                qs.add(str(ei["question"]).strip())
    if not qs and "prompt" in df.columns:
        import numpy as np
        for p in df["prompt"]:
            try:
                if isinstance(p, np.ndarray):
                    p = list(p)
                if isinstance(p, list):
                    for turn in p:
                        if isinstance(turn, dict) and turn.get("role") == "user":
                            qs.add(str(turn.get("content", "")).strip())
                            break
            except Exception:
                continue
    return qs


def _dedup_keep_order(rows: List[Dict]) -> List[Dict]:
    seen: set = set()
    out: List[Dict] = []
    for r in rows:
        q = str(r.get("question", "")).strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(r)
    return out


def _nonempty(s) -> bool:
    return isinstance(s, str) and len(s.strip()) > 0


def _coherence(s: str) -> float:
    """RM-independent 'not-garbage' score in [0, 1] for a solver answer.

    score = printable_ascii_ratio * distinct_char_4gram_ratio, catching the two
    dominant garbage modes of weak solvers: language drift / unicode spam (low
    ASCII ratio) and repetition spam (low distinct-4gram ratio). Both factors
    sit near 1.0 for genuine English, so the product ranks coherent answers up.
    """
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


# ---------------------------------------------------------------------------
# vLLM generation helper, used for both the optional fresh solver answers and
# the challenger-as-solver answers. Frees the model on return so two models can
# run sequentially in one process (same pattern as build_pairs_subtask._vllm).
# ---------------------------------------------------------------------------

def _vllm_generate(model_path: str, questions: List[str], *, n: int, seed: int,
                   max_tokens: int, temperature: float, top_p: float,
                   mem: float) -> List[List[str]]:
    import vllm
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = vllm.LLM(
        model=model_path,
        tokenizer=model_path,
        gpu_memory_utilization=mem,
        seed=int(seed),
    )
    sp = vllm.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=[tokenizer.eos_token_id],
        n=n,
    )
    chats = [
        [
            {"role": "system", "content": SOLVER_SYS},
            {"role": "user", "content": q},
        ]
        for q in questions
    ]
    if tokenizer.chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=True,
                add_special_tokens=True)
            for c in chats
        ]
    else:
        prompts = ["\n".join(f"{m['role']}: {m['content']}" for m in c)
                   for c in chats]
    responses = llm.generate(prompts, sampling_params=sp, use_tqdm=True)
    out = [[o.text for o in r.outputs] for r in responses]
    del llm
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    t0 = time.time()
    rng = random.Random(args.seed)

    # 0) Load + dedup pool ---------------------------------------------------
    pool_rows = _load_pool(args.in_dir, args.save_name)
    print(f"[chalneg_pairs] loaded {len(pool_rows)} (Q, A) rows from shards "
          f"({args.save_name}_*_answers.json in {args.in_dir})", flush=True)
    pool_rows = _dedup_keep_order(pool_rows)
    print(f"[chalneg_pairs] {len(pool_rows)} unique Qs after dedup", flush=True)

    # 1) Exclude solver-training Qs -----------------------------------------
    train_qs = _load_train_qs(args.exclude_parquet)
    print(f"[chalneg_pairs] {len(train_qs)} solver-training Qs to exclude",
          flush=True)

    remaining: List[Dict] = []
    n_excluded = 0
    for r in pool_rows:
        q = str(r.get("question", "")).strip()
        if q in train_qs:
            n_excluded += 1
            continue
        answers = [a for a in (r.get("answers", []) or []) if _nonempty(a)]
        if not answers:
            continue   # `chosen` needs at least one non-empty solver answer
        remaining.append({"question": q, "answers": answers})

    if train_qs:
        coverage_pct = 100.0 * n_excluded / max(1, len(train_qs))
        msg = (f"[chalneg_pairs] exclusion match: {n_excluded}/{len(train_qs)} "
               f"train_qs found in pool ({coverage_pct:.1f}%)")
        if coverage_pct < 80.0:
            print("[chalneg_pairs] WARN low train-Q coverage "
                  f"({coverage_pct:.1f}%) — exclude_parquet may be misaligned. "
                  + msg, flush=True)
        else:
            print(msg, flush=True)
    print(f"[chalneg_pairs] pool size after exclude+filter: {len(remaining)} Qs",
          flush=True)
    if not remaining:
        raise ValueError("[chalneg_pairs] FATAL: pool empty after excluding "
                         "solver-training Qs")

    target = int(args.target_pairs)
    # 2) Shuffle + take an oversampled candidate buffer ----------------------
    rng.shuffle(remaining)
    n_cand = min(len(remaining), max(target, math.ceil(target * args.oversample)))
    candidates = remaining[:n_cand]
    print(f"[chalneg_pairs] selected {len(candidates)} candidate Qs "
          f"(target={target}, oversample={args.oversample})", flush=True)
    if len(candidates) < target:
        raise ValueError(
            f"[chalneg_pairs] FATAL: only {len(candidates)} candidate Qs < "
            f"target {target}; pool too small after exclusion.")

    questions = [c["question"] for c in candidates]

    # 2b) OPTIONAL fresh chosen-side answers from the TRAINED solver ---------
    # With --solver_model, solver_t regenerates n answers per candidate Q using
    # gen_answers_rm.py's prompt + sampling family, and the top-quartile
    # selection below runs on those instead of the filter-stage pool (which
    # came from the pre-training solver_{t-1}).
    if args.solver_model:
        print(f"[chalneg_pairs] FRESH chosen mode: regenerating "
              f"n={args.n_solver_samples} solver answers per Q with trained "
              f"solver {args.solver_model} (pool answers ignored for chosen)",
              flush=True)
        fresh = _vllm_generate(
            args.solver_model, questions, n=int(args.n_solver_samples),
            seed=int(args.seed), max_tokens=args.max_tokens,
            temperature=1.0, top_p=0.99, mem=args.gpu_mem_util)
        n_empty_rows = 0
        for cand, ans in zip(candidates, fresh):
            ans = [a for a in ans if _nonempty(a)]
            if not ans:
                n_empty_rows += 1
            cand["answers"] = ans
        candidates = [c for c in candidates if c["answers"]]
        print(f"[chalneg_pairs] fresh solver answers ready "
              f"({len(candidates)} Qs kept, {n_empty_rows} all-empty dropped)",
              flush=True)
        questions = [c["question"] for c in candidates]

    # 3) vLLM: 1 challenger answer per candidate Q (SOLVER_SYS prompt) -------
    print(f"[chalneg_pairs] generating {len(questions)} challenger answers "
          f"(temp={args.temperature} top_p={args.top_p} n=1)", flush=True)
    chal_answers = [
        outs[0] if outs else ""
        for outs in _vllm_generate(
            args.challenger_model, questions, n=1, seed=int(args.seed),
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_p=args.top_p, mem=args.gpu_mem_util)
    ]

    # 4) Build pairs: chosen=random solver answer, rejected=challenger answer -
    pairs: List[Dict] = []
    chosen_cohs: List[float] = []
    n_skip_empty_chal = 0
    n_skip_tied = 0
    top_frac = float(args.chosen_top_frac)
    for cand, chal in zip(candidates, chal_answers):
        if not _nonempty(chal):
            n_skip_empty_chal += 1
            continue
        # Pair the challenger answer against a clean solver attempt, not a
        # garbage one: pick randomly within the top coherence fraction.
        ranked = sorted(cand["answers"], key=_coherence, reverse=True)
        k = max(1, math.ceil(len(ranked) * top_frac))
        chosen = rng.choice(ranked[:k])
        if not _nonempty(chosen):
            n_skip_tied += 1
            continue
        if chosen.strip() == chal.strip():
            n_skip_tied += 1
            continue
        pairs.append({
            "question": cand["question"],
            "chosen": chosen,
            "rejected": chal,
        })
        chosen_cohs.append(_coherence(chosen))
        if len(pairs) >= target:
            break

    print(f"[chalneg_pairs] built {len(pairs)} pairs "
          f"(chosen=random from top {top_frac:.0%} of solver answers by "
          f"coherence; skipped_empty_challenger={n_skip_empty_chal} "
          f"skipped_tied={n_skip_tied})", flush=True)

    # 5) Decide: success or short ------------------------------------------
    if len(pairs) < target:
        raise ValueError(
            f"[chalneg_pairs] FATAL: only {len(pairs)} valid pairs < target "
            f"{target}. skipped_empty_challenger={n_skip_empty_chal} "
            f"skipped_tied={n_skip_tied}. Raise --oversample or check that the "
            "challenger model produces non-empty answers.")

    pairs = pairs[:target]
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as fh:
        json.dump(pairs, fh)

    rej_cohs = [_coherence(p["rejected"]) for p in pairs]
    n_chosen_more_coherent = sum(1 for cc, rc in zip(chosen_cohs, rej_cohs)
                                 if cc >= rc)
    meta = {
        "policy": "challenger_negative",
        "solver_model": args.solver_model,
        "chosen_source": ("fresh trained-solver answers" if args.solver_model
                          else "pre-training filter-stage pool"),
        "chosen_selection": f"random from top {top_frac:.0%} of solver answers by coherence",
        "chosen_top_frac": top_frac,
        "target_pairs": target,
        "n_pairs": len(pairs),
        "n_unique_qs_pool": len(pool_rows),
        "n_excluded_train_qs": n_excluded,
        "n_remaining_after_exclude": len(remaining),
        "n_candidates": len(candidates),
        "skipped_empty_challenger": n_skip_empty_chal,
        "skipped_tied": n_skip_tied,
        "challenger_model": args.challenger_model,
        "seed": args.seed,
        "chosen_coherence_mean": (sum(chosen_cohs) / max(1, len(chosen_cohs))),
        "rejected_coherence_mean": (sum(rej_cohs) / max(1, len(rej_cohs))),
        "n_chosen_coherence_ge_rejected": n_chosen_more_coherent,
        "samples": [
            {
                "question": p["question"][:500],
                "chosen_head": p["chosen"][:300],
                "rejected_head": p["rejected"][:300],
                "chosen_coh": round(_coherence(p["chosen"]), 3),
                "rejected_coh": round(_coherence(p["rejected"]), 3),
            }
            for p in pairs[:8]
        ],
    }
    meta_path = args.out_path + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    chosen_len = [len(p["chosen"]) for p in pairs]
    rej_len = [len(p["rejected"]) for p in pairs]
    def _mean(x): return sum(x) / max(1, len(x))
    print(f"[chalneg_pairs] OK: wrote {len(pairs)} pairs -> {args.out_path}",
          flush=True)
    print(f"[chalneg_pairs] char-len: chosen mean={_mean(chosen_len):.0f} "
          f"rejected mean={_mean(rej_len):.0f}; meta -> {meta_path} "
          f"(elapsed={time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, required=True,
                   help="dir containing <save_name>_*_answers.json (this "
                        "iter's solver-train filtered/ pool)")
    p.add_argument("--save_name", type=str, required=True,
                   help="solver-side EXP_NAME, e.g. iter1_solver_chalneg_full")
    p.add_argument("--exclude_parquet", type=str, default="",
                   help="parquet whose extra_info.question lists the Qs that "
                        "trained solver_t; removed from the pool.")
    p.add_argument("--out_path", type=str, required=True,
                   help="output JSON of {question, chosen, rejected} pairs")
    p.add_argument("--challenger_model", type=str, required=True,
                   help="HF path/dir of challenger_t — generates the rejected "
                        "(challenger-as-solver) answers")
    p.add_argument("--solver_model", type=str, default="",
                   help="HF path/dir of the TRAINED solver_t. When set, the "
                        "chosen side is regenerated fresh (n_solver_samples "
                        "per Q, gen_answers_rm sampling) instead of reusing "
                        "the pre-training pool answers.")
    p.add_argument("--n_solver_samples", type=int, default=10,
                   help="fresh solver answers per Q when --solver_model is set")
    p.add_argument("--target_pairs", type=int,
                   default=int(os.getenv("RZERO_RM_MIN_PAIRS", "256")),
                   help="exact number of pairs to emit (rm_train.sh passes RZERO_RM_N_CHAL)")
    p.add_argument("--oversample", type=float, default=1.5,
                   help="candidate buffer = ceil(target * oversample) so a few "
                        "empty/tied challenger answers don't short the build")
    p.add_argument("--chosen_top_frac", type=float, default=0.25,
                   help="chosen is a random pick among the top this-fraction of "
                        "the Q's solver answers ranked by coherence (default "
                        "0.25 = top quartile). 1.0 reproduces random-from-all.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.99)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    main(p.parse_args())
