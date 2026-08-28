#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM-judge recheck on top of a rule-based grader (R-Zero `results_recheck.py`
style) against a SELF-HOSTED judge (vLLM OpenAI-compatible endpoint,
e.g. Qwen3.6-27B with enable_thinking=False). No external API, no key.

Pipeline (per record in `<dataset>.jsonl`):
  1. Base score via verl `math_verify.compute_score` over the raw solution.
  2. If base score < 0.5, pass the FULL raw model solution + ground truth to
     the judge. The prompt instructs it to locate the LAST `\\boxed{...}`
     itself and judge equivalence. "Yes" -> upgrade to 1.

Outputs `<dataset>_rechecked.jsonl` (per-record verdict trail) and
`recheck_summary.json` (per-dataset totals + comparison to reported).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as cf_wait, FIRST_COMPLETED
from typing import Dict, List

# Latched when the local vLLM judge is presumed deadlocked (vllm v1 EngineCore
# 0-tok/s stall). Once set, every later local_judge call in this run — including
# later datasets — short-circuits to the base score instead of hanging.
_JUDGE_DEAD = threading.Event()

# verl must be importable: installed editable into the venv, or via VERL_REPO.
VERL_REPO = os.environ.get("VERL_REPO")
if VERL_REPO:
    sys.path.insert(0, VERL_REPO)

# Used only against the self-hosted OpenAI-compatible vLLM endpoint; no key.
from openai import OpenAI                      # noqa: E402

from verl.utils.reward_score.math_verify import compute_score as verl_mv  # noqa: E402


REPORTED = {"math500": 68.20, "minervamath": 38.24, "olympiadbench": 41.04}

# We never extract the boxed answer ourselves: the judge sees the full raw
# solution and is instructed to look at the LAST \boxed{} in it.

# Prompt modified from xVerify (https://arxiv.org/pdf/2504.10481) Figure 15
_SYSTEM_PROMPT_NEW = "You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output solution, and the correct answer. Your task is to determine if the output solution accurately answers the question based on the provided correct answer. Respond with either 'Yes' or 'No'."

# Exact R-Zero original prompt (results_recheck.py), reproduced verbatim for a
# faithful comparison — including its semantically swapped template variables
# (`answer` holds the ground truth, `response` the model output).
_SYSTEM_PROMPT_RZERO = "You are a math answer checker."


def _user_text_new(question: str, gt: str, full_solution: str) -> str:
    # Missing-boxed cases never reach here — process_dataset's has_boxed
    # pre-filter handles them locally — so the prompt needs no branch for them.
    question_suffixed = question + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    return (
        "Determine whether the final answer of output solution is mathematically equivalent to the ground truth. "
        "Reply with only 'Yes' or 'No'."
        f"Question: \"\"\"{question_suffixed}\"\"\"\n\n"
        f"Output solution: \"\"\"{full_solution}\"\"\"\n\n"
        f"Ground truth: \"\"\"{gt}\"\"\""
    )


def _user_text_rzero(question: str, gt: str, full_solution: str) -> str:
    # Verbatim R-Zero results_recheck.py; `question` is unused because their
    # prompt does not include it.
    return (
        f"Hi, there is a answer: {gt}\n\n"
        f", and the ground truth answer is: {full_solution}\n\n"
        f", please check whether the answer is correct or not, and return the **only** Yes or No."
    )


_PROMPT_STYLES = {
    "new":            (_SYSTEM_PROMPT_NEW,   _user_text_new),
    "rzero_original": (_SYSTEM_PROMPT_RZERO, _user_text_rzero),
}


def has_boxed(text: str) -> bool:
    """Depth-balanced check for at least one well-formed \\boxed{...}.

    Local pre-filter: skips the LLM call when there is no boxed answer to judge.
    """
    marker = r"\boxed{"
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            return False
        depth = 1
        i = idx + len(marker)
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return True
            i += 1
        start = idx + len(marker)


# ---------------------------------------------------------------------------
# Self-hosted judge (vLLM OpenAI-compatible server). A recheck "fails" when the
# model hits max_tokens without finishing or emits no parseable Yes/No; those
# retry up to `max_retries` times, then count as a wrong answer.
# ---------------------------------------------------------------------------
_local_client_obj = None
_local_lock = threading.Lock()


def _local_client(base_url: str):
    global _local_client_obj
    with _local_lock:
        if _local_client_obj is None:
            _local_client_obj = OpenAI(base_url=base_url, api_key="dummy-local-key")
    return _local_client_obj


def _call_local(base_url: str, model_name: str, prompt_style: str,
                question: str, gt: str, full_solution: str, sp: Dict):
    """One chat.completions call to the local vLLM judge.
    Returns (verdict_text, finish_reason)."""
    sys_prompt, user_fn = _PROMPT_STYLES[prompt_style]
    user_text = user_fn(question, gt, full_solution)
    resp = _local_client(base_url).chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=sp["temperature"],
        top_p=sp["top_p"],
        max_tokens=sp["max_tokens"],
        presence_penalty=sp["presence_penalty"],
        # top_k is a vLLM extra. enable_thinking=False is redundant with the
        # nothink chat template the server is launched with.
        extra_body={"top_k": sp["top_k"],
                    "chat_template_kwargs": {"enable_thinking": False}},
        timeout=sp.get("timeout", 180),
    )
    choice = resp.choices[0]
    txt = (choice.message.content or "").strip()
    finish = getattr(choice, "finish_reason", None)
    return txt, finish


def classify_local_verdict(text: str, finish_reason) -> str:
    """Map a local-judge response to 'yes' | 'no' | 'degenerate'.

    'degenerate' = no usable verdict (hit the token cap, or nothing parseable).
    It triggers a retry; an exhausted budget falls back to a wrong answer.
    """
    if finish_reason == "length":
        return "degenerate"
    v = (text or "").strip().lower()
    if not v:
        return "degenerate"
    # Exact first-token match first, then a whole-WORD search. The word
    # boundaries are load-bearing: a substring match would read 'Now'/'None'/
    # 'cannot' as a 'no' verdict instead of falling through to 'degenerate'.
    toks = v.split()
    first = re.sub(r"[^a-z]", "", toks[0]) if toks else ""
    if first == "yes":
        return "yes"
    if first == "no":
        return "no"
    has_yes = re.search(r"\byes\b", v) is not None
    has_no = re.search(r"\bno\b", v) is not None
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "degenerate"


def local_judge(base_url: str, model_name: str, prompt_style: str,
                question: str, gt: str, model_resp: str,
                sp: Dict, max_retries: int = 2):
    """Run the local judge with degenerate-retry.

    Returns (decision, raw_text), decision in 'yes' | 'no' | 'degenerate'. Never
    raises: connection errors surviving the retries also become 'degenerate', so
    the sample stays wrong (matching R-Zero's recheck-reject behavior).
    """
    last_txt = ""
    last_err = None
    backoff = 2.0
    for _ in range(max_retries + 1):
        try:
            txt, finish = _call_local(base_url, model_name, prompt_style,
                                      question, gt, model_resp, sp)
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff *= 2
            continue
        last_txt = txt
        decision = classify_local_verdict(txt, finish)
        if decision in ("yes", "no"):
            return decision, txt
        # degenerate → retry with a fresh sample (temperature > 0)
    if last_err is not None and not last_txt:
        return "degenerate", f"[LOCAL_ERROR: {last_err}]"
    return "degenerate", (last_txt or "[DEGENERATE]")


# ---------------------------------------------------------------------------
# Base graders
# ---------------------------------------------------------------------------
def base_score_verl_mv(raw: str, gt: str) -> float:
    return float(verl_mv(raw, gt))


BASE_GRADERS = {"verl_mv": base_score_verl_mv}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_dataset(
    in_path: str,
    out_path: str,
    model_name: str,
    base_grader: str,
    prompt_style: str,
    max_workers: int,
    skip_judge: bool = False,
    local_base_url: str = None,
    sp: Dict = None,
    local_max_retries: int = 2,
    recheck_stall_secs: float = 150.0,
) -> Dict:
    grader_fn = BASE_GRADERS[base_grader]
    records: List[Dict] = []
    with open(in_path) as fh:
        for line in fh:
            records.append(json.loads(line))

    n = len(records)
    base_correct = 0
    needs_recheck: List[int] = []
    for i, r in enumerate(records):
        bs = grader_fn(r["raw"], r["ground_truth"])
        r["base_score"] = float(bs)
        r["base_grader"] = base_grader
        if bs > 0.5:
            r["final_score"] = 1.0
            r["recheck_reason"] = f"{base_grader}_pass"
            base_correct += 1
        else:
            r["final_score"] = 0.0
            r["recheck_reason"] = "pending"
            needs_recheck.append(i)
    print(
        f"  [{os.path.basename(in_path)}] base {base_grader}: {base_correct}/{n} "
        f"({base_correct/n*100:.2f}%); recheck queue: "
        f"{0 if skip_judge else len(needs_recheck)} (skip_judge={skip_judge})"
    )

    n_upgraded = 0
    n_no_boxed = 0
    n_degenerate = 0
    t0 = time.time()
    done = 0

    def _do_local(i: int):
        r = records[i]
        if _JUDGE_DEAD.is_set():
            return i, "[JUDGE_DEAD]", "judge_dead"
        # No well-formed \boxed{} means nothing to upgrade; skip the LLM call.
        if not has_boxed(r["raw"] or ""):
            return i, "[NO_BOXED]", "no_boxed"
        decision, raw_txt = local_judge(
            local_base_url,
            model_name,
            prompt_style,
            r["question"],
            str(r["ground_truth"]),
            r["raw"],
            sp,
            local_max_retries,
        )
        return i, raw_txt, decision

    if needs_recheck and not skip_judge:
        def _handle_local(fut):
            nonlocal done, n_upgraded, n_no_boxed, n_degenerate
            done += 1
            idx = fut_map[fut]
            try:
                i, verdict_text, decision = fut.result()
            except Exception as e:
                # local_judge swallows its own errors; this is defensive.
                records[idx]["judge_verdict"] = f"[ERROR: {e}]"
                records[idx]["recheck_reason"] = "local_error"
                n_degenerate += 1
            else:
                r = records[i]
                r["judge_verdict"] = verdict_text
                if decision == "no_boxed":
                    r["recheck_reason"] = "no_boxed"
                    n_no_boxed += 1
                elif decision == "yes":
                    r["final_score"] = 1.0
                    r["recheck_reason"] = "local_upgrade"
                    n_upgraded += 1
                elif decision == "no":
                    r["recheck_reason"] = "local_reject"
                else:  # degenerate / judge_dead: keep original (base) score
                    r["recheck_reason"] = "local_degenerate"
                    n_degenerate += 1
            if done % 25 == 0 or done == len(needs_recheck):
                print(
                    f"    progress: {done}/{len(needs_recheck)} "
                    f"({(time.time()-t0):.0f}s, upgraded={n_upgraded}, "
                    f"no_boxed={n_no_boxed}, degenerate={n_degenerate})"
                )

        # Stall-bounded wait loop: a healthy judge returns each verdict in <10s,
        # so no completion for recheck_stall_secs means the vllm v1 EngineCore
        # has deadlocked. Latch _JUDGE_DEAD, base-fallback the unfinished
        # samples, and bail rather than hanging the run.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_map = {pool.submit(_do_local, i): i for i in needs_recheck}
            pending = set(fut_map)
            last_progress = time.time()
            while pending:
                fin, pending = cf_wait(pending, timeout=30,
                                       return_when=FIRST_COMPLETED)
                if fin:
                    last_progress = time.time()
                    for fut in fin:
                        _handle_local(fut)
                elif _JUDGE_DEAD.is_set() or (time.time() - last_progress) > recheck_stall_secs:
                    if not _JUDGE_DEAD.is_set():
                        _JUDGE_DEAD.set()
                        print(f"    [recheck] JUDGE STALL: no verdict for "
                              f"{recheck_stall_secs:.0f}s → vllm v1 EngineCore presumed "
                              f"deadlocked. Base-fallback {len(pending)} remaining samples; "
                              f"later datasets skip the judge.")
                    for fut in pending:
                        idx = fut_map[fut]
                        fut.cancel()
                        records[idx]["judge_verdict"] = "[JUDGE_STALL]"
                        records[idx]["recheck_reason"] = "judge_stall"
                        n_degenerate += 1
                    pending = set()

    final_correct = sum(1 for r in records if r["final_score"] > 0.5)
    with open(out_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    return {
        "n": n,
        "base_grader": base_grader,
        "base_correct": base_correct,
        "base_pct": base_correct / n * 100,
        "rechecked": len(needs_recheck),
        "upgraded": n_upgraded,
        "no_boxed": n_no_boxed,
        "degenerate": n_degenerate,
        "final_correct": final_correct,
        "final_pct": final_correct / n * 100,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True, help="dir containing <dataset>.jsonl")
    p.add_argument("--out_dir", required=True, help="dir to write rechecked files + summary")
    p.add_argument("--datasets", nargs="+", default=["math500", "minervamath", "olympiadbench"])
    p.add_argument(
        "--base_grader",
        choices=list(BASE_GRADERS.keys()),
        default="verl_mv",
        help="Rule-based first-stage grader (verl_mv = verl math_verify)",
    )
    p.add_argument(
        "--prompt_style",
        choices=list(_PROMPT_STYLES.keys()),
        default="new",
        help="LLM judge prompt template (rzero_original = exact wording from "
        "R-Zero results_recheck.py; new = current xVerify-style prompt)",
    )
    p.add_argument("--skip_judge", action="store_true",
                   help="Compute base-grader score only — no LLM judge calls")
    p.add_argument("--max_workers", type=int, default=8)
    # --- self-hosted judge (Qwen3.6-27B via vLLM) ---
    p.add_argument("--local_base_url", required=True,
                   help="Local vLLM endpoint, e.g. http://127.0.0.1:8404/v1")
    p.add_argument("--judge_model", default="qwen36-judge",
                   help="served-model-name of the local judge")
    p.add_argument("--local_temperature", type=float, default=0.1)
    p.add_argument("--local_top_p", type=float, default=0.8)
    p.add_argument("--local_top_k", type=int, default=20)
    p.add_argument("--local_presence_penalty", type=float, default=1.5)
    p.add_argument("--local_max_tokens", type=int, default=8,
                   help="Output cap for the Yes/No judge. With enable_thinking=False the "
                   "verdict is 1-2 tokens (~99.85%% of samples), so a tiny cap suffices and "
                   "fails fast: a sample that ignores the 'reply Yes/No' instruction and "
                   "rambles hits the cap -> degenerate -> retry -> wrong. "
                   "Large caps (e.g. 32768) only let a runaway decode for minutes; "
                   "low-temp retries do NOT recover deterministic ramblers, so keep this small.")
    p.add_argument("--local_max_retries", type=int, default=2,
                   help="Retries on a degenerate local-judge response (no parseable "
                   "Yes/No or hit max_tokens). After this many retries the sample is "
                   "left as a wrong answer.")
    p.add_argument("--local_timeout", type=float, default=180,
                   help="Per-request timeout (s) to the local judge. Short on purpose: a "
                   "verdict returns in <5s, so a stalled/dead vllm EngineCore fails fast "
                   "instead of blocking for hours.")
    p.add_argument("--recheck_stall_secs", type=float, default=150,
                   help="If NO recheck verdict completes for this many seconds, the local "
                   "vllm judge is presumed deadlocked (v1 EngineCore 0-tok/s stall). The run "
                   "latches it dead, base-fallbacks the remaining samples, and skips the judge "
                   "for later datasets — so a wedged engine degrades to base scores in bounded "
                   "time instead of hanging the whole eval.")
    args = p.parse_args()

    model_name = args.judge_model
    sp = {
        "temperature": args.local_temperature,
        "top_p": args.local_top_p,
        "top_k": args.local_top_k,
        "presence_penalty": args.local_presence_penalty,
        "max_tokens": args.local_max_tokens,
        "timeout": args.local_timeout,
    }

    print(
        f"[recheck] model={model_name}  base_grader={args.base_grader}  "
        f"prompt_style={args.prompt_style}  skip_judge={args.skip_judge}  "
        f"local_sp={sp}  local_max_retries={args.local_max_retries}"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "model": model_name,
        "base_grader": args.base_grader,
        "prompt_style": args.prompt_style,
        "skip_judge": args.skip_judge,
        "datasets": {},
    }
    for ds in args.datasets:
        in_path = os.path.join(args.in_dir, f"{ds}.jsonl")
        out_path = os.path.join(args.out_dir, f"{ds}_rechecked.jsonl")
        if not os.path.isfile(in_path):
            print(f"[skip] {in_path} not found")
            continue
        print(f"[{ds}] start")
        s = process_dataset(in_path, out_path, model_name, args.base_grader,
                            args.prompt_style, args.max_workers,
                            skip_judge=args.skip_judge,
                            local_base_url=args.local_base_url,
                            sp=sp, local_max_retries=args.local_max_retries,
                            recheck_stall_secs=args.recheck_stall_secs)
        reported = REPORTED.get(ds)
        if reported is not None:
            s["reported"] = reported
            s["gap_vs_reported"] = s["final_pct"] - reported
        summary["datasets"][ds] = s
        print(
            f"[{ds}] final: {s['final_correct']}/{s['n']} = {s['final_pct']:.2f}%"
            + (f" (reported {reported:.2f}, gap {s['gap_vs_reported']:+.2f})" if reported else "")
        )

    with open(os.path.join(args.out_dir, "recheck_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
