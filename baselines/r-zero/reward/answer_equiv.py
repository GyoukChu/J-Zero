# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
Boxed-answer extraction + answer equivalence + self-consistency clustering.

Shared by the two places that need a majority vote over solver samples:

  - `vllm_service/start_vllm_server.py`  (challenger reward: p_hat per question)
  - `question_filter/evaluate.py`        (delta-band filter + pseudo-label)

Keeping both call sites here guarantees they agree with the grader used
everywhere else in the recipe, verl's math_verify. mathruler is not a dependency
anywhere; its boxed-extraction behavior is reproduced below.

Per-comparison timeout via RZERO_CLUSTER_GRADER_TIMEOUT (default 10s).
"""

from __future__ import annotations

import functools
import inspect
import os
from typing import Dict, List, Tuple

GRADER_TIMEOUT = float(os.getenv("RZERO_CLUSTER_GRADER_TIMEOUT", "10"))

# INVARIANT: nothing in this module may grade at import time. verl's math_verify
# grades in a *spawn* ProcessPoolExecutor whose children re-import the parent's
# __main__, so a grading call during import makes each child spawn its own pool.
from verl.utils.reward_score.math_verify import compute_score as _verl_mv_score

# Older verl has no per-call `timeout` kwarg; degrade quietly to its default
# rather than raising TypeError.
_VERL_MV_TIMEOUT_KWARG = "timeout" in inspect.signature(_verl_mv_score).parameters

_BOXED_TOKEN = "\\boxed{"


def extract_boxed_content(text: str) -> str:
    """Content of the LAST `\\boxed{...}` in `text`, brace-counted, stripped.

    Drop-in replacement for `mathruler.grader.extract_boxed_content`, including
    its return value when nothing parses: the *string* ``"None"``, not "". Do not
    "fix" that — R-Zero relies on it. Unparseable samples cluster together under
    the literal answer "None" and count toward the p_hat denominator; returning
    "" would drop them, shifting every p_hat and the delta-band survivors.
    """
    start_pos = text.rfind(_BOXED_TOKEN)
    if start_pos == -1:
        return "None"
    content = text[start_pos + len(_BOXED_TOKEN):]
    depth = 0
    for i, char in enumerate(content):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == -1:
            return content[:i].strip()
    return "None"


def _verl_mv_equiv(a: str, b: str) -> bool:
    """verl math_verify on `\\boxed{a}` vs ground truth `b`.

    compute_score wraps `ground_truth` itself but parses `model_output` as-is,
    so `a` must be wrapped here - as in solver_reward.py::_verl_mv_match.
    """
    try:
        if _VERL_MV_TIMEOUT_KWARG:
            score = _verl_mv_score(f"\\boxed{{{a}}}", b, timeout=GRADER_TIMEOUT)
        else:
            score = _verl_mv_score(f"\\boxed{{{a}}}", b)
        return float(score) > 0.5
    except Exception:
        return False


@functools.lru_cache(maxsize=262144)
def answers_equivalent(a: str, b: str) -> bool:
    """True if answer strings `a` and `b` denote the same value.

    Directional (math_verify treats the second argument as ground truth), so
    callers wanting symmetry must try both orders - `cluster_majority` does.

    Memoized because every call is a subprocess round trip and a filter shard's
    ~2000 boxed answers repeat heavily. The grader is deterministic, so the
    cache cannot change results.
    """
    if not a or not b:
        return False
    return _verl_mv_equiv(a, b)


_SELFTEST_DONE = False


def _selftest_once() -> None:
    """Fail loudly if the grader is silently returning "not equivalent".

    math_verify grades in a spawn subprocess pool and swallows every exception
    into a 0.0 score. If that pool cannot start - classically, a caller whose
    __main__ is not import-safe - every answer lands in its own cluster and
    p_hat collapses to 1/n, quietly corrupting both the challenger reward and
    the filter band. Two known pairs turn that into a crash instead.
    """
    global _SELFTEST_DONE
    if _SELFTEST_DONE:
        return
    _SELFTEST_DONE = True  # set first: the calls below must not recurse
    if not _verl_mv_equiv("\\frac{1}{2}", "0.5") or _verl_mv_equiv("2", "3"):
        raise RuntimeError(
            "verl math_verify self-test failed: \\frac{1}{2} vs 0.5 did not "
            "grade as equivalent. The math_verify subprocess pool is probably "
            "not starting - check that the calling script guards its entry "
            "point with `if __name__ == \"__main__\":` and that math-verify is "
            "installed. Refusing to run with a grader that reports everything "
            "as different."
        )


def cluster_majority(boxed: List[str]) -> Tuple[str, float, int]:
    """Cluster extracted answers by equivalence; return (majority, p_hat, n).

    `n` is the number of non-empty entries in `boxed`; `p_hat` is the majority
    cluster's share of those. Returns ("", 0.0, n) when nothing is usable.

    Loop structure is R-Zero's verbatim: cheap string equality, then the "no "
    heuristic, then the grader in both directions, first match wins.
    """
    _selftest_once()
    counts: Dict[str, int] = {}
    n_valid = 0
    for res in boxed:
        if not res:
            continue
        n_valid += 1
        matched = False
        for exist in list(counts.keys()):
            if res == exist or ("no " in res.lower() and "no " in exist.lower()):
                counts[exist] += 1
                matched = True
                break
            if answers_equivalent(res, exist) or answers_equivalent(exist, res):
                counts[exist] += 1
                matched = True
                break
        if not matched:
            counts[res] = 1
    if not counts:
        return "", 0.0, n_valid
    maj = max(counts, key=counts.get)
    return maj, counts[maj] / max(1, n_valid), n_valid
