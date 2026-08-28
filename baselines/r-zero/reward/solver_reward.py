# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
Solver reward: binary verl math_verify grade against the pseudo-label answer.

Multiple \\boxed{} blocks are allowed; only the LAST counts as the model's
answer, matching the LLM-judge convention in `llm_recheck.py`.

The grader, `verl.utils.reward_score.math_verify`, is the same one used by
`reward/answer_equiv.py` and `evaluate/llm_recheck.py`.

Supports two call signatures:
  - per-item (verl experimental NaiveRewardManager): keyword args
    `data_source, solution_str, ground_truth, extra_info`.
  - batched (legacy BatchRewardManager / RZeroBatchManager): keyword args
    `data_sources, solution_strs, ground_truths, extra_infos`.
"""

from __future__ import annotations

import re
from typing import Dict, List, Union

from verl.utils.reward_score.math_verify import compute_score as _verl_mv_score


_BOXED_RE = re.compile(r"\\boxed\s*\{")


def _extract_last_boxed(text: str) -> str:
    """Return the content of the LAST `\\boxed{...}` in `text`, or "".

    Brace-counted parse so nested `{}` (e.g. `\\boxed{\\frac{1}{2}}`) work.
    Tries matches from last to first; the last brace-balanced one wins.
    """
    if not text:
        return ""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return ""
    for m in reversed(matches):
        i = m.end()  # position right after '{'
        depth = 1
        buf: List[str] = []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
                buf.append(c)
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
                buf.append(c)
            else:
                buf.append(c)
            i += 1
        if depth == 0:
            return "".join(buf)
    return ""


def _verl_mv_match(pred_box: str, gt: str) -> bool:
    """Run verl math_verify on a synthetic `\\boxed{pred_box}` string.

    compute_score expects the full solution, so re-wrapping the last-boxed
    content forces it to grade exactly what we extracted rather than picking a
    different boxed block out of the raw solution.
    """
    if not pred_box:
        return False
    try:
        score = float(_verl_mv_score(f"\\boxed{{{pred_box}}}", gt))
    except Exception:
        return False
    return score > 0.5


def _score_one(sol: str, gt: str) -> Dict[str, float]:
    pred = _extract_last_boxed(sol)
    if not pred:
        return {"score": 0.0, "accuracy": 0.0, "format": 0.0}

    score = 1.0 if _verl_mv_match(pred, gt) else 0.0
    return {"score": score, "accuracy": score, "format": 1.0}


def compute_score(
    *args,
    **kwargs,
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    if "solution_strs" in kwargs:
        sols = kwargs["solution_strs"]
        gts = kwargs["ground_truths"]
        return [_score_one(s, g) for s, g in zip(sols, gts)]
    if "solution_str" in kwargs:
        return _score_one(kwargs["solution_str"], kwargs["ground_truth"])
    raise TypeError(f"compute_score got unexpected kwargs: {list(kwargs.keys())}")
