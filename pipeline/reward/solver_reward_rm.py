# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
R-Zero RM-variant solver reward.

For each solver rollout (Q, A) the reward = sigmoid(raw_RM_logit) ∈ [0,1].
No rule-based grader, no mathruler, no ground-truth comparison.

`compute_score` takes the batched signature `RZeroBatchManager` expects
(`data_sources`, `solution_strs`, `ground_truths`, `extra_infos`).
`ground_truths` is accepted but IGNORED — parquet rows from
`question_filter/upload_to_parquet_rm.py` carry no pseudo-label. The question
comes from `extra_infos[i]["question"]`, which that same upload step writes;
it is the single source of truth, since recovering the question otherwise
means decoding `data_item.batch["prompts"]`, unavailable here.

Env vars (same dispatch knobs as challenger_reward_rm):
    RZERO_RM_BASE_PORT          : RM server base port (default 19000)
    RZERO_RM_NUM_SERVERS_SOL    : RM server count for solver stage (default 2)
    RZERO_REWARD_CHUNK_SIZE     : sub-chunk size for HTTP fan-out (default 128)
    RZERO_RM_HTTP_TIMEOUT       : per-HTTP-request timeout (default 7200s)
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import requests

from rm_service.rm_io import sigmoid


NUM_RM_SERVERS = int(os.getenv("RZERO_RM_NUM_SERVERS_SOL", "2"))
RM_BASE_PORT = int(os.getenv("RZERO_RM_BASE_PORT", "19000"))
REWARD_CHUNK_SIZE = int(os.getenv("RZERO_REWARD_CHUNK_SIZE", "128"))
HTTP_TIMEOUT = float(os.getenv("RZERO_RM_HTTP_TIMEOUT", os.getenv("RZERO_HTTP_TIMEOUT", "7200")))

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"


def _split_list(lst: List, n: int) -> List[List]:
    if n <= 0:
        raise ValueError(f"n must be >0, got {n}")
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]


def _rm_post(port: int, items: List[Dict]) -> List[float]:
    resp = requests.post(
        f"http://127.0.0.1:{port}/score",
        json={"items": items},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    return [float(x) for x in body.get("scores", [])]


def _rm_score_once(items: List[Dict]) -> List[float]:
    if not items:
        return []
    n_servers = max(1, NUM_RM_SERVERS)
    shards = _split_list(items, n_servers)
    with ThreadPoolExecutor(max_workers=n_servers) as pool:
        futs = []
        for i, shard in enumerate(shards):
            if not shard:
                futs.append(None)
                continue
            futs.append(pool.submit(_rm_post, RM_BASE_PORT + i, shard))
        shard_scores: List[List[float]] = []
        for i, fut in enumerate(futs):
            shard_scores.append(fut.result() if fut is not None else [])
    # Per-shard length check: a partial RM response on shard k would otherwise
    # shift later items' rewards onto the wrong rollouts.
    out: List[float] = []
    for i, (s, shard) in enumerate(zip(shard_scores, shards)):
        if len(s) != len(shard):
            print(f"[solver-rm-reward] WARN RM shard {i} returned {len(s)} "
                  f"for input {len(shard)}; padding 0 to preserve order",
                  flush=True)
            if len(s) < len(shard):
                s = list(s) + [0.0] * (len(shard) - len(s))
            else:
                s = s[: len(shard)]
        out.extend(s)
    return out


def _rm_score(items: List[Dict]) -> List[float]:
    if not items:
        return []
    chunk_size = max(1, REWARD_CHUNK_SIZE)
    if len(items) <= chunk_size:
        return _rm_score_once(items)
    out: List[float] = []
    n_chunks = (len(items) + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        chunk = items[i * chunk_size: (i + 1) * chunk_size]
        out.extend(_rm_score_once(chunk))
    return out


def _get_question(extra_info, fallback: str = "") -> str:
    if isinstance(extra_info, dict):
        q = extra_info.get("question", None)
        if q:
            return str(q)
    return fallback


def compute_score(
    *args,
    data_sources: List[str] = None,
    solution_strs: List[str] = None,
    ground_truths: List = None,
    extra_infos: List = None,
    **kwargs,
) -> List[Dict[str, float]]:
    """Batched signature. Returns one dict per (Q, A) item with key 'score'."""
    if solution_strs is None:
        raise TypeError("solver_reward_rm.compute_score requires solution_strs kwarg")
    n = len(solution_strs)
    if extra_infos is None:
        extra_infos = [{} for _ in range(n)]

    items: List[Dict] = []
    for i in range(n):
        q = _get_question(extra_infos[i] if i < len(extra_infos) else {}, "")
        a = solution_strs[i] or ""
        items.append({"question": q, "answer": a})

    raw_scores = _rm_score(items)
    if len(raw_scores) != n:
        if len(raw_scores) < n:
            raw_scores = list(raw_scores) + [0.0] * (n - len(raw_scores))
        else:
            raw_scores = raw_scores[:n]

    out: List[Dict[str, float]] = []
    for raw in raw_scores:
        s = float(sigmoid(float(raw)))
        out.append({
            "score": s,
            "accuracy": s,        # for naive RewardManager logging compat
            "rm_raw": float(raw),
        })
    return out
