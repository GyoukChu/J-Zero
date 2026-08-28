# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
R-Zero RM-variant challenger reward (batched).

Per-prompt score:
    format OK:   max(0, 1 - mean(sigmoid(raw_RM_i)) - r_rep)
    format FAIL: -1 - r_rep                                # unchanged from R-Zero

For each challenger-generated question, the answers-only solver vLLM (GPU 2 in
the 4-GPU layout) produces n=10 answers, the RM HTTP servers (GPU 3) score each
(Q, A) pair as a RAW logit, and the per-Q "RM mean" is mean(sigmoid(raw_i)).
`r_rep` is challenger_reward.cluster_share over the batch's questions.

Env vars:
    RZERO_BASE_PORT              : answers-only vLLM port (default 18000)
    RZERO_NUM_SERVERS            : answers-only vLLM count (default 1)
    RZERO_RM_BASE_PORT           : RM server base port (default 19000)
    RZERO_RM_NUM_SERVERS_CHAL    : RM server count for challenger stage (default 1)
    RZERO_REWARD_CHUNK_SIZE      : sub-chunk size for HTTP fan-out (default 128)
    RZERO_RM_HTTP_TIMEOUT        : per-HTTP-request timeout (default 7200s)
    RZERO_BLEU_TAU               : BLEU clustering distance threshold (default 0.5)
    RZERO_TEMP_DIR               : tmp dir for vLLM file-payload fallback
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import regex as re
import requests

# Reuse the proven helpers from the rule-based variant.
from reward.challenger_reward import (  # noqa: F401  (boxed parser re-exported)
    _extract_boxed_depth_aware,
    cluster_share,
)
from rm_service.rm_io import SOLVER_SYS, sigmoid


# --- Config --------------------------------------------------------------
NUM_SOLVER_SERVERS = int(os.getenv("RZERO_NUM_SERVERS", "1"))
SOLVER_BASE_PORT = int(os.getenv("RZERO_BASE_PORT", "18000"))

NUM_RM_SERVERS = int(os.getenv("RZERO_RM_NUM_SERVERS_CHAL", "1"))
RM_BASE_PORT = int(os.getenv("RZERO_RM_BASE_PORT", "19000"))

REWARD_CHUNK_SIZE = int(os.getenv("RZERO_REWARD_CHUNK_SIZE", "128"))
HTTP_TIMEOUT = float(os.getenv("RZERO_RM_HTTP_TIMEOUT", os.getenv("RZERO_HTTP_TIMEOUT", "7200")))
BLEU_TAU = float(os.getenv("RZERO_BLEU_TAU", "0.5"))

# File-payload scratch dir shared with the answers-only vLLM servers; the
# training scripts export RZERO_TEMP_DIR (e.g. "$LOG_DIR/_reward_tmp").
TEMP_DIR = os.getenv("RZERO_TEMP_DIR") or os.path.join(
    tempfile.gettempdir(), "rzero_reward_tmp"
)
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"


# --- HTTP helpers --------------------------------------------------------
def _split_list(lst: List, n: int) -> List[List]:
    if n <= 0:
        raise ValueError(f"n must be >0, got {n}")
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]


def _tmpfile(prefix: str) -> str:
    ts = int(time.time() * 1000)
    rnd = random.randint(0, 99999)
    return os.path.join(TEMP_DIR, f"{prefix}_{ts}_{rnd}.json")


# --- Answers-only vLLM (file-payload, matches original /hello protocol) ---
def _solver_fetch(port: int, path: str) -> bool:
    resp = requests.get(
        f"http://127.0.0.1:{port}/hello",
        params={"name": path},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return True


def _query_solver_answers_once(items: List[Dict]) -> List[Dict]:
    """Send `items` (list of {question}) to the answers-only vLLM servers via
    file payloads. Returns list of {question, answers:[str,...]}.

    Each shard's response length is checked against its input before
    concatenation: an upstream drop on shard k would otherwise shift later
    shards' owner indices."""
    if not items:
        return []
    shards = _split_list(items, NUM_SOLVER_SERVERS)
    paths = [_tmpfile(f"chal_ans_{i}") for i in range(NUM_SOLVER_SERVERS)]
    for path, shard in zip(paths, shards):
        with open(path, "w") as fh:
            json.dump(shard, fh)
    with ThreadPoolExecutor(max_workers=max(1, NUM_SOLVER_SERVERS)) as pool:
        futs = [pool.submit(_solver_fetch, SOLVER_BASE_PORT + i, paths[i])
                for i in range(NUM_SOLVER_SERVERS)]
        for f in as_completed(futs):
            f.result()
    out: List[Dict] = []
    for i, (path, shard) in enumerate(zip(paths, shards)):
        rpath = path.replace(".json", "_results.json")
        with open(rpath) as fh:
            shard_out = json.load(fh)
        if len(shard_out) != len(shard):
            print(f"[rm-reward] WARN solver shard {i} returned "
                  f"{len(shard_out)} for input {len(shard)}; "
                  "padding/truncating to preserve owner alignment", flush=True)
            if len(shard_out) < len(shard):
                # Empty-answer rows mirroring the inputs, so the 0.5 fallback
                # in compute_score applies to them.
                for j in range(len(shard_out), len(shard)):
                    q = shard[j].get("question", "") if j < len(shard) else ""
                    shard_out.append({"question": q, "answers": []})
            else:
                shard_out = shard_out[: len(shard)]
        out.extend(shard_out)
        try:
            os.remove(rpath)
        except FileNotFoundError:
            pass
    return out


def _query_solver_answers(items: List[Dict]) -> List[Dict]:
    """Chunked variant of _query_solver_answers_once, mirroring
    challenger_reward._query_servers so the single solver server's memory
    pressure stays bounded."""
    if not items:
        return []
    chunk_size = max(1, REWARD_CHUNK_SIZE)
    if len(items) <= chunk_size:
        t0 = time.time()
        out = _query_solver_answers_once(items)
        print(f"[rm-reward] solver answers single n={len(items)} done in "
              f"{time.time()-t0:.1f}s", flush=True)
        return out
    out: List[Dict] = []
    n_chunks = (len(items) + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        chunk = items[i * chunk_size: (i + 1) * chunk_size]
        t0 = time.time()
        sub = _query_solver_answers_once(chunk)
        out.extend(sub)
        print(f"[rm-reward] solver chunk {i+1}/{n_chunks} n={len(chunk)} "
              f"responses={len(sub)} in {time.time()-t0:.1f}s", flush=True)
    return out


# --- RM HTTP scoring (JSON POST /score) ----------------------------------
def _rm_post(port: int, items: List[Dict]) -> List[float]:
    """One round-trip to a single RM server. Returns raw scores."""
    resp = requests.post(
        f"http://127.0.0.1:{port}/score",
        json={"items": items},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    return [float(x) for x in body.get("scores", [])]


def _rm_score_once(items: List[Dict]) -> List[float]:
    """Shard items across NUM_RM_SERVERS, parallel POST, reassemble in order.

    Each shard's response length is checked against its input before
    concatenation: a partial RM response would otherwise shift subsequent
    shards' owner indices."""
    if not items:
        return []
    n_servers = max(1, NUM_RM_SERVERS)
    shards = _split_list(items, n_servers)
    futures_idx = []
    with ThreadPoolExecutor(max_workers=n_servers) as pool:
        for i, shard in enumerate(shards):
            if not shard:
                continue
            futures_idx.append((i, pool.submit(_rm_post, RM_BASE_PORT + i, shard)))
        shard_scores: List[List[float]] = [[] for _ in range(n_servers)]
        for i, fut in futures_idx:
            shard_scores[i] = fut.result()
    # Concatenate in shard order to recover the original input order.
    out: List[float] = []
    for i, (s, shard) in enumerate(zip(shard_scores, shards)):
        if len(s) != len(shard):
            print(f"[rm-reward] WARN RM shard {i} returned {len(s)} for "
                  f"input {len(shard)}; padding 0 to preserve order",
                  flush=True)
            if len(s) < len(shard):
                s = list(s) + [0.0] * (len(shard) - len(s))
            else:
                s = s[: len(shard)]
        out.extend(s)
    return out


def _rm_score(items: List[Dict]) -> List[float]:
    """Chunked RM scoring (sub-chunk inside a single call). Returns RAW logits."""
    if not items:
        return []
    chunk_size = max(1, REWARD_CHUNK_SIZE)
    if len(items) <= chunk_size:
        t0 = time.time()
        out = _rm_score_once(items)
        print(f"[rm-reward] RM single n={len(items)} done in {time.time()-t0:.1f}s",
              flush=True)
        return out
    out: List[float] = []
    n_chunks = (len(items) + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        chunk = items[i * chunk_size: (i + 1) * chunk_size]
        t0 = time.time()
        sub = _rm_score_once(chunk)
        out.extend(sub)
        print(f"[rm-reward] RM chunk {i+1}/{n_chunks} n={len(chunk)} "
              f"responses={len(sub)} in {time.time()-t0:.1f}s", flush=True)
    return out


# --- Public entrypoint (verl custom_reward_function) ----------------------
def compute_score(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List,
    extra_infos: List[Dict],
    **kwargs,
) -> List[Dict[str, float]]:
    """verl BatchRewardManager signature.

    `solution_str` is a challenger rollout; the LAST <question>...</question>
    block is taken. Items with no parseable question are format-failed but
    still join the BLEU cluster as empty strings, so the batch's repetition
    penalty stays well-defined.
    """
    parsed_qs: List[str] = []
    fmt_flags: List[float] = []
    for s in solution_strs:
        qs = re.findall(r"<question>(.*?)</question>", s, re.DOTALL)
        # Unlike challenger_reward, no \boxed{...} is required: the RM variant
        # never reads the challenger's self-claimed answer. The R-Zero
        # format-fail shape (-1 - r_rep) is preserved.
        if qs and qs[-1].strip():
            parsed_qs.append(qs[-1].strip())
            fmt_flags.append(1.0)
        else:
            parsed_qs.append("")
            fmt_flags.append(0.0)

    n_prompts = len(parsed_qs)

    # 1) n=10 solver answers per question. Format-fail rows are forwarded
    #    in place — the vLLM script emits {question:"", answers:[]} for them
    #    without spending generation budget, which keeps the 1:1 input/output
    #    mapping the owner-index bookkeeping below relies on.
    solver_items = [{"question": q} for q in parsed_qs]
    solver_results = _query_solver_answers(solver_items)
    if len(solver_results) != n_prompts:
        print(f"[rm-reward] WARN solver returned {len(solver_results)} for "
              f"{n_prompts} prompts; padding", flush=True)
        while len(solver_results) < n_prompts:
            solver_results.append({"question": "", "answers": []})
        solver_results = solver_results[:n_prompts]

    # 2) Build the (Q, A) pair list flattened: pair_index -> (prompt_idx, a_idx).
    rm_items: List[Dict] = []
    pair_owner: List[int] = []  # which prompt each pair belongs to
    for pi, res in enumerate(solver_results):
        q = res.get("question", "") or parsed_qs[pi]
        for a in (res.get("answers", []) or []):
            rm_items.append({"question": q, "answer": a})
            pair_owner.append(pi)

    raw_scores = _rm_score(rm_items) if rm_items else []
    if len(raw_scores) != len(rm_items):
        print(f"[rm-reward] WARN RM returned {len(raw_scores)} for {len(rm_items)} pairs; "
              "padding with 0.0", flush=True)
        if len(raw_scores) < len(rm_items):
            raw_scores = list(raw_scores) + [0.0] * (len(rm_items) - len(raw_scores))
        else:
            raw_scores = raw_scores[: len(rm_items)]

    # 3) Aggregate per-prompt mean(sigmoid(raw_i)).
    per_prompt_sigs: List[List[float]] = [[] for _ in range(n_prompts)]
    for owner, raw in zip(pair_owner, raw_scores):
        per_prompt_sigs[owner].append(float(sigmoid(float(raw))))
    rm_means: List[float] = []
    for pi in range(n_prompts):
        vals = per_prompt_sigs[pi]
        if vals:
            rm_means.append(float(sum(vals) / len(vals)))
        else:
            # No answers: format-fail rows are overridden by fmt below, so
            # this 0.5 only applies to the rare OK row vLLM dropped.
            rm_means.append(0.5)

    # 4) Repetition penalty; format-fail items participate as empty strings.
    r_rep = cluster_share(parsed_qs, distance_threshold=BLEU_TAU) \
        if parsed_qs else [0.0] * n_prompts

    # 5) Per-prompt final score.
    scores: List[Dict[str, float]] = []
    for pi in range(n_prompts):
        fmt = fmt_flags[pi]
        rrep_i = float(r_rep[pi]) if pi < len(r_rep) else 0.0
        rm_mean_i = float(rm_means[pi])
        if fmt == 0.0:
            final = -1.0 - rrep_i
        else:
            final = max(0.0, 1.0 - rm_mean_i - rrep_i)
        scores.append({
            "score": float(final),
            "format": float(fmt),
            "rm_mean": rm_mean_i,
            "r_rep": rrep_i,
        })
    return scores
