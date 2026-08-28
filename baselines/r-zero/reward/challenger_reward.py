# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
R-Zero challenger reward (batched).

Per-prompt score:
    format OK:   max(0, 1 - 2 * |p_hat - 0.5| - r_rep)        # paper Eq. (Sec 3)
    format FAIL: -1 - r_rep                                    # R-Zero official

where
    - `p_hat` is the majority-vote fraction over m=10 solver answers for the
      generated question, obtained from RZERO_NUM_SERVERS Flask vLLM servers on
      ports RZERO_BASE_PORT+i (18000, 18001 as launched by the training scripts)
      that wrap the *current* solver.
    - `r_rep` is the BLEU-based repetition penalty: the share of this batch's
      problems in the same agglomerative cluster as this one, using
      distance = 1 - BLEU and `distance_threshold=tau` (default 0.5).
      Format-failed items cluster together as empty strings, so their penalty is
      proportional to the batch's format-failure count.

Returned dicts include `score` (used by GRPO), plus `format`, `p_hat`,
`r_rep` for logging.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import regex as re
import requests
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.cluster import AgglomerativeClustering


def _extract_boxed_depth_aware(text: str) -> List[str]:
    """Like R-Zero's extract_boxed: nested-brace aware."""
    results, i = [], 0
    prefix = r"\boxed{"
    plen = len(prefix)
    while True:
        start = text.find(prefix, i)
        if start == -1:
            break
        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        results.append(text[start + plen : j - 1])
        i = j
    return results

NUM_SERVERS = int(os.getenv("RZERO_NUM_SERVERS", "2"))
BASE_PORT = int(os.getenv("RZERO_BASE_PORT", "5000"))
# JSON handoff files exchanged with the Flask vLLM servers; the training scripts
# point RZERO_TEMP_DIR at the run's log dir.
TEMP_DIR = os.getenv(
    "RZERO_TEMP_DIR",
    os.path.join(tempfile.gettempdir(), "rzero_reward_tmp"),
)
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

BLEU_TAU = float(os.getenv("RZERO_BLEU_TAU", "0.5"))
HTTP_TIMEOUT = float(os.getenv("RZERO_HTTP_TIMEOUT", "3600"))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _split_list(lst: List, n: int) -> List[List]:
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def _tmpfile(prefix: str) -> str:
    ts = int(time.time() * 1000)
    rnd = random.randint(0, 99999)
    return os.path.join(TEMP_DIR, f"{prefix}_{ts}_{rnd}.json")


def _fetch(index: int, path: str) -> bool:
    port = BASE_PORT + index
    resp = requests.get(
        f"http://127.0.0.1:{port}/hello",
        params={"name": path},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return True


REWARD_CHUNK_SIZE = int(os.getenv("RZERO_REWARD_CHUNK_SIZE", "128"))


def _query_servers_once(items: List[Dict]) -> List[Dict]:
    """One round-trip: shard across NUM_SERVERS Flask vLLM servers, return
    concatenated list of results (each: {question, answer, score, results})."""
    if not items:
        return []
    shards = _split_list(items, NUM_SERVERS)
    paths = [_tmpfile(f"chal_{i}") for i in range(NUM_SERVERS)]
    for path, shard in zip(paths, shards):
        with open(path, "w") as fh:
            json.dump(shard, fh)
    with ThreadPoolExecutor(max_workers=NUM_SERVERS) as pool:
        futures = [pool.submit(_fetch, i, paths[i]) for i in range(NUM_SERVERS)]
        for f in as_completed(futures):
            f.result()
    out: List[Dict] = []
    for path in paths:
        rpath = path.replace(".json", "_results.json")
        with open(rpath) as fh:
            out.extend(json.load(fh))
        try:
            os.remove(rpath)
        except FileNotFoundError:
            pass
    return out


def _query_servers(items: List[Dict]) -> List[Dict]:
    """
    Spread `items` over NUM_SERVERS Flask vLLM servers and return concatenated
    results. Batches above RZERO_REWARD_CHUNK_SIZE are split into sequential
    chunks to cap each server's per-request prompt count, which otherwise
    reaches `len(items)/NUM_SERVERS * n=10` rollouts and can OOM or time out
    under v1's continuous batching. Sequential, not concurrent: the bottleneck
    is inside each server (single GPU, n=10 samples each).
    """
    if not items:
        return []
    chunk_size = max(1, REWARD_CHUNK_SIZE)
    if len(items) <= chunk_size:
        t0 = time.time()
        out = _query_servers_once(items)
        print(
            f"[reward] _query_servers single n={len(items)} done in {time.time()-t0:.1f}s",
            flush=True,
        )
        return out
    n_chunks = (len(items) + chunk_size - 1) // chunk_size
    out: List[Dict] = []
    for i in range(n_chunks):
        chunk = items[i * chunk_size : (i + 1) * chunk_size]
        t0 = time.time()
        sub = _query_servers_once(chunk)
        out.extend(sub)
        print(
            f"[reward] chunk {i+1}/{n_chunks} done in {time.time()-t0:.1f}s, "
            f"n={len(chunk)} responses={len(sub)}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# BLEU repetition penalty
# ---------------------------------------------------------------------------
def _bleu_distance_matrix(sentences: List[str]) -> np.ndarray:
    n = len(sentences)
    dist = np.zeros((n, n), dtype=np.float32)
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1.0 - float(score)
    return dist


MAX_CLUSTER_ITEMS = int(os.getenv("RZERO_MAX_CLUSTER_ITEMS", "256"))


def cluster_share(problems: List[str], distance_threshold: float = BLEU_TAU) -> List[float]:
    """
    Returns per-problem cluster-share = (#items in own cluster) / (total items),
    matching R-Zero official `cluster_share_per_problem`.

    O(n^2) BLEU is too slow for n>>100, so identical strings are deduped and
    clustering is capped at MAX_CLUSTER_ITEMS unique items by stride subsample.
    Cluster sizes are still counted from the FULL `problems` list, so
    format-failed (empty-string) items get a true proportion penalty; items whose
    representative missed the sample inherit the average ratio.
    """
    if not problems:
        return []
    n = len(problems)
    # Step 1: dedupe identical strings.
    uniq_to_idx: Dict[str, int] = {}
    rep_indices: List[int] = []  # indices into `problems` for representatives
    item_to_rep: List[int] = [-1] * n  # for each input item, which representative
    for i, p in enumerate(problems):
        if p in uniq_to_idx:
            item_to_rep[i] = uniq_to_idx[p]
        else:
            uniq_to_idx[p] = len(rep_indices)
            rep_indices.append(i)
            item_to_rep[i] = len(rep_indices) - 1
    unique_problems = [problems[idx] for idx in rep_indices]

    # Step 1.5: AgglomerativeClustering needs >=2 samples. With fewer, every item
    # is trivially in one cluster -> share=1.0, i.e. the max repetition penalty,
    # which is the right answer for a batch with no diversity.
    if n < 2 or len(unique_problems) < 2:
        print(
            f"[reward] clustering skipped: n_input={n} n_unique={len(unique_problems)} "
            f"(<2); returning share=1.0 for all",
            flush=True,
        )
        return [1.0] * n

    # Step 2: subsample if too many unique.
    if len(unique_problems) > MAX_CLUSTER_ITEMS:
        stride = len(unique_problems) / MAX_CLUSTER_ITEMS
        sampled_unique_indices = [int(i * stride) for i in range(MAX_CLUSTER_ITEMS)]
        sampled = [unique_problems[i] for i in sampled_unique_indices]
    else:
        sampled_unique_indices = list(range(len(unique_problems)))
        sampled = unique_problems

    print(
        f"[reward] clustering n_input={n} n_unique={len(unique_problems)} "
        f"n_clustered={len(sampled)}",
        flush=True,
    )
    t0 = time.time()
    dist = _bleu_distance_matrix(sampled)
    clu = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clu.fit_predict(dist)

    unique_to_label: Dict[int, int] = {}
    for k, ui in enumerate(sampled_unique_indices):
        unique_to_label[ui] = int(labels[k])

    # Count cluster sizes from FULL problems list (matches R-Zero official).
    item_labels: List[int] = [unique_to_label.get(item_to_rep[i], -1) for i in range(n)]
    sizes_full = Counter(lab for lab in item_labels if lab != -1)
    total = n
    ratio = {lab: sz / total for lab, sz in sizes_full.items()}
    avg_ratio = float(np.mean(list(ratio.values()))) if ratio else 0.0

    out: List[float] = []
    for lab in item_labels:
        if lab == -1:
            out.append(avg_ratio)
        else:
            out.append(min(1.0, float(ratio[lab])))
    print(f"[reward] clustering done in {time.time() - t0:.1f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# Public entrypoint expected by verl's custom_reward_function
# ---------------------------------------------------------------------------
def compute_score(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List,
    extra_infos: List[Dict],
    **kwargs,
) -> List[Dict[str, float]]:
    """
    verl BatchRewardManager signature.

    Each `solution_str` is a challenger rollout; we extract:
        <question>...</question>  and  \boxed{...}
    then query the solver servers for p_hat and compute the BLEU penalty.
    """
    parsed: List[Dict[str, str]] = []
    for s in solution_strs:
        qs = re.findall(r"<question>(.*?)</question>", s, re.DOTALL)
        ans = _extract_boxed_depth_aware(s)
        if qs and ans:
            q = qs[-1].strip()
            a = ans[-1].strip()
            parsed.append({"question": q, "answer": a})
        else:
            parsed.append({"question": "", "answer": ""})

    server_out = _query_servers(parsed)
    # One entry per input, in order, empty placeholders included (the server
    # emits score=-1 for those). Pad/truncate defensively if that ever breaks.
    if len(server_out) != len(parsed):
        print(
            f"[reward] WARN server returned {len(server_out)} for {len(parsed)} prompts; padding",
            flush=True,
        )
        if len(server_out) < len(parsed):
            server_out = server_out + [
                {"question": "", "answer": "", "score": -1, "results": []}
                for _ in range(len(parsed) - len(server_out))
            ]
        else:
            server_out = server_out[: len(parsed)]

    # Include ALL questions, empties included, to match R-Zero official
    # caller_penalty.py: BLEU(empty, *) = 0, so format-failed items form their
    # own cluster and each is penalised (#format_failed / n).
    all_q = [r["question"] for r in server_out]
    r_rep = cluster_share(all_q, distance_threshold=BLEU_TAU) if all_q else [0.0] * len(server_out)

    scores: List[Dict[str, float]] = []
    for i, r in enumerate(server_out):
        fmt = 1.0 if r["question"] else 0.0
        p_hat = float(r.get("score", 0.0))
        rrep_i = float(r_rep[i])
        if fmt == 0.0:
            # R-Zero official: format-fail base = -1, then subtract BLEU penalty.
            final = -1.0 - rrep_i
        else:
            # paper-spec: max(0, 1 - 2|p_hat - 0.5| - r_rep)
            final = max(0.0, 1.0 - 2.0 * abs(p_hat - 0.5) - rrep_i)
        scores.append(
            {
                "score": float(final),
                "format": fmt,
                "p_hat": p_hat,
                "r_rep": rrep_i,
            }
        )
    return scores
