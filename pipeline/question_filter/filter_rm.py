#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RM-variant question filter — aggregator step.

Reads all `<save_name>_*_answers.json` shards from `gen_answers_rm.py`, scores
every (question, answer) pair on the RM HTTP servers, reduces to per-question
(avg, std), keeps questions whose avg falls in [lo, hi], sorts survivors by std
DESC, and writes `<save_name>_filtered.json` of
    [{"question", "avg_rm", "std_rm", "num_answers"}, ...]
which run_parallel_rm.sh then feeds to `upload_to_parquet_rm.py` to build the
verl parquet.

The RM servers MUST already be listening on
[--rm_base_port, --rm_base_port + num_servers).

RZERO_FILTER_STD_MODE (default "sigmoid") selects the scale:
    sigmoid : per-question avg/std of sigmoid(raw); band on sigmoid-scale
              avg_rm ([--min_avg, --max_avg]); sort by std.
    raw     : NO sigmoid anywhere — avg/std of the RAW logits, and avg_rm /
              std_rm in the output JSON hold raw-scale values. The band reads
              RZERO_RM_FILTER_RAW_LO / _HI (default ±inf = pass-all) since a
              sigmoid-scale band is meaningless on logits.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import requests

# Allow `from rm_service.rm_io import sigmoid` when run as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from rm_service.rm_io import sigmoid  # noqa: E402


def _split_list(lst: List, n: int) -> List[List]:
    if n <= 0:
        raise ValueError(f"n must be >0, got {n}")
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]


def _rm_post(port: int, items: List[Dict], timeout: float) -> List[float]:
    resp = requests.post(
        f"http://127.0.0.1:{port}/score",
        json={"items": items},
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    return [float(x) for x in body.get("scores", [])]


def _rm_score_chunk(items: List[Dict], base_port: int, num_servers: int,
                    timeout: float) -> List[float]:
    if not items:
        return []
    shards = _split_list(items, num_servers)
    with ThreadPoolExecutor(max_workers=num_servers) as pool:
        futs = []
        for i, shard in enumerate(shards):
            if not shard:
                futs.append(None)
                continue
            futs.append(pool.submit(_rm_post, base_port + i, shard, timeout))
        out_shards = [fut.result() if fut else [] for fut in futs]
    # Per-shard length check: pad/truncate to expected so concatenation
    # preserves the (question_idx, answer_idx) -> pair_owner mapping.
    out: List[float] = []
    for i, (s, shard) in enumerate(zip(out_shards, shards)):
        if len(s) != len(shard):
            print(f"[filter_rm] WARN RM shard {i} returned {len(s)} for "
                  f"input {len(shard)}; padding 0 to preserve order",
                  flush=True)
            if len(s) < len(shard):
                s = list(s) + [0.0] * (len(shard) - len(s))
            else:
                s = s[: len(shard)]
        out.extend(s)
    return out


def main(args: argparse.Namespace) -> None:
    # 1) Aggregate all per-shard answer files.
    pattern = os.path.join(args.in_dir, f"{args.save_name}_*_answers.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[filter_rm] FATAL: no shard files matching {pattern}",
              file=sys.stderr)
        raise SystemExit(2)

    all_rows: List[Dict] = []
    for f in files:
        with open(f) as fh:
            all_rows.extend(json.load(fh))
    print(f"[filter_rm] loaded {len(all_rows)} questions from {len(files)} shards",
          flush=True)
    if not all_rows:
        # Write empty filtered JSON so the parquet step can no-op cleanly.
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        with open(args.out_path, "w") as fh:
            json.dump([], fh)
        return

    # 2) Flatten (q, a) pairs preserving the (question_idx, answer_idx) order.
    flat_items: List[Dict] = []
    pair_owner: List[int] = []
    for qi, row in enumerate(all_rows):
        q = row.get("question", "") or ""
        for a in row.get("answers", []) or []:
            flat_items.append({"question": q, "answer": a})
            pair_owner.append(qi)
    print(f"[filter_rm] {len(flat_items)} (Q, A) pairs to score", flush=True)

    # 3) Chunked RM scoring.
    chunk_size = max(1, args.chunk_size)
    n_chunks = (len(flat_items) + chunk_size - 1) // chunk_size
    raw_scores: List[float] = []
    for ci in range(n_chunks):
        chunk = flat_items[ci * chunk_size: (ci + 1) * chunk_size]
        t0 = time.time()
        chunk_scores = _rm_score_chunk(
            chunk, args.rm_base_port, args.num_servers, args.timeout
        )
        if len(chunk_scores) != len(chunk):
            print(f"[filter_rm] WARN chunk {ci+1}/{n_chunks}: got "
                  f"{len(chunk_scores)} scores for {len(chunk)} items; padding 0",
                  flush=True)
            if len(chunk_scores) < len(chunk):
                chunk_scores = list(chunk_scores) + [0.0] * (len(chunk) - len(chunk_scores))
            else:
                chunk_scores = chunk_scores[: len(chunk)]
        raw_scores.extend(chunk_scores)
        if ci == 0 or (ci + 1) % 5 == 0 or ci + 1 == n_chunks:
            print(f"[filter_rm] chunk {ci+1}/{n_chunks} ({len(chunk)} items) "
                  f"in {time.time()-t0:.1f}s", flush=True)

    # 4) Per-question stats. sigmoid (default) | raw (no sigmoid anywhere) |
    #    w, which sorts by the sample std of the within-question relative
    #    scores w_i = mean_{j!=i} sigmoid(r_i - r_j) while keeping the avg band
    #    on the sigmoid scale.
    std_mode = os.getenv("RZERO_FILTER_STD_MODE", "sigmoid").strip().lower()
    if std_mode not in ("sigmoid", "raw", "w"):
        print(f"[filter_rm] WARN unknown RZERO_FILTER_STD_MODE={std_mode!r}; "
              "falling back to 'sigmoid'", flush=True)
        std_mode = "sigmoid"
    print(f"[filter_rm] std_mode={std_mode}"
          + (" (NO sigmoid applied anywhere)" if std_mode == "raw" else ""),
          flush=True)

    per_q_raws: List[List[float]] = [[] for _ in range(len(all_rows))]
    for owner, raw in zip(pair_owner, raw_scores):
        per_q_raws[owner].append(float(raw))

    summaries: List[Dict] = []
    for qi, row in enumerate(all_rows):
        raws = per_q_raws[qi]
        if not raws:
            continue   # no answers -> skip entirely
        avg_raw = float(np.mean(raws))
        std_raw = float(np.std(raws))
        if std_mode == "raw":
            # avg_rm/std_rm carry raw-scale values in this mode.
            avg, std = avg_raw, std_raw
        else:
            sigs = [float(sigmoid(r)) for r in raws]
            avg = float(np.mean(sigs))
            std = float(np.std(sigs))
        std_w = 0.0
        if std_mode == "w" and len(raws) >= 2:
            g = len(raws)
            w = [sum(float(sigmoid(raws[i] - raws[j])) for j in range(g) if j != i) / (g - 1)
                 for i in range(g)]
            std_w = float(np.std(w, ddof=1))
        summaries.append({
            "question": row.get("question", ""),
            "avg_rm": avg,
            "std_rm": std,
            "avg_rm_raw": avg_raw,
            "std_rm_raw": std_raw,
            "std_rm_w": std_w,
            "num_answers": len(raws),
        })

    # Distribution log (helps tune the band on iter 1).
    if summaries:
        arr_avg = np.array([s["avg_rm"] for s in summaries])
        arr_std = np.array([s["std_rm"] for s in summaries])
        print(f"[filter_rm] avg_rm  min={arr_avg.min():.3f} mean={arr_avg.mean():.3f} "
              f"max={arr_avg.max():.3f}", flush=True)
        print(f"[filter_rm] std_rm  min={arr_std.min():.3f} mean={arr_std.mean():.3f} "
              f"max={arr_std.max():.3f}", flush=True)
        arr_std_raw = np.array([s["std_rm_raw"] for s in summaries])
        print(f"[filter_rm] std_rm_raw  min={arr_std_raw.min():.3f} "
              f"mean={arr_std_raw.mean():.3f} max={arr_std_raw.max():.3f}",
              flush=True)

    # Raw mode ignores the sigmoid-scale --min_avg/--max_avg and reads a
    # raw-scale band from env instead (default = pass-all).
    if std_mode == "raw":
        band_lo = float(os.getenv("RZERO_RM_FILTER_RAW_LO", "-1e30"))
        band_hi = float(os.getenv("RZERO_RM_FILTER_RAW_HI", "1e30"))
        print(f"[filter_rm] raw-mode band: [{band_lo}, {band_hi}] "
              "(sigmoid-scale --min_avg/--max_avg ignored)", flush=True)
    else:
        band_lo, band_hi = args.min_avg, args.max_avg

    # 4b) Band-split counts + full per-question score dump. Only survivors are
    # written otherwise, so #below/#above the band would need a re-score to
    # recover. Observability only — does NOT change which questions are kept.
    if summaries:
        n_below = int(sum(1 for s in summaries if s["avg_rm"] < band_lo))
        n_above = int(sum(1 for s in summaries if s["avg_rm"] > band_hi))
        n_in = len(summaries) - n_below - n_above
        print(f"[filter_rm] band-split: below(<{band_lo})={n_below} "
              f"in[{band_lo},{band_hi}]={n_in} "
              f"above(>{band_hi})={n_above} total={len(summaries)}", flush=True)
        all_path = os.path.join(os.path.dirname(args.out_path) or ".",
                                f"{args.save_name}_allscores.json")
        try:
            with open(all_path, "w") as fh:
                json.dump(summaries, fh)
            print(f"[filter_rm] wrote {len(summaries)} per-question scores -> "
                  f"{all_path}", flush=True)
        except Exception as e:  # never let logging break the filter
            print(f"[filter_rm] WARN could not write allscores: {e}", flush=True)

    # 5) Band filter + top-K by std (sort key selected by RZERO_FILTER_STD_MODE).
    sort_key = {"raw": "std_rm_raw", "w": "std_rm_w"}.get(std_mode, "std_rm")
    in_band = [s for s in summaries if band_lo <= s["avg_rm"] <= band_hi]
    print(f"[filter_rm] kept {len(in_band)}/{len(summaries)} in band "
          f"[{band_lo}, {band_hi}]; sort_key={sort_key}", flush=True)
    in_band.sort(key=lambda s: s[sort_key], reverse=True)
    top_k = in_band[: args.top_k]
    if len(top_k) < args.top_k:
        print(f"[filter_rm] WARN only {len(top_k)} survive band filter "
              f"(target top_k={args.top_k}). Solver trainer may see <15 effective "
              "steps.", flush=True)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as fh:
        json.dump(top_k, fh)
    print(f"[filter_rm] wrote {len(top_k)} survivors -> {args.out_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, required=True,
                   help="dir containing <save_name>_*_answers.json")
    p.add_argument("--save_name", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True,
                   help="output JSON of filtered survivors")
    p.add_argument("--rm_base_port", type=int,
                   default=int(os.getenv("RZERO_RM_BASE_PORT", "19000")))
    p.add_argument("--num_servers", type=int,
                   default=int(os.getenv("RZERO_RM_NUM_SERVERS_FILT", "4")))
    p.add_argument("--chunk_size", type=int,
                   default=int(os.getenv("RZERO_REWARD_CHUNK_SIZE", "256")))
    p.add_argument("--timeout", type=float,
                   default=float(os.getenv("RZERO_RM_HTTP_TIMEOUT", "7200")))
    # Band OFF by default, as in the released J-Zero run: avg_rm is a mean of
    # sigmoids, so [0.0, 1.0] passes everything and --top_k by std is the only
    # selection. Narrow it via --min_avg/--max_avg or RZERO_RM_FILTER_LO/HI;
    # R-Zero's [0.25, 0.75] band is a different method, kept in baselines/.
    # DUPLICATED DEFAULTS — same pair in pipeline/solver_train.sh and
    # question_filter/run_parallel_rm.sh.
    p.add_argument("--min_avg", type=float,
                   default=float(os.getenv("RZERO_RM_FILTER_LO", "0.0")))
    p.add_argument("--max_avg", type=float,
                   default=float(os.getenv("RZERO_RM_FILTER_HI", "1.0")))
    p.add_argument("--top_k", type=int,
                   default=int(os.getenv("RZERO_RM_FILTER_TOPN", "1920")))
    main(p.parse_args())
