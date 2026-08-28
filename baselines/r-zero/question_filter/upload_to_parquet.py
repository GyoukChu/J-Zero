#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Concatenate per-shard evaluator outputs, apply the delta-band filter, and
write a verl-format parquet for solver training.

Row schema matches the math500-style verl eval parquets: data_source / prompt
(chat messages) / ability / reward_model {ground_truth} / extra_info.
"""

import argparse
import glob
import json
import os
from typing import List

import pandas as pd


SOLVER_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def main(args: argparse.Namespace) -> None:
    rows: List[dict] = []
    for f in sorted(glob.glob(os.path.join(args.in_dir, f"{args.save_name}_*_results.json"))):
        with open(f) as fh:
            rows.extend(json.load(fh))
    n_total = len(rows)

    filtered = [
        r
        for r in rows
        if (
            args.min_score <= r["score"] <= args.max_score
            and r.get("answer", "") not in ("", "None", None)
        )
    ]
    n_kept = len(filtered)
    print(
        f"[upload] total={n_total} kept={n_kept} "
        f"({100 * n_kept / max(1, n_total):.1f}%) [band {args.min_score}, {args.max_score}]"
    )

    parquet_rows: List[dict] = []
    for i, r in enumerate(filtered):
        parquet_rows.append(
            {
                "data_source": args.data_source,
                "prompt": [
                    {"role": "system", "content": SOLVER_SYSTEM},
                    {"role": "user", "content": r["question"]},
                ],
                "ability": "math",
                "reward_model": {"ground_truth": r["answer"]},
                "extra_info": {
                    "index": i,
                    "p_hat": float(r["score"]),
                    "save_name": args.save_name,
                },
            }
        )

    df = pd.DataFrame(parquet_rows)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"[upload] wrote {len(df)} rows -> {args.out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, required=True)
    p.add_argument("--save_name", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--min_score", type=float, default=0.25)
    p.add_argument("--max_score", type=float, default=0.75)
    p.add_argument("--data_source", type=str, default="rzero_self_play")
    main(p.parse_args())
