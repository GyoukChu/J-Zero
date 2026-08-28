#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert `filter_rm.py`'s JSON survivors into a verl-format parquet.

Schema:
    data_source   : str
    prompt        : [ {role:user, content:q} ]   # user-only, no system turn
    ability       : "math"
    reward_model  : { ground_truth: "" }   # placeholder; solver_reward_rm ignores it
    extra_info    : { index, question, avg_rm, std_rm, num_answers, save_name }

`extra_info["question"]` is load-bearing: `solver_reward_rm.compute_score`
reads the question from there instead of re-decoding
`data_item.batch["prompts"]`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import pandas as pd


SOLVER_SYS = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def main(args: argparse.Namespace) -> None:
    with open(args.in_path) as fh:
        rows = json.load(fh)

    parquet_rows: List[dict] = []
    for i, r in enumerate(rows):
        q = r.get("question", "")
        if not q:
            continue
        parquet_rows.append({
            "data_source": args.data_source,
            "prompt": [
                {"role": "user", "content": q},
            ],
            "ability": "math",
            "reward_model": {"ground_truth": ""},
            "extra_info": {
                "index": i,
                "question": q,
                "avg_rm": float(r.get("avg_rm", 0.0)),
                "std_rm": float(r.get("std_rm", 0.0)),
                "num_answers": int(r.get("num_answers", 0)),
                "save_name": args.save_name,
            },
        })

    if not parquet_rows:
        print(f"[upload_rm] WARN empty parquet from {args.in_path}", flush=True)
    df = pd.DataFrame(parquet_rows)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"[upload_rm] wrote {len(df)} rows -> {args.out_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, required=True,
                   help="filter_rm.py JSON output")
    p.add_argument("--save_name", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--data_source", type=str, default="rzero_self_play_rm")
    main(p.parse_args())
