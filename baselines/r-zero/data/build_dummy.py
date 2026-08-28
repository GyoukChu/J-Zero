#!/usr/bin/env python
"""
Build a tiny verl-format parquet whose prompts ask the challenger to generate a
new problem. Every row is the same static chat; the row count only has to keep
the trainer from hitting an epoch end (challenger_train.sh passes --n 2048,
far above the GBS=16 x 5 steps it consumes).

Schema matches the verl math parquet format (data_source / prompt / ability /
reward_model / extra_info).
"""

import argparse
import os

import pandas as pd


SYSTEM_PROMPT = (
    "You are an expert competition-math problem setter.\n"
    "FIRST, in your private scratch-pad, think step-by-step to design a "
    "brand-new, non-trivial problem. The problem could come from any field of "
    "mathematics, including but not limited to algebra, geometry, number "
    "theory, combinatorics, prealgebra, probability, statistics, and calculus. "
    "Aim for a difficulty such that fewer than 30 % of advanced high-school "
    "students could solve it. Avoid re-using textbook cliches or famous "
    "contest problems.\n"
    "THEN, without revealing any of your private thoughts, output **exactly** "
    "the following two blocks:\n\n"
    "<question>\n{The full problem statement on one or more lines}\n"
    "</question>\n\n"
    r"\boxed{final_answer}"
    "\n\nDo NOT output anything else - no explanations, no extra markup."
)
USER_PROMPT = (
    "Generate one new, challenging reasoning question now. "
    "Remember to format the output exactly as instructed."
)


def main(args: argparse.Namespace) -> None:
    rows = []
    for i in range(args.n):
        rows.append(
            {
                "data_source": "rzero_challenger_seed",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
                "ability": "math",
                "reward_model": {"ground_truth": ""},
                "extra_info": {"index": i, "split": "train"},
            }
        )
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"wrote {len(df)} rows -> {args.out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--n", type=int, default=2048)
    main(p.parse_args())
