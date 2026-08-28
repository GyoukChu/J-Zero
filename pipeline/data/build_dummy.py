#!/usr/bin/env python
"""
Build the static prompt parquet that seeds J-Zero's Challenger GRPO trainer.

The challenger learns to produce one general-domain user request per rollout
(NOT a math problem). CHALLENGER_INSTRUCTION is imported from
pipeline/question_gen/question_generate.py, which owns the single definition:
the training prompt built here and the inference prompt used at
question-generation time must be the same string.

Relative to the G-Zero baseline's prompt
(baselines/g-zero/g_zero/prompts.py::CHALLENGER_INSTRUCTION), J-Zero's drops
the entire hint mechanism — the "plus a short hint ..." clause, both hint
rules, the <hint> blocks in the two few-shot examples — since J-Zero's
challenger emits a request only. The two strings are independent copies and
have also drifted cosmetically (em-dash vs hyphen in two places).

Chat shape: user = CHALLENGER_INSTRUCTION, single turn, no system role, so the
chat template owns any system slot — same as the G-Zero baseline. The R-Zero
math version (baselines/r-zero/data/build_dummy.py) instead pairs a
"competition-math problem setter" system turn with a short user turn.

Schema matches the verl val parquets (e.g. data/math500.parquet).
"""

import argparse
import os
import sys

import pandas as pd

# Both files are run as scripts by absolute path, so sys.path[0] is the
# script's own directory, not the recipe root (itself a copy installed as
# <verl>/recipe_J-Zero and invoked from several working directories). Derive
# the root from __file__ so the import resolves regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from question_gen.question_generate import CHALLENGER_INSTRUCTION  # noqa: E402


def main(args: argparse.Namespace) -> None:
    rows = []
    for i in range(args.n):
        rows.append(
            {
                "data_source": "jzero_challenger_seed",
                "prompt": [
                    {"role": "user", "content": CHALLENGER_INSTRUCTION},
                ],
                "ability": "general",
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
