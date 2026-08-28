#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare IFEval prompts (chat form) for gen_vllm.py.

Uses the official prompt set shipped with the google-research
instruction_following_eval module (data/input_data.jsonl: 541 prompts). The
prompt is the user turn verbatim -- no system prompt, no CoT instruction --
because IFEval measures direct instruction following and any extra reasoning
text counts against the constraints, exactly as in the official protocol.

Output JSONL line: {"id": key, "messages": [...], "prompt": <verbatim prompt>};
the verbatim prompt is kept because the official scorer matches responses to
inputs by exact prompt string.
"""
import argparse, json, os
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = str(_BENCH_ROOT /
                    "third_party/instruction_following_eval/data/input_data.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--input_data", default=DEFAULT_INPUT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.input_data) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if args.limit and n >= args.limit:
                break
            ex = json.loads(line)
            fout.write(json.dumps({
                "id": ex["key"],
                "messages": [{"role": "user", "content": ex["prompt"]}],
                "prompt": ex["prompt"],
            }) + "\n")
            n += 1
    print(f"[prep_ifeval] wrote {n} prompts -> {args.out}")


if __name__ == "__main__":
    main()