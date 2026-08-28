#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Join a gen_vllm.py/gen_http.py responses.jsonl with the prepared prompts.jsonl (by id) and
emit an AlpacaEval `model_outputs.json`: a list of
{"instruction", "output", "generator": <slug>}.

Usage:
  to_alpaca_outputs.py --prompts prompts.jsonl --responses responses.jsonl \
      --generator <slug> --out model_outputs.json
"""
from __future__ import annotations

import argparse
import json


def load_jsonl(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--responses", required=True)
    p.add_argument("--generator", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    prompt_recs = load_jsonl(args.prompts)
    prompts = {r["id"]: r["instruction"] for r in prompt_recs}
    order = {r["id"]: i for i, r in enumerate(prompt_recs)}
    resps = load_jsonl(args.responses)

    # keep order matching the prompts file so model_outputs and reference align
    resps.sort(key=lambda r: order.get(r["id"], 1 << 30))

    out = []
    missing = 0
    for r in resps:
        rid = r["id"]
        if rid not in prompts:
            missing += 1
            continue
        out.append({
            "instruction": prompts[rid],
            "output": r["raw"],
            "generator": args.generator,
        })

    with open(args.out, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"wrote {len(out)} outputs -> {args.out} "
          f"(generator={args.generator}, missing_ids={missing}, "
          f"n_resp={len(resps)}, n_prompts={len(prompts)})")


if __name__ == "__main__":
    main()