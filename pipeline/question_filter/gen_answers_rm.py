#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-shard answer generation for the RM-variant question filter.

Reads the challenger-generated questions for shard `--seed`, samples the solver
vLLM `--num_samples` times per question (default 10), and writes
`<save_name>_<seed>_answers.json` of

    [{"question": str, "answers": [str, ...]}, ...]

NO majority voting, NO mathruler grader — RM scoring happens downstream.
Sampling params, GPU mem, and seed handling mirror the R-Zero baseline's
`question_filter/evaluate.py` so behavior stays comparable.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import vllm
from transformers import AutoTokenizer


SOLVER_SYS = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def main(args: argparse.Namespace) -> None:
    in_path = os.path.join(args.in_dir, f"{args.save_name}_{args.seed}.json")
    out_path = os.path.join(args.out_dir, f"{args.save_name}_{args.seed}_answers.json")
    if not os.path.exists(in_path):
        print(f"[gen_answers seed={args.seed}] ERROR: no input {in_path}", flush=True)
        raise SystemExit(2)

    with open(in_path) as fh:
        data = json.load(fh)

    # score==0 is question_gen's "well-formed" marker; format-fail (-1) items
    # are dropped from the training pool.
    valid = [item for item in data if item.get("score") == 0]
    questions = [item["question"] for item in valid]
    if not questions:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump([], fh)
        print(f"[gen_answers seed={args.seed}] nothing to do", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        seed=int(args.seed),
    )
    sp = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=0.99,
        stop_token_ids=[tokenizer.eos_token_id],
        n=args.num_samples,
    )

    chats = [
        [
            {"role": "system", "content": SOLVER_SYS},
            {"role": "user", "content": q},
        ]
        for q in questions
    ]
    if tokenizer.chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=True, add_special_tokens=True,
            )
            for c in chats
        ]
    else:
        prompts = ["\n".join(f"{m['role']}: {m['content']}" for m in c) for c in chats]

    print(f"[gen_answers seed={args.seed}] {len(prompts)} Q × n={args.num_samples}",
          flush=True)
    responses = model.generate(prompts, sampling_params=sp, use_tqdm=True)

    out: List[dict] = []
    for q, response in zip(questions, responses):
        texts = [o.text for o in response.outputs]
        out.append({"question": q, "answers": texts})

    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    print(f"[gen_answers seed={args.seed}] wrote {len(out)} -> {out_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seed", type=str, default="0")
    p.add_argument("--save_name", type=str, required=True)
    p.add_argument("--in_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--max_tokens", type=int, default=4096)
    main(p.parse_args())
