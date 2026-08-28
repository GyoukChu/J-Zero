#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
For one shard of generated questions, run the *current solver* m=10 times per
question, compute majority-answer p_hat, drop heuristic-bad questions, write
results JSON.

Mirrors R-Zero question_evaluate/evaluate.py, except that boxed extraction and
answer equivalence go through reward/answer_equiv.py (verl math_verify).
"""

import argparse
import json
import os
import sys
from typing import List

import vllm
from transformers import AutoTokenizer

# Shared with vllm_service/start_vllm_server.py so the two cannot drift apart.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reward.answer_equiv import cluster_majority, extract_boxed_content  # noqa: E402


def main(args: argparse.Namespace) -> None:
    in_path = os.path.join(args.in_dir, f"{args.save_name}_{args.seed}.json")
    out_path = os.path.join(args.out_dir, f"{args.save_name}_{args.seed}_results.json")
    if not os.path.exists(in_path):
        print(f"[eval seed={args.seed}] ERROR: no input file {in_path}", flush=True)
        raise SystemExit(2)

    with open(in_path) as fh:
        data = json.load(fh)
    correct = [item for item in data if item.get("score") == 0]
    if not correct:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump([], fh)
        print(f"[eval seed={args.seed}] nothing to do")
        return

    questions = [item["question"] for item in correct]
    answers = [item["answer"] for item in correct]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        seed=int(args.seed),
    )
    # Paper Appendix B.1.1: top_p=0.99, top_k unspecified. R-Zero official adds
    # top_k=40 + top_p=1.0; we follow the paper.
    sp = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=0.99,
        stop_token_ids=[tokenizer.eos_token_id],
        n=args.num_samples,
    )

    chats = [
        [
            {
                "role": "system",
                "content": (
                    "Please reason step by step, and put your final answer "
                    "within \\boxed{}."
                ),
            },
            {"role": "user", "content": q},
        ]
        for q in questions
    ]
    if tokenizer.chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=True, add_special_tokens=True
            )
            for c in chats
        ]
    else:
        prompts = [
            f"system: {c[0]['content']}\nuser: {c[1]['content']}" for c in chats
        ]

    print(
        f"[eval seed={args.seed}] generating {args.num_samples} per q "
        f"for {len(prompts)} questions"
    )
    responses = model.generate(prompts, sampling_params=sp, use_tqdm=True)

    out: List[dict] = []
    for response, _golden, question in zip(responses, answers, questions):
        try:
            boxed = [extract_boxed_content(o.text) for o in response.outputs]
            boxed = [r for r in boxed if r]
            if not boxed:
                continue
            maj, score, _ = cluster_majority(boxed)
            if not maj:
                continue

            # Same heuristic drops as R-Zero.
            if (
                "证明" in question
                or "box" in question.lower()
                or "text" in maj.lower()
            ):
                continue

            out.append(
                {
                    "question": question,
                    "answer": maj,
                    "score": score,
                    "results": boxed,
                }
            )
        except Exception as exc:  # pragma: no cover
            print(f"[eval seed={args.seed}] error: {exc}")
            continue

    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[eval seed={args.seed}] wrote {len(out)} -> {out_path}")


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
