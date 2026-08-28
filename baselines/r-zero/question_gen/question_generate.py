#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate `--num_samples` problems with the challenger model using vLLM on a
single GPU. Output: JSON list of {question, answer, score=0 or -1}.

Mirrors R-Zero's question_generate/question_generate.py, except the output path
is a CLI arg (not env STORAGE_PATH) and temperature/top_p are configurable.
"""

import argparse
import json
import os
from typing import List

import regex as re
import vllm
from transformers import AutoTokenizer
from vllm.outputs import RequestOutput


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


def extract_boxed(text: str) -> List[str]:
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


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        seed=int(args.seed),
    )
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
    else:
        prompt = f"system: {chat[0]['content']}\nuser: {chat[1]['content']}"
    sp = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    print(f"[gen seed={args.seed}] generating {args.num_samples}")
    completions: List[RequestOutput] = model.generate(
        [prompt] * args.num_samples, sampling_params=sp
    )

    results = []
    for c in completions:
        response = c.outputs[0].text
        try:
            qs = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            ans = extract_boxed(response)
            if qs and ans:
                results.append(
                    {
                        "question": qs[-1].strip(),
                        "answer": ans[-1].strip(),
                        "score": 0,
                    }
                )
            else:
                results.append({"question": response, "answer": "", "score": -1})
        except Exception:
            results.append({"question": response, "answer": "", "score": -1})

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.save_name}_{args.seed}.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[gen seed={args.seed}] wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--seed", type=str, default="0", help="seed and shard suffix")
    p.add_argument("--save_name", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.99)
    main(p.parse_args())
