#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
J-Zero challenger question generator (general-domain).

Owns J-Zero's CHALLENGER_INSTRUCTION; pipeline/data/build_dummy.py imports it
from here so the challenger's GRPO training prompt and its inference prompt
cannot drift apart.

Differences vs the R-Zero math version
(baselines/r-zero/question_gen/question_generate.py):
  - Prompt: user-only and general-domain — the G-Zero baseline's prompt minus
    the hint mechanism (pipeline/data/build_dummy.py documents the diff).
  - Parser: only <question>...</question> is required, no \\boxed{} answer.
    Format OK -> {question, answer="", score=0}; format fail ->
    {question=response, answer="", score=-1}.

Downstream (gen_answers_rm.py, challenger_reward_rm.py, ...) ignores `answer`
and gates on score==0, so answer="" is the minimal-change adapter.
"""

import argparse
import json
import os
from typing import List

# vllm / transformers / regex are imported inside main() only: build_dummy.py
# imports CHALLENGER_INSTRUCTION from here before any GPU work, and pulling the
# vLLM stack in to read one string would cost it ~10s and break parquet builds
# on hosts without vLLM installed.


CHALLENGER_INSTRUCTION = (
    "Produce one challenging request that a real user might ask a capable "
    "assistant.\n\n"
    "The request should come from a general-domain distribution. Sample across "
    "task types, not from a single area. Examples of task types you can draw "
    "from:\n"
    "  - writing (email, story, essay, pitch, review, poem)\n"
    "  - explanation (make a concept clear to a specific audience)\n"
    "  - advice or planning (career, travel, project, learning)\n"
    "  - analysis (argument, text, dataset description, product)\n"
    "  - coding (small function, debugging, design question)\n"
    "  - role-play, dialogue, or creative tasks\n"
    "  - open-ended questions about ethics, science, everyday life\n"
    "  - reasoning, math, or logic problems (fine to include - roughly 1 in "
    "6 requests, no more)\n\n"
    "Weight the non-math categories above heavily. A little math is good "
    "for diversity, but it should not dominate - favor tasks where the "
    "response quality depends on tone, structure, audience-fit, clarity, or "
    "creativity, not just arithmetic correctness.\n\n"
    "Requirements:\n"
    "- The request must be self-contained and non-trivial to answer well.\n"
    "- Wrap the request in <question> and </question> tags.\n"
    "- Output nothing else before, between, or after the tagged blocks.\n\n"
    "Example 1 (writing):\n"
    "<question>Write a resignation email to my manager that keeps the door "
    "open for future collaboration. I've been at the company for 4 years "
    "and I'm leaving to join a competitor. Tone should be professional and "
    "warm without being effusive.</question>\n\n"
    "Example 2 (explanation):\n"
    "<question>Explain what a Kalman filter does to a software engineer who "
    "is comfortable with linear algebra but has never touched signal "
    "processing. Avoid control-theory jargon where possible.</question>\n\n"
    "Now produce one new request of your own:\n"
)


def main(args: argparse.Namespace) -> None:
    import regex as re
    import vllm
    from transformers import AutoTokenizer
    from vllm.outputs import RequestOutput  # noqa: F401  (annotation below)

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
        {"role": "user", "content": CHALLENGER_INSTRUCTION},
    ]
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
    else:
        prompt = f"user: {chat[0]['content']}"
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
            # G-Zero parser: <question>...</question> only, no \boxed{}. Take
            # the LAST match — the prompt's own examples carry <question> tags,
            # so an echoing model would otherwise shadow its new request. Same
            # rule as challenger_reward_rm.py.
            qs = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            if qs and qs[-1].strip():
                results.append(
                    {
                        "question": qs[-1].strip(),
                        "answer": "",
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
