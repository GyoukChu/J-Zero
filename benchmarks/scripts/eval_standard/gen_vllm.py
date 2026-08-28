#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Greedy embedded-vLLM generation driver (used by the IFEval step of run_eval.sh).

Input  : JSONL of {"id": <hashable>, "messages": [{"role", "content"}, ...]}.
Output : <out_path> JSONL of {"id": ..., "raw": <model text>}.

Generation matches the repo's math eval (recipe_R-Zero/evaluate/
gen_eval_responses.py): greedy (temperature=0.0, top_p=1.0, n=1), bf16,
tensor_parallel_size=1, and apply_chat_template(add_generation_prompt=True,
add_special_tokens=False) with NO enable_thinking, so the Qwen3 ChatML prompt
ends at "<|im_start|>assistant\\n" with no forced "<think>" block ("no <think>
prepend on base eval"). Unlike the math driver we also stop on "<|im_end|>"
(eos is honored too) so a turn cannot run on into a fabricated next one, which
would break multiple-choice answer extraction.

DP-by-question sharding mirrors gen_eval_responses.py: shard `dp_shard` of
`dp_total` processes only records where (line_index % dp_total == dp_shard).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def load_records(path: str) -> List[Dict]:
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def build_prompts(records: List[Dict], tokenizer) -> List[str]:
    if not tokenizer.chat_template:
        raise RuntimeError("tokenizer has no chat_template; these checkpoints "
                           "are chat-formatted and require one")
    prompts = []
    for r in records:
        msgs = r["messages"]
        # add_special_tokens=False: the chat_template already inserts them
        # (True double-prepends BOS). No enable_thinking — see module docstring.
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            add_special_tokens=False,
        ))
    return prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--dp_shard", type=int, default=0)
    p.add_argument("--dp_total", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_mem_util", type=float, default=0.9)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    args = p.parse_args()
    assert 0 <= args.dp_shard < args.dp_total, "dp_shard must be in [0, dp_total)"

    tag = f"[gen {args.dp_shard}/{args.dp_total}]" if args.dp_total > 1 else "[gen]"

    if os.path.isfile(args.out_path):
        print(f"{tag} output exists, skip: {args.out_path}", flush=True)
        return

    import vllm
    from transformers import AutoTokenizer

    full = load_records(args.in_path)
    if args.dp_total > 1:
        recs = [r for i, r in enumerate(full) if i % args.dp_total == args.dp_shard]
    else:
        recs = full
    if not recs:
        print(f"{tag} empty shard, writing empty file", flush=True)
        open(args.out_path, "w").close()
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = build_prompts(recs, tokenizer)

    model = vllm.LLM(
        model=args.model, tokenizer=args.model,
        tensor_parallel_size=1,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=args.trust_remote_code,
    )
    sp = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, n=1,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
        stop_token_ids=[tokenizer.eos_token_id],
    )
    print(f"{tag} {len(prompts)} prompts (of {len(full)}), greedy, "
          f"max_tokens={args.max_tokens}", flush=True)
    responses = model.generate(prompts, sampling_params=sp, use_tqdm=True)

    tmp = args.out_path + ".tmp"
    with open(tmp, "w") as fh:
        for r, resp in zip(recs, responses):
            fh.write(json.dumps({"id": r["id"], "raw": resp.outputs[0].text}) + "\n")
    os.replace(tmp, args.out_path)
    print(f"{tag} wrote {len(recs)} records -> {args.out_path}", flush=True)


if __name__ == "__main__":
    main()