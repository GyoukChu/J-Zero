#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate solver-model responses (--model, default Qwen3-4B-Base) for one or
more verl-format eval parquets, using vLLM. Per-dataset mode:
  - greedy:   temperature=0.0, n=1
  - sampling: temperature=0.6, top_p=1.0, n=N   (default N=32, for AMC/AIME mean@N)

Always:
  - max_tokens=4096
  - System prompt "Please reason step by step, and put your final answer within \\boxed{}."
  - apply_chat_template; user prompt = question only

Output: <dataset>.jsonl with one record per (question, sample_idx):
  {"data_source", "question", "ground_truth", "sample_idx", "raw"}
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import vllm
from transformers import AutoTokenizer


SAMPLING_DATASETS = {"amc23", "aime24", "aime25"}  # mean@N
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def load_parquet(path: str) -> List[Dict]:
    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        q = ""
        for m in row["prompt"]:
            if m["role"] == "user":
                q = m["content"]
                break
        if not q:
            q = row["prompt"][-1]["content"]
        out.append({
            "data_source": row["data_source"],
            "question": q,
            "ground_truth": row["reward_model"]["ground_truth"],
        })
    return out


def build_prompts(records: List[Dict], tokenizer) -> List[str]:
    chats = [
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": r["question"]}]
        for r in records
    ]
    if tokenizer.chat_template:
        # add_special_tokens=False: the Qwen3 chat_template already inserts BOS,
        # and True would double-prepend it and silently shift tokenization.
        return [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=True, add_special_tokens=False,
            )
            for c in chats
        ]
    return [f"system: {SYSTEM}\nuser: {r['question']}" for r in records]


def gen_one_dataset(model, tokenizer, ds_name: str, parquet_path: str,
                    out_path: str, n_samples: int, max_tokens: int,
                    dp_shard: int = 0, dp_total: int = 1) -> Dict:
    """Generate responses for one benchmark.

    DP-by-question: shard `dp_shard` of `dp_total` handles only the records where
    `original_idx % dp_total == dp_shard`. dp_total=1 -> single-shard.
    """
    full_recs = load_parquet(parquet_path)
    if dp_total > 1:
        recs = [r for i, r in enumerate(full_recs) if i % dp_total == dp_shard]
        tag = f"[gen {dp_shard}/{dp_total}]"
    else:
        recs = full_recs
        tag = "[gen]"
    if not recs:
        print(f"{tag} {ds_name}: empty shard, skip", flush=True)
        return {"n_questions": 0, "n_records": 0, "mode": "skip-empty"}
    prompts = build_prompts(recs, tokenizer)

    if ds_name in SAMPLING_DATASETS:
        sp = vllm.SamplingParams(
            temperature=0.6,
            top_p=1.0,
            n=n_samples,
            max_tokens=max_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
        )
        mode = f"sampling(n={n_samples}, temp=0.6, top_p=1.0)"
    else:
        sp = vllm.SamplingParams(
            temperature=0.0,
            top_p=1.0,
            n=1,
            max_tokens=max_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
        )
        mode = "greedy(temp=0.0, n=1)"

    print(f"{tag} {ds_name}: {len(prompts)} questions (of {len(full_recs)}), mode={mode}", flush=True)
    responses = model.generate(prompts, sampling_params=sp, use_tqdm=True)

    # Write via tmp + atomic rename: the rerun path skips a dataset whenever
    # out_path exists, so a shard truncated by a mid-write kill must never be
    # left under the final name (it would silently shrink the merged n).
    n_records = 0
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as fh:
        for r, resp in zip(recs, responses):
            for idx, o in enumerate(resp.outputs):
                fh.write(json.dumps({
                    "data_source": r["data_source"],
                    "question": r["question"],
                    "ground_truth": r["ground_truth"],
                    "sample_idx": idx,
                    "raw": o.text,
                }) + "\n")
                n_records += 1
    os.replace(tmp_path, out_path)
    print(f"{tag} {ds_name}: wrote {n_records} records -> {out_path}", flush=True)
    return {"n_questions": len(recs), "n_records": n_records, "mode": mode}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--data_dir", required=True,
                   help="dir containing <dataset>.parquet files (verl format)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--datasets", nargs="+", required=True,
                   help="dataset basenames (e.g., math500 gsm8k amc23 aime24)")
    p.add_argument("--n_samples", type=int, default=32,
                   help="N samples for sampling-mode datasets")
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--gpu_mem_util", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=8192,
                   help="vLLM max_model_len. Bounds the KV cache and packed "
                        "batch size; must be >= max_tokens + longest prompt.")
    p.add_argument("--dtype", default="bfloat16",
                   help="vLLM compute dtype.")
    p.add_argument("--trust_remote_code", action="store_true", default=True,
                   help="Pass trust_remote_code to vLLM + tokenizer.")
    p.add_argument("--dp_shard", type=int, default=0,
                   help="DP-by-question shard id (0-indexed).")
    p.add_argument("--dp_total", type=int, default=1,
                   help="Total DP shards (=number of GPUs). 1 = single-shard "
                        "(default, backwards compat).")
    args = p.parse_args()
    assert 0 <= args.dp_shard < args.dp_total, "dp_shard must be in [0, dp_total)"

    shard_suffix = f".shard{args.dp_shard}of{args.dp_total}" if args.dp_total > 1 else ""
    tag = f"[gen {args.dp_shard}/{args.dp_total}]" if args.dp_total > 1 else "[gen]"

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = vllm.LLM(
        model=args.model, tokenizer=args.model,
        tensor_parallel_size=1,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=args.trust_remote_code,
    )

    summary = {"model": args.model, "max_tokens": args.max_tokens,
               "n_samples_for_sampling": args.n_samples,
               "dp_shard": args.dp_shard, "dp_total": args.dp_total,
               "datasets": {}}
    for ds in args.datasets:
        parquet = os.path.join(args.data_dir, f"{ds}.parquet")
        if not os.path.isfile(parquet):
            print(f"{tag} skip {ds} ({parquet} not found)", flush=True)
            continue
        out_path = os.path.join(args.out_dir, f"{ds}{shard_suffix}.jsonl")
        merged_path = os.path.join(args.out_dir, f"{ds}.jsonl")
        if os.path.isfile(out_path) or (args.dp_total > 1 and os.path.isfile(merged_path)):
            print(f"{tag} skip {ds} (output exists)", flush=True)
            continue
        summary["datasets"][ds] = gen_one_dataset(
            model, tokenizer, ds, parquet, out_path,
            args.n_samples, args.max_tokens,
            dp_shard=args.dp_shard, dp_total=args.dp_total,
        )

    gen_summary_name = f"gen_summary{shard_suffix}.json"
    with open(os.path.join(args.out_dir, gen_summary_name), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
