#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HTTP-based gen driver — drop-in replacement for gen_vllm.py against a
persistent vLLM OpenAI-compatible endpoint. Sends prompts in parallel from a
thread pool so one persistent DP=N server batches them across replicas.

/v1/completions, not /v1/chat/completions: prompts are built client-side
exactly as gen_vllm.py builds them (add_generation_prompt=True,
add_special_tokens=False, no enable_thinking) and POSTed as raw text. Letting
vLLM re-apply the chat template server-side could inject the <think> block on
Qwen3, which the project's "no <think> prepend on base eval" rule forbids.

Output JSONL: {"id": ..., "raw": <model text>} — same as gen_vllm.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import requests
from transformers import AutoTokenizer


def load_records(path: str) -> List[Dict]:
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def build_prompt(rec, tokenizer):
    return tokenizer.apply_chat_template(
        rec["messages"], tokenize=False,
        add_generation_prompt=True, add_special_tokens=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default="http://127.0.0.1:7451/v1",
                   help="vLLM OpenAI-compatible base URL (must end without /completions)")
    p.add_argument("--model_name", required=True,
                   help="served-model-name as registered on the endpoint")
    p.add_argument("--tokenizer", required=True,
                   help="HF path or local dir used to build chat-template prompts. "
                        "MUST match the served model.")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--dp_shard", type=int, default=0)
    p.add_argument("--dp_total", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--stop", action="append", default=None,
                   help="extra stop sequences. Default: ['<|im_end|>']")
    p.add_argument("--workers", type=int, default=128,
                   help="thread-pool size for concurrent HTTP requests")
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-request timeout seconds")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--progress_every", type=int, default=200)
    args = p.parse_args()

    if args.stop is None:
        args.stop = ["<|im_end|>"]

    assert 0 <= args.dp_shard < args.dp_total, "dp_shard must be in [0, dp_total)"
    tag = f"[gen_http {args.dp_shard}/{args.dp_total}]" if args.dp_total > 1 else "[gen_http]"

    if os.path.isfile(args.out_path):
        print(f"{tag} output exists, skip: {args.out_path}", flush=True)
        return

    full = load_records(args.in_path)
    if args.dp_total > 1:
        recs = [r for i, r in enumerate(full) if i % args.dp_total == args.dp_shard]
    else:
        recs = full
    if not recs:
        print(f"{tag} empty shard, writing empty file", flush=True)
        open(args.out_path, "w").close()
        return

    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    except AttributeError:
        # Checkpoints saved by some transformers versions serialize
        # extra_special_tokens as a LIST in tokenizer_config.json; the pinned
        # 4.56 expects a dict and dies on .keys(). Override the field away —
        # it is empty in those checkpoints.
        tok = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True, extra_special_tokens={})
    if not tok.chat_template:
        raise RuntimeError("tokenizer has no chat_template; this driver "
                           "expects chat-formatted checkpoints")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = [build_prompt(r, tok) for r in recs]
    url = args.base_url.rstrip("/") + "/completions"
    session = requests.Session()
    headers = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}

    def call_one(i: int):
        payload = {
            "model": args.model_name,
            "prompt": prompts[i],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "stop": args.stop,
            "n": 1,
        }
        last_err = ""
        for attempt in range(args.retries):
            try:
                r = session.post(url, json=payload, headers=headers, timeout=args.timeout)
                r.raise_for_status()
                data = r.json()
                return i, data["choices"][0]["text"]
            except Exception as exc:  # noqa: BLE001
                last_err = repr(exc)
                time.sleep(1.0 * (attempt + 1))
        print(f"{tag} WARN giving up on idx {i}: {last_err}", flush=True)
        return i, ""

    out = [None] * len(recs)
    done = 0
    start = time.monotonic()
    print(f"{tag} {len(prompts)} prompts -> {url} (workers={args.workers}, "
          f"max_tokens={args.max_tokens})", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(call_one, i) for i in range(len(prompts))]
        for fut in as_completed(futs):
            i, text = fut.result()
            out[i] = text
            done += 1
            if done % args.progress_every == 0:
                rate = done / max(time.monotonic() - start, 1e-6)
                print(f"{tag} {done}/{len(prompts)} ({rate:.1f} req/s)", flush=True)

    tmp = args.out_path + ".tmp"
    with open(tmp, "w") as fh:
        for r, text in zip(recs, out):
            fh.write(json.dumps({"id": r["id"], "raw": text or ""}) + "\n")
    os.replace(tmp, args.out_path)
    elapsed = time.monotonic() - start
    print(f"{tag} wrote {len(recs)} -> {args.out_path} (took {elapsed:.1f}s)",
          flush=True)


if __name__ == "__main__":
    main()