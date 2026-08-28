#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variant of the R-Zero baseline's start_vllm_server.py
(baselines/r-zero/vllm_service/) returning the RAW n candidate answers per
question, WITHOUT majority voting or pseudo-label extraction, because the
reward signal here is a learned RM rather than a vote.

Single GPU per server; the challenger stage runs one on GPU 2 (4-GPU layout).

I/O contract (mirrors the baseline's shape):
  - GET /health           -> {"status":"ok"}
  - GET /hello?name=<jsonpath>
        Input file: JSON list of {"question": str, ...} items.
        Writes `<jsonpath>_results.json` (same convention) as
            [{"question": str, "answers": [str, ...]}, ...]
        The baseline's `answer`, `score`, `results` fields are deliberately
        omitted — the RM reward fn scores downstream.

Sampling params match the paper-compliant solver settings: temperature=1.0,
top_p=0.99, n=10 (overridable via --n), max_tokens=4096.
"""

from __future__ import annotations

import argparse
import json
import os
import threading

import torch
from flask import Flask, jsonify, request
from transformers import AutoTokenizer


# MUST match question_filter/gen_answers_rm.py:SOLVER_SYS, so reward-stage
# solver answers come from the same answer distribution as the filter pool.
SOLVER_SYS = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def make_app(model, tokenizer, sample_params, pause_event):
    app = Flask(__name__)
    # `LLM.generate()` is NOT re-entrant against the same engine: a second
    # concurrent call corrupts the first's request queue and its output buffer
    # is never flushed. Flask runs threaded=True and RZeroBatchManager can
    # submit two reward chunks back-to-back at a step boundary, so two /hello
    # requests do land together. This lock keeps the engine single-tenant at no
    # throughput cost, since the engine is the bottleneck anyway.
    inference_lock = threading.Lock()

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/hello", methods=["GET"])
    def hello():
        name = request.args.get("name", "")
        if not name or not os.path.exists(name):
            return jsonify({"error": f"missing file: {name}"}), 400
        print(f"[server] processing {name}", flush=True)

        with open(name, "r") as fh:
            data = json.load(fh)

        questions = [item.get("question", "") for item in data]

        valid_chats = []
        valid_idx_set = set()
        for i, q in enumerate(questions):
            if q:
                valid_idx_set.add(i)
                valid_chats.append([
                    {"role": "system", "content": SOLVER_SYS},
                    {"role": "user", "content": q},
                ])

        if valid_chats:
            if tokenizer.chat_template:
                prompts = [
                    tokenizer.apply_chat_template(
                        chat,
                        tokenize=False,
                        add_generation_prompt=True,
                        add_special_tokens=True,
                    )
                    for chat in valid_chats
                ]
            else:
                prompts = [
                    "\n".join(f"{m['role']}: {m['content']}" for m in c)
                    for c in valid_chats
                ]
            # pause_event only signals to external probes that a generate is
            # in flight; correctness rests on inference_lock alone.
            with inference_lock:
                pause_event.set()
                try:
                    torch.cuda.synchronize()
                    print(f"[server] generate start name={name} n_prompts={len(prompts)}", flush=True)
                    responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
                    print(f"[server] generate done  name={name} n_responses={len(responses)}", flush=True)
                finally:
                    pause_event.clear()
        else:
            responses = []

        out_list = []
        cursor = 0
        for i, q in enumerate(questions):
            if i in valid_idx_set and cursor < len(responses):
                texts = [o.text for o in responses[cursor].outputs]
                out_list.append({"question": q, "answers": texts})
                cursor += 1
            else:
                # Format-failed input still emits a row, so downstream RM
                # scoring sees the row count it expects.
                out_list.append({"question": q, "answers": []})

        out_path = name.replace(".json", "_results.json")
        with open(out_path, "w") as fh:
            json.dump(out_list, fh)
        try:
            os.remove(name)
        except FileNotFoundError:
            pass

        print(f"[server] wrote {out_path}", flush=True)
        return jsonify({"message": f"Processed {name}", "out": out_path})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="5000")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--gpu_mem_util",
        type=float,
        default=float(os.getenv("RZERO_REWARD_MEM_UTIL", "0.7")),
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    import vllm  # under __main__ for vLLM v1 multiproc detection.

    print(f"[init] Loading model from {args.model_path} on port {args.port} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = vllm.LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    sample_params = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=0.99,
        stop_token_ids=[tokenizer.eos_token_id],
        n=args.n,
    )

    pause_event = threading.Event()
    app = make_app(model, tokenizer, sample_params, pause_event)
    app.run(host="127.0.0.1", port=int(args.port), threaded=True)


if __name__ == "__main__":
    main()
