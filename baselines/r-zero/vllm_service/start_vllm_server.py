#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask + vLLM server: receives a JSON file of (question, answer) items, generates
n candidate answers per question, returns majority-vote `p_hat` (= count of
majority answer / n) per question. Ported from the official R-Zero
`vllm_service_init/start_vllm_server.py`, stripped to two servers and adapted to
this directory layout; answer equivalence goes through reward/answer_equiv.py.

Single GPU per server. Two servers run on GPU 2, 3 (ports RZERO_BASE_PORT+i).

vLLM v1 multiprocessing requires the engine to be initialized under
`if __name__ == "__main__":`, or child workers re-execute module-level code and
crash on `_check_not_importing_main`.
"""

import argparse
import json
import os
import sys
import threading
import time

import torch
from flask import Flask, jsonify, request
from transformers import AutoTokenizer

# Shared with question_filter/evaluate.py so the two cannot drift apart.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reward.answer_equiv import cluster_majority, extract_boxed_content  # noqa: E402


# Flask runs threaded=True, but `vllm.LLM.generate()` is NOT thread-safe: two
# request threads calling it on one engine wedge the v1 EngineCore in a futex
# deadlock (0% GPU, memory still resident). The batch manager triggers exactly
# that when overlapping reward chunks both GET /hello here. Serializing only
# queues generation, so results and hyperparameters are unaffected, and the
# servers still run in parallel as separate processes.
_GENERATE_LOCK = threading.Lock()


def make_app(model, tokenizer, sample_params, pause_event):
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/hello", methods=["GET"])
    def hello():
        pause_event.set()
        torch.cuda.synchronize()

        name = request.args.get("name", "")
        if not name or not os.path.exists(name):
            pause_event.clear()
            return jsonify({"error": f"missing file: {name}"}), 400
        print(f"[server] processing {name}")

        with open(name, "r") as fh:
            data = json.load(fh)

        questions = [item.get("question", "") for item in data]
        answers = [item.get("answer", "") for item in data]

        valid_chats: list = []
        valid_idx_set: set = set()
        for i, (q, a) in enumerate(zip(questions, answers)):
            if q and a:
                valid_idx_set.add(i)
                valid_chats.append(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Please reason step by step, and put your final "
                                "answer within \\boxed{}."
                            ),
                        },
                        {"role": "user", "content": q},
                    ]
                )

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
                    f"system: {chat[0]['content']}\nuser: {chat[1]['content']}"
                    for chat in valid_chats
                ]
            # See _GENERATE_LOCK: concurrent generate() deadlocks the v1 engine.
            with _GENERATE_LOCK:
                responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
        else:
            responses = []

        def process_single(question: str, golden: str, response) -> dict:
            boxed = [extract_boxed_content(out.text) for out in response.outputs]
            maj, p_hat, _ = cluster_majority(boxed)
            if not maj:
                return {"question": question, "answer": "", "score": 0.0, "results": boxed}
            return {"question": question, "answer": maj, "score": p_hat, "results": boxed}

        out_list: list = []
        cursor = 0
        for i, (q, a) in enumerate(zip(questions, answers)):
            if i in valid_idx_set and cursor < len(responses):
                try:
                    out_list.append(process_single(q, a, responses[cursor]))
                except Exception as exc:  # pragma: no cover - defensive
                    out_list.append(
                        {"question": q, "answer": a, "score": -1, "results": [], "error": str(exc)}
                    )
                cursor += 1
            else:
                out_list.append({"question": q, "answer": a, "score": -1, "results": []})

        out_path = name.replace(".json", "_results.json")
        with open(out_path, "w") as fh:
            json.dump(out_list, fh, indent=2)
        try:
            os.remove(name)
        except FileNotFoundError:
            pass

        pause_event.clear()
        print(f"[server] wrote {out_path}")
        return jsonify({"message": f"Processed {name}", "out": out_path})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="5000")
    parser.add_argument("--model_path", type=str, required=True)
    # 0.7 leaves headroom for the compile workspace and v1 EngineCore memory
    # growth under large batches (n=10 samples per prompt).
    parser.add_argument(
        "--gpu_mem_util",
        type=float,
        default=float(os.getenv("RZERO_REWARD_MEM_UTIL", "0.7")),
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    import vllm  # imported here so the v1 engine subprocess detection works.

    print(f"[init] Loading model from {args.model_path} on port {args.port} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = vllm.LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    # Paper Appendix B.1.1/B.1.2: top_p=0.99, top_k unspecified. R-Zero official
    # adds top_k=40 + top_p=1.0; we follow the paper.
    sample_params = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=0.99,
        stop_token_ids=[tokenizer.eos_token_id],
        n=args.n,
    )

    pause_event = threading.Event()  # make_app expects it; unused (no keep-alive thread)

    app = make_app(model, tokenizer, sample_params, pause_event)
    app.run(host="127.0.0.1", port=int(args.port), threaded=True)
    print("[main] server stopped")


if __name__ == "__main__":
    main()
