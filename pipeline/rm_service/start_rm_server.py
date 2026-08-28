#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask + transformers Skywork reward-model HTTP server.

Mirrors the I/O contract of `vllm_service/start_vllm_server_answers.py` so the
existing challenger reward / batch-manager dispatch code can talk to it:

  - GET  /health                  -> {"status":"ok"} (probe)
  - POST /score   JSON body:
        {"items": [{"question": str, "answer": str}, ...]}
    Returns:
        {"scores": [raw_logit_0, raw_logit_1, ...]}      # SAME ORDER
    The scores are RAW logits (NOT sigmoid). Callers apply sigmoid.
  - GET  /hello?name=<path_to_input_json>  — file-payload fallback for
    symmetry with the vLLM answers server. Input is a JSON list of
    {"question", "answer"}; the server writes `<path>_results.json` as
        [{"question": str, "answer": str, "raw_score": float}, ...]
    (a list, not a top-level dict — same shape vllm_service emits).

Single GPU per server (one transformers model in bf16 + FA2); `start.sh` spawns
N of them and the caller shards items across ports.

Items are sorted by length before batching so each forward pass has a uniform
shape. Batch size is capped by `--max_batch` (default 8) and by total tokens
(`--max_batch_tokens`, default 32768), whichever binds first. Padding and
truncation sides come from rm_io.load_skywork_tokenizer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import Dict, List, Tuple

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForSequenceClassification

# Make `from rm_service.rm_io import ...` resolve when this file is invoked
# directly as `python rm_service/start_rm_server.py` (the recipe layout).
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rm_service.rm_io import (  # noqa: E402
    DEFAULT_RM_MAX_LEN,
    DEFAULT_RM_PATH,
    format_rm_input,
    load_skywork_tokenizer,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rm_server")


# Module-level singletons populated by main(); functions read them.
_MODEL = None
_TOKENIZER = None
_DEVICE = None
_DTYPE = None
_MAX_LEN = DEFAULT_RM_MAX_LEN
_MAX_BATCH = 8
_MAX_BATCH_TOKENS = 32768
_INFER_LOCK = threading.Lock()


@torch.inference_mode()
def _score_batch(formatted_texts: List[str]) -> List[float]:
    """Tokenize a batch of pre-formatted strings (right-pad, left-trunc)
    and return raw classifier logits as floats."""
    enc = _TOKENIZER(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_MAX_LEN,
    )
    enc = {k: v.to(_DEVICE, non_blocking=True) for k, v in enc.items()}
    out = _MODEL(**enc)
    # logits are (B, 1) for num_labels=1.
    logits = out.logits.view(-1).float().detach().cpu().tolist()
    return logits


def _truncated_count(formatted_texts: List[str]) -> int:
    """How many of these would be truncated at _MAX_LEN tokens. Observability
    only, so it returns -1 rather than raising on failure."""
    try:
        enc = _TOKENIZER(formatted_texts, padding=False, truncation=False)
        return sum(1 for ids in enc["input_ids"] if len(ids) > _MAX_LEN)
    except Exception:  # pragma: no cover - never raise from logging path
        return -1


def _score_items(items: List[Dict[str, str]]) -> List[float]:
    """One raw RM logit per {question, answer} item, in INPUT order — internal
    batching reorders by length but restores the order before returning."""
    n = len(items)
    if n == 0:
        return []

    formatted: List[str] = [
        format_rm_input(_TOKENIZER, it.get("question", "") or "",
                        it.get("answer", "") or "")
        for it in items
    ]
    # Char count as the length proxy: the tokenizer is exact but ~10x slower.
    lengths = [len(s) for s in formatted]
    order = sorted(range(n), key=lambda i: lengths[i])

    batches: List[List[int]] = []
    cur: List[int] = []
    cur_tokens = 0
    for idx in order:
        # Coarse char->token estimate for the batch bound only; the real
        # tokenizer enforces _MAX_LEN at encode time.
        approx = max(1, lengths[idx] // 4)
        if cur and (
            len(cur) >= _MAX_BATCH
            or cur_tokens + approx > _MAX_BATCH_TOKENS
        ):
            batches.append(cur)
            cur = []
            cur_tokens = 0
        cur.append(idx)
        cur_tokens += approx
    if cur:
        batches.append(cur)

    n_trunc = _truncated_count(formatted)
    if n_trunc > 0:
        log.warning("[rm_server] %d/%d items will be left-truncated to %d tokens",
                    n_trunc, n, _MAX_LEN)

    scores: List[float] = [0.0] * n
    t0 = time.time()
    # Single global lock — only one CUDA forward at a time per server.
    with _INFER_LOCK:
        for bi, batch_indices in enumerate(batches):
            sub_texts = [formatted[i] for i in batch_indices]
            try:
                sub_scores = _score_batch(sub_texts)
            except torch.cuda.OutOfMemoryError as oom:
                log.error("[rm_server] OOM on batch size %d (%s); falling back per-item",
                          len(sub_texts), oom)
                torch.cuda.empty_cache()
                sub_scores = []
                for txt in sub_texts:
                    try:
                        sub_scores.extend(_score_batch([txt]))
                    except Exception as exc:
                        log.exception("[rm_server] per-item fallback also failed: %s", exc)
                        sub_scores.append(0.0)
            for j, idx in enumerate(batch_indices):
                scores[idx] = float(sub_scores[j])
    dt = time.time() - t0
    log.info("[rm_server] scored %d items in %.1fs across %d batches (avg bs %.1f)",
             n, dt, len(batches), n / max(1, len(batches)))
    return scores


def build_app() -> Flask:
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model_loaded": _MODEL is not None})

    @app.route("/score", methods=["POST"])
    def score():
        payload = request.get_json(force=True, silent=False) or {}
        items = payload.get("items", [])
        if not isinstance(items, list):
            return jsonify({"error": "items must be a list"}), 400
        try:
            scores = _score_items(items)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("[rm_server] /score failed")
            return jsonify({"error": str(exc)[:200]}), 500
        return jsonify({"scores": scores})

    @app.route("/hello", methods=["GET"])
    def hello():
        """File-payload fallback (matches the vLLM answers server's shape)."""
        name = request.args.get("name", "")
        if not name or not os.path.exists(name):
            return jsonify({"error": f"missing file: {name}"}), 400
        with open(name, "r") as fh:
            items = json.load(fh)
        if not isinstance(items, list):
            return jsonify({"error": "input file must be a JSON list"}), 400
        try:
            scores = _score_items(items)
        except Exception as exc:  # pragma: no cover
            log.exception("[rm_server] /hello failed")
            return jsonify({"error": str(exc)[:200]}), 500

        out = []
        for it, s in zip(items, scores):
            out.append({
                "question": it.get("question", ""),
                "answer": it.get("answer", ""),
                "raw_score": float(s),
            })
        out_path = name.replace(".json", "_results.json")
        with open(out_path, "w") as fh:
            json.dump(out, fh)
        try:
            os.remove(name)
        except FileNotFoundError:
            pass
        return jsonify({"message": f"Processed {name}", "out": out_path})

    return app


def main() -> None:
    global _MODEL, _TOKENIZER, _DEVICE, _DTYPE, _MAX_LEN, _MAX_BATCH, _MAX_BATCH_TOKENS

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--model_path", type=str, default=DEFAULT_RM_PATH)
    parser.add_argument("--dtype", type=str, default=os.getenv("RZERO_RM_DTYPE", "bfloat16"))
    parser.add_argument("--max_len", type=int,
                        default=int(os.getenv("RZERO_RM_MAX_LEN", str(DEFAULT_RM_MAX_LEN))))
    parser.add_argument("--max_batch", type=int,
                        default=int(os.getenv("RZERO_RM_MAX_BATCH", "8")))
    parser.add_argument("--max_batch_tokens", type=int,
                        default=int(os.getenv("RZERO_RM_MAX_BATCH_TOKENS", "32768")))
    parser.add_argument("--attn_impl", type=str,
                        default=os.getenv("RZERO_RM_ATTN_IMPL", "flash_attention_2"))
    args = parser.parse_args()

    _MAX_LEN = int(args.max_len)
    _MAX_BATCH = int(args.max_batch)
    _MAX_BATCH_TOKENS = int(args.max_batch_tokens)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    _DTYPE = dtype_map.get(args.dtype, torch.bfloat16)

    if not torch.cuda.is_available():
        raise RuntimeError("rm_server requires CUDA")
    _DEVICE = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES selects which physical GPU

    log.info("[rm_server] loading tokenizer from %s ...", args.model_path)
    _TOKENIZER = load_skywork_tokenizer(args.model_path)
    log.info("[rm_server] tokenizer padding_side=%s truncation_side=%s pad=%s",
             _TOKENIZER.padding_side, _TOKENIZER.truncation_side,
             _TOKENIZER.pad_token)

    log.info("[rm_server] loading model from %s dtype=%s attn=%s ...",
             args.model_path, args.dtype, args.attn_impl)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,
        torch_dtype=_DTYPE,
        attn_implementation=args.attn_impl,
    )
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    # force-sync to tokenizer pad to avoid SC head reading at left-padded positions
    _MODEL.config.pad_token_id = _TOKENIZER.pad_token_id
    log.info("[rm_server] model ready. max_len=%d max_batch=%d max_batch_tokens=%d",
             _MAX_LEN, _MAX_BATCH, _MAX_BATCH_TOKENS)

    app = build_app()
    # threaded=False: the HF Rust-backed tokenizer is not thread-safe, and
    # _INFER_LOCK guards only the CUDA forward — concurrent /score requests
    # racing in format_rm_input / _truncated_count raise "Already borrowed".
    # Costs no throughput, since the forward is serialized anyway.
    app.run(host="127.0.0.1", port=int(args.port), threaded=False)


if __name__ == "__main__":
    main()
