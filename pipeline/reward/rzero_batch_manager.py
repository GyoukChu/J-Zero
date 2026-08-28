# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
Custom RewardManager that BATCHES the per-sample async calls
RewardLoopWorker.compute_score_batch issues (one asyncio task per item). The
challenger reward needs the whole chunk: BLEU clustering runs across the
batch's generated problems, and bulked HTTP calls to the vLLM solver server
are far cheaper than per-item ones.

Configure via:
    reward.reward_manager.source=importlib
    reward.reward_manager.module.path=file:///abs/path/rzero_batch_manager.py
    reward.reward_manager.name=RZeroBatchManager
    reward.num_workers=1

Each call decodes its item synchronously (no run_in_executor, so every task
registers before the leader settles) and appends its payload to a shared chunk
buffer. The first registrant (leader) waits SETTLE_MS, then runs compute_score
over whatever accumulated. Followers hold local refs to the buffer/event so a
chunk created meanwhile cannot strand them.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any, Dict, List

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

logger = logging.getLogger(__name__)

# How long the leader waits without new submissions before closing the chunk.
# Higher = bigger batches (better BLEU cluster signal, fewer round-trips) but
# more leader latency; 5s covers decode + asyncio scheduling for GBS 16 × n 4.
SETTLE_MS = int(os.getenv("RZERO_SETTLE_MS", "5000"))
# Ceiling on the leader's total wait, so a hung compute_score (e.g. downstream
# HTTP stall) force-fails the chunk with zero scores and releases followers
# instead of blocking the run. 30s covers the normal fan-out with slack.
MAX_WAIT_MS = int(os.getenv("RZERO_MAX_WAIT_MS", "30000"))
# Hard cap on prompts per chunk: at this size the leader triggers immediately
# and later submissions start a fresh chunk. 64 keeps one compute_score call's
# HTTP fan-out manageable; pair with RZERO_REWARD_CHUNK_SIZE, which sub-chunks
# inside a single call.
CHUNK_MAX = max(1, int(os.getenv("RZERO_CHUNK_MAX", "64")))
# Expected chunk size (0 = disabled, e.g. 64 = GBS 16 × n 4). When set, the
# leader will not settle on a stable count below it and waits for stragglers
# up to MAX_WAIT_MS. Otherwise a rollout that misses one SETTLE_MS window
# forms its own 1-2 item chunk, whose BLEU cluster-share is meaningless (a
# 1-item chunk scores r_rep=1.0 whatever its content).
CHUNK_EXPECT = max(0, int(os.getenv("RZERO_CHUNK_EXPECT", "0")))
# Process-local counter so interleaved logs can be traced to one chunk.
_CHUNK_SEQ = 0
# Extra keys stamped as 0.0 on FAILURE/padding result dicts so every item in a
# step carries the same key set as the success dicts — agent_loop._postprocess
# reads the key set off the first info only. Set per stage, e.g.
#   RZERO_FAIL_EXTRA_KEYS="format_reward,rep_reward,difficulty_reward"
_FAIL_EXTRA_KEYS = [
    k.strip() for k in os.getenv("RZERO_FAIL_EXTRA_KEYS", "").split(",") if k.strip()
]


def _failure_result(err: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"score": 0.0, "_failed": True, "_error": err}
    for k in _FAIL_EXTRA_KEYS:
        out.setdefault(k, 0.0)
    return out


class _Chunk:
    """Per-chunk shared state."""

    def __init__(self) -> None:
        global _CHUNK_SEQ
        _CHUNK_SEQ += 1
        self.cid = _CHUNK_SEQ  # log identification only
        self.payloads: List[Dict[str, Any]] = []
        self.results: List[Any] = []
        self.done: asyncio.Event = asyncio.Event()
        self.snapshot_taken: bool = False
        self.lock: asyncio.Lock = asyncio.Lock()


class RZeroBatchManager(RewardManagerBase):
    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        # global lock guards `_current_chunk` reassignment
        self._global_lock: asyncio.Lock | None = None
        self._current_chunk: _Chunk | None = None

    @classmethod
    def init_class(cls, config, tokenizer):
        super().init_class(config, tokenizer)
        cust = (config.reward.get("custom_reward_function") or {})
        cls._reward_kwargs = dict(cust.get("reward_kwargs", {}) or {})

    def _decode_sync(self, data_item) -> str:
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_len = data_item.batch["attention_mask"][-response_length:].sum()
        valid_ids = response_ids[:valid_len]
        return self.tokenizer.decode(valid_ids, skip_special_tokens=True)

    async def _get_or_make_chunk(self) -> _Chunk:
        if self._global_lock is None:
            self._global_lock = asyncio.Lock()
        async with self._global_lock:
            chunk = self._current_chunk
            if chunk is None or chunk.snapshot_taken:
                chunk = _Chunk()
                self._current_chunk = chunk
            return chunk

    async def run_single(self, data: DataProto) -> dict:
        loop = asyncio.get_event_loop()

        # Decode synchronously to avoid executor contention.
        data_item = data[0]
        solution_str = self._decode_sync(data_item)
        ground_truth = data_item.non_tensor_batch["reward_model"].get("ground_truth", None)
        data_source = data_item.non_tensor_batch.get("data_source", "")
        extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
        payload = {
            "data_source": data_source,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "extra_info": dict(extra_info),
        }

        chunk = await self._get_or_make_chunk()
        async with chunk.lock:
            if chunk.snapshot_taken:
                # Submitted after the leader snapshotted; start a fresh chunk.
                chunk = await self._get_or_make_chunk()
                async with chunk.lock:
                    my_idx = len(chunk.payloads)
                    chunk.payloads.append(payload)
                    is_leader = (my_idx == 0)
            else:
                my_idx = len(chunk.payloads)
                chunk.payloads.append(payload)
                is_leader = (my_idx == 0)

        if is_leader:
            print(
                f"[batch] leader chunk#{chunk.cid} start (settle={SETTLE_MS}ms "
                f"max_wait={MAX_WAIT_MS}ms chunk_max={CHUNK_MAX})",
                flush=True,
            )
            # Stop on the first of: no new submission for SETTLE_MS, chunk
            # reaching CHUNK_MAX, or the MAX_WAIT_MS ceiling.
            prev_count = -1
            total_wait = 0
            while total_wait < MAX_WAIT_MS:
                await asyncio.sleep(SETTLE_MS / 1000.0)
                total_wait += SETTLE_MS
                cur = len(chunk.payloads)
                if cur >= CHUNK_MAX:
                    logger.info(
                        "RZeroBatchManager: chunk#%d hit CHUNK_MAX=%d (settle %dms), triggering early",
                        chunk.cid,
                        CHUNK_MAX,
                        total_wait,
                    )
                    break
                if cur == prev_count and (CHUNK_EXPECT <= 0 or cur >= CHUNK_EXPECT):
                    break
                if cur == prev_count and CHUNK_EXPECT > 0:
                    logger.info(
                        "RZeroBatchManager: chunk#%d stable at %d < expect %d; "
                        "waiting for stragglers (%dms/%dms)",
                        chunk.cid, cur, CHUNK_EXPECT, total_wait, MAX_WAIT_MS,
                    )
                prev_count = cur

            async with chunk.lock:
                chunk.snapshot_taken = True
                payloads = list(chunk.payloads)
            print(
                f"[batch] leader chunk#{chunk.cid} settled n={len(payloads)} after {total_wait}ms",
                flush=True,
            )

            import time as _time
            _cs_t0 = _time.time()
            print(
                f"[batch] chunk#{chunk.cid} compute_score start n={len(payloads)}",
                flush=True,
            )
            try:
                if inspect.iscoroutinefunction(self.compute_score):
                    out = await self.compute_score(
                        data_sources=[p["data_source"] for p in payloads],
                        solution_strs=[p["solution_str"] for p in payloads],
                        ground_truths=[p["ground_truth"] for p in payloads],
                        extra_infos=[p["extra_info"] for p in payloads],
                        **self._reward_kwargs,
                    )
                else:
                    out = await loop.run_in_executor(
                        None,
                        lambda: self.compute_score(
                            data_sources=[p["data_source"] for p in payloads],
                            solution_strs=[p["solution_str"] for p in payloads],
                            ground_truths=[p["ground_truth"] for p in payloads],
                            extra_infos=[p["extra_info"] for p in payloads],
                            **self._reward_kwargs,
                        ),
                    )
                print(
                    f"[batch] chunk#{chunk.cid} compute_score done n={len(payloads)} "
                    f"in {_time.time()-_cs_t0:.1f}s",
                    flush=True,
                )
                if not isinstance(out, list) or len(out) != len(payloads):
                    logger.error(
                        "compute_score returned %s of len %s for %d payloads; padding zeros",
                        type(out).__name__,
                        len(out) if hasattr(out, "__len__") else "?",
                        len(payloads),
                    )
                    pad_err = (
                        f"length_mismatch out={len(out) if hasattr(out, '__len__') else '?'} "
                        f"expected={len(payloads)}"
                    )
                    out = (list(out) if isinstance(out, list) else []) + [
                        _failure_result(pad_err)
                    ] * (len(payloads) - (len(out) if isinstance(out, list) else 0))
                    out = out[: len(payloads)]
            except Exception as exc:
                # Never silently zero rewards: tag each item (_failed, _error)
                # so failures show in rollout dumps and aggregate through
                # reward_extra_info -> wandb as reward/error_count.
                logger.exception("RZeroBatchManager compute_score failed: %s", exc)
                err_msg = str(exc)[:200]
                out = [_failure_result(err_msg) for _ in payloads]
                logger.error(
                    "RZeroBatchManager: chunk#%d error_count=%d / chunk_size=%d (error=%s)",
                    chunk.cid,
                    len(payloads),
                    len(payloads),
                    err_msg,
                )
                print(
                    f"[batch] chunk#{chunk.cid} compute_score FAILED in "
                    f"{_time.time()-_cs_t0:.1f}s (error={err_msg})",
                    flush=True,
                )
            chunk.results = out
            n_failed = sum(
                1 for r in out if isinstance(r, dict) and bool(r.get("_failed", False))
            )
            if n_failed > 0:
                print(
                    f"!!! [reward] CHUNK ERRORS: {n_failed}/{len(out)} items failed",
                    flush=True,
                )
            else:
                print(
                    f"[reward] chunk ok: {len(out)} items, 0 failures",
                    flush=True,
                )
            chunk.done.set()
        else:
            await chunk.done.wait()

        if my_idx >= len(chunk.results):
            logger.warning(
                "RZeroBatchManager: my_idx=%d outside results len=%d; returning 0",
                my_idx,
                len(chunk.results),
            )
            return {"reward_score": 0.0, "reward_extra_info": {}}

        result = chunk.results[my_idx]
        if isinstance(result, dict):
            score = float(result.get("score", 0.0))
            extra = {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool))}
            # _failed and _error are set unconditionally: verl's
            # agent_loop._postprocess reads the key set off the FIRST info and
            # then fetches those keys from every info, so a step mixing failed
            # and successful chunks would KeyError.
            extra["_failed"] = bool(result.get("_failed", False))
            # 0/1 per item, so its per-step mean is the error rate.
            extra["error_count"] = 1 if extra["_failed"] else 0
            extra.setdefault("_error", "")
        else:
            score = float(result)
            extra = {"acc": score, "_failed": False, "error_count": 0}
        return {"reward_score": score, "reward_extra_info": extra}
