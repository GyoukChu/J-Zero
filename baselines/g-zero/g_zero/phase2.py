"""Phase 2: Build the DPO training pool (local vLLM backend).

1. Challenger generates `num_questions` (q, h) pairs.
2. Solver samples a_hard ~ π(·|q) and a_assisted ~ π(·|q,h); compute δ.
3. Cache the full scored pool to `raw_pool.jsonl` (so re-filtering is free).
4. Apply δ-percentile filter [pct_low, pct_high] + structural quality filters.
5. Write {prompt=q, chosen=a_assisted, rejected=a_hard} to `dpo_data.jsonl`.

Challenger and Solver may be different models (round k pairs the newly trained
Challenger with round k-1's Solver). Generation fully precedes scoring, so the
two samplers load sequentially — only one vLLM engine is ever resident.
"""
from __future__ import annotations

import json
import logging
import time
import zlib
from pathlib import Path

import numpy as np

from .config import Config
from .hint_delta import QHScore, score_batch
from .local_backend import LocalSampler
from .local_render import SamplingParams, get_renderer, get_tokenizer
from .parse import extract_qh
from .prompts import build_challenger_convo

logger = logging.getLogger(__name__)


def _make_sampler(config: Config, model_path: str | None) -> LocalSampler:
    return LocalSampler(
        model_path or config.model_name,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.vllm_max_model_len,
        dtype=config.dtype,
    )


def _challenger_generate_qh(
    *,
    sampler: LocalSampler,
    renderer,
    tokenizer,
    num_samples: int,
    sampling_params: SamplingParams,
    chunk_size: int = 256,
) -> list[tuple[str, str]]:
    prompt = renderer.build_generation_prompt(build_challenger_convo())
    pairs: list[tuple[str, str]] = []
    done = 0
    total_chunks = (num_samples + chunk_size - 1) // chunk_size
    chunk_idx = 0
    t0 = time.time()
    remaining = num_samples
    while remaining > 0:
        n = min(chunk_size, remaining)
        t_chunk = time.time()
        res = sampler.sample(prompt=prompt, num_samples=n, sampling_params=sampling_params)
        for seq in res.sequences:
            q, h = extract_qh(tokenizer.decode(list(seq.tokens)))
            if q and h:
                pairs.append((q, h))
        done += n
        chunk_idx += 1
        elapsed = time.time() - t0
        logger.info(
            "  [challenger q/h] %d/%d (%.1f%%) chunk=%d/%d valid=%d "
            "chunk_time=%.1fs elapsed=%.1fs",
            done, num_samples, 100 * done / num_samples,
            chunk_idx, total_chunks, len(pairs),
            time.time() - t_chunk, elapsed,
        )
        remaining -= n
    return pairs


def _dump_raw_pool(pool: list[QHScore], path: Path) -> None:
    # tmp + atomic rename: build_dpo_dataset trusts these cache files by
    # existence alone, so a write cut short (OOM-kill, disk full) must never
    # leave a truncated file under the final name.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for s in pool:
            f.write(json.dumps({
                "question": s.question,
                "hint": s.hint,
                "a_hard": s.a_hard,
                "a_assisted": s.a_assisted,
                "logp_q": s.logp_q,
                "logp_qh": s.logp_qh,
                "delta": s.delta,
            }, ensure_ascii=False) + "\n")
    tmp.replace(path)
    logger.info("  wrote raw pool: %s (%d records)", path, len(pool))


def _load_raw_pool(path: Path) -> list[QHScore]:
    pool: list[QHScore] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pool.append(QHScore(
                question=r.get("question", ""),
                hint=r.get("hint", ""),
                a_hard=r.get("a_hard", ""),
                a_assisted=r.get("a_assisted", ""),
                a_hard_tokens=[],
                logp_q=float(r.get("logp_q", 0.0)),
                logp_qh=float(r.get("logp_qh", 0.0)),
                delta=float(r.get("delta", 0.0)),
            ))
    logger.info("  loaded cached raw pool: %s (%d records)", path, len(pool))
    return pool


def _is_repetitive(text: str, threshold: float) -> bool:
    if len(text) < 500:
        return False
    b = text.encode("utf-8")
    compressed = zlib.compress(b, level=1)
    return len(compressed) / max(1, len(b)) < threshold


def _common_prefix_len(a: str, b: str) -> int:
    a, b = a.lstrip(), b.lstrip()
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_dpo_dataset(
    config: Config,
    *,
    challenger_sampling_path: str | None = None,
    solver_sampling_path: str | None = None,
) -> Path:
    """Build (and cache) the DPO pool. Returns path to dpo_data.jsonl.

    challenger_sampling_path / solver_sampling_path:
      local HF model directory for a trained checkpoint, or None for base model.
    """
    out_path = config.dpo_data_path()
    raw_pool_path = config.raw_pool_path()

    # Fast path: filtered DPO data already exists.
    if out_path.exists():
        logger.info("Phase 2: %s already exists, skipping", out_path)
        return out_path

    # Load cached raw pool, or generate it.
    if raw_pool_path.exists():
        logger.info("Phase 2: found cached raw pool at %s, skipping generation", raw_pool_path)
        scored = _load_raw_pool(raw_pool_path)
    else:
        tokenizer = get_tokenizer(config.model_name)
        renderer = get_renderer(config.renderer_name, tokenizer)

        challenger_sp = SamplingParams(
            max_tokens=config.challenger_max_tokens, temperature=1.0,
            stop=renderer.get_stop_sequences(),
        )
        solver_sp = SamplingParams(
            max_tokens=config.solver_max_tokens,
            temperature=config.solver_sample_temperature,
            stop=renderer.get_stop_sequences(),
        )

        # ── Step 1: Challenger generates (q, h) ──────────────────────────────
        logger.info("Phase 2a: generating (q, h) pool (challenger=%s)...",
                    challenger_sampling_path or config.model_name)
        ch_sampler = _make_sampler(config, challenger_sampling_path)
        try:
            qh_pairs = _challenger_generate_qh(
                sampler=ch_sampler, renderer=renderer, tokenizer=tokenizer,
                num_samples=config.num_questions, sampling_params=challenger_sp,
            )
        finally:
            ch_sampler.shutdown()
        logger.info("  got %d valid (q, h) pairs", len(qh_pairs))
        if not qh_pairs:
            raise SystemExit("Phase 2: challenger produced no valid (q, h) pairs.")

        # ── Step 2: Solver scores (samples a_hard / a_assisted, computes δ) ──
        logger.info("Phase 2b: scoring with Solver (solver=%s)...",
                    solver_sampling_path or config.model_name)
        sv_sampler = _make_sampler(config, solver_sampling_path)
        try:
            scored = score_batch(
                sv_sampler, renderer, tokenizer, qh_pairs,
                mode="full", sampling_params=solver_sp,
            )
        finally:
            sv_sampler.shutdown()

        scored = [s for s in scored if s.a_hard.strip() and s.a_assisted.strip()]
        if not scored:
            raise SystemExit("Phase 2: no valid scored records.")
        _dump_raw_pool(scored, raw_pool_path)

    # Filter by δ-percentile
    deltas = np.array([s.delta for s in scored], dtype=np.float64)
    lo = float(np.percentile(deltas, config.pct_low))
    hi = float(np.percentile(deltas, config.pct_high))
    logger.info(
        "  delta stats mean=%.3f std=%.3f min=%.3f max=%.3f; keeping [%.3f, %.3f] (p%.0f, p%.0f)",
        deltas.mean(), deltas.std(), deltas.min(), deltas.max(),
        lo, hi, config.pct_low, config.pct_high,
    )
    pct_kept = [s for s in scored if lo <= s.delta <= hi]
    logger.info("  percentile filter: kept %d / %d (%.1f%%)",
                len(pct_kept), len(scored),
                100 * len(pct_kept) / max(1, len(scored)))

    # Quality filters
    drop_too_long = drop_too_short = drop_ratio = drop_repeat = 0
    drop_prompt_echo = drop_role_marker = 0
    kept: list[QHScore] = []
    for s in pct_kept:
        lc, lr = len(s.a_assisted), len(s.a_hard)
        if lc > config.chosen_max_chars:
            drop_too_long += 1; continue
        if lc < config.chosen_min_chars:
            drop_too_short += 1; continue
        if lc / max(1, lr) > config.chosen_rejected_ratio_max:
            drop_ratio += 1; continue
        if _is_repetitive(s.a_assisted, config.chosen_repetition_zlib_threshold):
            drop_repeat += 1; continue
        if _common_prefix_len(s.a_assisted, s.question) >= config.chosen_prompt_prefix_overlap:
            drop_prompt_echo += 1; continue
        stripped = s.a_assisted.lstrip()
        if any(stripped.startswith(p) for p in config.chosen_role_marker_prefixes):
            drop_role_marker += 1; continue
        kept.append(s)
    logger.info(
        "  quality filter: too_long=%d too_short=%d ratio_bloat=%d repetition=%d "
        "prompt_echo=%d role_marker=%d -> kept %d",
        drop_too_long, drop_too_short, drop_ratio, drop_repeat,
        drop_prompt_echo, drop_role_marker, len(kept),
    )
    if not kept:
        raise SystemExit("Phase 2: all pairs dropped by quality filters; relax thresholds.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Same tmp+rename contract as _dump_raw_pool: the fast path above skips on
    # bare existence, so a truncated dpo_data.jsonl must never be adopted.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for s in kept:
            f.write(json.dumps({
                "prompt": s.question,
                "chosen": s.a_assisted,
                "rejected": s.a_hard,
                "delta": s.delta,
            }, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    logger.info("  wrote %s", out_path)
    return out_path
