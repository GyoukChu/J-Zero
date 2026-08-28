"""Compute hint-effectiveness δ via a local Solver sampler (vLLM).

    δ(q, h, a_hard) = mean_t log π_S(a_hard_t | q, a_hard_<t)
                    − mean_t log π_S(a_hard_t | q, h, a_hard_<t)

Positive δ ⇒ the hint shifts the Solver away from its own unassisted response
a_hard, i.e. the hint carries non-trivial structural signal.

All prompts are batched through one `LocalSampler`. Two modes:
  - "delta_only": sample a_hard, compute δ            (Phase 1 reward)
  - "full":       also sample a_assisted for the DPO pool (Phase 2)
"""
from dataclasses import dataclass

from .local_backend import LocalSampler
from .local_render import SamplingParams
from .prompts import build_solver_convo


@dataclass
class QHScore:
    question: str
    hint: str
    a_hard: str
    a_assisted: str        # "" when mode == "delta_only"
    a_hard_tokens: list[int]
    logp_q: float
    logp_qh: float
    delta: float


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_batch(
    sampler: LocalSampler,
    renderer,
    tokenizer,
    qh_pairs: list[tuple[str, str]],
    *,
    mode: str = "delta_only",
    sampling_params: SamplingParams,
) -> list[QHScore]:
    """Score every (q, h) pair. Sampling + teacher-forced logprobs are batched."""
    if not qh_pairs:
        return []

    ctx_q = [renderer.build_generation_prompt(build_solver_convo(q)) for q, _ in qh_pairs]
    ctx_qh = [renderer.build_generation_prompt(build_solver_convo(q, h)) for q, h in qh_pairs]

    # a_hard ~ π(·|q)  (one sample each, all batched)
    hard_results = sampler.generate_batch(ctx_q, sampling_params, num_samples=1)
    a_hard_tokens = [list(r.sequences[0].tokens) if r.sequences else [] for r in hard_results]
    a_hard_text = [tokenizer.decode(t) for t in a_hard_tokens]

    # a_assisted ~ π(·|q,h)  (Phase 2 only)
    a_assisted_text = [""] * len(qh_pairs)
    if mode == "full":
        assist_results = sampler.generate_batch(ctx_qh, sampling_params, num_samples=1)
        a_assisted_text = [
            tokenizer.decode(list(r.sequences[0].tokens)) if r.sequences else ""
            for r in assist_results
        ]

    # Teacher-forced log-probs of a_hard under both contexts.
    seq_q = [list(ctx_q[i].tokens) + a_hard_tokens[i] for i in range(len(qh_pairs))]
    seq_qh = [list(ctx_qh[i].tokens) + a_hard_tokens[i] for i in range(len(qh_pairs))]
    lp_q_all = sampler.compute_logprobs_batch(seq_q)
    lp_qh_all = sampler.compute_logprobs_batch(seq_qh)

    out: list[QHScore] = []
    for i, (q, h) in enumerate(qh_pairs):
        n = len(a_hard_tokens[i])
        if n == 0:
            out.append(QHScore(q, h, a_hard_text[i], a_assisted_text[i], [], 0.0, 0.0, 0.0))
            continue
        cq, cqh = ctx_q[i].length, ctx_qh[i].length
        tgt_q = lp_q_all[i][cq: cq + n]
        tgt_qh = lp_qh_all[i][cqh: cqh + n]
        logp_q = _mean(tgt_q)
        logp_qh = _mean(tgt_qh)
        out.append(QHScore(
            question=q, hint=h,
            a_hard=a_hard_text[i], a_assisted=a_assisted_text[i],
            a_hard_tokens=a_hard_tokens[i],
            logp_q=float(logp_q), logp_qh=float(logp_qh),
            delta=float(logp_q - logp_qh),
        ))
    return out
