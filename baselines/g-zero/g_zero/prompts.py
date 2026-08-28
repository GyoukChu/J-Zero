"""Challenger and Solver prompts for G-Zero.

The Solver turn deliberately carries no system prompt: base models echo system
messages back into the response, and that prefix then contaminates both sides
of every DPO pair.
"""
from __future__ import annotations


CHALLENGER_INSTRUCTION = (
    "Produce one challenging request that a real user might ask a capable "
    "assistant, plus a short hint that helps the assistant give a noticeably "
    "better response.\n\n"
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
    "  - reasoning, math, or logic problems (fine to include — roughly 1 in "
    "6 requests, no more)\n\n"
    "Weight the non-math categories above heavily. A little math is good "
    "for diversity, but it should not dominate — favor tasks where the "
    "response quality depends on tone, structure, audience-fit, clarity, or "
    "creativity, not just arithmetic correctness.\n\n"
    "Requirements:\n"
    "- The request must be self-contained and non-trivial to answer well.\n"
    "- The hint must guide the approach (e.g. tone, structure, what to "
    "  include, what to avoid) but must not give away the full answer.\n"
    "- Wrap the request in <question> and </question> tags.\n"
    "- Wrap the hint in <hint> and </hint> tags.\n"
    "- Output nothing else before, between, or after the tagged blocks.\n\n"
    "Example 1 (writing):\n"
    "<question>Write a resignation email to my manager that keeps the door "
    "open for future collaboration. I've been at the company for 4 years "
    "and I'm leaving to join a competitor. Tone should be professional and "
    "warm without being effusive.</question>\n"
    "<hint>Lead with gratitude for specific experiences rather than generic "
    "thanks, keep the departure reason brief and non-defensive, and close "
    "with a concrete offer to help during the transition.</hint>\n\n"
    "Example 2 (explanation):\n"
    "<question>Explain what a Kalman filter does to a software engineer who "
    "is comfortable with linear algebra but has never touched signal "
    "processing. Avoid control-theory jargon where possible.</question>\n"
    "<hint>Frame it as 'recursive Bayesian least-squares on a noisy "
    "dynamical system' and build up from a 1-D example (estimating "
    "position from noisy GPS) before generalizing.</hint>\n\n"
    "Now produce one new request and hint of your own:\n"
)


def build_challenger_convo() -> list[dict]:
    return [{"role": "user", "content": CHALLENGER_INSTRUCTION}]


def build_solver_convo(question: str, hint: str | None = None) -> list[dict]:
    user = question if not hint else f"{question}\n\nHint: {hint}"
    return [{"role": "user", "content": user}]
