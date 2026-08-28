# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
Shared IO helpers for the Skywork reward model (Skywork-Reward-V2-Llama-3.1-8B).

All reward / filter / training code in this recipe MUST import these symbols
rather than re-implementing them:

  - SOLVER_SYS — the solver system prompt; byte-identical everywhere the RM
    scores an answer, else the score sits on a different input distribution
    than the solver was prompted with.
  - `load_skywork_tokenizer(path)` — the only permitted way to load the
    Skywork tokenizer. Sets two independent sides:
      padding_side="right" matches the model's shipped default (its pad token
        is named `<|finetune_right_pad_id|>`), under which the SC head's
        `((input_ids==pad).argmax(-1)-1) % L` picks the last non-pad position
        without modulo wrap-around.
      truncation_side="left" keeps the answer tail and chat-template close
        tokens when Q + A exceeds max_length; dropping the front only costs
        context, dropping the tail corrupts what the SC head reads.
  - `format_rm_input(tokenizer, question, answer)` — builds and templates the
    (user, assistant) chat, stripping a leading BOS so the tokenizer re-adds
    exactly one (the double-BOS bug Skywork's README warns about).
  - `sigmoid(x)` — numerically stable scalar/array sigmoid.
"""

from __future__ import annotations

from typing import Iterable, List, Union

import numpy as np
from transformers import AutoTokenizer


# Canonical solver answer-generation instruction, shared so every RM input is
# formatted with the same prefix. Skywork's README says NOT to include their
# system prompt for V2, so `format_rm_input` emits no `role=system` turn.
SOLVER_SYS = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

DEFAULT_RM_PATH = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
DEFAULT_RM_MAX_LEN = 8192


def load_skywork_tokenizer(model_path: str):
    """Load the Skywork tokenizer with right padding + left truncation.

    Every site (reward fn, filter, trainer) MUST call this rather than
    `AutoTokenizer.from_pretrained(...)` directly; see the module docstring for
    why each side is set that way. Llama-3.1 reserves <|finetune_right_pad_id|>
    (id 128004) for padding so EOS is never masked, and Skywork-Reward-V2 ships
    with it set — re-asserted here defensively, falling back to EOS.
    """
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "right"
    tok.truncation_side = "left"
    if tok.pad_token is None or tok.pad_token == tok.eos_token:
        reserved_pad = "<|finetune_right_pad_id|>"
        try:
            ids = tok.convert_tokens_to_ids(reserved_pad)
        except Exception:
            ids = None
        # convert_tokens_to_ids returns the unk id (or None) on a miss.
        if ids is not None and ids != tok.unk_token_id and ids >= 0:
            tok.pad_token = reserved_pad
            tok.pad_token_id = ids
        elif tok.pad_token is None:
            # No reserved pad token: reuse EOS so pad_token_id at least exists.
            tok.pad_token = tok.eos_token
    return tok


def format_rm_input(tokenizer, question: str, answer: str) -> str:
    """Build the text fed to the RM tokenizer: the (user, assistant) chat run
    through the chat template, with any leading BOS stripped so the tokenizer
    re-adds exactly one at encode time (Skywork's README warns about double
    BOS). Returns a string, NOT token ids — the caller feeds it to
    `tokenizer(..., truncation=True, max_length=8192)`.
    """
    user_content = f"{question}"
    conv = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(conv, tokenize=False)
    else:
        # Never triggers for Skywork-Llama-3.1; kept for parity with the rest
        # of the recipe's defensive code.
        text = f"user: {user_content}\nassistant: {answer}"

    bos = tokenizer.bos_token
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text


def sigmoid(x: Union[float, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
    """Numerically stable sigmoid mapping Skywork's raw classifier logits to
    [0, 1], for a scalar or an array.
    """
    arr = np.asarray(x, dtype=np.float64)
    # Branch before the exp rather than via np.where, which evaluates BOTH
    # branches eagerly and warns on overflow for large |arr|.
    out = np.empty_like(arr, dtype=np.float64)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    out[~pos] = np.exp(arr[~pos]) / (1.0 + np.exp(arr[~pos]))
    if out.ndim == 0:
        return float(out)
    return out


def split_evenly(lst: List, n: int) -> List[List]:
    """Split into n nearly-even chunks, order preserved. Duplicated as
    `_split_list` in challenger_reward.py."""
    if n <= 0:
        raise ValueError(f"n must be >0, got {n}")
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]
