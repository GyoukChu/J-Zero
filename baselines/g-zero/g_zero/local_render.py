"""Local replacements for the `tinker` / `tinker_cookbook` surface used by G-Zero.

Reimplements only the subset the pipeline touches, backed by a HuggingFace
tokenizer instead of the cloud client:
  - `get_tokenizer(model)`                         (was tinker_cookbook.tokenizer_utils)
  - `get_renderer(name, tok)` / `Renderer`         (was tinker_cookbook.renderers)
  - `TokenSeq` / `EncodedTextChunk`                (was tinker.types ModelInput/EncodedTextChunk)
  - `SamplingParams`                               (was tinker.types.SamplingParams)

The renderer is a single chat turn rendered with the model's own chat template.
Qwen3 base-model templates carry no thinking-mode injection, so
`apply_chat_template(..., add_generation_prompt=True)` reproduces exactly the
format G-Zero's prompts.py was written for (Llama-3.2 analogously).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from transformers import AutoTokenizer

# Qwen3 chat turn terminator; also the eos for Qwen3 chat format.
QWEN3_IM_END = "<|im_end|>"


def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


@dataclass(frozen=True)
class EncodedTextChunk:
    """A run of already-encoded token ids (mirrors tinker.types.EncodedTextChunk)."""
    tokens: list[int]


@dataclass(frozen=True)
class TokenSeq:
    """Immutable token sequence mirroring the slice of tinker's ModelInput G-Zero uses.

    `append` / `append_int` return a NEW sequence; originals are reused across
    contexts, exactly as in the Tinker code paths.
    """
    tokens: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.tokens)

    def append(self, chunk: EncodedTextChunk) -> "TokenSeq":
        return replace(self, tokens=self.tokens + list(chunk.tokens))

    def append_int(self, token: int) -> "TokenSeq":
        return replace(self, tokens=self.tokens + [int(token)])


@dataclass
class SamplingParams:
    """Generation parameters (mirrors the tinker.types.SamplingParams fields used)."""
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)


class Renderer:
    """Single-turn chat renderer over a HuggingFace chat template."""

    def __init__(self, tokenizer, name: str = "qwen3"):
        self.tokenizer = tokenizer
        self.name = name

    def build_generation_prompt(self, convo: list[dict]) -> TokenSeq:
        ids = self.tokenizer.apply_chat_template(
            convo, add_generation_prompt=True, tokenize=True
        )
        # transformers 4.x returns list[int]; 5.x returns a BatchEncoding dict.
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return TokenSeq(tokens=[int(t) for t in ids])

    def get_stop_sequences(self) -> list[str]:
        # Chat-format turn terminator plus the model's own eos (which covers
        # Llama-3's <|eot_id|>).
        stops = [QWEN3_IM_END]
        eot = getattr(self.tokenizer, "eos_token", None)
        if eot and eot not in stops:
            stops.append(eot)
        return stops


def get_renderer(name: str, tokenizer) -> Renderer:
    # The name is informational; the model's own chat template supplies the format.
    return Renderer(tokenizer, name=name)
