"""Local vLLM backend — drop-in replacement for the Tinker `SamplingClient`.

`LocalSampler` covers the three things G-Zero needed from Tinker — sampling
completions, teacher-forced log-probs of a fixed continuation (the δ score),
and loading a trained checkpoint — on local GPUs via one offline `vllm.LLM`.
Sampled-token log-probs come from `SamplingParams(logprobs=0)`; teacher-forced
ones from `SamplingParams(prompt_logprobs=0)`, which yields
`log p(token_i | tokens_<i)` at every prompt position, exactly what δ needs.

Result objects mimic Tinker's shape (`result.sequences[i].tokens` /
`.logprobs`) so the phase code ports with minimal edits.
"""
from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field

import torch

from .local_render import SamplingParams, TokenSeq

logger = logging.getLogger(__name__)

# On hosts without a C/CUDA compiler every vLLM path that JIT-compiles a kernel
# fails, so pick precompiled backends: prebuilt FLASH_ATTN over FlashInfer, the
# native top-k/top-p sampler, no DeepGEMM fp8 warmup (bf16 model), and eager
# torch.compile (the native sampler is @torch.compile-wrapped, so inductor would
# still need gcc). Must be set before vLLM is imported; setdefault lets the
# caller override.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


@dataclass
class _Seq:
    tokens: list[int]
    logprobs: list[float] = field(default_factory=list)


@dataclass
class _SampleResult:
    sequences: list[_Seq]


def _import_vllm():
    from vllm import LLM
    from vllm import SamplingParams as VSamplingParams
    try:
        from vllm import TokensPrompt
    except ImportError:  # older/newer layout
        from vllm.inputs import TokensPrompt
    return LLM, VSamplingParams, TokensPrompt


class LocalSampler:
    """Offline vLLM sampler/logprob engine over a single model (base or merged ckpt)."""

    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        dtype: str = "bfloat16",
        seed: int = 0,
        enforce_eager: bool = True,
    ):
        LLM, VSamplingParams, TokensPrompt = _import_vllm()
        self._V = VSamplingParams
        self._TokensPrompt = TokensPrompt
        self._max_len = max_model_len
        logger.info(
            "LocalSampler: loading %s (tp=%d, gpu_mem=%.2f, max_len=%d, %s, eager=%s)",
            model_path, tensor_parallel_size, gpu_memory_utilization, max_model_len,
            dtype, enforce_eager,
        )
        # enforce_eager avoids torch.compile / triton JIT (needs a host C
        # compiler); inference is fast enough eager.
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            seed=seed,
            enforce_eager=enforce_eager,
            trust_remote_code=False,
        )

    # ── sampling ─────────────────────────────────────────────────────────────
    def _vsp(self, sp: SamplingParams, num_samples: int):
        return self._V(
            n=num_samples,
            max_tokens=sp.max_tokens,
            temperature=sp.temperature,
            top_p=sp.top_p,
            stop=list(sp.stop) if sp.stop else None,
            logprobs=0,  # log-prob of each sampled token
        )

    @staticmethod
    def _sampled_logprobs(comp) -> list[float]:
        if comp.logprobs is None:
            return []
        out = []
        for tid, lpdict in zip(comp.token_ids, comp.logprobs):
            lp = lpdict.get(tid) if lpdict else None
            out.append(float(lp.logprob) if lp is not None else 0.0)
        return out

    def _result(self, request_output) -> _SampleResult:
        return _SampleResult(sequences=[
            _Seq(tokens=list(c.token_ids), logprobs=self._sampled_logprobs(c))
            for c in request_output.outputs
        ])

    def _clamp_gen(self, ids: list[int]) -> list[int]:
        """Keep room for at least some generated tokens (defensive vs. over-long prompts)."""
        cap = self._max_len - 256
        return ids[:cap] if len(ids) > cap else ids

    def sample(self, prompt: TokenSeq, num_samples: int, sampling_params: SamplingParams) -> _SampleResult:
        """N independent completions of one prompt (mirrors SamplingClient.sample)."""
        req = self._TokensPrompt(prompt_token_ids=self._clamp_gen(list(prompt.tokens)))
        outs = self.llm.generate([req], self._vsp(sampling_params, num_samples))
        return self._result(outs[0])

    def generate_batch(
        self, prompts: list[TokenSeq], sampling_params: SamplingParams, num_samples: int = 1
    ) -> list[_SampleResult]:
        """One completion-set per prompt, all batched through vLLM in a single call."""
        if not prompts:
            return []
        reqs = [self._TokensPrompt(prompt_token_ids=self._clamp_gen(list(p.tokens))) for p in prompts]
        outs = self.llm.generate(reqs, self._vsp(sampling_params, num_samples))
        return [self._result(o) for o in outs]

    # ── teacher-forced log-probs (δ) ─────────────────────────────────────────
    def compute_logprobs_batch(self, token_id_lists: list[list[int]]) -> list[list[float]]:
        """For each id-list, return [logp(tok_i | tok_<i)] aligned to input (index 0 = 0.0)."""
        if not token_id_lists:
            return []
        sp = self._V(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        # Clamp to the context window so an over-long δ sequence truncates instead of crashing.
        clamped = [list(ids)[: self._max_len] for ids in token_id_lists]
        reqs = [self._TokensPrompt(prompt_token_ids=ids) for ids in clamped]
        outs = self.llm.generate(reqs, sp)
        results: list[list[float]] = []
        for ids, out in zip(clamped, outs):
            pl = out.prompt_logprobs or []
            row = [0.0] * len(ids)
            for j in range(1, len(ids)):
                entry = pl[j] if j < len(pl) else None
                lp = entry.get(ids[j]) if entry else None
                row[j] = float(lp.logprob) if lp is not None else 0.0
            results.append(row)
        return results

    def compute_logprobs(self, token_ids: list[int]) -> list[float]:
        return self.compute_logprobs_batch([list(token_ids)])[0]

    # ── teardown (free the GPUs before the training phase) ───────────────────
    def shutdown(self) -> None:
        try:
            del self.llm
        except AttributeError:
            pass
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
