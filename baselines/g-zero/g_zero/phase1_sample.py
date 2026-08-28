"""Phase-1 sampling worker (run as a subprocess, one GRPO step at a time).

Loads the *current* Challenger policy and the frozen Solver in vLLM, samples
n_roll Challenger rollouts (q/h), then for the valid pairs samples a_hard from
the Solver and computes the Hint-δ reward. It runs in its own process so all
vLLM GPU memory is released on exit, leaving the GPU free for the HF LoRA
gradient step that the parent (`phase1.py`) does.

Input  (CLI args): base model, optional LoRA adapter dir, Solver path, sizes.
Output (JSON at --out): {"rollouts": [[token_ids]...], "q": [...], "h": [...],
                         "valid": [bool...], "delta": [float...]}  (len == n_roll)
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .hint_delta import score_batch
from .local_backend import LocalSampler
from .local_render import SamplingParams, get_renderer, get_tokenizer
from .parse import extract_qh
from .prompts import build_challenger_convo


def _build_challenger_path(base_model: str, adapter: str, tokenizer, work_dir: Path) -> str:
    """Return an HF dir/id for vLLM: base as-is, or base+adapter merged to disk."""
    if adapter in ("", "none", None):
        return base_model
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from .lora_compat import patch_peft_torchao, save_merged_checkpoint
    patch_peft_torchao()
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, trust_remote_code=False)
    pm = PeftModel.from_pretrained(base, adapter)
    merged = pm.merge_and_unload()
    out = work_dir / "merged_challenger"
    save_merged_checkpoint(merged, tokenizer, out)
    del base, pm, merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--solver_model", required=True)
    ap.add_argument("--model_name", required=True)      # tokenizer/renderer source
    ap.add_argument("--renderer", default="qwen3")
    ap.add_argument("--n_roll", type=int, required=True)
    ap.add_argument("--challenger_max_tokens", type=int, required=True)
    ap.add_argument("--phase1_max_tokens", type=int, required=True)
    ap.add_argument("--solver_temperature", type=float, default=0.7)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu_mem", type=float, default=0.55)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    work_dir = Path(args.work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer(args.model_name)
    renderer = get_renderer(args.renderer, tokenizer)

    def _mk(path: str) -> LocalSampler:
        return LocalSampler(
            path, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
            max_model_len=args.max_model_len, dtype=args.dtype,
        )

    # ── 1. Challenger rollouts (q/h) via vLLM ───────────────────────────────
    ch_path = _build_challenger_path(args.base_model, args.adapter, tokenizer, work_dir)
    ch_sampler = _mk(ch_path)
    prompt = renderer.build_generation_prompt(build_challenger_convo())
    ch_sp = SamplingParams(
        max_tokens=args.challenger_max_tokens, temperature=1.0,
        stop=renderer.get_stop_sequences(),
    )
    res = ch_sampler.sample(prompt=prompt, num_samples=args.n_roll, sampling_params=ch_sp)
    rollouts = [list(seq.tokens) for seq in res.sequences]
    ch_sampler.shutdown()
    del ch_sampler; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    qh = [extract_qh(tokenizer.decode(r)) for r in rollouts]
    valid_idx = [i for i, (q, h) in enumerate(qh) if q and h]

    # ── 2. Solver a_hard + δ for valid rollouts via vLLM ────────────────────
    deltas = [0.0] * len(rollouts)
    if valid_idx:
        sv = _mk(args.solver_model)
        sv_sp = SamplingParams(
            max_tokens=args.phase1_max_tokens, temperature=args.solver_temperature,
            stop=renderer.get_stop_sequences(),
        )
        scores = score_batch(
            sv, renderer, tokenizer, [qh[i] for i in valid_idx],
            mode="delta_only", sampling_params=sv_sp,
        )
        sv.shutdown()
        for i, sc in zip(valid_idx, scores):
            deltas[i] = float(sc.delta)

    valid = [bool(qh[i][0] and qh[i][1]) for i in range(len(rollouts))]
    payload = {
        "rollouts": rollouts,
        "q": [qh[i][0] for i in range(len(rollouts))],
        "h": [qh[i][1] for i in range(len(rollouts))],
        "valid": valid,
        "delta": deltas,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"[phase1_sample] wrote {args.out}: n_roll={len(rollouts)} valid={sum(valid)}", flush=True)


if __name__ == "__main__":
    main()
