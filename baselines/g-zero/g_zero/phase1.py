"""Phase 1: Challenger GRPO training — local LoRA Dr.GRPO, vLLM sampling.

The Challenger (Proposer) is trained to maximize:

    reward(q, h) = δ(q, h, a_hard)
                 − λ_len · max(0, |h| − L_tgt) / 100        (length hinge)
                 − BLEU-cluster-share(q, batch)              (diversity penalty)

with Dr.GRPO **mean-only** advantages (centered within each group, no std
normalization). Invalid rollouts (missing/empty <question>/<hint>) get a −1
floor and skip the δ computation.

Backend split:
  * **Sampling** — Challenger rollouts and the frozen Solver's a_hard + Hint-δ
    run in a per-step vLLM subprocess (`g_zero/phase1_sample.py`), which exits
    after each step so all its GPU memory is released.
  * **Training** — the Challenger is a base model + trainable LoRA adapter
    resident on cuda:0; each GRPO step is one on-policy REINFORCE update on the
    adapter, merged into the base before saving.
"""
from __future__ import annotations

import gc
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .bleu_penalty import cluster_share
from .config import Config
from .local_render import get_renderer, get_tokenizer
from .lora_compat import checkpoint_complete, merge_lora, save_merged_checkpoint, wrap_lora
from .prompts import build_challenger_convo

logger = logging.getLogger(__name__)

# vLLM KV fraction for the sampling subprocess. Kept low so it coexists with the
# resident HF Challenger on cuda:0 and survives the fragmentation that builds up
# across many spawn/teardown cycles (a higher fraction fails vLLM's "free <
# desired" startup check); at tp=4 each shard's KV cache is tiny.
_SAMPLER_GPU_MEM = 0.35


def _free_cuda_all() -> None:
    gc.collect()
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            with torch.cuda.device(d):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


def _run_sampling_step(config: Config, *, base_model: str, adapter_dir: str,
                       solver_model: str, n_roll: int, work_dir: Path, out_json: Path) -> dict:
    """Launch the vLLM sampling subprocess for one GRPO step; return its JSON."""
    cmd = [
        sys.executable, "-m", "g_zero.phase1_sample",
        "--base_model", base_model,
        "--adapter", adapter_dir,
        "--solver_model", solver_model,
        "--model_name", config.model_name,
        "--renderer", config.renderer_name,
        "--n_roll", str(n_roll),
        "--challenger_max_tokens", str(config.challenger_max_tokens),
        "--phase1_max_tokens", str(config.phase1_max_tokens),
        "--solver_temperature", str(config.solver_sample_temperature),
        "--tp", str(config.tensor_parallel_size),
        "--gpu_mem", str(_SAMPLER_GPU_MEM),
        "--max_model_len", str(config.vllm_max_model_len),
        "--dtype", config.dtype,
        "--work_dir", str(work_dir),
        "--out", str(out_json),
    ]
    # Release the parent's cached (unused) GPU blocks so the subprocess has room.
    _free_cuda_all()
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"phase1_sample subprocess failed (rc={rc}) — see stdout above")
    with open(out_json) as f:
        return json.load(f)


def train_challenger(
    config: Config,
    *,
    challenger_init_path: str | None = None,
    solver_sampling_path: str | None = None,
    save_path: str,
) -> str:
    """One Phase-1 stage. Returns the trained (merged) Challenger HF model dir."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    out_dir = save_dir / "final_model"
    if checkpoint_complete(out_dir):
        logger.info("Phase 1: cached Challenger at %s, skipping training", out_dir)
        return str(out_dir)
    if out_dir.exists():
        # Torso from an interrupted save — retrain rather than hand it to vLLM.
        logger.warning("Phase 1: incomplete checkpoint at %s, removing and retraining", out_dir)
        shutil.rmtree(out_dir)

    tokenizer = get_tokenizer(config.model_name)
    renderer = get_renderer(config.renderer_name, tokenizer)
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    ch_device = "cuda:0"
    base_model = challenger_init_path or config.model_name
    solver_model = solver_sampling_path or config.model_name

    logger.info("Phase 1 (LoRA Dr.GRPO, vLLM sampling): challenger=%s solver=%s",
                base_model, solver_model)

    # Challenger = base + trainable adapter, resident on cuda:0 for the GRPO
    # weight update; sampling happens out-of-process.
    ch_base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, trust_remote_code=False,
    ).to(ch_device)
    ch_base.config.use_cache = False
    challenger = wrap_lora(ch_base, config)
    # A single full-length rollout backpropagated through an 8B model OOMs on
    # activations without checkpointing (wrap_lora already did the
    # enable_input_require_grads() that a frozen base requires).
    challenger.gradient_checkpointing_enable()
    challenger.train()
    ch_trainable = [p for p in challenger.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(ch_trainable, lr=config.challenger_lr, betas=(0.9, 0.95), eps=1e-8)

    challenger_prompt = list(renderer.build_generation_prompt(build_challenger_convo()).tokens)
    ob_len = len(challenger_prompt)
    n_roll = config.challenger_batch_size * config.challenger_group_size
    work_root = save_dir / "_sample_work"
    adapter_dir = save_dir / "_adapter"

    mf = open(metrics_path, "w")
    for step in range(config.challenger_steps):
        t0 = time.time()

        # 1) Snapshot current policy for sampling. Step 0's adapter is the
        #    identity (B=0), so the base model already == the policy.
        if step == 0:
            adapter_arg = "none"
        else:
            challenger.save_pretrained(str(adapter_dir))   # adapter only (small)
            adapter_arg = str(adapter_dir)

        # 2) vLLM sampling subprocess → rollouts + δ.
        data = _run_sampling_step(
            config, base_model=base_model, adapter_dir=adapter_arg,
            solver_model=solver_model, n_roll=n_roll,
            work_dir=work_root / f"step{step}", out_json=work_root / f"step{step}.json",
        )
        rollouts = data["rollouts"]
        qs_raw, hs, valids, deltas_raw = data["q"], data["h"], data["valid"], data["delta"]
        n = len(rollouts)

        # 3) δ (clipped) + diversity/length penalties → reward.
        deltas = [max(deltas_raw[i], config.delta_clip_min) if valids[i] else 0.0 for i in range(n)]
        qs = [qs_raw[i] if valids[i] else "__INVALID__" for i in range(n)]
        penalties = cluster_share(qs, config.bleu_distance_threshold)
        rewards, length_pens = [], []
        for i in range(n):
            if valids[i]:
                over = max(0, len(hs[i]) - config.hint_length_target_chars)
                lp = config.hint_length_penalty_lambda * (over / 100.0)
                length_pens.append(lp)
                rewards.append(deltas[i] - penalties[i] - lp)
            else:
                length_pens.append(0.0)
                rewards.append(-1.0 - penalties[i])

        # 4) Dr.GRPO advantages: mean-center within each group (no std norm).
        group_of = [i // config.challenger_group_size for i in range(n)]
        advantages = [0.0] * n
        for g in range(config.challenger_batch_size):
            idx = [i for i in range(n) if group_of[i] == g]
            if not idx:
                continue
            mean = sum(rewards[i] for i in idx) / len(idx)
            for i in idx:
                advantages[i] = rewards[i] - mean

        # 5) Policy-gradient update (REINFORCE surrogate; one on-policy step).
        optim.zero_grad(set_to_none=True)
        n_used = 0
        for i in range(n):
            if abs(advantages[i]) < 1e-8 or len(rollouts[i]) < 1:
                continue
            full = torch.tensor([challenger_prompt + rollouts[i]], device=ch_device)
            logits = challenger(input_ids=full, use_cache=False).logits[:, :-1, :].float()
            targets = full[:, 1:]
            logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            resp_logp = logp[0, ob_len - 1:]
            loss = -(advantages[i] * resp_logp.sum()) / n
            loss.backward()
            n_used += 1
        if n_used:
            torch.nn.utils.clip_grad_norm_(ch_trainable, 1.0)
            optim.step()

        n_valid = sum(valids)
        m = {
            "challenger/step": step,
            "challenger/reward_mean": sum(rewards) / max(1, n),
            "challenger/delta_mean_valid": sum(deltas[i] for i in range(n) if valids[i]) / max(1, n_valid),
            "challenger/valid_frac": n_valid / max(1, n),
            "challenger/penalty_mean": sum(penalties) / max(1, len(penalties)),
            "challenger/length_penalty_mean": sum(length_pens) / max(1, len(length_pens)),
            "challenger/hint_len_mean_chars": (
                sum(len(hs[i]) for i in range(n) if valids[i]) / max(1, n_valid)
            ),
            "challenger/n_datums": n_used,
            "time/step": time.time() - t0,
        }
        mf.write(json.dumps(m) + "\n"); mf.flush()
        logger.info(
            "  challenger step %d/%d reward=%.4f delta=%.4f valid=%.2f hint_len=%.0f n=%d (%.1fs)",
            step, config.challenger_steps, m["challenger/reward_mean"],
            m["challenger/delta_mean_valid"], m["challenger/valid_frac"],
            m["challenger/hint_len_mean_chars"], n_used, m["time/step"],
        )
        _free_cuda_all()
    mf.close()

    logger.info("Merging LoRA + saving Challenger -> %s", out_dir)
    challenger.eval()
    merged = merge_lora(challenger)
    merged.config.use_cache = True
    save_merged_checkpoint(merged, tokenizer, out_dir)

    merged.to("cpu"); challenger.to("cpu")
    del merged, challenger, optim
    _free_cuda_all()
    return str(out_dir)
