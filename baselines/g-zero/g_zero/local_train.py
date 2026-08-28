"""Local LoRA DPO trainer — replaces Tinker's training client (Phase 3).

Offline DPO on the Solver (Generator) in a self-contained HuggingFace loop:

  * One model on cuda:0 carries the trainable LoRA adapter; π_ref (the Solver
    at round start) is that same model with the adapter disabled, so no
    separate frozen copy is resident.
  * DPO-sigmoid loss over length-normalized log-ratios, so a longer `chosen`
    cannot dominate the gradient.
  * The prompt is `q` only — hint-assisted behavior is distilled into the
    Solver's q-only conditional.

The adapter is merged into the base before saving, so the vLLM sampling/eval
path loads an ordinary HF model directory.
"""
from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .config import Config
from .local_render import get_renderer, get_tokenizer
from .lora_compat import merge_lora, save_merged_checkpoint, wrap_lora

logger = logging.getLogger(__name__)


def _load_pairs(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _response_logprob_lengthnorm(model, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Mean log p(token_t | token_<t) over the response tokens (positions >= prompt_len).

    input_ids: [1, T] on the model's device. Returns a scalar tensor.
    """
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()           # predicts positions 1..T-1
    targets = input_ids[:, 1:]                         # tokens   1..T-1
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    resp_logp = logp[:, prompt_len - 1:]               # response token log-probs (length R)
    if resp_logp.shape[1] == 0:
        return logp.new_zeros(logp.shape[0])
    return resp_logp.mean(dim=-1)                       # length-normalized


def _encode_side(renderer, tokenizer, prompt: str, response: str, max_len: int):
    """Return (input_ids[1,T] on CPU, prompt_len) for one DPO side."""
    gen_prompt = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
    prompt_ids = list(gen_prompt.tokens)
    resp_ids = tokenizer.encode(response, add_special_tokens=False)
    if not resp_ids:
        resp_ids = [tokenizer.eos_token_id or 0]
    budget = max_len - len(prompt_ids)
    if budget < 1:
        prompt_ids = prompt_ids[: max_len - 1]
        budget = 1
    resp_ids = resp_ids[:budget]
    ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long)
    return ids, len(prompt_ids)


def train_dpo(config: Config, *, data_path: Path, save_path: str, init_path: str | None = None) -> str:
    """Run LoRA DPO on the Solver. Returns the saved (merged) HF model directory.

    init_path: HF model dir to initialize the Solver from (e.g. the previous
    round's merged checkpoint); defaults to the base model.
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    base_model = init_path or config.model_name
    tokenizer = get_tokenizer(config.model_name)
    renderer = get_renderer(config.renderer_name, tokenizer)
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

    policy_device = "cuda:0"
    logger.info("DPO (LoRA r=%d) init from: %s  | device=%s", config.lora_rank, base_model, policy_device)

    # Base on GPU0 + trainable adapter; the adapter-disabled base doubles as
    # π_ref, so only one model is resident.
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, trust_remote_code=False,
    ).to(policy_device)
    base.config.use_cache = False
    model = wrap_lora(base, config)
    model.gradient_checkpointing_enable()
    model.train()

    rows = _load_pairs(data_path)
    logger.info("Loaded %d DPO pairs from %s", len(rows), data_path)

    bs = config.dpo_batch_size
    n_batches = len(rows) // bs if bs else 0
    if n_batches == 0:
        raise SystemExit(f"Phase 3: not enough pairs ({len(rows)}) for batch size {bs}.")
    # dpo_max_steps is authoritative: exactly that many steps, cycling the pair
    # set if it holds fewer batches (no epoch cap).
    total_steps = config.dpo_max_steps if config.dpo_max_steps else n_batches

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=config.dpo_lr, betas=(0.9, 0.95), eps=1e-8)
    max_len = config.vllm_max_model_len
    beta = config.dpo_beta

    mf = open(metrics_path, "w")
    for step in range(total_steps):
        batch_idx = step % n_batches
        batch = rows[batch_idx * bs: (batch_idx + 1) * bs]

        lr = config.dpo_lr * max(0.0, 1.0 - step / max(1, total_steps))
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
        losses, accs, margins, c_rewards, r_rewards = [], [], [], [], []
        for r in batch:
            c_ids, c_plen = _encode_side(renderer, tokenizer, r["prompt"], r["chosen"], max_len)
            rj_ids, rj_plen = _encode_side(renderer, tokenizer, r["prompt"], r["rejected"], max_len)

            # π_ref = base with the adapter disabled (Solver at round start).
            with torch.no_grad(), model.disable_adapter():
                c_ref = _response_logprob_lengthnorm(model, c_ids.to(policy_device), c_plen)
                rj_ref = _response_logprob_lengthnorm(model, rj_ids.to(policy_device), rj_plen)

            # Policy (adapter active).
            c_lp = _response_logprob_lengthnorm(model, c_ids.to(policy_device), c_plen)
            rj_lp = _response_logprob_lengthnorm(model, rj_ids.to(policy_device), rj_plen)

            chosen_ratio = c_lp - c_ref
            rejected_ratio = rj_lp - rj_ref
            loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()
            (loss / len(batch)).backward()

            losses.append(float(loss.item()))
            accs.append(float((chosen_ratio > rejected_ratio).float().mean().item()))
            margins.append(float(((chosen_ratio - rejected_ratio) * beta).mean().item()))
            c_rewards.append(float((beta * chosen_ratio).mean().item()))
            r_rewards.append(float((beta * rejected_ratio).mean().item()))

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()

        m = {
            "solver_dpo/step": step,
            "solver_dpo/lr": lr,
            "dpo_loss": sum(losses) / len(losses),
            "accuracy": sum(accs) / len(accs),
            "margin": sum(margins) / len(margins),
            "chosen_reward": sum(c_rewards) / len(c_rewards),
            "rejected_reward": sum(r_rewards) / len(r_rewards),
        }
        mf.write(json.dumps(m) + "\n"); mf.flush()
        logger.info(
            "  dpo step %d/%d lr=%.2e loss=%.4f acc=%.3f margin=%.3f",
            step, total_steps, lr, m["dpo_loss"], m["accuracy"], m["margin"],
        )
    mf.close()

    # Merge the adapter and save a plain HF model for the vLLM sampling/eval path.
    out_dir = save_dir / "final_model"
    logger.info("Merging LoRA + saving Solver -> %s", out_dir)
    model.eval()
    merged = merge_lora(model)
    merged.config.use_cache = True
    save_merged_checkpoint(merged, tokenizer, out_dir)

    merged.to("cpu"); model.to("cpu")
    del merged, model, optim
    gc.collect()
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            with torch.cuda.device(d):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    return str(out_dir)
