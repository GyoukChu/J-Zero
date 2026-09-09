#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSDP pairwise Bradley-Terry trainer for the Skywork reward model.

Hand-rolled rather than reusing verl: verl's model_merger targets
AutoModelForCausalLM, not SequenceClassification. Must run under torchrun:

    torchrun --nproc_per_node=4 rm_train_loop.py \
        --pairs_json <path> \
        --init_model <Skywork or prev RM dir> \
        --save_dir <ckpt out dir>/global_step_8/actor/huggingface \
        --batch_size 64 --num_steps 8 --lr 5e-7 --max_len 8192

Loss: -log sigmoid(s_chosen - s_rejected), mean per micro-batch; AdamW at a
constant LR (no warmup, no weight decay). One epoch over the (chosen,
rejected) pair JSON. After `--num_steps` steps the full state dict is gathered
on rank 0 (offload_to_cpu=True, rank0_only=True) and written via
save_pretrained together with the tokenizer, so the dir is HF-loadable as the
next iter's RM.

Tokenization matches the inference server (rm_service/rm_io.py): right
padding, left truncation, max_length=8192. Score = `model(**enc).logits` — the
SC head already reduces over the sequence at its last-non-pad position.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from functools import partial
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForSequenceClassification
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from rm_service.rm_io import format_rm_input, load_skywork_tokenizer  # noqa: E402


def _log0(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _build_micro_batches(pairs: List[Dict], batch_size: int, rank: int, world_size: int):
    """Yield one per-rank slice of `batch_size // world_size` pairs per
    optimizer step, so each step sees `batch_size` pairs across all ranks.
    world_size MUST divide batch_size. The caller splits each slice further
    into micro-batches for gradient accumulation."""
    if batch_size % world_size != 0:
        raise ValueError(f"batch_size {batch_size} not divisible by world_size {world_size}")
    per_rank = batch_size // world_size
    n_pairs = len(pairs)
    n_steps = n_pairs // batch_size
    for step in range(n_steps):
        start = step * batch_size + rank * per_rank
        end = start + per_rank
        yield pairs[start:end]


def _split_into_micro(pair_batch: List[Dict], micro_size: int) -> List[List[Dict]]:
    """Split a per-rank batch into micro-batches of size <= micro_size."""
    if micro_size <= 0:
        return [pair_batch]
    out: List[List[Dict]] = []
    for i in range(0, len(pair_batch), micro_size):
        out.append(pair_batch[i : i + micro_size])
    return out


def _tokenize_pair_batch(tokenizer, pair_batch: List[Dict], max_len: int, device):
    """Build (chosen_enc, rejected_enc) tensor dicts on `device`."""
    chosen_texts = [format_rm_input(tokenizer, p["question"], p["chosen"]) for p in pair_batch]
    rejected_texts = [format_rm_input(tokenizer, p["question"], p["rejected"]) for p in pair_batch]
    enc_c = tokenizer(chosen_texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
    enc_r = tokenizer(rejected_texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
    enc_c = {k: v.to(device, non_blocking=True) for k, v in enc_c.items()}
    enc_r = {k: v.to(device, non_blocking=True) for k, v in enc_r.items()}
    return enc_c, enc_r


def _score(model, enc) -> torch.Tensor:
    out = model(**enc)
    # SC head with num_labels=1 -> logits shape (B, 1).
    return out.logits.view(-1)


def main(args: argparse.Namespace) -> None:
    # --- DDP/FSDP init ---------------------------------------------------
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    _log0(rank, f"[rm_train] world_size={world_size} bs={args.batch_size} "
                f"steps={args.num_steps} lr={args.lr} max_len={args.max_len}")

    # --- Wandb init (rank 0 only) ----------------------------------------
    # verl instruments challenger/solver itself; this loop is hand-rolled, so
    # log here to land in the same project.
    _wandb_active = False
    if rank == 0 and not args.disable_wandb:
        try:
            import wandb as _wandb  # noqa: WPS433 (intentional lazy import)
            # Write to <recipe_dir>/wandb/ next to verl's challenger/solver
            # runs, so one `wandb sync` catches all three for this iter.
            wandb_dir = HERE
            run_name = args.exp_name if args.exp_name else f"rm_train_step{args.num_steps}"
            _wandb.init(
                project=os.getenv("WANDB_PROJECT", "J-Zero"),
                name=run_name,
                config={
                    "batch_size": args.batch_size,
                    "num_steps": args.num_steps,
                    "lr": args.lr,
                    "max_len": args.max_len,
                    "micro_batch": args.micro_batch,
                    "gradient_checkpointing": bool(args.gradient_checkpointing),
                    "init_model": args.init_model,
                    "pairs_json": args.pairs_json,
                    "shuffle_seed": args.shuffle_seed,
                    "attn_impl": args.attn_impl,
                    "world_size": world_size,
                },
                mode=os.getenv("WANDB_MODE", "online"),
                dir=wandb_dir,
            )
            _wandb_active = True
            _log0(rank, f"[rm_train] wandb run initialized: project={os.getenv('WANDB_PROJECT','J-Zero')} name={run_name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[rm_train] WARN wandb init failed: {exc}", flush=True)
            _wandb_active = False

    # --- Load pairs ------------------------------------------------------
    with open(args.pairs_json) as fh:
        pairs: List[Dict] = json.load(fh)
    # Shuffle BEFORE sharding so ranks see iid pairs rather than a block of a
    # possibly question-sorted file; fixed seed => same order on every rank.
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(pairs)
    expected = args.batch_size * args.num_steps
    if len(pairs) < expected:
        if os.environ.get("RZERO_RM_ALLOW_SHORT", "0") != "1":
            raise SystemExit(f"[rm_train] FATAL: only {len(pairs)} pairs (need {expected}) — "
                             f"set RZERO_RM_ALLOW_SHORT=1 to train anyway")
        _log0(rank, f"[rm_train] WARN: only {len(pairs)} pairs (need {expected}); "
                    f"taking {len(pairs) // args.batch_size} effective steps")
        n_batches = len(pairs) // args.batch_size
        pairs = pairs[: n_batches * args.batch_size]
        effective_steps = n_batches
    else:
        pairs = pairs[:expected]
        effective_steps = args.num_steps
    _log0(rank, f"[rm_train] using {len(pairs)} pairs -> {effective_steps} steps "
                f"(shuffle_seed={args.shuffle_seed})")

    # --- Tokenizer + model ----------------------------------------------
    _log0(rank, f"[rm_train] loading tokenizer from {args.init_model}")
    tokenizer = load_skywork_tokenizer(args.init_model)

    _log0(rank, f"[rm_train] loading model from {args.init_model}")
    # Keep FP32 master weights; FSDP casts to bf16 for forward/backward.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init_model,
        num_labels=1,
        torch_dtype=torch.float32,
        attn_implementation=args.attn_impl,
    )
    # force-sync to tokenizer pad to avoid SC head reading at left-padded positions
    model.config.pad_token_id = tokenizer.pad_token_id

    # KV cache off: with use_cache=True, HF silently disables gradient
    # checkpointing instead of using it.
    model.config.use_cache = False

    # Gradient checkpointing MUST be enabled BEFORE FSDP.wrap so the hooks land
    # on the unwrapped Llama modules; use_reentrant=False is required for
    # compatibility with FSDP's parameter resharding.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        _log0(rank, "[rm_train] gradient_checkpointing ENABLED (use_reentrant=False)")
    else:
        _log0(rank, "[rm_train] gradient_checkpointing DISABLED")

    wrap_policy = partial(transformer_auto_wrap_policy,
                          transformer_layer_cls={LlamaDecoderLayer})

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        cpu_offload=CPUOffload(offload_params=False),
        use_orig_params=True,
        limit_all_gathers=True,
    )
    model.train()
    _log0(rank, "[rm_train] FSDP wrap done")

    # --- Optimizer (constant LR, AdamW, no warmup, no decay) -------------
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    # --- Train loop (micro-batch + gradient accumulation) ----------------
    step = 0
    losses: List[float] = []
    accs: List[float] = []
    per_rank = args.batch_size // world_size
    micro_size = max(1, args.micro_batch)
    if micro_size > per_rank:
        micro_size = per_rank
    num_micro = math.ceil(per_rank / micro_size)
    _log0(rank, f"[rm_train] per_rank={per_rank} micro_batch={micro_size} "
                f"num_micro_per_step={num_micro} "
                f"(effective global batch = {args.batch_size})")

    # Reset peak memory tracking so the final report reflects training only.
    torch.cuda.reset_peak_memory_stats(device)

    micro_iter = _build_micro_batches(pairs, args.batch_size, rank, world_size)
    for pair_batch in micro_iter:
        step += 1
        if not pair_batch:
            continue
        t0 = time.time()
        optim.zero_grad(set_to_none=True)

        # Backward each micro-batch separately to free activations; FSDP
        # accumulates grads across them, so each loss is scaled by 1/num_micro
        # to keep the step gradient a mean over the full per-rank batch.
        micros = _split_into_micro(pair_batch, micro_size)
        # Sums, not means: ranks may hold unequal numel, so all-reduce counts.
        step_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        step_correct = torch.zeros((), device=device, dtype=torch.float32)
        step_count = torch.zeros((), device=device, dtype=torch.float32)

        for mi, mb in enumerate(micros):
            enc_c, enc_r = _tokenize_pair_batch(tokenizer, mb, args.max_len, device)
            s_c = _score(model, enc_c)
            s_r = _score(model, enc_r)
            margin = s_c - s_r
            # logsigmoid rather than log(sigmoid(...)) for numerical stability.
            loss_mb = -F.logsigmoid(margin).mean() / num_micro
            loss_mb.backward()
            with torch.no_grad():
                step_loss_sum += (-F.logsigmoid(margin).sum()).float()
                step_correct += (margin > 0).float().sum()
                step_count += float(margin.numel())
            del enc_c, enc_r, s_c, s_r, margin, loss_mb

        # MUST be FSDP's bound clip_grad_norm_: the bare torch.nn.utils one
        # sees only the local shard and over-clips where grads concentrate.
        model.clip_grad_norm_(max_norm=1.0)
        optim.step()

        with torch.no_grad():
            dist.all_reduce(step_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(step_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(step_count, op=dist.ReduceOp.SUM)
            denom = step_count.clamp_min(1.0)
            loss_val = (step_loss_sum / denom).item()
            acc = (step_correct / denom).item()

        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        losses.append(float(loss_val))
        accs.append(float(acc))
        _log0(rank, f"[rm_train] step {step}/{effective_steps} loss={loss_val:.4f} "
                    f"pair_acc={acc:.3f} dt={dt:.1f}s "
                    f"peak_mem_gb={peak_gb:.2f}")

        if rank == 0 and _wandb_active:
            try:
                import wandb as _wandb  # noqa: WPS433
                _wandb.log({
                    "rm/loss": float(loss_val),
                    "rm/pair_acc": float(acc),
                    "rm/peak_mem_gb": float(peak_gb),
                    "rm/step_seconds": float(dt),
                }, step=step)
            except Exception as exc:  # noqa: BLE001
                print(f"[rm_train] WARN wandb.log step={step} failed: {exc}", flush=True)

        if step >= effective_steps:
            break

    final_peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_t = torch.tensor([final_peak_gb], device=device, dtype=torch.float32)
    dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
    _log0(rank, f"[rm_train] max_memory_allocated_gb (across ranks) = {peak_t.item():.2f}")

    _log0(rank, f"[rm_train] training done. mean_loss={sum(losses)/max(1,len(losses)):.4f} "
                f"mean_pair_acc={sum(accs)/max(1,len(accs)):.3f}")

    # --- Save HF checkpoint via FULL_STATE_DICT --------------------------
    _log0(rank, f"[rm_train] gathering full state dict on rank 0 ...")
    save_dir = args.save_dir
    save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
        full_state = model.state_dict()
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        _log0(rank, f"[rm_train] writing HF checkpoint to {save_dir}")
        # Preserve FP32 master weights for the next round.
        cpu_model = AutoModelForSequenceClassification.from_pretrained(
            args.init_model,
            num_labels=1,
            torch_dtype=torch.float32,
            # Skip FA2 here — bare CPU build, only used as a state_dict shell.
            attn_implementation="eager",
        )
        cpu_model.load_state_dict(full_state, strict=True)
        cpu_model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "rm_train_summary.json"), "w") as fh:
            json.dump({
                "steps_run": len(losses),
                "mean_loss": sum(losses) / max(1, len(losses)),
                "mean_pair_acc": sum(accs) / max(1, len(accs)),
                "per_step_loss": losses,
                "per_step_pair_acc": accs,
                "config": {
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "num_steps": args.num_steps,
                    "max_len": args.max_len,
                    "init_model": args.init_model,
                    "micro_batch": args.micro_batch,
                    "gradient_checkpointing": bool(args.gradient_checkpointing),
                    "attn_impl": args.attn_impl,
                },
                "max_memory_allocated_gb_max_across_ranks": float(peak_t.item()),
            }, fh, indent=2)
        _log0(rank, "[rm_train] save complete")
    dist.barrier()

    # --- Wandb finalize (rank 0, after save + barrier) ---------------------
    # Before destroy_process_group so wandb has a clean python state.
    if rank == 0 and _wandb_active:
        try:
            import wandb as _wandb  # noqa: WPS433
            _wandb.log({
                "rm/mean_loss": sum(losses) / max(1, len(losses)),
                "rm/mean_pair_acc": sum(accs) / max(1, len(accs)),
                "rm/max_memory_gb_across_ranks": float(peak_t.item()),
                "rm/steps_run": len(losses),
            }, step=max(1, len(losses)))
            _wandb.finish()
            _log0(rank, "[rm_train] wandb.finish() called")
        except Exception as exc:  # noqa: BLE001
            print(f"[rm_train] WARN wandb.finish() failed: {exc}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_json", type=str, required=True,
                   help="JSON list of {question, chosen, rejected} pairs")
    p.add_argument("--init_model", type=str, required=True,
                   help="HF path or local dir for RM init weights")
    p.add_argument("--save_dir", type=str, required=True,
                   help="HF save_pretrained target (e.g. .../global_step_8/actor/huggingface)")
    p.add_argument("--batch_size", type=int,
                   default=int(os.getenv("RZERO_RM_BATCH", "64")))
    p.add_argument("--num_steps", type=int,
                   default=int(os.getenv("RZERO_RM_STEPS", "4")))
    p.add_argument("--lr", type=float,
                   default=float(os.getenv("RZERO_RM_LR", "5e-7")))
    p.add_argument("--max_len", type=int,
                   default=int(os.getenv("RZERO_RM_MAX_LEN", "8192")))
    p.add_argument("--attn_impl", type=str,
                   default=os.getenv("RZERO_RM_ATTN_IMPL", "flash_attention_2"))
    p.add_argument("--shuffle_seed", type=int,
                   default=int(os.getenv("RZERO_RM_SHUFFLE_SEED", "42")))
    p.add_argument("--micro_batch", type=int,
                   default=int(os.getenv("RZERO_RM_MICRO_BATCH", "4")),
                   help="per-rank micro batch size; grads accumulate across "
                        "ceil(per_rank/micro_batch) micros before optim.step()")
    p.add_argument("--gradient_checkpointing", type=int,
                   default=int(os.getenv("RZERO_RM_GRAD_CKPT", "1")),
                   help="1 to enable activation checkpointing (default), 0 off")
    p.add_argument("--disable_wandb", action="store_true",
                   default=bool(int(os.getenv("RZERO_RM_DISABLE_WANDB", "0"))),
                   help="if set, skip wandb init/log/finish entirely (rank 0)")
    p.add_argument("--exp_name", type=str,
                   default=os.getenv("RZERO_RM_EXP_NAME", ""),
                   help="wandb run name (default derived from --num_steps)")
    main(p.parse_args())
