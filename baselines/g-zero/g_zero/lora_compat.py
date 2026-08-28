"""LoRA helpers for G-Zero training (Phase-1 GRPO + DPO), paper rank 32.

Environment gotcha: peft 0.19 calls `is_torchao_available()` while dispatching
every LoRA Linear, and that raises on the pinned torchao 0.9.0. torchao is
irrelevant here (bf16, no quantization), so the check is neutralized to False
before `get_peft_model` to keep peft on the normal Linear-LoRA path.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def patch_peft_torchao() -> None:
    """Force peft's torchao availability check to False (avoids the 0.9.0 raise)."""
    try:
        import peft.import_utils as _iu
        _iu.is_torchao_available = lambda: False
        import peft.tuners.lora.torchao as _lt
        _lt.is_torchao_available = lambda: False
    except Exception as e:  # noqa: BLE001
        logger.warning("peft torchao patch skipped: %s", e)


# Qwen3 / Llama-3 attention + MLP projections — standard LoRA target set.
DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


def wrap_lora(model, config):
    """Attach a fresh LoRA adapter to `model` and return the PeftModel.

    Only the adapter trains; the frozen base weights (== πG / πP at the start of
    the round) stay recoverable via `model.disable_adapter()`, which the DPO
    path uses as the reference π_ref.
    """
    patch_peft_torchao()
    from peft import LoraConfig, get_peft_model

    targets = list(getattr(config, "lora_target_modules", None) or DEFAULT_TARGET_MODULES)
    lc = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lc)
    # Gradient checkpointing with a frozen base needs grads forced onto the
    # (frozen) input-embedding outputs.
    peft_model.enable_input_require_grads()
    return peft_model


def merge_lora(peft_model):
    """Merge the adapter into the base, returning a plain HF model that the
    vLLM sampling path loads like any ordinary checkpoint."""
    return peft_model.merge_and_unload()


def save_merged_checkpoint(merged, tokenizer, out_dir) -> None:
    """save_pretrained the merged model + tokenizer as a plain HF checkpoint."""
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


def checkpoint_complete(ckpt_dir) -> bool:
    """True iff ``ckpt_dir`` holds a complete save_merged_checkpoint output.

    save_pretrained writes config.json BEFORE the weight shards, and the
    tokenizer files land last, so an interrupted save (Ctrl-C, OOM kill)
    leaves a directory that still has config.json. A cache check keyed on
    config.json alone then skips retraining and hands the torso to vLLM,
    which dies later on the missing tokenizer/vocab. Require the pieces
    written after config.json instead: complete weights (single-file, or an
    index whose every shard exists) plus tokenizer_config.json (written
    last, so its presence implies the save ran to the end).
    """
    d = Path(ckpt_dir)
    if not (d / "config.json").is_file():
        return False
    if not (d / "model.safetensors").is_file():
        index = d / "model.safetensors.index.json"
        if not index.is_file():
            return False
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
        except (OSError, ValueError):
            return False
        shards = set(weight_map.values())
        if not shards or any(not (d / s).is_file() for s in shards):
            return False
    return (d / "tokenizer_config.json").is_file()
