"""All G-Zero hyperparameters in one place.

Defaults reproduce the paper-main configuration on Qwen3-8B-Base.
Override any field via CLI: `python -m g_zero.multi_round --field value`.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ─── Run identity ────────────────────────────────────────────────────────
    tag: str = "g_zero_demo"
    storage_path: str = field(
        default_factory=lambda: os.environ.get(
            "G_ZERO_STORAGE", str(Path(__file__).resolve().parents[1] / "runs")
        )
    )

    # ─── Model ───────────────────────────────────────────────────────────────
    # Local HF model id/path; the renderer wraps the model's own chat template
    # ("qwen3" for Qwen3, "llama3" for Llama-3.2). Paper-main model (Table 4).
    model_name: str = "Qwen/Qwen3-8B-Base"
    renderer_name: str = "qwen3"
    base_url: str | None = None
    # ─── LoRA (paper-faithful; Table 4: rank 32) ─────────────────────────────
    # Phase-1 GRPO and Phase-3 DPO both train adapters, merged into the base
    # before saving so the vLLM path loads an ordinary checkpoint (lora_compat).
    lora_rank: int = 32
    lora_alpha: int = 64                    # 2×rank (standard)
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )

    # ─── Local backend (one 4-GPU node) ──────────────────────────────────────
    tensor_parallel_size: int = 4          # vLLM TP across the 4-GPU node (paper runs)
    gpu_memory_utilization: float = 0.7    # vLLM KV-cache fraction (headroom for training residual)
    vllm_max_model_len: int = 32768        # fits q/h ctx + a_hard(16384) + overhead for δ logprobs
    dtype: str = "bfloat16"

    # ─── Phase 1: Challenger GRPO (local LoRA Dr.GRPO; see phase1.py) ─────────
    run_phase1: bool = True
    challenger_steps: int = 6                       # GRPO steps (Table 4)
    challenger_batch_size: int = 128                # GRPO groups per step (Table 4: batch size 128)
    challenger_group_size: int = 16                 # rollouts per group (Table 4)
    challenger_lr: float = 4e-5                     # paper LoRA rate (Table 4)
    challenger_gen_sub_batch: int = 8               # unused: Phase-1 sampling runs in the vLLM subprocess (phase1_sample.py)
    # Generation cap for a_hard, the δ-reward signal (Table 4: 8192). Sampling
    # a_hard dominates Phase-1 cost; lower it to trade fidelity for speed.
    phase1_max_tokens: int = 8192
    delta_clip_min: float = -5.0                   # floor for δ in reward
    hint_length_target_chars: int = 200             # length-hinge target
    hint_length_penalty_lambda: float = 0.03        # λ in length penalty
    bleu_distance_threshold: float = 0.5            # for cluster_share()

    # ─── Phase 2: data generation ────────────────────────────────────────────
    num_questions: int = 2000                       # paper-main: 2000
    challenger_max_tokens: int = 8192               # Proposer max_tokens (Table 4)
    solver_max_tokens: int = 16384                  # Generator max_tokens (Table 4)
    solver_sample_temperature: float = 0.7

    # δ percentile filter [pct_low, pct_high]. paper-main = bot50 = [0, 50].
    pct_low: float = 0.0
    pct_high: float = 50.0

    # Quality filters on chosen (= a_assisted)
    chosen_min_chars: int = 100
    chosen_max_chars: int = 10_000
    chosen_rejected_ratio_max: float = 2.5
    chosen_repetition_zlib_threshold: float = 0.15
    chosen_prompt_prefix_overlap: int = 30
    chosen_role_marker_prefixes: tuple[str, ...] = (
        "Assistant:", "AI:", "Bot:", "User:", "Human:"
    )

    # ─── Phase 3: DPO ────────────────────────────────────────────────────────
    # LoRA DPO (Table 4). Steps cycle the pair set as needed; the loss is a
    # length-normalized log-ratio on a q-only prompt (local_train.py).
    dpo_beta: float = 2.0
    dpo_lr: float = 1e-5
    dpo_batch_size: int = 8
    dpo_max_steps: int = 50

    # ─── Per-run paths ───────────────────────────────────────────────────────
    def run_dir(self) -> Path:
        return Path(self.storage_path) / self.tag

    def raw_pool_path(self) -> Path:
        return self.run_dir() / "raw_pool.jsonl"

    def dpo_data_path(self) -> Path:
        return self.run_dir() / "dpo_data.jsonl"

    def challenger_save_dir(self) -> Path:
        return self.run_dir() / "challenger"

    def solver_save_dir(self) -> Path:
        return self.run_dir() / "solver"


def parse_cli_overrides(argv: list[str] | None = None) -> Config:
    """Build a Config and overlay CLI overrides like `--tag foo --num_questions 100`."""
    cfg = Config()
    parser = argparse.ArgumentParser(description="G-Zero pipeline")
    for f in cfg.__dataclass_fields__.values():
        if f.type in (bool, "bool"):
            parser.add_argument(f"--{f.name}", type=lambda x: str(x).lower() in ("1","true","y"))
        elif f.type in (int, "int"):
            parser.add_argument(f"--{f.name}", type=int)
        elif f.type in (float, "float"):
            parser.add_argument(f"--{f.name}", type=float)
        else:
            parser.add_argument(f"--{f.name}", type=str)
    args = parser.parse_args(argv)
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg
