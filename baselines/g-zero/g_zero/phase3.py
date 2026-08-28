"""Phase 3: DPO-train the Solver on (q, a_assisted, a_hard) pairs.

Thin wrapper over the local LoRA DPO loop in `local_train.py`, which holds the
DPO math; this module plumbs config/paths and keeps the `train_solver_dpo`
signature that multi_round.py calls.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .local_train import train_dpo

logger = logging.getLogger(__name__)


def train_solver_dpo(
    config: Config,
    *,
    solver_init_path: str | None = None,
    data_path: Path | None = None,
    save_path: str | None = None,
) -> str:
    """Run DPO. Returns the path to the merged Solver HF model directory."""
    save_path = save_path or str(config.solver_save_dir())
    data_path = Path(data_path or config.dpo_data_path())
    logger.info("Phase 3: DPO on %s -> %s", data_path, save_path)
    return train_dpo(
        config, data_path=data_path, save_path=save_path, init_path=solver_init_path,
    )
