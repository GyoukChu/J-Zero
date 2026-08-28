"""Multi-round Challenger ↔ Solver co-evolution.

Round 1 bootstraps both Challenger and Solver from the base model. In
round k > 1, the Challenger is initialized from round k-1's Challenger
and scored against round k-1's Solver; round k's Solver is initialized
from round k-1's Solver and DPO-trained on data built by the new
Challenger together with round k-1's Solver.

Per-round artifacts land in ``runs/<tag>/round_<k>/``, with a
``resume_state.json`` at the run root so re-running picks up where it
crashed.
"""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

from .config import Config, parse_cli_overrides
from .phase1 import train_challenger
from .phase2 import build_dpo_dataset
from .phase3 import train_solver_dpo

logger = logging.getLogger(__name__)


def _state(result) -> str | None:
    if isinstance(result, dict):
        return result.get("state_path") or result.get("path")
    return str(result) if result else None


def _sampler(result) -> str | None:
    if isinstance(result, dict):
        return result.get("sampler_path") or result.get("state_path")
    return str(result) if result else None


def _round_config(base: Config, round_idx: int) -> Config:
    """Return a copy of ``base`` whose paths are scoped to ``round_<k>``."""
    cfg = deepcopy(base)
    cfg.tag = f"{base.tag}/round_{round_idx}"
    return cfg


def _load_resume(run_dir: Path) -> dict | None:
    p = run_dir / "resume_state.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        logger.warning("resume_state.json unreadable; starting fresh")
        return None


def _save_resume(run_dir: Path, state: dict) -> None:
    (run_dir / "resume_state.json").write_text(json.dumps(state, indent=2))


def run_multi_round(config: Config, num_rounds: int) -> dict[int, dict[str, float]]:
    run_dir = Path(config.storage_path) / config.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Multi-round run dir: %s  (rounds=%d)", run_dir, num_rounds)

    prev_challenger_state: str | None = None
    prev_solver_state: str | None = None
    prev_solver_sampling: str | None = None
    start_round = 1

    rs = _load_resume(run_dir)
    if rs:
        start_round = int(rs.get("next_round", 1))
        prev_challenger_state = rs.get("challenger_state")
        prev_solver_state = rs.get("solver_state")
        prev_solver_sampling = rs.get("solver_sampling")
        logger.info("RESUME: starting from round %d", start_round)

    results: dict[int, dict[str, float]] = {}

    for k in range(start_round, num_rounds + 1):
        logger.info("========= Round %d / %d =========", k, num_rounds)
        rcfg = _round_config(config, k)
        rcfg.run_dir().mkdir(parents=True, exist_ok=True)

        # Phase 1
        if rcfg.run_phase1:
            ch_result = train_challenger(
                rcfg,
                challenger_init_path=prev_challenger_state,
                solver_sampling_path=prev_solver_sampling,
                save_path=str(rcfg.challenger_save_dir()),
            )
            challenger_sampler = _sampler(ch_result)
            challenger_state = _state(ch_result)
        else:
            logger.info("round %d: Phase 1 skipped (run_phase1=False)", k)
            challenger_sampler = None
            challenger_state = None

        # Phase 2: NEW challenger × prev solver
        build_dpo_dataset(
            rcfg,
            challenger_sampling_path=challenger_sampler,
            solver_sampling_path=prev_solver_sampling,
        )

        # Phase 3: prev solver → DPO
        sv_result = train_solver_dpo(
            rcfg,
            solver_init_path=prev_solver_state,
            data_path=rcfg.dpo_data_path(),
            save_path=str(rcfg.solver_save_dir()),
        )
        solver_sampler = _sampler(sv_result)
        solver_state = _state(sv_result)

        prev_challenger_state = challenger_state
        prev_solver_state = solver_state
        prev_solver_sampling = solver_sampler

        _save_resume(run_dir, {
            "last_completed_round": k,
            "next_round": k + 1,
            "challenger_state": challenger_state,
            "solver_state": solver_state,
            "solver_sampling": solver_sampler,
        })

        results[k] = {"solver": str(solver_sampler), "challenger": str(challenger_state)}

    summary_path = run_dir / "rounds_summary.json"
    summary_path.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=2))
    logger.info("Multi-round summary -> %s", summary_path)
    return results


def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Pre-parse only --num_rounds; everything else flows into Config.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--num_rounds", type=int, default=3)
    args, rest = pre.parse_known_args()
    config = parse_cli_overrides(rest)
    run_multi_round(config, num_rounds=args.num_rounds)


if __name__ == "__main__":
    main()
