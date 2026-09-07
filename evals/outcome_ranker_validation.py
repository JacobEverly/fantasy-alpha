#!/usr/bin/env python3
"""Fit-free later-season evaluation of the frozen outcome ranker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from envs.draftgym import realistic_roster_points
from evals.agents_execution import SurvivalSequencer
from evals.draftbench import (
    DEFAULT_ROSTER,
    DEFAULT_TEAMS,
    AutopickADP,
    DraftSimulator,
    load_board,
)
from evals.outcome_headroom import robust_metrics
from harness.league import LeagueConfig
from harness.scoring import PRESETS
from training.outcome_decisions import (
    PLANNER_CONFIG,
    VALIDATION_SEASONS,
    generate_episode,
    validate_rows,
)
from training.outcome_ranker import (
    ARTIFACT_DIR,
    MODEL_PATH,
    OutcomeRankerAgent,
    load_artifact,
    turn_metrics,
)

ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = ARTIFACT_DIR / "later-season-validation.json"
ROWS_PATH = ARTIFACT_DIR / "later-season-decisions.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate() -> dict:
    development_report = json.loads((ARTIFACT_DIR / "development-report.json").read_text())
    expected_hash = development_report["artifact_sha256"]
    if sha256(MODEL_PATH) != expected_hash:
        raise ValueError("frozen ranker artifact changed before later-season evaluation")
    artifact = load_artifact()
    model = artifact["model"]

    decision_rows = []
    for season in VALIDATION_SEASONS:
        for slot in (1, 6, 12):
            for seed in range(3):
                for trajectory in ("adp", "planner"):
                    decision_rows.extend(
                        generate_episode(
                            season=season,
                            preset="ppr",
                            slot=slot,
                            seed=seed,
                            trajectory=trajectory,
                            candidates_per_state=5,
                        )
                    )
    validate_rows(decision_rows)
    ROWS_PATH.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decision_rows)
    )

    ranker_deltas = []
    planner_deltas = []
    for season in VALIDATION_SEASONS:
        loaded = load_board(season, "ppr")
        assert loaded is not None
        pool, weekly = loaded
        league = LeagueConfig(
            teams=DEFAULT_TEAMS,
            roster=dict(DEFAULT_ROSTER),
            scoring=PRESETS["ppr"],
        )
        simulator = DraftSimulator(pool, league)
        for slot in (1, 6, 12):
            for seed in range(5):
                control = simulator.simulate(AutopickADP(), slot, seed)
                ranker = simulator.simulate(OutcomeRankerAgent(model), slot, seed)
                planner = simulator.simulate(
                    SurvivalSequencer(**PLANNER_CONFIG), slot, seed
                )
                control_points = realistic_roster_points(control.agent_roster, weekly, league)
                ranker_deltas.append(
                    realistic_roster_points(ranker.agent_roster, weekly, league)
                    - control_points
                )
                planner_deltas.append(
                    realistic_roster_points(planner.agent_roster, weekly, league)
                    - control_points
                )

    result = {
        "protocol": "fantasy-alpha-outcome-ranker-later-season-validation-v1",
        "fit_or_tuning_during_evaluation": False,
        "model_artifact_sha256_before": expected_hash,
        "model_artifact_sha256_after": sha256(MODEL_PATH),
        "seasons": list(VALIDATION_SEASONS),
        "sealed_seasons_untouched": [2013, 2018, 2023, 2024, 2025],
        "turn_level": turn_metrics(model, decision_rows),
        "full_draft": {
            "outcome_ranker_vs_adp": robust_metrics(ranker_deltas),
            "handwritten_planner_vs_adp": robust_metrics(planner_deltas),
        },
        "catastrophic_drafts_below_minus_100": {
            "outcome_ranker": sum(value <= -100 for value in ranker_deltas),
            "handwritten_planner": sum(value <= -100 for value in planner_deltas),
            "adp": 0,
        },
        "decision_rows_sha256": sha256(ROWS_PATH),
        "paid_spend_usd": 0.0,
    }
    if result["model_artifact_sha256_before"] != result["model_artifact_sha256_after"]:
        raise AssertionError("evaluation mutated the frozen model")
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    result = evaluate()
    print(json.dumps(result["full_draft"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
