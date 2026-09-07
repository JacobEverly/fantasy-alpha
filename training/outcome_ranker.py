#!/usr/bin/env python3
"""Train the smallest outcome-aligned candidate ranker before any LLM run."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from envs.draftgym import realistic_roster_points
from evals.draftbench import (
    DEFAULT_ROSTER,
    DEFAULT_TEAMS,
    AutopickADP,
    DraftAgent,
    DraftSimulator,
    _board_key,
    feasible_players,
    load_board,
    team_on_the_clock,
)
from evals.outcome_headroom import robust_metrics
from harness.league import LeagueConfig
from harness.scoring import PRESETS
from training.outcome_decisions import (
    DEVELOPMENT_SEASONS,
    FEATURE_ALLOWLIST,
    ROWS_PATH,
    SEALED_SEASONS,
    decision_features,
    validate_rows,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts/outcome-ranker-v1"
MODEL_PATH = ARTIFACT_DIR / "ranker.joblib"
REPORT_PATH = ARTIFACT_DIR / "development-report.json"
POSITIONS = ("QB", "RB", "WR", "TE")
MODEL_CONFIG = {
    "loss": "squared_error on negative candidate regret",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": 20260906,
}


def encode(features: Mapping[str, Any]) -> list[float]:
    if set(features) != set(FEATURE_ALLOWLIST):
        raise ValueError("ranker received an unexpected feature schema")
    position = str(features["candidate_position"])
    return [float(position == value) for value in POSITIONS] + [
        float(features[name])
        for name in FEATURE_ALLOWLIST
        if name != "candidate_position"
    ]


def load_rows(path: Path = ROWS_PATH) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate_rows(rows)
    if any(row["split"] != "development" for row in rows):
        raise ValueError("trainer accepts development rows only")
    return rows


def fit(rows: Sequence[dict[str, Any]]):
    from sklearn.ensemble import HistGradientBoostingRegressor

    x = np.asarray([encode(row["features"]) for row in rows], dtype=float)
    y = np.asarray(
        [-float(row["label"]["regret_to_best_candidate"]) for row in rows],
        dtype=float,
    )
    state_sizes: dict[str, int] = defaultdict(int)
    for row in rows:
        state_sizes[row["state_id"]] += 1
    weights = np.asarray([1.0 / state_sizes[row["state_id"]] for row in rows], dtype=float)
    model = HistGradientBoostingRegressor(
        learning_rate=MODEL_CONFIG["learning_rate"],
        max_iter=MODEL_CONFIG["max_iter"],
        max_leaf_nodes=MODEL_CONFIG["max_leaf_nodes"],
        min_samples_leaf=MODEL_CONFIG["min_samples_leaf"],
        l2_regularization=MODEL_CONFIG["l2_regularization"],
        random_state=MODEL_CONFIG["random_state"],
    )
    model.fit(x, y, sample_weight=weights)
    return model


def turn_metrics(model: Any, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["state_id"]].append(row)
    chosen_regret = []
    adp_regret = []
    for group in groups.values():
        predictions = model.predict(np.asarray([encode(row["features"]) for row in group]))
        chosen = group[int(np.argmax(predictions))]
        adp = min(group, key=lambda row: row["features"]["candidate_board_rank"])
        chosen_regret.append(float(chosen["label"]["regret_to_best_candidate"]))
        adp_regret.append(float(adp["label"]["regret_to_best_candidate"]))
    return {
        "states": len(groups),
        "mean_candidate_regret": float(np.mean(chosen_regret)),
        "adp_mean_candidate_regret": float(np.mean(adp_regret)),
        "zero_regret_rate": float(np.mean(np.asarray(chosen_regret) == 0)),
        "adp_zero_regret_rate": float(np.mean(np.asarray(adp_regret) == 0)),
    }


class OutcomeRankerAgent(DraftAgent):
    name = "outcome_ranker_v1"

    def __init__(self, model: Any, candidates_per_state: int = 5):
        self.model = model
        self.candidates_per_state = candidates_per_state
        self.slot: int | None = None

    def pick(self, board, my_roster, league, pick_number):
        if self.slot is None:
            self.slot = team_on_the_clock(league, pick_number)
        legal = feasible_players(board, my_roster, league, pick_number)
        candidates = sorted(legal, key=_board_key)[: self.candidates_per_state]
        if len(candidates) == 1:
            return candidates[0].player_id
        state = {
            "board": tuple(board),
            "roster": tuple(my_roster),
            "pick_number": pick_number,
            "candidates": tuple(legal),
        }
        x = np.asarray(
            [
                encode(decision_features(state, player, league=league, slot=self.slot))
                for player in candidates
            ]
        )
        scores = self.model.predict(x)
        best = max(
            range(len(candidates)),
            key=lambda index: (float(scores[index]), -candidates[index].adp),
        )
        return candidates[best].player_id


def full_draft_metrics(model: Any, seasons: Sequence[int], seeds: int = 5) -> dict:
    if set(seasons) & SEALED_SEASONS:
        raise ValueError("refusing sealed-season ranker evaluation")
    deltas: list[float] = []
    for season in seasons:
        loaded = load_board(season, "ppr")
        if loaded is None:
            continue
        pool, weekly = loaded
        league = LeagueConfig(
            teams=DEFAULT_TEAMS,
            roster=dict(DEFAULT_ROSTER),
            scoring=PRESETS["ppr"],
        )
        simulator = DraftSimulator(pool, league)
        for slot in (1, 6, 12):
            for seed in range(seeds):
                control = simulator.simulate(AutopickADP(), slot, seed)
                ranked = simulator.simulate(OutcomeRankerAgent(model), slot, seed)
                deltas.append(
                    realistic_roster_points(ranked.agent_roster, weekly, league)
                    - realistic_roster_points(control.agent_roster, weekly, league)
                )
    return robust_metrics(deltas)


def train() -> dict[str, Any]:
    rows = load_rows()
    seasons = sorted({int(row["season"]) for row in rows})
    if tuple(seasons) != DEVELOPMENT_SEASONS:
        raise ValueError("development dataset season drift")
    folds = []
    for season in seasons:
        train_rows = [row for row in rows if int(row["season"]) != season]
        test_rows = [row for row in rows if int(row["season"]) == season]
        model = fit(train_rows)
        turn = turn_metrics(model, test_rows)
        draft = full_draft_metrics(model, [season], seeds=5)
        folds.append({"season": season, "turn": turn, "full_draft": draft})

    # One honest OOF policy is represented by one fold-specific model per
    # season; pool its exact episode deltas with a direct replay.
    oof_deltas: list[float] = []
    for season in seasons:
        model = fit([row for row in rows if int(row["season"]) != season])
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
                ranked = simulator.simulate(OutcomeRankerAgent(model), slot, seed)
                oof_deltas.append(
                    realistic_roster_points(ranked.agent_roster, weekly, league)
                    - realistic_roster_points(control.agent_roster, weekly, league)
                )

    final_model = fit(rows)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": final_model,
        "feature_allowlist": FEATURE_ALLOWLIST,
        "positions": POSITIONS,
        "config": MODEL_CONFIG,
        "development_seasons": DEVELOPMENT_SEASONS,
        "dataset_sha256": hashlib.sha256(ROWS_PATH.read_bytes()).hexdigest(),
    }
    joblib.dump(payload, MODEL_PATH)
    report = {
        "protocol": "fantasy-alpha-outcome-ranker-v1",
        "selection": "single predeclared HistGradientBoostingRegressor recipe; no held-out tuning",
        "model_config": MODEL_CONFIG,
        "folds": folds,
        "oof_turn": {
            key: sum(fold["turn"][key] * fold["turn"]["states"] for fold in folds)
            / sum(fold["turn"]["states"] for fold in folds)
            for key in (
                "mean_candidate_regret",
                "adp_mean_candidate_regret",
                "zero_regret_rate",
                "adp_zero_regret_rate",
            )
        }
        | {"states": sum(fold["turn"]["states"] for fold in folds)},
        "oof_full_draft": robust_metrics(oof_deltas),
        "artifact_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "dataset_sha256": payload["dataset_sha256"],
        "paid_spend_usd": 0.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def load_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    payload = joblib.load(path)
    if tuple(payload["feature_allowlist"]) != FEATURE_ALLOWLIST:
        raise ValueError("ranker artifact feature schema drift")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = train()
    print(json.dumps(report["oof_full_draft"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
