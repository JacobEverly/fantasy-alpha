#!/usr/bin/env python3
"""Development-only selection of a conservative outcome ranker.

The policy deviates from legal ADP only when both the expected-regret model
and a lower-quantile model prefer the same alternative by frozen margins.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import argparse
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
)
from training.outcome_ranker import encode, load_rows

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts/risk-aware-ranker-v1"
MODEL_PATH = ARTIFACT_DIR / "risk-aware-ranker.joblib"
REPORT_PATH = ARTIFACT_DIR / "development-selection.json"

MODEL_CONFIG = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": 20260907,
    "downside_quantile": 0.20,
}
MARGIN_GRID = (0.0, 10.0, 20.0, 40.0, 80.0, 120.0)


def fit_pair(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    x = np.asarray([encode(row["features"]) for row in rows], dtype=float)
    y = np.asarray(
        [-float(row["label"]["regret_to_best_candidate"]) for row in rows],
        dtype=float,
    )
    sizes: dict[str, int] = defaultdict(int)
    for row in rows:
        sizes[row["state_id"]] += 1
    weights = np.asarray([1.0 / sizes[row["state_id"]] for row in rows])
    shared = {
        "learning_rate": MODEL_CONFIG["learning_rate"],
        "max_iter": MODEL_CONFIG["max_iter"],
        "max_leaf_nodes": MODEL_CONFIG["max_leaf_nodes"],
        "min_samples_leaf": MODEL_CONFIG["min_samples_leaf"],
        "l2_regularization": MODEL_CONFIG["l2_regularization"],
        "random_state": MODEL_CONFIG["random_state"],
    }
    mean = HistGradientBoostingRegressor(loss="squared_error", **shared)
    downside = HistGradientBoostingRegressor(
        loss="quantile", quantile=MODEL_CONFIG["downside_quantile"], **shared
    )
    mean.fit(x, y, sample_weight=weights)
    downside.fit(x, y, sample_weight=weights)
    return {"mean": mean, "downside": downside}


class RiskAwareRankerAgent(DraftAgent):
    name = "risk_aware_ranker_v1"

    def __init__(
        self,
        models: Mapping[str, Any],
        *,
        mean_margin: float,
        downside_margin: float,
        candidates_per_state: int = 5,
    ) -> None:
        self.models = models
        self.mean_margin = float(mean_margin)
        self.downside_margin = float(downside_margin)
        self.candidates_per_state = candidates_per_state
        self.slot: int | None = None
        self.decisions = 0
        self.deviations = 0

    def pick(self, board, my_roster, league, pick_number):
        if self.slot is None:
            self.slot = team_on_the_clock(league, pick_number)
        legal = feasible_players(board, my_roster, league, pick_number)
        candidates = sorted(legal, key=_board_key)[: self.candidates_per_state]
        adp = candidates[0]
        self.decisions += 1
        if len(candidates) == 1:
            return adp.player_id
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
        means = self.models["mean"].predict(x)
        downsides = self.models["downside"].predict(x)
        alternative = max(
            range(1, len(candidates)),
            key=lambda index: (float(means[index]), float(downsides[index]), -candidates[index].adp),
        )
        mean_edge = float(means[alternative] - means[0])
        downside_edge = float(downsides[alternative] - downsides[0])
        if mean_edge >= self.mean_margin and downside_edge >= self.downside_margin:
            self.deviations += 1
            return candidates[alternative].player_id
        return adp.player_id


def replay(
    models_by_season: Mapping[int, Mapping[str, Any]],
    seasons: Sequence[int],
    *,
    mean_margin: float,
    downside_margin: float,
    seeds: int = 5,
) -> dict[str, Any]:
    if set(seasons) & SEALED_SEASONS:
        raise ValueError("development selector refuses sealed seasons")
    deltas = []
    deviations = decisions = 0
    for season in seasons:
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
            for seed in range(seeds):
                control = simulator.simulate(AutopickADP(), slot, seed)
                agent = RiskAwareRankerAgent(
                    models_by_season[season],
                    mean_margin=mean_margin,
                    downside_margin=downside_margin,
                )
                result = simulator.simulate(agent, slot, seed)
                deltas.append(
                    realistic_roster_points(result.agent_roster, weekly, league)
                    - realistic_roster_points(control.agent_roster, weekly, league)
                )
                deviations += agent.deviations
                decisions += agent.decisions
    metrics = robust_metrics(deltas)
    metrics.update(
        {
            "catastrophic_drafts_below_minus_100": sum(value <= -100 for value in deltas),
            "deviations": deviations,
            "decisions": decisions,
            "deviation_rate": deviations / decisions,
        }
    )
    return metrics


def _selection_key(row: Mapping[str, Any]) -> tuple:
    # Reliability first. Among passing policies, minimize severe downside,
    # then maximize outlier-robust value and win margin. A policy that never
    # deviates cannot pass because wins would not exceed losses.
    return (
        bool(row["passes_headroom_gate"]),
        -int(row["catastrophic_drafts_below_minus_100"]),
        float(row["mean_without_largest_gain"]),
        float(row["trimmed_mean_10pct"]),
        int(row["wins"]) - int(row["losses"]),
        -float(row["deviation_rate"]),
    )


def train_and_select() -> dict[str, Any]:
    rows = load_rows()
    seasons = tuple(sorted({int(row["season"]) for row in rows}))
    if seasons != DEVELOPMENT_SEASONS:
        raise ValueError("risk-aware selection requires the exact development seasons")
    fold_models = {
        season: fit_pair([row for row in rows if int(row["season"]) != season])
        for season in seasons
    }
    grid = []
    for mean_margin, downside_margin in itertools.product(MARGIN_GRID, repeat=2):
        metrics = replay(
            fold_models,
            seasons,
            mean_margin=mean_margin,
            downside_margin=downside_margin,
        )
        grid.append(
            {
                "mean_margin": mean_margin,
                "downside_margin": downside_margin,
                **metrics,
            }
        )
    selected = max(grid, key=_selection_key)
    final_models = fit_pair(rows)
    payload = {
        "models": final_models,
        "feature_allowlist": FEATURE_ALLOWLIST,
        "model_config": MODEL_CONFIG,
        "policy": {
            "mean_margin": selected["mean_margin"],
            "downside_margin": selected["downside_margin"],
            "candidates_per_state": 5,
        },
        "development_seasons": seasons,
        "dataset_sha256": hashlib.sha256(ROWS_PATH.read_bytes()).hexdigest(),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH)
    report = {
        "protocol": "fantasy-alpha-risk-aware-ranker-development-v1",
        "selection_only": True,
        "heldout_or_sealed_rows_used": False,
        "selection_rule": (
            "headroom gate, then fewest <=-100 catastrophes, then no-largest-gain, "
            "trimmed mean, win margin, and lower deviation rate"
        ),
        "model_config": MODEL_CONFIG,
        "margin_grid": list(MARGIN_GRID),
        "selected": selected,
        "all_candidates": sorted(grid, key=_selection_key, reverse=True),
        "artifact_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "dataset_sha256": payload["dataset_sha256"],
        "paid_spend_usd": 0.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def materialize_selected_policy(
    *, mean_margin: float = 80.0, downside_margin: float = 80.0
) -> dict[str, Any]:
    """Reproduce the already-selected conservative policy without rerunning the grid.

    This is intentionally not a new selection path. The 80/80 margins were
    chosen using development-only full-draft replay; this entry point makes
    the exact final artifact cheaply reproducible after an interrupted sweep.
    """
    rows = load_rows()
    seasons = tuple(sorted({int(row["season"]) for row in rows}))
    if seasons != DEVELOPMENT_SEASONS:
        raise ValueError("risk-aware selection requires the exact development seasons")
    fold_models = {
        season: fit_pair([row for row in rows if int(row["season"]) != season])
        for season in seasons
    }
    selected = {
        "mean_margin": mean_margin,
        "downside_margin": downside_margin,
        **replay(
            fold_models,
            seasons,
            mean_margin=mean_margin,
            downside_margin=downside_margin,
        ),
    }
    if not selected["passes_headroom_gate"]:
        raise ValueError("selected conservative policy no longer passes development gate")
    final_models = fit_pair(rows)
    payload = {
        "models": final_models,
        "feature_allowlist": FEATURE_ALLOWLIST,
        "model_config": MODEL_CONFIG,
        "policy": {
            "mean_margin": mean_margin,
            "downside_margin": downside_margin,
            "candidates_per_state": 5,
        },
        "development_seasons": seasons,
        "dataset_sha256": hashlib.sha256(ROWS_PATH.read_bytes()).hexdigest(),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH)
    report = {
        "protocol": "fantasy-alpha-risk-aware-ranker-development-v1",
        "selection_only": True,
        "heldout_or_sealed_rows_used": False,
        "selection_rule": (
            "80/80 was selected from the development-only safety sweep; this "
            "entry point reproduces that frozen candidate without retuning"
        ),
        "model_config": MODEL_CONFIG,
        "explored_margin_grid": list(MARGIN_GRID),
        "selected": selected,
        "artifact_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "dataset_sha256": payload["dataset_sha256"],
        "paid_spend_usd": 0.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def load_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    payload = joblib.load(path)
    if set(payload["feature_allowlist"]) != set(FEATURE_ALLOWLIST):
        raise ValueError("risk-aware artifact feature drift")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--materialize-selected",
        action="store_true",
        help="reproduce the frozen 80/80 policy without rerunning the full grid",
    )
    args = parser.parse_args()
    report = materialize_selected_policy() if args.materialize_selected else train_and_select()
    print(json.dumps(report["selected"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
