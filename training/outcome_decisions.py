#!/usr/bin/env python3
"""Generate masked, outcome-aligned draft decisions from complete rollouts.

Inputs contain only state available at draft time.  Realized season results
appear only in labels produced after each counterfactual draft is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from envs.draftgym import realistic_roster_points
from evals.agents_execution import SurvivalSequencer, survival_probability
from evals.draftbench import (
    DEFAULT_ROSTER,
    DEFAULT_TEAMS,
    AutopickADP,
    DraftAgent,
    DraftPlayer,
    DraftSimulator,
    _board_key,
    feasible_players,
    load_board,
)
from harness.league import LeagueConfig
from harness.scoring import PRESETS

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/processed/outcome_decisions_v1"
ROWS_PATH = OUT_DIR / "decisions.jsonl"
MANIFEST_PATH = OUT_DIR / "manifest.json"
SEALED_SEASONS = frozenset({2013, 2018, 2023, 2024, 2025})
DEVELOPMENT_SEASONS = (2015, 2016, 2017, 2019, 2020)
VALIDATION_SEASONS = (2021, 2022)
PLANNER_CONFIG = {"take_band": 0.55, "pass_band": 0.65, "scarcity_ratio": 1.5}
FEATURE_ALLOWLIST = (
    "round",
    "overall_pick",
    "draft_slot",
    "picks_until_next_turn",
    "candidate_board_rank",
    "candidate_position",
    "candidate_adp",
    "candidate_adp_stdev",
    "candidate_prev_season_points",
    "candidate_survival_to_next_turn",
    "roster_qb",
    "roster_rb",
    "roster_wr",
    "roster_te",
)


class RecordingAgent(DraftAgent):
    def __init__(self, delegate: DraftAgent):
        self.delegate = delegate
        self.name = f"record_{delegate.name}"
        self.states: list[dict[str, Any]] = []

    def pick(self, board, my_roster, league, pick_number):
        candidates = feasible_players(board, my_roster, league, pick_number)
        choice = self.delegate.pick(board, my_roster, league, pick_number)
        self.states.append(
            {
                "board": tuple(board),
                "roster": tuple(my_roster),
                "pick_number": pick_number,
                "candidates": tuple(candidates),
                "choice": choice,
            }
        )
        return choice


class PrefixCounterfactualAgent(DraftAgent):
    """Replay a fixed prefix, make one branch choice, then legally use ADP."""

    name = "prefix_counterfactual"

    def __init__(self, prefix: Sequence[str], branch: str):
        self.prefix = tuple(prefix)
        self.branch = branch
        self.calls = 0

    def pick(self, board, my_roster, league, pick_number):
        candidates = feasible_players(board, my_roster, league, pick_number)
        legal = {player.player_id for player in candidates}
        if self.calls < len(self.prefix):
            choice = self.prefix[self.calls]
        elif self.calls == len(self.prefix):
            choice = self.branch
        else:
            choice = min(candidates, key=_board_key).player_id
        self.calls += 1
        if choice not in legal:
            raise AssertionError("counterfactual prefix no longer reaches the matched state")
        return choice


def _next_pick(league: LeagueConfig, slot: int, pick_number: int) -> int | None:
    return next((pick for pick in league.snake_picks(slot) if pick > pick_number), None)


def _masked_ids(pool: Sequence[DraftPlayer]) -> dict[str, str]:
    return {
        player.player_id: f"B{index + 1:03d}"
        for index, player in enumerate(sorted(pool, key=_board_key))
    }


def decision_features(
    state: dict[str, Any],
    candidate: DraftPlayer,
    *,
    league: LeagueConfig,
    slot: int,
) -> dict[str, Any]:
    roster = Counter(player.position for player in state["roster"])
    nxt = _next_pick(league, slot, int(state["pick_number"]))
    ordered = sorted(state["candidates"], key=_board_key)
    rank = next(i for i, player in enumerate(ordered, 1) if player.player_id == candidate.player_id)
    return {
        "round": len(state["roster"]) + 1,
        "overall_pick": int(state["pick_number"]),
        "draft_slot": slot,
        "picks_until_next_turn": 0 if nxt is None else nxt - int(state["pick_number"]),
        "candidate_board_rank": rank,
        "candidate_position": candidate.position,
        "candidate_adp": candidate.adp,
        "candidate_adp_stdev": candidate.stdev if candidate.stdev is not None else 0.15 * candidate.adp,
        "candidate_prev_season_points": candidate.prev_points,
        "candidate_survival_to_next_turn": 0.0 if nxt is None else survival_probability(candidate, nxt),
        "roster_qb": roster["QB"],
        "roster_rb": roster["RB"],
        "roster_wr": roster["WR"],
        "roster_te": roster["TE"],
    }


def generate_episode(
    *,
    season: int,
    preset: str,
    slot: int,
    seed: int,
    trajectory: str,
    candidates_per_state: int = 5,
) -> list[dict[str, Any]]:
    if season in SEALED_SEASONS:
        raise ValueError("refusing to generate labels from a sealed season")
    loaded = load_board(season, preset)
    if loaded is None:
        return []
    pool, weekly = loaded
    league = LeagueConfig(
        teams=DEFAULT_TEAMS,
        roster=dict(DEFAULT_ROSTER),
        scoring=PRESETS[preset],
    )
    if trajectory == "adp":
        delegate: DraftAgent = AutopickADP()
    elif trajectory == "planner":
        delegate = SurvivalSequencer(**PLANNER_CONFIG)
    else:
        raise ValueError(f"unknown trajectory: {trajectory}")
    recorder = RecordingAgent(delegate)
    DraftSimulator(pool, league).simulate(recorder, slot, seed)
    masked = _masked_ids(pool)
    rows: list[dict[str, Any]] = []
    prefix: list[str] = []
    episode_id = f"{season}:{preset}:slot{slot}:seed{seed}:{trajectory}"
    for state_index, state in enumerate(recorder.states):
        candidates = sorted(state["candidates"], key=_board_key)[:candidates_per_state]
        # Forced endgame states contain no ranking decision and therefore no
        # preference signal. Preserve the reference pick in the replay prefix
        # but do not manufacture duplicate alternatives.
        if len(candidates) < 2:
            prefix.append(str(state["choice"]))
            continue
        outcomes = []
        for candidate in candidates:
            branch = DraftSimulator(pool, league).simulate(
                PrefixCounterfactualAgent(prefix, candidate.player_id), slot, seed
            )
            points = realistic_roster_points(branch.agent_roster, weekly, league)
            outcomes.append((candidate, points))
        best_points = max(points for _, points in outcomes)
        state_id = hashlib.sha256(f"{episode_id}:{state_index}".encode()).hexdigest()[:20]
        for candidate, points in outcomes:
            features = decision_features(state, candidate, league=league, slot=slot)
            if set(features) != set(FEATURE_ALLOWLIST):
                raise AssertionError("feature schema drift")
            rows.append(
                {
                    "state_id": state_id,
                    "episode_id": episode_id,
                    "split": "development" if season in DEVELOPMENT_SEASONS else "validation",
                    "season": season,
                    "format": preset,
                    "trajectory": trajectory,
                    "candidate_id": masked[candidate.player_id],
                    "features": features,
                    "label": {
                        "completed_roster_points": points,
                        "regret_to_best_candidate": round(best_points - points, 2),
                        "is_best_candidate": points == best_points,
                    },
                }
            )
        prefix.append(str(state["choice"]))
    return rows


def validate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty outcome-decision dataset")
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row["season"]) in SEALED_SEASONS:
            raise ValueError("sealed season in outcome-decision dataset")
        if set(row["features"]) != set(FEATURE_ALLOWLIST):
            raise ValueError("unexpected model feature")
        serialized_features = json.dumps(row["features"], sort_keys=True).lower()
        for forbidden in ("completed_roster_points", "regret", "is_best", "name", "team"):
            if forbidden in serialized_features:
                raise ValueError(f"label or identity leaked into features: {forbidden}")
        by_state.setdefault(row["state_id"], []).append(row)
    if any(len(group) < 2 for group in by_state.values()):
        raise ValueError("every state must retain multiple plausible actions")
    episodes_by_split: dict[str, set[str]] = {}
    for row in rows:
        episodes_by_split.setdefault(row["split"], set()).add(row["episode_id"])
    if episodes_by_split.get("development", set()) & episodes_by_split.get("validation", set()):
        raise ValueError("episode crosses development and validation")
    return {
        "rows": len(rows),
        "states": len(by_state),
        "episodes": len({row["episode_id"] for row in rows}),
        "split_rows": dict(Counter(row["split"] for row in rows)),
        "split_states": {
            split: len({row["state_id"] for row in rows if row["split"] == split})
            for split in sorted({row["split"] for row in rows})
        },
        "mean_candidates_per_state": len(rows) / len(by_state),
        "zero_regret_rate": sum(row["label"]["is_best_candidate"] for row in rows) / len(rows),
    }


def generate(
    *,
    include_validation: bool = False,
    seeds: int = 3,
    candidates_per_state: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seasons = DEVELOPMENT_SEASONS + (VALIDATION_SEASONS if include_validation else ())
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for slot in (1, 6, 12):
            for seed in range(seeds):
                for trajectory in ("adp", "planner"):
                    rows.extend(
                        generate_episode(
                            season=season,
                            preset="ppr",
                            slot=slot,
                            seed=seed,
                            trajectory=trajectory,
                            candidates_per_state=candidates_per_state,
                        )
                    )
    audit = validate_rows(rows)
    manifest = {
        "protocol": "fantasy-alpha-outcome-decisions-v1",
        "masked_primary_track": True,
        "label": "realistic completed-roster points after a matched counterfactual branch",
        "features": list(FEATURE_ALLOWLIST),
        "sealed_seasons_untouched": sorted(SEALED_SEASONS),
        "development_seasons": list(DEVELOPMENT_SEASONS),
        "validation_seasons": list(VALIDATION_SEASONS) if include_validation else [],
        "formats": ["ppr"],
        "slots": [1, 6, 12],
        "seeds_per_cell": seeds,
        "reference_trajectories": ["adp", "planner"],
        "candidates_per_state": candidates_per_state,
        "audit": audit,
    }
    return rows, manifest


def write(rows: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ROWS_PATH.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = dict(manifest)
    manifest["dataset_sha256"] = hashlib.sha256(ROWS_PATH.read_bytes()).hexdigest()
    manifest["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=5)
    args = parser.parse_args()
    rows, manifest = generate(
        include_validation=args.include_validation,
        seeds=args.seeds,
        candidates_per_state=args.candidates,
    )
    write(rows, manifest)
    print(json.dumps(manifest["audit"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
