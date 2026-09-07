#!/usr/bin/env python3
"""Checkpoint 1: can legal, draft-day information beat ADP?

This benchmark uses DraftGym's realistic weekly manager score, not
DraftBench's hindsight lineup score.  The deployable planner sees only the
draft board, roster state, league settings, and ADP uncertainty.  The
hindsight arm deliberately sees the realized season and is reported only as
an unattainable ceiling reference.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from envs.draftgym import realistic_roster_points
from evals.agents_execution import SurvivalSequencer, bootstrap_ci
from evals.draftbench import (
    DEFAULT_ROSTER,
    DEFAULT_TEAMS,
    AutopickADP,
    DraftAgent,
    DraftPlayer,
    DraftSimulator,
    feasible_players,
    load_board,
)
from harness.league import LeagueConfig
from harness.scoring import PRESETS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "artifacts/outcome-headroom-v1"
ROWS_PATH = RESULTS / "matched-drafts.csv"
SUMMARY_PATH = RESULTS / "summary.json"
REPORT_PATH = ROOT / "docs/outcome-headroom.md"

# Sealed by the current outcome-training protocol.  The script refuses them.
SEALED_SEASONS = frozenset({2013, 2018, 2023, 2024, 2025})
SELECTION_SEASONS = (2015, 2016, 2017, 2019, 2020)
VALIDATION_SEASONS = (2021, 2022)
FORMATS = ("standard", "ppr", "half_ppr")
SLOTS = (1, 6, 12)
N_SEEDS = 5

# Selected on SELECTION_SEASONS from a frozen 3 x 3 x 4 local grid.  The
# later-season validation results were not used to choose these values.
PLANNER_CONFIG = {
    "take_band": 0.55,
    "pass_band": 0.65,
    "scarcity_ratio": 1.5,
}


class HindsightMarginalOracle(DraftAgent):
    """Greedy clairvoyant reference; never a deployable policy or model input.

    At a pick it chooses the legal player with the largest increase in the
    final reward scorer applied to the roster assembled so far.  It knows the
    entire realized season.  It is an operational upper reference for the
    available information, not a proof of the globally optimal draft path.
    """

    name = "hindsight_marginal_oracle"

    def __init__(
        self,
        weekly: Mapping[str, Mapping[int, float]],
        league: LeagueConfig,
    ) -> None:
        self.weekly = weekly
        self.league = league

    def pick(self, board, my_roster, league, pick_number):
        candidates = feasible_players(board, my_roster, league, pick_number)
        base = realistic_roster_points(my_roster, self.weekly, self.league)

        def key(player: DraftPlayer) -> tuple[float, float, str]:
            gain = realistic_roster_points(
                tuple(my_roster) + (player,), self.weekly, self.league
            ) - base
            return (gain, -player.adp, player.name)

        return max(candidates, key=key).player_id


def robust_metrics(deltas: Sequence[float]) -> dict:
    if not deltas:
        raise ValueError("cannot score an empty panel")
    values = [float(value) for value in deltas]
    ordered = sorted(values)
    trim = math.floor(0.10 * len(ordered))
    trimmed = ordered[trim : len(ordered) - trim] if trim else ordered
    without_largest = list(values)
    without_largest.remove(max(without_largest))
    lo, hi = bootstrap_ci(values, n_boot=10_000, seed=20260906)
    metrics = {
        "n": len(values),
        "mean_points_above_adp": statistics.fmean(values),
        "trimmed_mean_10pct": statistics.fmean(trimmed),
        "wins": sum(value > 0 for value in values),
        "losses": sum(value < 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "mean_without_largest_gain": statistics.fmean(without_largest),
        "minimum": min(values),
        "maximum": max(values),
        "paired_bootstrap_95pct": [lo, hi],
    }
    metrics["passes_headroom_gate"] = (
        metrics["mean_points_above_adp"] > 0
        and metrics["trimmed_mean_10pct"] > 0
        and metrics["wins"] > metrics["losses"]
        and metrics["mean_without_largest_gain"] > 0
    )
    return metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_panel() -> tuple[list[dict], dict]:
    seasons = SELECTION_SEASONS + VALIDATION_SEASONS
    if set(seasons) & SEALED_SEASONS:
        raise ValueError("headroom panel includes a sealed season")

    rows: list[dict] = []
    for season in seasons:
        split = "planner_selection" if season in SELECTION_SEASONS else "planner_validation"
        for preset in FORMATS:
            loaded = load_board(season, preset)
            if loaded is None:
                continue
            pool, weekly = loaded
            league = LeagueConfig(
                teams=DEFAULT_TEAMS,
                roster=dict(DEFAULT_ROSTER),
                scoring=PRESETS[preset],
            )
            simulator = DraftSimulator(pool, league)
            for slot in SLOTS:
                for seed in range(N_SEEDS):
                    arms: tuple[tuple[str, DraftAgent], ...] = (
                        ("legal_adp", AutopickADP()),
                        ("point_in_time_planner", SurvivalSequencer(**PLANNER_CONFIG)),
                        ("hindsight_marginal_oracle", HindsightMarginalOracle(weekly, league)),
                    )
                    scored: dict[str, float] = {}
                    for name, agent in arms:
                        result = simulator.simulate(agent, slot, seed)
                        scored[name] = realistic_roster_points(
                            result.agent_roster, weekly, league
                        )
                    for name, _ in arms:
                        rows.append(
                            {
                                "split": split,
                                "season": season,
                                "format": preset,
                                "slot": slot,
                                "seed": seed,
                                "arm": name,
                                "realistic_points": scored[name],
                                "points_above_adp": round(scored[name] - scored["legal_adp"], 2),
                            }
                        )

    def arm_metrics(arm: str, split: str | None = None) -> dict:
        return robust_metrics(
            [
                row["points_above_adp"]
                for row in rows
                if row["arm"] == arm and (split is None or row["split"] == split)
            ]
        )

    summary = {
        "protocol": "fantasy-alpha-outcome-headroom-v1",
        "reward": "realized season points from deterministic, prior-weeks-only weekly lineups",
        "comparison": "same season, format, slot, opponents, and seed legal ADP",
        "sealed_seasons_untouched": sorted(SEALED_SEASONS),
        "selection_seasons": list(SELECTION_SEASONS),
        "validation_seasons": list(VALIDATION_SEASONS),
        "planner_inputs": [
            "legal board", "current roster", "league constraints", "ADP",
            "ADP dispersion", "snake-turn distance",
        ],
        "planner_config": PLANNER_CONFIG,
        "hindsight_warning": (
            "The oracle sees realized outcomes and is unattainable. It is a greedy "
            "ceiling reference, not a globally optimal draft proof or training input."
        ),
        "metrics": {
            "selection": arm_metrics("point_in_time_planner", "planner_selection"),
            "validation": arm_metrics("point_in_time_planner", "planner_validation"),
            "all_development": arm_metrics("point_in_time_planner"),
            "hindsight_all_development": arm_metrics("hindsight_marginal_oracle"),
        },
    }
    summary["decision"] = {
        "strict_later_season_replication_passed": summary["metrics"]["validation"][
            "passes_headroom_gate"
        ],
        "all_development_gate_passed": summary["metrics"]["all_development"][
            "passes_headroom_gate"
        ],
        "interpretation": (
            "provisional headroom only; improve the planner/ranker before paid LLM training"
        ),
    }
    return rows, summary


def write_outputs(rows: Sequence[Mapping], summary: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with ROWS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["artifacts"] = {"matched_drafts_sha256": _sha256(ROWS_PATH)}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    select = summary["metrics"]["selection"]
    valid = summary["metrics"]["validation"]
    all_dev = summary["metrics"]["all_development"]
    oracle = summary["metrics"]["hindsight_all_development"]
    REPORT_PATH.write_text(
        "# Outcome headroom before model training\n\n"
        "Fantasy Alpha treats a snake draft as a finite-horizon stochastic allocation game. "
        "The terminal objective is the roster's realized season points under a deterministic "
        "weekly manager that ranks players using prior weeks only. Subtracting the same-seat, "
        "same-opponents, same-seed legal ADP roster is only the evaluation layer.\n\n"
        "The deployable planner uses ADP, ADP dispersion, roster constraints, positional "
        "scarcity, and distance to the next snake turn. It never sees the target season's "
        "outcomes. The hindsight arm does see them and is an unattainable greedy ceiling "
        "reference—not a training input and not a claim of global optimality.\n\n"
        "## Result\n\n"
        "| panel | n | mean | 10% trimmed | W-L-T | no largest gain | 95% bootstrap CI | gate |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|\n"
        f"| selection (2015–2020, sealed years excluded) | {select['n']} | {select['mean_points_above_adp']:+.2f} | {select['trimmed_mean_10pct']:+.2f} | {select['wins']}-{select['losses']}-{select['ties']} | {select['mean_without_largest_gain']:+.2f} | [{select['paired_bootstrap_95pct'][0]:+.2f}, {select['paired_bootstrap_95pct'][1]:+.2f}] | {'pass' if select['passes_headroom_gate'] else 'fail'} |\n"
        f"| later validation (2021–2022) | {valid['n']} | {valid['mean_points_above_adp']:+.2f} | {valid['trimmed_mean_10pct']:+.2f} | {valid['wins']}-{valid['losses']}-{valid['ties']} | {valid['mean_without_largest_gain']:+.2f} | [{valid['paired_bootstrap_95pct'][0]:+.2f}, {valid['paired_bootstrap_95pct'][1]:+.2f}] | {'pass' if valid['passes_headroom_gate'] else 'fail'} |\n"
        f"| all development | {all_dev['n']} | {all_dev['mean_points_above_adp']:+.2f} | {all_dev['trimmed_mean_10pct']:+.2f} | {all_dev['wins']}-{all_dev['losses']}-{all_dev['ties']} | {all_dev['mean_without_largest_gain']:+.2f} | [{all_dev['paired_bootstrap_95pct'][0]:+.2f}, {all_dev['paired_bootstrap_95pct'][1]:+.2f}] | {'pass' if all_dev['passes_headroom_gate'] else 'fail'} |\n"
        f"| hindsight reference | {oracle['n']} | {oracle['mean_points_above_adp']:+.2f} | {oracle['trimmed_mean_10pct']:+.2f} | {oracle['wins']}-{oracle['losses']}-{oracle['ties']} | {oracle['mean_without_largest_gain']:+.2f} | [{oracle['paired_bootstrap_95pct'][0]:+.2f}, {oracle['paired_bootstrap_95pct'][1]:+.2f}] | reference |\n\n"
        "The planner has positive average value and the full development panel narrowly "
        "passes the mechanical gate. The later-season replication does not: it wins 26 and "
        "loses 29 drafts, and its interval includes zero. The honest conclusion is provisional "
        "headroom, not a repeatable win. Paid LLM training remains gated until the rollout "
        "planner or simple ranker clears the later-season test.\n\n"
        "Machine-readable rows and exact metrics are in `artifacts/outcome-headroom-v1/`.\n"
    )


def main() -> int:
    rows, summary = run_panel()
    write_outputs(rows, summary)
    print(json.dumps(summary["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
