#!/usr/bin/env python3
"""Board-execution agents for DraftBench — execution dynamics WITHOUT projection skill.

Motivation (research repo: docs/draft-board.md, docs/edge-framework.md §10):
Jacob's board separates *valuation* (projected points minus replacement) from
*execution* (ADP-driven sequencing at each snake turn): "fall watch" / "take
at early pick" / "wait for late pick" survival-conditional ordering, mid-turn
QB/TE roster guardrails, and positional-scarcity discipline ("preserve paths
to fill three WR, one TE, and two FLEX without forcing a position ahead of a
stronger tier"). This module isolates that execution layer:

- SurvivalSequencer values players by MARKET CONSENSUS ONLY (ADP order — no
  projections, no hindsight). Any edge over the same-seed AutopickADP control
  is therefore pure execution dynamics.
- SurvivalSequencerVORP runs the identical execution logic but values players
  by GreedyVORP's naive projection (previous-season points minus replacement)
  — does execution rescue bad projections? (Plain GreedyVORP scores ~-141 vs
  the market on the default grid.)

Execution logic per pick (thresholds are constructor parameters):
1. My next snake pick number (edge-framework §10: at pick 19, survival to 22
   matters little; at pick 22, survival to 39 matters enormously).
2. Survival to that pick via the normal CDF: a player drafted around
   Normal(adp, max(stdev, 1)) survives past pick q with probability
   1 - Phi((q - adp) / max(stdev, 1)). Take the best candidate UNLIKELY to
   survive (survival < take_band, default 0.40); pass on candidates who
   LIKELY survive (> pass_band, default 0.70) unless no scarce alternative.
3. Positional scarcity: remaining league-wide demand (teams x required still
   unmet + proportional FLEX pressure) vs remaining supply above a depth
   cutoff (players inside starter-round ADP). Prefer positions whose quality
   tier is about to evaporate.
4. Roster guardrails: never a 2nd QB or 2nd TE while required RB/WR/FLEX
   starters are unfilled; no QB/TE at all before round 5 unless that
   position's scarcity triggers (draft-board.md "QB/TE roster guardrail").
5. Feasibility always wins: guardrail filters degrade to the simulator's
   feasible set rather than returning an illegal pick.

Run `python -m evals.agents_execution` for the standard grid ->
data/processed/draftbench/results_execution.csv (results_v0.csv untouched)
and evals/results/execution_dynamics.md. 2025 stays holdout (GAMEPLAN §9).
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from evals.draftbench import (
    DEFAULT_FORMATS,
    DEFAULT_SEASONS,
    DEFAULT_SLOTS,
    OUT_DIR,
    DraftAgent,
    DraftPlayer,
    _board_key,
    feasible_players,
    position_counts,
    run,
    team_on_the_clock,
)
from harness.league import LeagueConfig
from harness.valuation import ProjectedPlayer, assign_starters

EXECUTION_RESULTS_PATH = OUT_DIR / "results_execution.csv"
REPORT_PATH = Path(__file__).resolve().parent / "results" / "execution_dynamics.md"

_SQRT2 = math.sqrt(2.0)


def survival_probability(player: DraftPlayer, pick: int) -> float:
    """P(player still on the board at overall pick `pick`).

    Draft position modeled as Normal(adp, max(stdev, 1)); stdev falls back to
    the same 0.15*adp proxy the simulator's opponents use when FFC has none.
    """
    sigma = player.stdev if player.stdev and player.stdev > 0 else 0.15 * player.adp
    sigma = max(sigma, 1.0)
    z = (pick - player.adp) / sigma
    return 1.0 - 0.5 * (1.0 + math.erf(z / _SQRT2))


class SurvivalSequencer(DraftAgent):
    """Market-consensus values + survival-conditional execution (see module docstring).

    take_band / pass_band: survival probabilities bounding "gone before my
    next pick, take now" and "safely waits, pass". scarcity_ratio: a position
    is scarce when quality supply <= scarcity_ratio x remaining league demand.
    depth_cutoff: ADP bound for "quality" supply; defaults to
    teams x total-starter-slots (the starter rounds of the draft).
    """

    NO_EARLY = frozenset({"QB", "TE"})
    EARLIEST_ROUND = 5

    def __init__(
        self,
        take_band: float = 0.40,
        pass_band: float = 0.70,
        scarcity_ratio: float = 1.0,
        depth_cutoff: float | None = None,
        name: str = "survival_seq",
    ) -> None:
        if not 0.0 < take_band <= pass_band < 1.0:
            raise ValueError("need 0 < take_band <= pass_band < 1")
        self.take_band = take_band
        self.pass_band = pass_band
        self.scarcity_ratio = scarcity_ratio
        self.depth_cutoff = depth_cutoff
        self.name = name
        self._slot: int | None = None
        self._my_picks: tuple[int, ...] = ()
        self._initial_counts: dict[str, int] = {}

    # -- value model (overridden by the VORP variant) -----------------------

    def _setup(self, board: Sequence[DraftPlayer], league: LeagueConfig) -> None:
        """Hook: snapshot anything value-related from the first board seen."""

    def _value_key(self, player: DraftPlayer):
        """Sort key, ascending = better. Market consensus only: ADP order."""
        return _board_key(player)

    # -- scarcity model ------------------------------------------------------

    def _quality_cutoff(self, league: LeagueConfig) -> float:
        if self.depth_cutoff is not None:
            return self.depth_cutoff
        starters = sum(league.required_positions().values()) + league.flex_per_team()
        return float(league.teams * starters)

    def scarce_positions(
        self, board: Sequence[DraftPlayer], league: LeagueConfig
    ) -> set[str]:
        """Positions whose quality tier is about to evaporate.

        Demand: unmet league-wide required slots (teams x required minus
        players drafted at the position since the first board we saw) plus a
        proportional share of unmet FLEX pressure. Supply: remaining board
        players above the depth cutoff. Scarce when
        supply <= scarcity_ratio x demand.
        """
        required = league.required_positions()
        flex_set = set(league.flex_eligible)
        cutoff = self._quality_cutoff(league)

        seen = position_counts(board)
        drafted = {
            pos: max(0, self._initial_counts.get(pos, 0) - seen.get(pos, 0))
            for pos in self._initial_counts
        }
        quality: dict[str, int] = defaultdict(int)
        for p in board:
            if p.adp <= cutoff:
                quality[p.position] += 1

        # Unmet required demand per position; drafted players fill required
        # slots first (optimistic — real leagues bench some, so this
        # underestimates demand slightly and scarcity fires conservatively).
        demand = {
            pos: max(0.0, league.teams * n - drafted.get(pos, 0))
            for pos, n in required.items()
        }
        # FLEX pressure: unfilled flex slots, allocated to flex-eligible
        # positions in proportion to their remaining quality supply.
        flex_total = league.teams * league.flex_per_team()
        flex_filled = sum(
            max(0, drafted.get(pos, 0) - league.teams * required.get(pos, 0))
            for pos in flex_set
        )
        flex_left = max(0, flex_total - flex_filled)
        flex_quality = sum(quality.get(pos, 0) for pos in flex_set)

        scarce: set[str] = set()
        for pos, base in demand.items():
            need = base
            if pos in flex_set and flex_quality > 0:
                need += flex_left * (quality.get(pos, 0) / flex_quality)
            if need > 0 and quality.get(pos, 0) <= self.scarcity_ratio * need:
                scarce.add(pos)
        return scarce

    # -- guardrails ------------------------------------------------------

    def _guardrail_ok(
        self,
        position: str,
        counts: Mapping[str, int],
        league: LeagueConfig,
        round_number: int,
        scarce: set[str],
    ) -> bool:
        if position not in self.NO_EARLY:
            return True
        required = league.required_positions()
        # Never a 2nd QB / 2nd TE while required RB/WR/FLEX starters unfilled.
        if counts.get(position, 0) >= 1:
            core_missing = sum(
                max(0, required.get(pos, 0) - counts.get(pos, 0))
                for pos in league.flex_eligible
                if pos not in self.NO_EARLY
            )
            flex_extra = sum(
                max(0, counts.get(pos, 0) - required.get(pos, 0))
                for pos in league.flex_eligible
            )
            if core_missing > 0 or flex_extra < league.flex_per_team():
                return False
        # No QB/TE before round 5 at all — unless that tier is evaporating.
        if round_number < self.EARLIEST_ROUND and position not in scarce:
            return False
        return True

    # -- the pick ----------------------------------------------------------

    def pick(self, board, my_roster, league, pick_number):
        if self._slot is None:
            self._slot = team_on_the_clock(league, pick_number)
            self._my_picks = league.snake_picks(self._slot)
            self._initial_counts = position_counts(board)
            self._setup(board, league)

        counts = position_counts(my_roster)
        candidates = feasible_players(board, my_roster, league, pick_number)
        scarce = self.scarce_positions(board, league)
        round_number = len(my_roster) + 1
        guarded = [
            p
            for p in candidates
            if self._guardrail_ok(p.position, counts, league, round_number, scarce)
        ]
        pool = guarded or candidates  # feasibility always wins
        next_pick = next((q for q in self._my_picks if q > pick_number), None)

        def band(p: DraftPlayer) -> int:
            if next_pick is None:  # last pick: pure value
                return 0
            s = survival_probability(p, next_pick)
            if s < self.take_band:
                return 0  # unlikely to survive: take now
            if s <= self.pass_band:
                return 1  # coin flip
            return 2  # safely waits: pass unless scarce (or nothing better)

        # Urgent picks first; within a band, scarce positions before safe
        # ones; within that, best value. A band-2 scarce player therefore
        # beats nothing in band 0/1 ("pass ... unless no scarce alternative"),
        # while an evaporating tier wins any tie.
        best = min(
            pool,
            key=lambda p: (
                band(p),
                0 if p.position in scarce else 1,
                self._value_key(p),
            ),
        )
        return best.player_id


class SurvivalSequencerVORP(SurvivalSequencer):
    """Same execution logic, but values GreedyVORP's naive projections.

    Projection = previous season's total points at the board's preset (0 for
    rookies), replacement from harness.valuation over the first board seen —
    exactly GreedyVORP's value model. Tests whether execution discipline
    rescues bad projections.
    """

    def __init__(self, name: str = "survival_seq_vorp", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self._replacement: dict[str, float] = {}

    def _setup(self, board, league):
        projected = [
            ProjectedPlayer(p.player_id, p.name, p.position, p.prev_points) for p in board
        ]
        self._replacement = dict(assign_starters(projected, league).replacement_points)

    def _value_key(self, player: DraftPlayer):
        vorp = player.prev_points - self._replacement.get(player.position, 0.0)
        return (-vorp, _board_key(player))


# ---------------------------------------------------------------------------
# Benchmark configs: defaults +- one alternative setting each (3 configs),
# plus the VORP-valued variant at defaults.

EXECUTION_AGENTS: dict[str, Callable[[], DraftAgent]] = {
    "survival_seq": lambda: SurvivalSequencer(name="survival_seq"),
    "survival_seq_bands55_85": lambda: SurvivalSequencer(
        take_band=0.55, pass_band=0.85, name="survival_seq_bands55_85"
    ),
    "survival_seq_scarcity1.5": lambda: SurvivalSequencer(
        scarcity_ratio=1.5, name="survival_seq_scarcity1.5"
    ),
    "survival_seq_vorp": lambda: SurvivalSequencerVORP(name="survival_seq_vorp"),
}


# ---------------------------------------------------------------------------
# Report: mean +- std per agent/format + paired bootstrap 95% CI on per-run
# deltas (points_above_control is already a same-seed paired difference).


def bootstrap_ci(
    deltas: Sequence[float], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def _cell(deltas: Sequence[float]) -> str:
    if not deltas:
        return "—"
    mean = statistics.mean(deltas)
    std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    lo, hi = bootstrap_ci(deltas)
    return f"{mean:+.1f} ± {std:.1f} [{lo:+.1f}, {hi:+.1f}] (n={len(deltas)})"


def _verdict(deltas: Sequence[float]) -> str:
    lo, hi = bootstrap_ci(deltas)
    if lo > 0:
        return "yes"
    if hi < 0:
        return "no"
    return "unresolved"


def write_report(rows: Sequence[Mapping], path: Path = REPORT_PATH) -> str:
    agents = list(dict.fromkeys(r["agent"] for r in rows))
    formats = list(dict.fromkeys(r["format"] for r in rows))

    def deltas(agent: str, fmt: str | None = None) -> list[float]:
        return [
            float(r["points_above_control"])
            for r in rows
            if r["agent"] == agent and (fmt is None or r["format"] == fmt)
        ]

    lines = [
        "# Execution dynamics vs the market (DraftBench)",
        "",
        "Does draft *execution* alone — survival-conditional sequencing, turn",
        "awareness, scarcity discipline, QB/TE guardrails (research board:",
        "draft-board.md, edge-framework.md §10) — beat ADP autopick, with no",
        "projection skill at all?",
        "",
        "- `survival_seq*`: market-consensus values (ADP order) + execution logic.",
        "- `survival_seq_vorp`: same execution, GreedyVORP's naive prev-season",
        "  projections as values (plain GreedyVORP: ~-141 vs market).",
        "- Grid: 2015–2024 × standard/ppr/half_ppr × slots 1/6/12 × 5 seeds,",
        "  paired same-seed AutopickADP control. half_ppr ADP exists 2018+.",
        "- Cells: mean ± std of points-above-control, paired bootstrap 95% CI",
        "  (10k resamples) in brackets.",
        "",
        "| agent | " + " | ".join(formats) + " | overall | beats market |",
        "|---|" + "---|" * (len(formats) + 2),
    ]
    for agent in agents:
        overall = deltas(agent)
        cells = [_cell(deltas(agent, fmt)) for fmt in formats]
        lines.append(
            f"| {agent} | " + " | ".join(cells) + f" | {_cell(overall)} | **{_verdict(overall)}** |"
        )
    lines += [
        "",
        "Verdicts use the overall paired bootstrap 95% CI: yes = CI entirely",
        "above 0, no = entirely below 0, unresolved otherwise.",
        "",
        "Rows: data/processed/draftbench/results_execution.csv",
        "",
    ]
    report = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DraftBench execution-dynamics run")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=str(EXECUTION_RESULTS_PATH))
    ap.add_argument("--report", default=str(REPORT_PATH))
    args = ap.parse_args(argv)

    rows = run(
        seasons=DEFAULT_SEASONS,
        formats=DEFAULT_FORMATS,
        slots=DEFAULT_SLOTS,
        n_seeds=args.seeds,
        agents=EXECUTION_AGENTS,
        out_path=Path(args.out),
        verbose=True,
    )
    print()
    print(write_report(rows, Path(args.report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
