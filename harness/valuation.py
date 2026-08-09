"""Replacement level and signed VORP for arbitrary league shapes.

Lifted from the research repo's valuation.py with one behavioral change:
thin player pools degrade (replacement falls back to the pool minimum, with a
status flag) instead of raising, so a product request never 500s on a sparse
projection table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from harness.league import LeagueConfig


@dataclass(frozen=True)
class ProjectedPlayer:
    player_id: str
    name: str
    position: str
    points: float


@dataclass(frozen=True)
class StarterAssignment:
    starters: dict[str, list[ProjectedPlayer]]
    flex_starters: list[ProjectedPlayer]
    replacement_points: dict[str, float]
    coverage_status: dict[str, str] = field(default_factory=dict)  # position -> ok | pool_exhausted


def assign_starters(players: list[ProjectedPlayer], league: LeagueConfig) -> StarterAssignment:
    required = league.required_positions()
    by_position: dict[str, list[ProjectedPlayer]] = {}
    for p in sorted(players, key=lambda p: p.points, reverse=True):
        by_position.setdefault(p.position, []).append(p)

    starters: dict[str, list[ProjectedPlayer]] = {}
    coverage: dict[str, str] = {}
    taken: set[str] = set()
    for pos, per_team in required.items():
        pool = by_position.get(pos, [])
        needed = league.teams * per_team
        chosen = pool[:needed]
        starters[pos] = chosen
        coverage[pos] = "ok" if len(chosen) == needed else "pool_exhausted"
        taken.update(p.player_id for p in chosen)

    flex_pool = sorted(
        (
            p
            for pos in league.flex_eligible
            for p in by_position.get(pos, [])
            if p.player_id not in taken
        ),
        key=lambda p: p.points,
        reverse=True,
    )
    flex_needed = league.teams * league.flex_per_team()
    flex_starters = flex_pool[:flex_needed]
    if len(flex_starters) < flex_needed:
        coverage["FLEX"] = "pool_exhausted"
    taken.update(p.player_id for p in flex_starters)

    replacement: dict[str, float] = {}
    next_flex = flex_pool[flex_needed].points if len(flex_pool) > flex_needed else None
    positions = set(required) | set(league.flex_eligible) | set(by_position)
    for pos in positions:
        remaining = [p for p in by_position.get(pos, []) if p.player_id not in taken]
        if remaining:
            replacement[pos] = remaining[0].points
        elif pos in league.flex_eligible and next_flex is not None:
            replacement[pos] = next_flex
        else:
            pool = by_position.get(pos, [])
            replacement[pos] = pool[-1].points if pool else 0.0
            coverage.setdefault(pos, "pool_exhausted")

    return StarterAssignment(
        starters=starters,
        flex_starters=flex_starters,
        replacement_points=replacement,
        coverage_status=coverage,
    )


def value_over_replacement(player: ProjectedPlayer, assignment: StarterAssignment) -> float:
    """Signed by design — negative VORP is information, never clamp to zero."""
    return round(player.points - assignment.replacement_points.get(player.position, 0.0), 2)
