"""League scoring over the nflverse stat vocabulary.

Stat keys are nflverse `stats_player_week` column names, plus a few derived
aggregates (see DERIVED). A ScoringConfig maps stat -> points-per-unit, with
optional per-position overrides (TE premium, per-position pass TD values, ...).
This one engine scores live projections, 2011-2025 historical weeks (labels,
DraftGym rewards), and third-party component lines re-scored under league rules.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping

# Derived stats resolved from raw columns when the key itself is absent.
DERIVED: dict[str, tuple[str, ...]] = {
    "fumbles_lost": ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"),
    "two_point_conversions": (
        "passing_2pt_conversions",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
    ),
}


def _num(stats: Mapping[str, object], key: str) -> float:
    v = stats.get(key)
    if v is None or v == "" or v == "NA":
        return 0.0
    return float(v)


def stat_value(stats: Mapping[str, object], key: str) -> float:
    if key in stats:
        return _num(stats, key)
    parts = DERIVED.get(key)
    if parts:
        return sum(_num(stats, p) for p in parts)
    return 0.0


@dataclass(frozen=True)
class ScoringConfig:
    """base: stat -> points per unit; position_overrides merge over base."""

    base: Mapping[str, float]
    position_overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def for_position(self, position: str | None) -> Mapping[str, float]:
        if not position or position not in self.position_overrides:
            return self.base
        merged = dict(self.base)
        merged.update(self.position_overrides[position])
        return merged

    def contributions(self, stats: Mapping[str, object], position: str | None = None) -> dict[str, float]:
        rules = self.for_position(position)
        out: dict[str, float] = {}
        for stat, pts in rules.items():
            value = stat_value(stats, stat)
            if value:
                out[stat] = round(value * pts, 4)
        return out

    def score(self, stats: Mapping[str, object], position: str | None = None) -> float:
        return round(sum(self.contributions(stats, position).values()), 2)

    def config_hash(self) -> str:
        payload = {
            "base": dict(sorted(self.base.items())),
            "position_overrides": {
                p: dict(sorted(v.items())) for p, v in sorted(self.position_overrides.items())
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _preset(receptions: float) -> ScoringConfig:
    return ScoringConfig(
        base={
            "passing_yards": 0.04,
            "passing_tds": 4.0,
            "passing_interceptions": -2.0,
            "rushing_yards": 0.1,
            "rushing_tds": 6.0,
            "receptions": receptions,
            "receiving_yards": 0.1,
            "receiving_tds": 6.0,
            "fumbles_lost": -2.0,
            "two_point_conversions": 2.0,
        }
    )


PRESETS: dict[str, ScoringConfig] = {
    "standard": _preset(0.0),
    "half_ppr": _preset(0.5),
    "ppr": _preset(1.0),
}

# Research-repo scoring vocabulary (config/league.yaml) -> nflverse stat keys.
_RESEARCH_KEYS = {
    "passing_yard": "passing_yards",
    "passing_td": "passing_tds",
    "interception": "passing_interceptions",
    "rushing_yard": "rushing_yards",
    "rushing_td": "rushing_tds",
    "reception": "receptions",
    "receiving_yard": "receiving_yards",
    "receiving_td": "receiving_tds",
    "fumble_lost": "fumbles_lost",
}


def from_research_scoring(scoring: Mapping[str, float]) -> ScoringConfig:
    return ScoringConfig(base={_RESEARCH_KEYS[k]: float(v) for k, v in scoring.items()})
