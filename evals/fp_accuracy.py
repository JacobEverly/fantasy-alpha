#!/usr/bin/env python3
"""FP-Accuracy v0 — replication of the FantasyPros draft-rankings accuracy method.

The external anchor against named human experts (GAMEPLAN eval suite;
docs/benchmark-landscape.md "Industry accuracy evaluations"). Public
methodology replicated from
https://www.fantasypros.com/about/faq/football-draft-accuracy-methodology/
(verified 2026-08-08):

- Input: one preseason positional ranking list per position, frozen before
  week 1.
- Player pool per position = (consensus top N) UNION (actual finish top N);
  N = 25 QB, 50 RB, 60 WR, 20 TE. K/DST/IDP are tracked by FantasyPros but
  excluded from the overall score; v0 skips them entirely.
- Every rank converts to points via a "slot curve": the rolling 3-year
  average of the fantasy points scored by the player who FINISHED at that
  positional rank. Both sides of the comparison go through the curve:
  Accuracy Gap = |slot_points(expert_rank) - slot_points(actual_rank)|.
  Lower is better.
- Pool players the expert left unranked: if the player is in the pool via
  consensus, assigned rank = expert's last ranked player + 1; if in the pool
  only via actual finish, assigned rank = whichever is WORSE of
  (consensus rank + 1) and (expert's last ranked player + 1).
- Players the expert ranked inside the top-N range that ended up NOT in the
  pool draw a penalty equal to the amount by which the expert's gap exceeds
  the "average expert's" gap for that player (only when positive).
- Weighting (their 2021+ contest): each player's gap is multiplied by a
  factor of 1.0 down to 0.5 driven by the player's CONSENSUS rank —
  1.0 at/above maxrank, 0.5 at/below minrank, linear in between
  (QB 18/30, RB 72/96, WR 84/112, TE 18/24).
- Overall score = sum of the weighted position gaps for QB+RB+WR+TE.

Replication deviations (full list in evals/results/fp_accuracy_replication.md):
- Consensus (their preseason ECR) is proxied by the FFC ADP board
  (data/raw/ffc_adp/) — historical ECR snapshots are not public.
- Scoring preset defaults to PPR (project decision); FantasyPros grades on
  half-PPR. Scores are therefore NOT numerically comparable to their
  published leaderboards (which publish ordinal ranks only anyway).
- The "average expert" in the out-of-pool penalty is proxied by the
  consensus list itself.
- The slot curve for season S uses seasons S-3..S-1 ONLY (never S): their
  wording ("rolling 3-year average", "the past few years") does not pin the
  window, so we choose the leakage-free reading.

IMPORTANT: 2025 is the untouched eval holdout (GAMEPLAN §4/§9).
DEFAULT_SEASONS stops at 2024 on purpose; score_rankings refuses holdout
seasons unless include_holdout=True is passed explicitly at gate time.

Pluggable interface for future contenders (e.g. our model's preseason ranks):

    score_rankings(rankings, season, position) -> PositionScore
    score_overall({pos: rankings, ...}, season)  -> OverallScore

where `rankings` is an ordered list of player names (normalized internally
via evals.names.norm_name, so nflverse/FFC spelling variants both work).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from evals.names import norm_name

ROOT = Path(__file__).resolve().parent.parent
ADP_DIR = ROOT / "data" / "raw" / "ffc_adp"
LABELS = ROOT / "data" / "processed" / "labels"
RESULTS_DIR = ROOT / "evals" / "results"
REPORT_PATH = RESULTS_DIR / "fp_accuracy_replication.md"

VERSION = "v0"

POSITIONS = ("QB", "RB", "WR", "TE")
POOL_N = {"QB": 25, "RB": 50, "WR": 60, "TE": 20}
# (maxrank, minrank) for the 1.0 -> 0.5 consensus-rank multiplier (their step 4).
WEIGHT_RANKS = {"QB": (18, 30), "RB": (72, 96), "WR": (84, 112), "TE": (18, 24)}
MAX_MULT, MIN_MULT = 1.0, 0.5
CURVE_SEASONS = 3  # slot curve = trailing 3-season average

PRESET_TO_FFC = {"standard": "standard", "half_ppr": "half-ppr", "ppr": "ppr"}

# 2025 is the untouched holdout (GAMEPLAN §4/§9): excluded from defaults,
# only ever scored at gate time via explicit include_holdout.
HOLDOUT_SEASON = 2025
DEFAULT_SEASONS = tuple(range(2015, 2025))


# ---------------------------------------------------------------------------
# Pure scoring core (fully injectable — tests build tiny hand cases)


def multiplier(consensus_rank: int, position: str) -> float:
    """Positional relevance weight from the player's consensus rank:
    1.0 at/above maxrank, 0.5 at/below minrank, linear in between."""
    maxrank, minrank = WEIGHT_RANKS[position]
    if consensus_rank <= maxrank:
        return MAX_MULT
    if consensus_rank >= minrank:
        return MIN_MULT
    frac = (consensus_rank - maxrank) / (minrank - maxrank)
    return MAX_MULT - (MAX_MULT - MIN_MULT) * frac


def slot_points(curve: Mapping[int, float], rank: int) -> float:
    """Points implied by a positional rank slot. Ranks beyond the deepest
    observed slot clamp to the deepest slot's value (a near-zero tail —
    ASSUMPTION A5 in the report)."""
    if not curve:
        raise ValueError("empty slot curve")
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    return curve[min(rank, max(curve))]


def build_slot_curve(
    rows: Iterable[Mapping[str, str]], season: int, position: str, scoring: str = "ppr"
) -> dict[int, float]:
    """rank -> mean total points at that positional finish rank over seasons
    S-3..S-1 ONLY (strictly before `season`: the leakage-free reading of
    their 'rolling 3-year average'). Each source season has dense ranks
    1..n, so the union is dense; ranks present in fewer seasons average over
    the seasons that reached that depth."""
    rank_col, pts_col = f"pos_season_rank_{scoring}", f"total_{scoring}"
    by_rank: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        s = int(r["season"])
        if r["position"] != position or not (season - CURVE_SEASONS <= s <= season - 1):
            continue
        by_rank[int(r[rank_col])].append(float(r[pts_col]))
    if not by_rank:
        raise ValueError(f"no seasons {season - CURVE_SEASONS}..{season - 1} for {position} slot curve")
    return {rank: statistics.mean(v) for rank, v in by_rank.items()}


@dataclass(frozen=True)
class PlayerEval:
    """One pool player's contribution to a contender's position score."""

    key: str  # normalized name
    consensus_rank: int | None  # None = not on the consensus board at all
    actual_rank: int | None  # None = scored no fantasy points that season
    expert_rank: int  # the contender's (possibly assigned) rank
    ranked_by_expert: bool
    projected_pts: float  # slot_points(expert_rank)
    actual_pts: float  # slot_points(effective actual rank)
    gap: float  # |projected - actual|
    weight: float
    weighted_gap: float


@dataclass(frozen=True)
class PenaltyEval:
    """Out-of-pool player the expert ranked inside the top-N range."""

    key: str
    expert_rank: int
    gap: float  # expert's gap for the player
    consensus_gap: float  # the 'average expert' proxy's gap
    weight: float
    penalty: float  # max(0, gap - consensus_gap) * weight


@dataclass(frozen=True)
class PositionScore:
    position: str
    season: int | None  # None for synthetic/core-only scoring
    pool_size: int
    accuracy_gap: float  # sum of weighted gaps + penalties (LOWER IS BETTER)
    penalty_total: float
    players: tuple[PlayerEval, ...]
    penalties: tuple[PenaltyEval, ...]


@dataclass(frozen=True)
class OverallScore:
    season: int | None
    accuracy_gap: float  # QB+RB+WR+TE weighted gap total (LOWER IS BETTER)
    by_position: dict[str, PositionScore]


def _first_occurrence_index(names: Sequence[str]) -> dict[str, int]:
    """normalized name -> 0-based index of first occurrence."""
    out: dict[str, int] = {}
    for i, name in enumerate(names):
        out.setdefault(norm_name(name), i)
    return out


def score_position_core(
    rankings: Sequence[str],
    consensus_order: Sequence[str],
    actual_ranks: Mapping[str, int],
    curve: Mapping[int, float],
    position: str,
    pool_n: int | None = None,
    season: int | None = None,
) -> PositionScore:
    """The FantasyPros accuracy formula on explicit inputs.

    rankings / consensus_order: ordered player-name lists (any spelling —
    normalized internally). actual_ranks: normalized name -> realized
    positional finish rank (dense, 1 = best); players absent scored no
    points. curve: finish rank -> trailing-average points.
    """
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}, got {position!r}")
    n = POOL_N[position] if pool_n is None else pool_n
    expert_idx = _first_occurrence_index(rankings)
    consensus_idx = _first_occurrence_index(consensus_order)
    n_ranked = len(expert_idx)
    consensus_size = len(consensus_idx)
    # Effective ranks off the board: one past the end of the respective list.
    off_consensus = consensus_size + 1
    worst_actual = max(actual_ranks.values(), default=0) + 1

    pool = {k for k, i in consensus_idx.items() if i < n}
    pool |= {k for k, rank in actual_ranks.items() if rank <= n}

    def consensus_rank_eff(key: str) -> int:
        i = consensus_idx.get(key)
        return off_consensus if i is None else i + 1

    def actual_rank_eff(key: str) -> int:
        return actual_ranks.get(key, worst_actual)

    players: list[PlayerEval] = []
    for key in sorted(pool):
        in_consensus_pool = consensus_idx.get(key, n) < n
        if key in expert_idx:
            expert_rank, ranked = expert_idx[key] + 1, True
        elif in_consensus_pool:
            expert_rank, ranked = n_ranked + 1, False  # last ranked + 1
        else:  # in pool via actual finish only: whichever rank is WORSE
            expert_rank, ranked = max(consensus_rank_eff(key) + 1, n_ranked + 1), False
        projected = slot_points(curve, expert_rank)
        actual = slot_points(curve, actual_rank_eff(key))
        gap = abs(projected - actual)
        weight = multiplier(consensus_rank_eff(key), position)
        players.append(
            PlayerEval(
                key=key,
                consensus_rank=consensus_idx[key] + 1 if key in consensus_idx else None,
                actual_rank=actual_ranks.get(key),
                expert_rank=expert_rank,
                ranked_by_expert=ranked,
                projected_pts=projected,
                actual_pts=actual,
                gap=gap,
                weight=weight,
                weighted_gap=gap * weight,
            )
        )

    # Out-of-pool penalty: expert ranked the player inside the top-N range
    # but the player is in neither pool half. Penalized only for being worse
    # than the 'average expert' — proxied here by the consensus list itself
    # (ASSUMPTION A4).
    penalties: list[PenaltyEval] = []
    for key, i in expert_idx.items():
        if i >= n or key in pool:
            continue
        actual = slot_points(curve, actual_rank_eff(key))
        gap = abs(slot_points(curve, i + 1) - actual)
        consensus_gap = abs(slot_points(curve, consensus_rank_eff(key)) - actual)
        weight = multiplier(consensus_rank_eff(key), position)
        penalty = max(0.0, gap - consensus_gap) * weight
        penalties.append(PenaltyEval(key, i + 1, gap, consensus_gap, weight, penalty))

    penalty_total = sum(p.penalty for p in penalties)
    total = sum(p.weighted_gap for p in players) + penalty_total
    return PositionScore(
        position=position,
        season=season,
        pool_size=len(pool),
        accuracy_gap=round(total, 2),
        penalty_total=round(penalty_total, 2),
        players=tuple(players),
        penalties=tuple(sorted(penalties, key=lambda p: p.expert_rank)),
    )


# ---------------------------------------------------------------------------
# Data loading (season labels + FFC ADP consensus proxy)


@lru_cache(maxsize=None)
def _season_rows() -> tuple[dict, ...]:
    with open(LABELS / "season_points.csv", newline="") as f:
        return tuple(csv.DictReader(f))


@lru_cache(maxsize=None)
def slot_curve(season: int, position: str, scoring: str = "ppr") -> Mapping[int, float]:
    return build_slot_curve(_season_rows(), season, position, scoring)


@lru_cache(maxsize=None)
def actual_position_ranks(season: int, position: str, scoring: str = "ppr") -> Mapping[str, int]:
    """normalized name -> realized positional finish rank. On the rare
    normalized-name collision within a position-season the better (lower)
    rank wins (ASSUMPTION A6)."""
    rank_col = f"pos_season_rank_{scoring}"
    out: dict[str, int] = {}
    for r in _season_rows():
        if int(r["season"]) != season or r["position"] != position:
            continue
        key = norm_name(r["player"])
        rank = int(r[rank_col])
        if key not in out or rank < out[key]:
            out[key] = rank
    return out


def _adp_path(season: int, scoring: str) -> Path | None:
    fmt = PRESET_TO_FFC[scoring]
    for teams in (12, 10, 14, 8):  # historical years only have *_8t_* (12-team data)
        p = ADP_DIR / f"{fmt}_{teams}t_{season}.json"
        if p.exists():
            return p
    return None


@lru_cache(maxsize=None)
def consensus_order(season: int, position: str, scoring: str = "ppr") -> tuple[str, ...]:
    """The ECR proxy: FFC ADP board for that season/format, one position,
    ADP-ascending (ties: more-drafted first, then name)."""
    path = _adp_path(season, scoring)
    if path is None:
        raise FileNotFoundError(f"no FFC ADP snapshot for {scoring} {season} in {ADP_DIR}")
    payload = json.loads(path.read_text())
    players = [p for p in payload["players"] if p.get("position") == position]
    players.sort(key=lambda p: (float(p["adp"]), -int(p.get("times_drafted") or 0), p["name"]))
    return tuple(p["name"] for p in players)


def _check_holdout(season: int, include_holdout: bool) -> None:
    if season >= HOLDOUT_SEASON and not include_holdout:
        raise ValueError(
            f"{season} is the untouched eval holdout (GAMEPLAN §9); "
            "pass include_holdout=True only at gate time"
        )


# ---------------------------------------------------------------------------
# Pluggable public interface


def score_rankings(
    rankings: Sequence[str],
    season: int,
    position: str,
    scoring: str = "ppr",
    include_holdout: bool = False,
) -> PositionScore:
    """FantasyPros-style accuracy for one preseason positional ranking list.

    rankings: ordered player names, best first (any source's spelling).
    Returns the weighted Accuracy Gap total — LOWER IS BETTER.
    """
    _check_holdout(season, include_holdout)
    return score_position_core(
        rankings,
        consensus_order(season, position, scoring),
        actual_position_ranks(season, position, scoring),
        slot_curve(season, position, scoring),
        position,
        season=season,
    )


def score_overall(
    rankings_by_position: Mapping[str, Sequence[str]],
    season: int,
    scoring: str = "ppr",
    include_holdout: bool = False,
) -> OverallScore:
    """Overall = sum of the weighted QB/RB/WR/TE position gaps."""
    by_pos = {
        pos: score_rankings(rankings_by_position[pos], season, pos, scoring, include_holdout)
        for pos in POSITIONS
    }
    return OverallScore(
        season=season,
        accuracy_gap=round(sum(s.accuracy_gap for s in by_pos.values()), 2),
        by_position=by_pos,
    )


# ---------------------------------------------------------------------------
# Reference contenders (scale calibration)


def adp_rankings(season: int, position: str, scoring: str = "ppr") -> list[str]:
    """(a) ADP-implied rankings — identical to the consensus proxy, so its
    gap isolates pure market error (and it never draws penalties)."""
    return list(consensus_order(season, position, scoring))


def momentum_rankings(season: int, position: str, scoring: str = "ppr") -> list[str]:
    """(b) Naive momentum: last season's finish order."""
    prev = actual_position_ranks(season - 1, position, scoring)
    return [key for key, _ in sorted(prev.items(), key=lambda kv: kv[1])]


def avg3_rankings(season: int, position: str, scoring: str = "ppr") -> list[str]:
    """(c) 3-yr-weighted-average baseline: players ordered by
    (3*pts[S-1] + 2*pts[S-2] + 1*pts[S-3]) / 6, missing seasons = 0
    (recency-weighted; seasons >= S never consulted)."""
    pts_col = f"total_{scoring}"
    weights = {season - 1: 3.0, season - 2: 2.0, season - 3: 1.0}
    scores: dict[str, float] = defaultdict(float)
    for r in _season_rows():
        s = int(r["season"])
        if r["position"] != position or s not in weights:
            continue
        scores[norm_name(r["player"])] += weights[s] * float(r[pts_col])
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [key for key, total in ordered if total > 0]


CONTENDERS = {
    "adp_ffc": adp_rankings,
    "momentum_last_season": momentum_rankings,
    "avg3_weighted": avg3_rankings,
}


# ---------------------------------------------------------------------------
# Benchmark runner + report


def run(
    seasons: Sequence[int] = DEFAULT_SEASONS,
    scoring: str = "ppr",
    include_holdout: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Score every reference contender for every season; one row per
    (contender, season) with per-position and overall weighted gaps."""
    rows: list[dict] = []
    for season in seasons:
        _check_holdout(season, include_holdout)
        for name, fn in CONTENDERS.items():
            by_pos = {pos: fn(season, pos, scoring) for pos in POSITIONS}
            overall = score_overall(by_pos, season, scoring, include_holdout)
            row: dict = {"contender": name, "season": season, "scoring": scoring}
            for pos in POSITIONS:
                row[pos] = overall.by_position[pos].accuracy_gap
                row[f"{pos}_penalty"] = overall.by_position[pos].penalty_total
            row["overall"] = overall.accuracy_gap
            rows.append(row)
            if verbose:
                cells = "  ".join(f"{pos} {row[pos]:8.1f}" for pos in POSITIONS)
                print(f"{season} {name:<22} {cells}  overall {row['overall']:9.1f}")
    return rows


def summarize(rows: Sequence[Mapping]) -> str:
    """Contender table: per-season overall gaps + per-position means."""
    contenders = list(dict.fromkeys(r["contender"] for r in rows))
    seasons = sorted({r["season"] for r in rows})
    lines = ["| Contender | " + " | ".join(map(str, seasons)) + " | Mean overall | " + " | ".join(f"Mean {p}" for p in POSITIONS) + " |"]
    lines.append("|" + "---|" * (len(seasons) + 5))
    for c in contenders:
        crows = {r["season"]: r for r in rows if r["contender"] == c}
        per_season = [f"{crows[s]['overall']:.0f}" if s in crows else "-" for s in seasons]
        mean_overall = statistics.mean(r["overall"] for r in crows.values())
        pos_means = [f"{statistics.mean(r[p] for r in crows.values()):.0f}" for p in POSITIONS]
        lines.append(f"| {c} | " + " | ".join(per_season) + f" | {mean_overall:.0f} | " + " | ".join(pos_means) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FP-Accuracy v0 — FantasyPros accuracy-method replication")
    ap.add_argument("--seasons", default=None, help="e.g. 2015-2024 or 2018,2021 (default: 2015-2024)")
    ap.add_argument("--scoring", default="ppr", choices=sorted(PRESET_TO_FFC))
    ap.add_argument(
        "--include-holdout", action="store_true",
        help="allow the 2025 holdout season (gate-time evaluation ONLY — see GAMEPLAN §9)",
    )
    args = ap.parse_args(argv)

    seasons: tuple[int, ...] = DEFAULT_SEASONS
    if args.seasons:
        out: list[int] = []
        for part in args.seasons.split(","):
            part = part.strip()
            if "-" in part[1:]:
                lo, hi = part.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            elif part:
                out.append(int(part))
        seasons = tuple(out)
    rows = run(seasons=seasons, scoring=args.scoring, include_holdout=args.include_holdout)
    print()
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
