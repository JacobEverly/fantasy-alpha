#!/usr/bin/env python3
"""DraftBench v0 — the draft-replay benchmark (GAMEPLAN §4.1).

Replays 12-team snake drafts over historical FFC ADP boards. Every non-agent
seat is a noisy-ADP autopicker; the agent seat is whatever policy is under
test. Finished rosters are scored against what actually happened that season:
hindsight-optimal weekly lineups over realized weekly fantasy points
(data/processed/labels/weekly_points.csv), summed over weeks 1-17, plus a
fixed "fantasy playoff" subtotal for weeks 15-17.

Metrics per run (season, format, slot, seed):
- roster points (total + playoff weeks),
- points above the league-median opponent,
- points above the ADP-autopick control drafted in the same seat with the
  same seed (the market baseline — AutopickADP's own delta is 0 by
  construction).

Data notes:
- Boards come from data/raw/ffc_adp/<fmt>_<N>t_<year>.json, positions
  QB/RB/WR/TE only. Historical files are saved as *_8t_*.json but the FFC API
  coerces old years to 12-team drafts (meta.teams == 12) — they are 12-team
  boards, which matches the 12-team league simulated here.
- Boards join to realized outcomes by normalized name + position with a team
  tiebreak (evals/names.py, same approach as build_breakout_labels.py).
  Unmatched ADP players (mostly non-appearing rookies/retirees) stay on the
  board and score zero — drafting them is a real, realized cost.
- Pools filtered to QB/RB/WR/TE are frequently smaller than teams*rounds
  (real drafts spend late picks on K/DST, out of v0 scope). The simulator
  therefore uses a board-aware picks-remaining count plus a league-level
  scarcity guard (see DraftSimulator) so starters still get filled before the
  board runs out, and ends the draft early if it empties.

IMPORTANT: 2025 is the untouched eval holdout (GAMEPLAN §4/§9). DEFAULT_SEASONS
stops at 2024 on purpose — never add 2025 during development; it is evaluated
once per contender at gate time only.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from evals.names import norm_name
from evals.normalize import normalized_metrics
from harness.league import NON_STARTING_SLOTS, LeagueConfig
from harness.scoring import PRESETS
from harness.valuation import ProjectedPlayer, assign_starters

ROOT = Path(__file__).resolve().parent.parent
ADP_DIR = ROOT / "data" / "raw" / "ffc_adp"
LABELS = ROOT / "data" / "processed" / "labels"
OUT_DIR = ROOT / "data" / "processed" / "draftbench"
RESULTS_PATH = OUT_DIR / "results_v0.csv"

POSITIONS = ("QB", "RB", "WR", "TE")
PRESET_TO_FFC = {"standard": "standard", "half_ppr": "half-ppr", "ppr": "ppr"}
_PRESET_IDX = {"standard": 0, "half_ppr": 1, "ppr": 2}

# 2025 is the untouched holdout (GAMEPLAN §4/§9): excluded from defaults,
# only ever run at gate time via an explicit --include-holdout.
DEFAULT_SEASONS = tuple(range(2015, 2025))
DEFAULT_FORMATS = ("standard", "half_ppr", "ppr")  # half_ppr ADP exists 2018+
DEFAULT_SLOTS = (1, 6, 12)
DEFAULT_TEAMS = 12
DEFAULT_ROSTER = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2, "BENCH": 6}  # 15 rounds


# ---------------------------------------------------------------------------
# Board players


@dataclass(frozen=True)
class DraftPlayer:
    """One player on the draft board (market view only — no realized data)."""

    player_id: str  # FFC id, unique within a board
    name: str
    position: str
    team: str
    adp: float
    stdev: float | None  # FFC per-player draft-position spread
    times_drafted: int
    prev_points: float  # previous season's total at this board's preset (0 = rookie/no history)
    nflverse_id: str | None  # realized-data join key; None = never appeared that season


def adp_sigma(player: DraftPlayer) -> float:
    """Noise scale for opponent picks: FFC stdev when present, else 0.15*ADP floored at 1.0."""
    if player.stdev and player.stdev > 0:
        return player.stdev
    return max(1.0, 0.15 * player.adp)


def _board_key(p: DraftPlayer) -> tuple:
    return (p.adp, -p.times_drafted, p.name, p.player_id)


# ---------------------------------------------------------------------------
# Roster feasibility math (shared by simulator and agents)


def position_counts(roster: Iterable[DraftPlayer]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in roster:
        counts[p.position] = counts.get(p.position, 0) + 1
    return counts


def startable_count(counts: Mapping[str, int], league: LeagueConfig) -> int:
    """How many rostered players fit in starting slots (required + FLEX)."""
    required = league.required_positions()
    starters = sum(min(counts.get(pos, 0), n) for pos, n in required.items())
    flex_extra = sum(max(0, counts.get(pos, 0) - required.get(pos, 0)) for pos in league.flex_eligible)
    return starters + min(league.flex_per_team(), flex_extra)


def starter_deficit(counts: Mapping[str, int], league: LeagueConfig) -> int:
    """Starting slots (required + FLEX) still unfilled by the current roster."""
    required = league.required_positions()
    missing = sum(max(0, n - counts.get(pos, 0)) for pos, n in required.items())
    flex_extra = sum(max(0, counts.get(pos, 0) - required.get(pos, 0)) for pos in league.flex_eligible)
    return missing + max(0, league.flex_per_team() - flex_extra)


def _bench_capacity(league: LeagueConfig) -> int:
    return sum(int(league.roster.get(slot, 0)) for slot in NON_STARTING_SLOTS)


def can_add(position: str, counts: Mapping[str, int], league: LeagueConfig, picks_left: int) -> bool:
    """Legal to draft this position now?

    - must still be able to fill every required starting slot with the picks
      left after this one (never strand a starting slot), and
    - the player must fit a starting or bench slot (never roster a position
      beyond starter need + bench capacity).
    """
    after = dict(counts)
    after[position] = after.get(position, 0) + 1
    if starter_deficit(after, league) > picks_left - 1:
        return False
    return sum(after.values()) - startable_count(after, league) <= _bench_capacity(league)


def team_on_the_clock(league: LeagueConfig, pick_number: int) -> int:
    """1-based slot picking at this 1-based overall pick (snake order)."""
    rnd, idx = divmod(pick_number - 1, league.teams)
    return idx + 1 if rnd % 2 == 0 else league.teams - idx


def picks_left(league: LeagueConfig, pick_number: int, board_size: int) -> int:
    """Picks (including this one) the on-clock team still gets before the
    board empties. Boards can be smaller than teams*rounds, so scheduled
    picks beyond the board's supply don't count as capacity."""
    team = team_on_the_clock(league, pick_number)
    return sum(1 for q in league.snake_picks(team) if q >= pick_number and q - pick_number < board_size)


def feasible_players(
    board: Sequence[DraftPlayer],
    my_roster: Sequence[DraftPlayer],
    league: LeagueConfig,
    pick_number: int,
) -> list[DraftPlayer]:
    """Board players whose position is legal to draft right now.

    Degrades in stages when the board runs thin. If no position passes
    can_add (capacity math says the team is cornered), fall back to the
    positions that still reduce the starter deficit — keep filling starting
    slots while players that do so remain — and only then to the whole
    board: a pick is never skipped while players remain, mirroring real
    autopick.
    """
    counts = position_counts(my_roster)
    left = picks_left(league, pick_number, len(board))
    positions = {p.position for p in board}
    legal = {pos for pos in positions if can_add(pos, counts, league, left)}
    if legal:
        return [p for p in board if p.position in legal]
    deficit = starter_deficit(counts, league)
    reducers = set()
    for pos in positions:
        after = dict(counts)
        after[pos] = after.get(pos, 0) + 1
        if starter_deficit(after, league) < deficit:
            reducers.add(pos)
    if reducers:
        return [p for p in board if p.position in reducers]
    return list(board)


# ---------------------------------------------------------------------------
# Agents


class DraftAgent:
    """Interface: pick(board, my_roster, league, pick_number) -> player_id.

    board: available players, ADP-ascending. my_roster: players already drafted
    by this seat. Must return the player_id of a feasible board player.
    """

    name = "agent"

    def pick(
        self,
        board: Sequence[DraftPlayer],
        my_roster: Sequence[DraftPlayer],
        league: LeagueConfig,
        pick_number: int,
    ) -> str:
        raise NotImplementedError


class AutopickADP(DraftAgent):
    """Lowest-ADP feasible player every time. This IS the market baseline."""

    name = "autopick_adp"

    def pick(self, board, my_roster, league, pick_number):
        candidates = feasible_players(board, my_roster, league, pick_number)
        return min(candidates, key=_board_key).player_id


class BestBallADP(DraftAgent):
    """AutopickADP that refuses QB/TE before round 5 — a simple structural
    heuristic (load up on RB/WR early) to show strategy effects."""

    name = "bestball_adp"
    NO_EARLY = frozenset({"QB", "TE"})
    EARLIEST_ROUND = 5

    def pick(self, board, my_roster, league, pick_number):
        candidates = feasible_players(board, my_roster, league, pick_number)
        if len(my_roster) + 1 < self.EARLIEST_ROUND:
            filtered = [p for p in candidates if p.position not in self.NO_EARLY]
            candidates = filtered or candidates
        return min(candidates, key=_board_key).player_id


class GreedyVORP(DraftAgent):
    """Naive-projection VORP maximizer.

    Projection = previous season's total points at the board's preset (0 for
    rookies/no history). Replacement level comes from harness.valuation over
    the first board this agent sees (the full pool minus at most teams-1
    already-picked players — a negligible shift at replacement rank). Picks
    the max value-over-replacement feasible player, preferring players that
    still add a starter while starting slots remain open.
    """

    name = "greedy_vorp"

    def __init__(self) -> None:
        self._replacement: dict[str, float] | None = None

    def pick(self, board, my_roster, league, pick_number):
        if self._replacement is None:
            projected = [ProjectedPlayer(p.player_id, p.name, p.position, p.prev_points) for p in board]
            self._replacement = dict(assign_starters(projected, league).replacement_points)
        replacement = self._replacement

        candidates = feasible_players(board, my_roster, league, pick_number)
        counts = position_counts(my_roster)
        base = startable_count(counts, league)
        adds_starter = set()
        for pos in {p.position for p in candidates}:
            after = dict(counts)
            after[pos] = after.get(pos, 0) + 1
            if startable_count(after, league) > base:
                adds_starter.add(pos)
        pool = [p for p in candidates if p.position in adds_starter] or candidates
        return max(
            pool, key=lambda p: (p.prev_points - replacement.get(p.position, 0.0), -p.adp, p.name)
        ).player_id


BASELINE_AGENTS: dict[str, Callable[[], DraftAgent]] = {
    "autopick_adp": AutopickADP,
    "greedy_vorp": GreedyVORP,
    "bestball_adp": BestBallADP,
}


# ---------------------------------------------------------------------------
# Draft simulator


@dataclass(frozen=True)
class PickRecord:
    pick_number: int
    team: int  # 1-based slot
    player_id: str


@dataclass(frozen=True)
class DraftResult:
    slot: int
    seed: int
    agent_name: str
    rosters: tuple[tuple[DraftPlayer, ...], ...]  # index 0 = slot 1
    picks: tuple[PickRecord, ...]

    @property
    def agent_roster(self) -> tuple[DraftPlayer, ...]:
        return self.rosters[self.slot - 1]


class DraftSimulator:
    """Snake draft over a LeagueConfig: one agent seat, noisy-ADP opponents.

    Opponents score every feasible player as adp + Normal(0, adp_sigma) and
    take the minimum; sigma is the player's real FFC draft spread. On top of
    own-roster feasibility, every seat (agent included, via its offered
    board) obeys a league-level scarcity guard maintaining two invariants:

    - per position: board supply >= the league's unmet *required* need, and
    - across flex-eligible positions: combined supply >= unmet required need
      at those positions + unmet FLEX slots.

    A team is only offered a player its roster doesn't need when taking him
    keeps both invariants intact. Together with the picks-left forcing in
    can_add, this makes "every team fills its starting slots" hold even on
    boards thinner than teams*rounds. The guard applies to agent and control
    identically, so it never biases the points-above-control metric. All
    randomness comes from a single random.Random(seed), so an (agent, slot,
    seed) triple replays identically.
    """

    def __init__(self, pool: Sequence[DraftPlayer], league: LeagueConfig):
        ids = [p.player_id for p in pool]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate player_id in pool")
        self.pool = tuple(sorted(pool, key=_board_key))
        self.league = league

    def simulate(self, agent: DraftAgent, slot: int, seed: int) -> DraftResult:
        league = self.league
        if not 1 <= slot <= league.teams:
            raise ValueError(f"slot must be within 1..{league.teams}")
        rng = random.Random(seed)
        available: list[DraftPlayer] = list(self.pool)
        by_id = {p.player_id: p for p in available}
        rosters: list[list[DraftPlayer]] = [[] for _ in range(league.teams)]
        picks: list[PickRecord] = []

        required = league.required_positions()
        flex_set = set(league.flex_eligible)
        flex_n = league.flex_per_team()
        supply: dict[str, int] = {}
        for p in available:
            supply[p.position] = supply.get(p.position, 0) + 1
        unmet_req = {pos: league.teams * n for pos, n in required.items()}
        unmet_flex = league.teams * flex_n
        flex_supply = sum(supply.get(pos, 0) for pos in flex_set)
        flex_extra = [0] * league.teams  # flex-eligible players beyond required, capped at flex_n

        for pick_number in range(1, league.teams * league.rounds + 1):
            if not available:
                break  # QB/RB/WR/TE pools can be smaller than teams*rounds
            team = team_on_the_clock(league, pick_number)
            roster = rosters[team - 1]
            counts = position_counts(roster)

            def offered(pos: str) -> bool:
                # League-level scarcity guard (see class docstring).
                if counts.get(pos, 0) < required.get(pos, 0):
                    return True  # fills own required slot: always offered
                if supply.get(pos, 0) <= unmet_req.get(pos, 0):
                    return False  # protect required slots league-wide
                if pos in flex_set and flex_extra[team - 1] >= flex_n:
                    group_need = sum(unmet_req.get(q, 0) for q in flex_set) + unmet_flex
                    return flex_supply > group_need  # protect FLEX slots league-wide
                return True

            # Never hide the whole board — a pick is never skipped while
            # players remain (in that degenerate corner someone starves
            # regardless of who picks).
            offer = tuple(p for p in available if offered(p.position)) or tuple(available)
            candidates = feasible_players(offer, roster, league, pick_number)
            if team == slot:
                pid = agent.pick(offer, tuple(roster), league, pick_number)
                if pid not in {p.player_id for p in candidates}:
                    raise ValueError(
                        f"{agent.name} returned illegal pick {pid!r} at pick {pick_number}"
                    )
                player = by_id[pid]
            else:
                player = self._autopick(candidates, rng)

            pos = player.position
            if counts.get(pos, 0) < required.get(pos, 0):
                unmet_req[pos] -= 1
            elif pos in flex_set and flex_extra[team - 1] < flex_n:
                unmet_flex -= 1
                flex_extra[team - 1] += 1
            supply[pos] -= 1
            if pos in flex_set:
                flex_supply -= 1
            available.remove(player)
            roster.append(player)
            picks.append(PickRecord(pick_number, team, player.player_id))

        return DraftResult(
            slot=slot,
            seed=seed,
            agent_name=agent.name,
            rosters=tuple(tuple(r) for r in rosters),
            picks=tuple(picks),
        )

    @staticmethod
    def _autopick(candidates: Sequence[DraftPlayer], rng: random.Random) -> DraftPlayer:
        best: DraftPlayer | None = None
        best_score = 0.0
        for p in candidates:  # fixed ADP-ascending order -> deterministic draw sequence
            score = p.adp + rng.gauss(0.0, adp_sigma(p))
            if best is None or score < best_score:
                best, best_score = p, score
        assert best is not None
        return best


# ---------------------------------------------------------------------------
# Roster scoring (realized weekly points, hindsight-optimal lineups)


@dataclass(frozen=True)
class RosterScore:
    total: float  # weeks 1..league.season_weeks
    playoff: float  # league.playoff_weeks subtotal (15-17 by default)


def optimal_lineup_points(
    week_scores: Sequence[tuple[str, float]], league: LeagueConfig
) -> float:
    """Best starting-lineup total for one week from (position, points) pairs.

    Greedy — top scorers fill each required slot, best remaining
    flex-eligibles fill FLEX — which is exactly optimal for a required+FLEX
    roster. Slots with no rostered player at the position score 0; a slot is
    never left empty while a candidate exists (negative weeks still start).
    """
    by_pos: dict[str, list[float]] = defaultdict(list)
    for pos, pts in week_scores:
        by_pos[pos].append(pts)
    for scores in by_pos.values():
        scores.sort(reverse=True)

    required = league.required_positions()
    total = sum(sum(by_pos.get(pos, [])[:n]) for pos, n in required.items())
    flex_pool: list[float] = []
    for pos in league.flex_eligible:
        flex_pool.extend(by_pos.get(pos, [])[required.get(pos, 0):])
    flex_pool.sort(reverse=True)
    return total + sum(flex_pool[: league.flex_per_team()])


_NO_WEEKS: dict[int, float] = {}


def weekly_roster_points(
    roster: Sequence[DraftPlayer],
    weekly: Mapping[str, Mapping[int, float]],
    league: LeagueConfig,
) -> list[float]:
    """Per-week hindsight-optimal lineup points, weeks 1..season_weeks.

    Reporting helper (league index / championship odds): sums to
    score_roster(...).total exactly — same lineup math, week by week."""
    return [
        optimal_lineup_points(
            [(p.position, weekly.get(p.player_id, _NO_WEEKS).get(week, 0.0)) for p in roster],
            league,
        )
        for week in range(1, league.season_weeks + 1)
    ]


def score_roster(
    roster: Sequence[DraftPlayer],
    weekly: Mapping[str, Mapping[int, float]],
    league: LeagueConfig,
) -> RosterScore:
    """Realized season points: optimal lineup each week, summed across weeks
    1..season_weeks (17 — week 18 of 2021+ seasons is deliberately excluded).
    Players missing a week (bye/injury/inactive/unmatched) score 0."""
    playoff_weeks = set(league.playoff_weeks)
    total = playoff = 0.0
    for week in range(1, league.season_weeks + 1):
        pts = optimal_lineup_points(
            [(p.position, weekly.get(p.player_id, _NO_WEEKS).get(week, 0.0)) for p in roster],
            league,
        )
        total += pts
        if week in playoff_weeks:
            playoff += pts
    return RosterScore(round(total, 2), round(playoff, 2))


# ---------------------------------------------------------------------------
# Data loading: FFC ADP boards joined to realized points


def _adp_path(preset: str, season: int) -> Path | None:
    fmt = PRESET_TO_FFC[preset]
    for teams in (12, 10, 14, 8):  # historical years only have *_8t_* (coerced to 12-team data)
        p = ADP_DIR / f"{fmt}_{teams}t_{season}.json"
        if p.exists():
            return p
    return None


@lru_cache(maxsize=None)
def _season_rows() -> dict[int, list[dict]]:
    by_season: dict[int, list[dict]] = defaultdict(list)
    with open(LABELS / "season_points.csv", newline="") as f:
        for r in csv.DictReader(f):
            by_season[int(r["season"])].append(r)
    return dict(by_season)


@lru_cache(maxsize=None)
def _season_name_index(season: int) -> dict[tuple[str, str], list[dict]]:
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in _season_rows().get(season, []):
        index[(norm_name(r["player"]), r["position"])].append(r)
    return dict(index)


@lru_cache(maxsize=None)
def _weekly_points() -> dict[int, dict[str, dict[int, tuple[float, float, float]]]]:
    """season -> nflverse player_id -> week -> (standard, half_ppr, ppr)."""
    out: dict[int, dict[str, dict[int, tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with open(LABELS / "weekly_points.csv", newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["season"])][r["player_id"]][int(r["week"])] = (
                float(r["pts_standard"]),
                float(r["pts_half_ppr"]),
                float(r["pts_ppr"]),
            )
    return {s: dict(players) for s, players in out.items()}


def _match_row(
    index: Mapping[tuple[str, str], list[dict]], name: str, position: str, team: str
) -> dict | None:
    """Same matching rule as build_breakout_labels: normalized name + position,
    team tiebreak when several players share the key."""
    candidates = index.get((norm_name(name), position), [])
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        team_hits = [c for c in candidates if team and team in c["team"].split("/")]
        return team_hits[0] if team_hits else candidates[0]
    return None


@lru_cache(maxsize=None)
def load_board(
    season: int, preset: str
) -> tuple[tuple[DraftPlayer, ...], dict[str, dict[int, float]]] | None:
    """Build the (pool, realized-weekly-points) pair for one season+format.

    Returns None when no ADP snapshot exists (e.g. half_ppr before 2018).
    weekly maps board player_id -> {week: points at this preset} for players
    who actually appeared; unmatched board players simply have no entry.
    """
    path = _adp_path(preset, season)
    if path is None:
        return None
    payload = json.loads(path.read_text())
    idx = _PRESET_IDX[preset]
    this_year = _season_name_index(season)
    prev_year = _season_name_index(season - 1)
    weekly_by_nflverse = _weekly_points().get(season, {})

    pool: list[DraftPlayer] = []
    weekly: dict[str, dict[int, float]] = {}
    seen: set[str] = set()
    for p in payload["players"]:
        if p.get("position") not in POSITIONS:
            continue
        board_id = str(p["player_id"])
        if board_id in seen:  # defensive: FFC ids should be unique per file
            continue
        seen.add(board_id)
        team = p.get("team") or ""
        match = _match_row(this_year, p["name"], p["position"], team)
        prev = _match_row(prev_year, p["name"], p["position"], team)
        stdev = float(p["stdev"]) if p.get("stdev") not in (None, "") else None
        pool.append(
            DraftPlayer(
                player_id=board_id,
                name=p["name"],
                position=p["position"],
                team=team,
                adp=float(p["adp"]),
                stdev=stdev,
                times_drafted=int(p.get("times_drafted") or 0),
                prev_points=float(prev[f"total_{preset}"]) if prev else 0.0,
                nflverse_id=match["player_id"] if match else None,
            )
        )
        if match:
            weeks = weekly_by_nflverse.get(match["player_id"], {})
            weekly[board_id] = {week: pts[idx] for week, pts in weeks.items()}
    return tuple(sorted(pool, key=_board_key)), weekly


# ---------------------------------------------------------------------------
# Benchmark runner


RUN_COLUMNS = [
    "agent", "season", "format", "slot", "seed", "roster_size",
    "points", "playoff_points", "median_opponent_points", "points_above_median",
    "control_points", "points_above_control",
    # Normalized reporting layer (evals/normalize.py) — human-readable index
    # anchored to the same simulated league; the paired points metrics above
    # remain the statistical backbone.
    "league_index", "league_rank", "playoff_pct", "title_pct",
]


def run(
    seasons: Sequence[int] = DEFAULT_SEASONS,
    formats: Sequence[str] = DEFAULT_FORMATS,
    slots: Sequence[int] = DEFAULT_SLOTS,
    n_seeds: int = 5,
    agents: Mapping[str, Callable[[], DraftAgent]] | None = None,
    out_path: Path | None = RESULTS_PATH,
    verbose: bool = True,
    champ_sims: int = 2000,
    on_result: Callable | None = None,
) -> list[dict]:
    """Run the benchmark grid; one result row per (agent, season, format, slot, seed).

    The AutopickADP draft for each (season, format, slot, seed) is the shared
    control: every agent's points_above_control subtracts that same roster's
    realized points. champ_sims sizes the championship-odds Monte Carlo
    (reporting only — see evals/normalize.py).

    on_result is a REPORTING-ONLY hook (evals/value_attribution.py's ledger
    collector): called once per simulated agent draft as
    on_result(agent_name, season, preset, slot, seed, result, control,
    pool_by_id, league) with the full DraftResult and its same-seed
    AutopickADP control (result IS control for the autopick agent). It never
    affects simulation, scoring, or the returned rows."""
    agents = dict(agents or BASELINE_AGENTS)
    rows: list[dict] = []
    skipped: list[str] = []

    for season in seasons:
        for preset in formats:
            if preset not in PRESET_TO_FFC:
                raise ValueError(f"unknown format {preset!r}; expected one of {sorted(PRESET_TO_FFC)}")
            loaded = load_board(season, preset)
            if loaded is None:
                skipped.append(f"{season}/{preset}")
                continue
            pool, weekly = loaded
            league = LeagueConfig(teams=DEFAULT_TEAMS, roster=dict(DEFAULT_ROSTER), scoring=PRESETS[preset])
            sim = DraftSimulator(pool, league)
            if verbose:
                matched = sum(1 for p in pool if p.nflverse_id)
                print(f"{season} {preset}: board {len(pool)} players ({matched} matched to outcomes)")

            for slot in slots:
                for seed in range(n_seeds):
                    control = sim.simulate(AutopickADP(), slot, seed)
                    control_points = score_roster(control.agent_roster, weekly, league).total
                    for agent_name, factory in agents.items():
                        result = (
                            control
                            if factory is AutopickADP
                            else sim.simulate(factory(), slot, seed)
                        )
                        if on_result is not None:
                            on_result(
                                agent_name, season, preset, slot, seed,
                                result, control,
                                {p.player_id: p for p in pool}, league,
                            )
                        score = score_roster(result.agent_roster, weekly, league)
                        weekly_by_team = [
                            weekly_roster_points(r, weekly, league) for r in result.rosters
                        ]
                        opponents = [
                            round(sum(w), 2)
                            for i, w in enumerate(weekly_by_team)
                            if i != slot - 1
                        ]
                        median_opp = statistics.median(opponents)
                        norm = normalized_metrics(
                            weekly_by_team, slot - 1, n_sims=champ_sims, seed=seed
                        )
                        rows.append({
                            "agent": agent_name,
                            "season": season,
                            "format": preset,
                            "slot": slot,
                            "seed": seed,
                            "roster_size": len(result.agent_roster),
                            "points": score.total,
                            "playoff_points": score.playoff,
                            "median_opponent_points": round(median_opp, 2),
                            "points_above_median": round(score.total - median_opp, 2),
                            "control_points": control_points,
                            "points_above_control": round(score.total - control_points, 2),
                            "league_index": norm["league_index"],
                            "league_rank": norm["league_rank"],
                            "playoff_pct": norm["playoff_pct"],
                            "title_pct": norm["title_pct"],
                        })

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
            w.writeheader()
            w.writerows(rows)

    if verbose:
        if skipped:
            print(f"skipped (no ADP snapshot): {', '.join(skipped)}")
        if out_path is not None:
            print(f"wrote {len(rows)} rows -> {out_path}")
        print()
        print(summarize(rows))
    return rows


def _mean_std(values: Sequence[float]) -> str:
    if not values:
        return "-"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:+8.1f} ±{std:6.1f} (n={len(values)})"


def summarize(rows: Sequence[Mapping]) -> str:
    """Agent × format table of mean points above the ADP-autopick control,
    plus overall aggregates (and points above league-median opponent)."""
    agent_order = list(dict.fromkeys(r["agent"] for r in rows))
    format_order = list(dict.fromkeys(r["format"] for r in rows))
    cell = lambda vals: _mean_std(vals)  # noqa: E731

    lines = ["DraftBench v0 — points above ADP-autopick control (same seat/seed)"]
    header = f"{'agent':<14}" + "".join(f"{fmt:>26}" for fmt in format_order) + f"{'overall':>26}"
    lines.append(header)
    for agent in agent_order:
        agent_rows = [r for r in rows if r["agent"] == agent]
        parts = [
            cell([r["points_above_control"] for r in agent_rows if r["format"] == fmt])
            for fmt in format_order
        ]
        parts.append(cell([r["points_above_control"] for r in agent_rows]))
        lines.append(f"{agent:<14}" + "".join(f"{p:>26}" for p in parts))

    lines.append("")
    lines.append("points above league-median opponent")
    lines.append(header)
    for agent in agent_order:
        agent_rows = [r for r in rows if r["agent"] == agent]
        parts = [
            cell([r["points_above_median"] for r in agent_rows if r["format"] == fmt])
            for fmt in format_order
        ]
        parts.append(cell([r["points_above_median"] for r in agent_rows]))
        lines.append(f"{agent:<14}" + "".join(f"{p:>26}" for p in parts))

    if rows and all(k in rows[0] for k in ("league_index", "title_pct", "playoff_pct")):
        lines.append("")
        lines.append(
            "normalized (evals/normalize.py): League Index (100 = league-best "
            "opponent) / playoff% / title%"
        )
        for agent in agent_order:
            agent_rows = [r for r in rows if r["agent"] == agent]
            idx = statistics.mean(r["league_index"] for r in agent_rows)
            playoff = statistics.mean(r["playoff_pct"] for r in agent_rows)
            title = statistics.mean(r["title_pct"] for r in agent_rows)
            lines.append(
                f"{agent:<14} index {idx:6.1f}   playoff {100 * playoff:5.1f}%   "
                f"title {100 * title:5.1f}%   (n={len(agent_rows)})"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI


def _parse_ints(spec: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return tuple(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DraftBench v0 — draft-replay benchmark")
    ap.add_argument("--seasons", default=None, help="e.g. 2015-2024 or 2018,2021 (default: 2015-2024)")
    ap.add_argument("--formats", default=",".join(DEFAULT_FORMATS), help="comma-separated presets")
    ap.add_argument("--slots", default=",".join(map(str, DEFAULT_SLOTS)), help="e.g. 1,6,12")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per cell (default 5)")
    ap.add_argument("--out", default=str(RESULTS_PATH), help="results CSV path")
    ap.add_argument(
        "--include-holdout", action="store_true",
        help="allow the 2025 holdout season (gate-time evaluation ONLY — see GAMEPLAN §9)",
    )
    args = ap.parse_args(argv)

    seasons = _parse_ints(args.seasons) if args.seasons else DEFAULT_SEASONS
    if 2025 in seasons and not args.include_holdout:
        ap.error("2025 is the untouched eval holdout (GAMEPLAN §9); pass --include-holdout only at gate time")
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    slots = _parse_ints(args.slots)
    run(seasons=seasons, formats=formats, slots=slots, n_seeds=args.seeds, out_path=Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
