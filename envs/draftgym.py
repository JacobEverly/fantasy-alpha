#!/usr/bin/env python3
"""DraftGym — the draft RL environment (GAMEPLAN §5).

A gym-style step API over the same historical FFC boards, opponent model, and
feasibility/scarcity guards as DraftBench (imported from evals.draftbench — one
simulator, two consumers). The agent owns one snake-draft seat; every other
seat is the noisy-ADP autopicker. Between agent turns opponents auto-play.

Actions (dict):
    {"pick": "<player_id>"}                    — draft a player (optional
        "rationale": str is recorded for SFT-compatible traces)
    {"tool": {...openai-style tool call...}}   — evidence lookup, dispatched
        through a harness.evidence.EvidenceStore pinned to as_of = Sept 1 of
        the episode season. Results are appended to the observation. Capped at
        MAX_LOOKUPS_PER_PICK per pick; over-cap calls return an error envelope
        and are counted in info["cap_hits"].

Reward (terminal only — risk register #7 amendment):
    realized season points of the agent's roster under a REALISTIC weekly
    manager policy, minus the same-seed AutopickADP control roster scored
    under the SAME policy. The realistic manager sets each week-W lineup by
    season-to-date points-per-game entering that week (weeks < W only, ADP
    order before week 1) — a manager who cannot see the future. DraftBench's
    hindsight-optimal scorer stays a benchmark-only diagnostic; it is NEVER
    the RL reward (hindsight overpays unharvestable variance — reward-hacking
    vector, docs/training-risk-register.md risk 7 hypothesis 3).

Leakage guards:
    - Board features expose ADP/stdev and the PREVIOUS season's total only
      (DraftPlayer carries no realized data; OBS_PLAYER_FIELDS is the exact
      whitelist, unit-tested).
    - Evidence tools hard-filter published < Sept 1 of the season inside
      EvidenceStore (the gate lives in the store, never in the prompt).
    - season == 2025 is refused: untouched gate-time holdout (GAMEPLAN §9).
    - Anonymized mode (``mask_names=True``): render_json() replaces player
      names with stable per-episode board ids (B001... in ADP order) and picks
      use those ids; features (adp/stdev/position/prior-season line) are
      unchanged. Free-text evidence tools (player_news/search) are disabled —
      their text reveals names — and structured lookups (depth_chart/
      injury_status) take a board id and return rows re-keyed to it, with
      errors sanitized. Counter to risk register #8: on historical seasons a
      pretrained model can win by drafting players it REMEMBERS finishing
      well; masked boards force strategy to come from features. Named-minus-
      masked reward is itself a memorization measurement.
    - Season-split discipline (risk register #8): RL rollouts train on
      2015-2022; 2023-2024 are checkpoint-selection val only — enforced by
      the verifiers adapter's taskset splits (envs/verifiers_v1_adapter.py).

Zero third-party dependencies (stdlib + this repo only).
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

from evals.draftbench import (
    DEFAULT_ROSTER,
    DEFAULT_TEAMS,
    AutopickADP,
    DraftPlayer,
    DraftSimulator,
    PickRecord,
    adp_sigma,
    feasible_players,
    load_board,
    position_counts,
    score_roster,
    team_on_the_clock,
    _board_key,
)
from evals.normalize import normalized_metrics
from harness.league import LeagueConfig
from harness.scoring import PRESETS

HOLDOUT_SEASON = 2025
CHAMP_SIMS = 2000  # championship-odds Monte Carlo size (reporting only)
MAX_LOOKUPS_PER_PICK = 5
TOP_AVAILABLE = 25

# The exact per-player observation surface. Anything realized-in-season is a
# leakage bug; tests/test_draftgym.py pins this whitelist.
OBS_PLAYER_FIELDS = (
    "player_id", "name", "position", "team", "adp", "adp_stdev",
    "prev_season_points",
)


# ---------------------------------------------------------------------------
# Realistic weekly manager policy (the RL reward's lineup setter)


def realistic_weekly_lineups(
    roster: Sequence[DraftPlayer],
    weekly: Mapping[str, Mapping[int, float]],
    league: LeagueConfig,
) -> list[float]:
    """Per-week realized points under a manager with no future knowledge.

    Entering week W the manager ranks rostered players by season-to-date
    points per elapsed week (sum of weeks < W divided by W-1 — absences count
    as zeros, so a one-boom wonder decays instead of starting forever; the
    week-1 lineup is the draft-market ADP lineup), fills required slots then
    FLEX by that ranking, and collects whatever those starters ACTUALLY score
    in week W (0 for a bye/injury/dud — the manager eats it, exactly like a
    real one). Deterministic: ties break by lower ADP then name.
    """
    required = league.required_positions()
    flex_set = tuple(league.flex_eligible)
    flex_n = league.flex_per_team()
    perf = {p.player_id: weekly.get(p.player_id, {}) for p in roster}
    week_points: list[float] = []

    for week in range(1, league.season_weeks + 1):
        def key(p: DraftPlayer) -> tuple:
            past = sum(pts for wk, pts in perf[p.player_id].items() if wk < week)
            ppg = past / (week - 1) if week > 1 else 0.0
            # ppg desc, then ADP asc (week 1 is a pure ADP lineup), then name.
            return (-ppg, p.adp, p.name)

        by_pos: dict[str, list[DraftPlayer]] = defaultdict(list)
        for p in roster:
            by_pos[p.position].append(p)
        for players in by_pos.values():
            players.sort(key=key)

        starters: list[DraftPlayer] = []
        for pos, n in required.items():
            starters.extend(by_pos.get(pos, [])[:n])
        flex_pool: list[DraftPlayer] = []
        for pos in flex_set:
            flex_pool.extend(by_pos.get(pos, [])[required.get(pos, 0):])
        flex_pool.sort(key=key)
        starters.extend(flex_pool[:flex_n])

        week_points.append(
            sum(perf[p.player_id].get(week, 0.0) for p in starters)
        )
    return week_points


def realistic_roster_points(
    roster: Sequence[DraftPlayer],
    weekly: Mapping[str, Mapping[int, float]],
    league: LeagueConfig,
) -> float:
    """Season total under the realistic manager policy (weeks 1..season_weeks)."""
    return round(sum(realistic_weekly_lineups(roster, weekly, league)), 2)


def hindsight_roster_points(
    roster: Sequence[DraftPlayer],
    weekly: Mapping[str, Mapping[int, float]],
    league: LeagueConfig,
) -> float:
    """DraftBench's hindsight-optimal scorer (diagnostic only — never reward)."""
    return score_roster(roster, weekly, league).total


# ---------------------------------------------------------------------------
# Observation


class Observation(dict):
    """Plain dict plus render_json() for direct LLM consumption."""

    def render_json(self, indent: int | None = None) -> str:
        return json.dumps(self, indent=indent, default=str)


# ---------------------------------------------------------------------------
# The environment


class DraftGym:
    """One snake-draft episode as a step environment.

    >>> gym = DraftGym(season=2021, preset="ppr", agent_slot=7, seed=3)
    >>> obs = gym.reset()
    >>> obs, reward, done, info = gym.step({"pick": obs["top_available"][0]["player_id"]})

    Determinism: the opponent RNG stream is identical to
    evals.draftbench.DraftSimulator for the same (slot, seed) — an agent that
    always takes the lowest-ADP feasible player reproduces the AutopickADP
    control exactly, so the autopick policy's reward is 0.0 by construction.
    """

    def __init__(
        self,
        season: int,
        preset: str = "ppr",
        league: LeagueConfig | None = None,
        agent_slot: int = 1,
        seed: int = 0,
        max_lookups_per_pick: int = MAX_LOOKUPS_PER_PICK,
        evidence_index_dir: Any | None = None,
        enable_evidence: bool = True,
        mask_names: bool = False,
        pool: Sequence[DraftPlayer] | None = None,
        weekly: Mapping[str, Mapping[int, float]] | None = None,
    ):
        if season == HOLDOUT_SEASON:
            raise ValueError(
                f"season {HOLDOUT_SEASON} is the untouched eval holdout "
                "(GAMEPLAN §9) — DraftGym refuses it during development"
            )
        if preset not in PRESETS:
            raise ValueError(f"unknown scoring preset {preset!r}")
        self.season = int(season)
        self.preset = preset
        self.league = league or LeagueConfig(
            teams=DEFAULT_TEAMS, roster=dict(DEFAULT_ROSTER), scoring=PRESETS[preset]
        )
        if not 1 <= agent_slot <= self.league.teams:
            raise ValueError(f"agent_slot must be within 1..{self.league.teams}")
        self.agent_slot = int(agent_slot)
        self.seed = int(seed)
        self.max_lookups_per_pick = int(max_lookups_per_pick)
        self._enable_evidence = enable_evidence
        self._evidence_index_dir = evidence_index_dir
        self._store: Any | None = None
        self._store_error: str | None = None

        if pool is not None:  # test/synthetic injection
            self.pool = tuple(sorted(pool, key=_board_key))
            self.weekly = dict(weekly or {})
        else:
            loaded = load_board(self.season, preset)
            if loaded is None:
                raise FileNotFoundError(f"no ADP board for {season}/{preset}")
            self.pool, self.weekly = loaded
        self._by_id = {p.player_id: p for p in self.pool}

        # Anonymized observation mode (memorization counter — risk register #8):
        # names are replaced with stable per-episode board ids (B001... in ADP
        # order) so a pretrained model can't win by drafting players it
        # REMEMBERS finishing well. Feature content is unchanged.
        self.mask_names = bool(mask_names)
        self._to_masked = {
            p.player_id: f"B{i + 1:03d}" for i, p in enumerate(self.pool)
        }
        self._from_masked = {m: pid for pid, m in self._to_masked.items()}

        self._done = True  # call reset() first

    # -- evidence -----------------------------------------------------------

    @property
    def as_of(self) -> str:
        """Evidence cutoff: everything published on/after Sept 1 of the
        episode season does not exist for this episode."""
        return f"{self.season}-09-01"

    def _evidence_store(self):
        if not self._enable_evidence:
            self._store_error = "evidence tools disabled for this episode"
            return None
        if self._store is None and self._store_error is None:
            try:
                from harness.evidence import EvidenceStore

                kwargs = {"as_of": self.as_of}
                if self._evidence_index_dir is not None:
                    kwargs["index_dir"] = self._evidence_index_dir
                self._store = EvidenceStore(**kwargs)
            except FileNotFoundError as e:
                self._store_error = f"evidence index unavailable: {e}"
        return self._store

    def tool_definitions(self) -> list[dict]:
        store = self._evidence_store()
        if store is None:
            return []
        return store.as_tool_definitions()

    # -- episode lifecycle ----------------------------------------------------

    def reset(self) -> Observation:
        import random

        self._rng = random.Random(self.seed)
        self._available: list[DraftPlayer] = list(self.pool)
        self._rosters: list[list[DraftPlayer]] = [[] for _ in range(self.league.teams)]
        self._picks: list[PickRecord] = []
        self._pick_number = 1
        self._done = False
        self._lookups_this_pick = 0
        self._total_lookups = 0
        self._cap_hits = 0
        self._tool_results: list[dict] = []
        self._rationales: dict[int, str] = {}

        # League-level scarcity-guard state (mirrors DraftSimulator.simulate —
        # identical guard + RNG stream so the same-seed control replays).
        league = self.league
        self._required = league.required_positions()
        self._flex_set = set(league.flex_eligible)
        self._flex_n = league.flex_per_team()
        self._supply: dict[str, int] = {}
        for p in self._available:
            self._supply[p.position] = self._supply.get(p.position, 0) + 1
        self._unmet_req = {pos: league.teams * n for pos, n in self._required.items()}
        self._unmet_flex = league.teams * self._flex_n
        self._flex_supply = sum(self._supply.get(pos, 0) for pos in self._flex_set)
        self._flex_extra = [0] * league.teams

        self._advance_opponents()
        return self._observe()

    # -- internal draft mechanics (state kept in lockstep with DraftSimulator) --

    def _offered(self, team: int) -> tuple[DraftPlayer, ...]:
        counts = position_counts(self._rosters[team - 1])

        def offered(pos: str) -> bool:
            if counts.get(pos, 0) < self._required.get(pos, 0):
                return True
            if self._supply.get(pos, 0) <= self._unmet_req.get(pos, 0):
                return False
            if pos in self._flex_set and self._flex_extra[team - 1] >= self._flex_n:
                group_need = (
                    sum(self._unmet_req.get(q, 0) for q in self._flex_set)
                    + self._unmet_flex
                )
                return self._flex_supply > group_need
            return True

        return tuple(p for p in self._available if offered(p.position)) or tuple(
            self._available
        )

    def _apply_pick(self, team: int, player: DraftPlayer) -> None:
        counts = position_counts(self._rosters[team - 1])
        pos = player.position
        if counts.get(pos, 0) < self._required.get(pos, 0):
            self._unmet_req[pos] -= 1
        elif pos in self._flex_set and self._flex_extra[team - 1] < self._flex_n:
            self._unmet_flex -= 1
            self._flex_extra[team - 1] += 1
        self._supply[pos] -= 1
        if pos in self._flex_set:
            self._flex_supply -= 1
        self._available.remove(player)
        self._rosters[team - 1].append(player)
        self._picks.append(PickRecord(self._pick_number, team, player.player_id))
        self._pick_number += 1

    def _autopick(self, candidates: Sequence[DraftPlayer]) -> DraftPlayer:
        best: DraftPlayer | None = None
        best_score = 0.0
        for p in candidates:  # ADP-ascending order → same draw sequence as DraftSimulator
            score = p.adp + self._rng.gauss(0.0, adp_sigma(p))
            if best is None or score < best_score:
                best, best_score = p, score
        assert best is not None
        return best

    def _advance_opponents(self) -> None:
        """Auto-play opponent picks until the agent is on the clock or the
        draft ends (board empty or all picks made)."""
        league = self.league
        total = league.teams * league.rounds
        while self._pick_number <= total and self._available:
            team = team_on_the_clock(league, self._pick_number)
            if team == self.agent_slot:
                return
            offer = self._offered(team)
            candidates = feasible_players(
                offer, tuple(self._rosters[team - 1]), league, self._pick_number
            )
            self._apply_pick(team, self._autopick(candidates))
        self._done = True

    def _agent_candidates(self) -> list[DraftPlayer]:
        offer = self._offered(self.agent_slot)
        return feasible_players(
            offer, tuple(self._rosters[self.agent_slot - 1]), self.league, self._pick_number
        )

    # -- observation ------------------------------------------------------------

    def _player_view(self, p: DraftPlayer) -> dict:
        # prev_points is the PREVIOUS season's total only (leakage-safe);
        # never expose nflverse_id or anything realized in-season.
        # In masked mode both id and name become the stable board id.
        pid = self._to_masked[p.player_id] if self.mask_names else p.player_id
        return {
            "player_id": pid,
            "name": pid if self.mask_names else p.name,
            "position": p.position,
            "team": p.team,
            "adp": p.adp,
            "adp_stdev": p.stdev,
            "prev_season_points": p.prev_points,
        }

    def _observe(self) -> Observation:
        league = self.league
        roster = self._rosters[self.agent_slot - 1]
        offer = () if self._done else self._offered(self.agent_slot)
        top = sorted(offer, key=_board_key)[:TOP_AVAILABLE]
        my_picks = league.snake_picks(self.agent_slot)
        future = [q for q in my_picks if q > self._pick_number]
        return Observation(
            season=self.season,
            format=self.preset,
            league={
                "teams": league.teams,
                "roster": dict(league.roster),
                "flex_eligible": list(league.flex_eligible),
                "rounds": league.rounds,
                "scoring": self.preset,
            },
            agent_slot=self.agent_slot,
            pick_number=None if self._done else self._pick_number,
            round=None if self._done else (self._pick_number - 1) // league.teams + 1,
            picks_until_next_turn=(future[0] - self._pick_number) if future else None,
            my_roster=[self._player_view(p) for p in roster],
            feasible_positions=sorted({p.position for p in self._agent_candidates()})
            if not self._done
            else [],
            top_available=[self._player_view(p) for p in top],
            lookups_used_this_pick=self._lookups_this_pick,
            lookups_remaining=max(0, self.max_lookups_per_pick - self._lookups_this_pick),
            tool_results=list(self._tool_results),
            evidence_as_of=self.as_of,
            anonymized=self.mask_names,
            done=self._done,
        )

    # -- step -----------------------------------------------------------------

    def step(self, action: Mapping[str, Any]) -> tuple[Observation, float, bool, dict]:
        if self._done:
            raise RuntimeError("episode is done — call reset()")
        if not isinstance(action, Mapping):
            raise ValueError("action must be a dict with 'pick' or 'tool'")

        if "tool" in action and "pick" not in action:
            return self._step_tool(action["tool"])
        if "pick" in action:
            return self._step_pick(action)
        raise ValueError("action must contain 'pick' (player_id) or 'tool' (tool call)")

    def _step_tool(self, tool_call: Any) -> tuple[Observation, float, bool, dict]:
        if self._lookups_this_pick >= self.max_lookups_per_pick:
            self._cap_hits += 1
            envelope = {
                "ok": False,
                "error": f"lookup cap reached ({self.max_lookups_per_pick} per pick) — make your pick",
            }
        else:
            self._lookups_this_pick += 1
            self._total_lookups += 1
            store = self._evidence_store()
            if store is None:
                envelope = {"ok": False, "error": self._store_error or "no evidence store"}
            elif self.mask_names:
                envelope = self._masked_dispatch(store, tool_call)
            else:
                envelope = store.dispatch(tool_call)
        self._tool_results.append({"call": tool_call, "response": envelope})
        info = self._info(tool_dispatched=envelope["ok"])
        return self._observe(), 0.0, False, info

    def _masked_dispatch(self, store: Any, tool_call: Any) -> dict:
        """Anonymized-mode tool surface: free-text tools are disabled (their
        results would reveal names); structured lookups accept a board id and
        return rows with the same masked id. Errors are sanitized — an error
        message must never de-anonymize a board id either."""
        try:
            call = json.loads(tool_call) if isinstance(tool_call, str) else tool_call
            call = call.get("function", call)
            name = call["name"]
            args = call.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            return {"ok": False, "error": f"bad tool call: {type(e).__name__}"}
        if name in ("player_news", "search"):
            return {
                "ok": False,
                "error": (
                    "tool disabled in anonymized mode (free-text results reveal "
                    "player names); use depth_chart/injury_status with a board id"
                ),
            }
        if name not in ("depth_chart", "injury_status"):
            return {"ok": False, "error": f"unknown tool: {name}"}
        arg_key = "team_or_player" if name == "depth_chart" else "player"
        masked = str(args.get(arg_key, "")).strip()
        real_pid = self._from_masked.get(masked)
        if real_pid is None:
            return {
                "ok": False,
                "error": "anonymized mode: pass a board id (e.g. 'B012') from top_available",
            }
        player = self._by_id[real_pid]
        if name == "injury_status":
            envelope = store.dispatch({"name": "injury_status", "arguments": {"player": player.name}})
        else:
            envelope = store.dispatch({
                "name": "depth_chart",
                "arguments": {"team_or_player": player.name, "latest": bool(args.get("latest", True))},
            })
        if not envelope.get("ok"):
            return {"ok": False, "error": "lookup failed"}  # sanitized: no name echo
        return {"ok": True, "result": self._mask_result(envelope["result"], masked)}

    @staticmethod
    def _mask_result(result: Any, masked_id: str) -> Any:
        def mask_row(row: dict) -> dict:
            row = dict(row)
            if "player_name" in row:
                row["player_name"] = masked_id
            row.pop("gsis_id", None)
            return row

        if result is None:
            return None
        if isinstance(result, list):
            return [mask_row(r) for r in result if isinstance(r, dict)]
        if isinstance(result, dict):
            return mask_row(result)
        return result

    def _step_pick(self, action: Mapping[str, Any]) -> tuple[Observation, float, bool, dict]:
        pid = str(action["pick"])
        if self.mask_names and pid in self._from_masked:
            pid = self._from_masked[pid]
        candidates = self._agent_candidates()
        if pid not in {p.player_id for p in candidates}:
            raise ValueError(
                f"illegal pick {pid!r} at pick {self._pick_number} "
                f"(feasible positions: {sorted({p.position for p in candidates})})"
            )
        if action.get("rationale"):
            self._rationales[self._pick_number] = str(action["rationale"])
        self._apply_pick(self.agent_slot, self._by_id[pid])
        self._lookups_this_pick = 0
        self._tool_results = []
        self._advance_opponents()

        reward = 0.0
        info = self._info()
        if self._done:
            reward = self._terminal_reward(info)
        return self._observe(), reward, self._done, info

    # -- reward -----------------------------------------------------------------

    def _terminal_reward(self, info: dict) -> float:
        league = self.league
        agent_roster = tuple(self._rosters[self.agent_slot - 1])
        control = DraftSimulator(self.pool, league).simulate(
            AutopickADP(), self.agent_slot, self.seed
        )
        agent_pts = realistic_roster_points(agent_roster, self.weekly, league)
        control_pts = realistic_roster_points(control.agent_roster, self.weekly, league)
        reward = round(agent_pts - control_pts, 2)

        # Normalized reporting layer (evals/normalize.py) — league-anchored
        # index + championship odds over the SAME realistic-manager weekly
        # scores. Reporting only: never part of the reward.
        try:
            weekly_by_team = [
                realistic_weekly_lineups(tuple(r), self.weekly, league)
                for r in self._rosters
            ]
            norm = normalized_metrics(
                weekly_by_team, self.agent_slot - 1, n_sims=CHAMP_SIMS, seed=self.seed
            )
        except ValueError:  # degenerate league (odd/too-few teams, zero scores)
            norm = {
                "league_index": None, "league_rank": None,
                "playoff_pct": None, "title_pct": None,
            }
        # Value-ledger telemetry (evals/value_attribution.py) — per-pick
        # capture rows for the agent's picks: realized season VORP minus the
        # slot-price curve. ATTRIBUTION ONLY, never reward: the terminal
        # reward above already prices value capture implicitly; shaping with
        # this would double-count it. Powers the post-draft user receipt
        # ("your best value: X at pick 68, +87 vs slot"). None when the
        # ledger inputs are unavailable (e.g. no slot curve on disk).
        try:
            from evals.value_attribution import value_ledger_rows

            value_ledger = value_ledger_rows(
                [
                    (r.pick_number, self._by_id[r.player_id])
                    for r in self._picks
                    if r.team == self.agent_slot
                ],
                self.season,
                self.preset,
                teams=league.teams,
            )
        except (KeyError, FileNotFoundError):
            value_ledger = None
        info.update(
            value_ledger=value_ledger,
            agent_points_realistic=agent_pts,
            control_points_realistic=control_pts,
            # diagnostics only — hindsight is NEVER the reward (risk 7):
            agent_points_hindsight=hindsight_roster_points(agent_roster, self.weekly, league),
            league_index=norm["league_index"],
            league_rank=norm["league_rank"],
            playoff_pct=norm["playoff_pct"],
            title_pct=norm["title_pct"],
            reward=reward,
            agent_roster=[p.player_id for p in agent_roster],
            rationales=dict(self._rationales),
            picks=[
                {"pick_number": r.pick_number, "team": r.team, "player_id": r.player_id}
                for r in self._picks
            ],
        )
        return reward

    def _info(self, **extra: Any) -> dict:
        info = {
            "pick_number": self._pick_number,
            "lookups_used_this_pick": self._lookups_this_pick,
            "total_lookups": self._total_lookups,
            "cap_hits": self._cap_hits,
        }
        info.update(extra)
        return info
