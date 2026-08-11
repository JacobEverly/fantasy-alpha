#!/usr/bin/env python3
"""Packet-v2 feature extraction: per-player enrichment tables for BreakoutBench dossiers.

Emits data/processed/packet_features/features_<S>.csv — one row per
(player_id, season S) for every QB/RB/WR/TE who appears in season S-1 weekly
stats, in the season-S FFC ADP pool, or in the season-S draft class.

The as-of gate: every value is computed from data dated (or occurring)
STRICTLY BEFORE Sept 1 of season S. Channel builders assert this on their
inputs ("leakage" AssertionError), mirroring evals/harness_breakout.py, and
tests/test_packet_features.py poisons each channel with future rows.

Missing != zero: every channel carries a *_status column. Statuses:
  ok                    computed from in-gate data
  no_prior_season       player has no S-1 stats (e.g. rookie)
  unavailable_pre_asof  no pre-Sept-1 source exists for that season
                        (historical depth charts are weekly REG-only; the
                        dated-snapshot schema starts with the 2025 file)
  pre_coverage          news archive does not reach back that far (~2016-05)
  no_id_match           player could not be joined to the source's id space
  not_derivable         source exists but lacks the needed team/season rows

Usage:
  .venv/bin/python -m evals.packet_features build --season 2023
  .venv/bin/python -m evals.packet_features build --seasons 2015-2024
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from evals.names import norm_name

ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = ROOT / "data" / "raw" / "nflverse" / "stats_player"
DEPTH_DIR = ROOT / "data" / "raw" / "nflverse" / "depth_charts"
DRAFT_CSV = ROOT / "data" / "raw" / "nflverse" / "draft_picks" / "draft_picks.csv"
ADP_DIR = ROOT / "data" / "raw" / "ffc_adp"
NEWS_DIR = ROOT / "data" / "raw" / "evidence" / "sleeper_backfill_internal"
SLEEPER_CACHE = ROOT / "data" / "raw" / "evidence" / "sleeper_players_cache.json"
PLAY_CALLER_CSV = (ROOT / "fantasy-football-alpha-2026" / "data" / "manual"
                   / "play_caller_history.csv")
OUT_DIR = ROOT / "data" / "processed" / "packet_features"

POSITIONS = {"QB", "RB", "WR", "TE"}
NEWS_WINDOW_DAYS = 90

# keyword flags over news titles/descriptions (case-insensitive), versioned.
#
# packet-news-lexicon-v1 used bare \bstarter\b|\bstarting\b, which
# evals/source_alpha.py measured as firing on preseason-game logistics prose
# ("starters will play a quarter Thursday") on ~82% of matches — no
# discrimination. v2 (2026-08-11) ports source_alpha's promotion-specific
# CAMP_PROMOTION_RE (source-alpha-lexicon-v1) verbatim: the word
# starter/starting only counts inside promotion phrasing. Counts built under
# different lexicon versions are not comparable.
NEWS_LEXICON_VERSION = "packet-news-lexicon-v2"

# retired v1 pattern, kept for provenance/diffing only — do not use
CAMP_PROMOTION_RE_V1 = re.compile(
    r"first[- ]team|\bstarter\b|\bstarting\b|\bWR1\b|\bRB1\b", re.IGNORECASE)

# v2 == evals/source_alpha.py CAMP_PROMOTION_RE (source-alpha-lexicon-v1)
CAMP_PROMOTION_RE = re.compile(
    r"first[- ]team|with the (?:1s|ones|starters)\b"
    r"|named (?:the |him )?(?:\w+\s)?starter|listed as (?:the |a )?starter"
    r"|won the (?:starting )?job|starting job is his"
    r"|\bWR1\b|\bRB1\b|\bTE1\b|\bQB1\b"
    r"|promot(?:ed|ion)|lead\s+(?:role|back|receiver)"
    r"|no\.\s*1\s+(?:receiver|wideout|back|running back|tight end|option)"
    r"|top of the depth chart|atop the depth chart",
    re.IGNORECASE)
INJURY_RE = re.compile(
    r"injur|\bACL\b|achilles|hamstring|concussion|sprain|fractur|surgery"
    r"|\bMCL\b|high[- ]ankle|\bIR\b", re.IGNORECASE)
HYPE_RE = re.compile(r"breakout|impress", re.IGNORECASE)

COLUMNS = [
    "season", "as_of", "player_id", "player", "position",
    "in_prior_stats", "in_adp_pool", "rookie_flag",
    # 1. usage (from weekly stats <= S-1)
    "games_s1", "targets_pg_s1", "carries_pg_s1", "opps_pg_s1",
    "rec_yards_pg_s1", "target_share_s1", "air_yards_share_s1",
    "opps_pg_s2", "opps_pg_yoy", "usage_status",
    # 1b. role quality (weekly stats <= S-1; receiving-side air yards).
    #     The 145-col nflverse weekly schema has NO red-zone / end-zone
    #     opportunity columns (rz_targets, endzone_targets, ...) — TD counts
    #     are outcomes, not role, so no proxy is emitted.
    "adot_s1", "wopr_s1", "tgt_games_s1", "route_part_proxy_s1",
    "target_share_s2", "target_share_yoy",
    "air_yards_share_s2", "air_yards_share_yoy",
    "adot_s2", "adot_yoy", "wopr_s2", "wopr_yoy", "role_status",
    # 2. depth chart
    "team_s1_end", "pos_rank_s1_end", "depth_pos_s1_end",
    "team_asof", "pos_rank_asof", "pos_rank_delta", "changed_team",
    "depth_status",
    # 3. vacated opportunity (player's pre-Sept-1 team)
    "vacated_targets", "vacated_carries", "vacated_opps", "vacated_status",
    # 4. coaching
    "team_changed_play_caller", "play_caller_s", "play_caller_s1",
    "coaching_status",
    # 5. pedigree
    "birth_date", "age_at_sept1", "age_source",
    "draft_year", "draft_round", "draft_pick", "undrafted",
    "years_since_draft", "pedigree_status",
    # 6. news (90 days before Sept 1 of S)
    "n_news_90d", "camp_promotion_flags", "injury_mention_flags",
    "hype_flags", "news_status",
]


# ---------------------------------------------------------------------------
# as-of gate helpers

def as_of_date(season: int) -> date:
    """The gate: Sept 1 of season S. Everything must be strictly before it."""
    return date(season, 9, 1)


def as_of_epoch_ms(season: int) -> int:
    return int(datetime(season, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)


def assert_in_gate(ok: bool, channel: str, detail: str = "") -> None:
    assert ok, f"leakage: {channel} received data at/after the as-of gate {detail}"


# ---------------------------------------------------------------------------
# 1. usage (weekly stats)

def season_usage(weekly_rows: list[dict], season: int, eval_season: int) -> dict:
    """Aggregate one player-season of weekly REG stat rows (already grouped).

    weekly_rows carry int `season`; the gate demands season < eval_season
    (season-N games all end by early Feb of N+1 < Sept 1 of eval_season).
    """
    out: dict = {"games": 0, "targets": 0.0, "carries": 0.0,
                 "rec_yards": 0.0, "tgt_share": [], "ay_share": [], "team": "",
                 "rec_air_yards": 0.0, "rec_air_yards_seen": False,
                 "wopr": [], "tgt_games": 0}
    for r in weekly_rows:
        assert_in_gate(int(r["season"]) < eval_season, "usage",
                       f"(row season {r['season']} vs eval {eval_season})")
        if int(r["season"]) != season:
            continue
        out["games"] += 1
        out["targets"] += _f(r.get("targets"))
        out["carries"] += _f(r.get("carries"))
        out["rec_yards"] += _f(r.get("receiving_yards"))
        if _f(r.get("targets")) > 0:
            out["tgt_games"] += 1
        ay = r.get("receiving_air_yards")
        if ay not in (None, "", "NA"):
            out["rec_air_yards"] += float(ay)
            out["rec_air_yards_seen"] = True
        for key, acc in (("target_share", "tgt_share"),
                         ("air_yards_share", "ay_share"),
                         ("wopr", "wopr")):
            v = r.get(key)
            if v not in (None, "", "NA"):
                out[acc].append(float(v))
        out["team"] = r.get("team") or out["team"]
    return out


def _mean(vals: list, nd: int = 4):
    return round(sum(vals) / len(vals), nd) if vals else None


def role_metrics(u1: dict | None, u2: dict | None,
                 team_games_s1: int | None) -> dict:
    """Role-quality aggregates from season_usage() outputs (missing != zero).

    u1/u2: S-1 / S-2 season_usage dicts (None = no rows that season).
      adot_sN            receiving air yards per target (aDOT-ish); only when
                         the season had >0 targets AND non-NA air-yards data
                         (pre-2006 files carry the column zero-filled -> the
                         seen flag alone does not protect; caller restricts
                         to seasons with real data, 2010+ here)
      wopr_sN            mean weekly WOPR (nflverse precomputed)
      tgt_games_s1       S-1 games with >=1 target
      route_part_proxy   tgt_games_s1 / team REG games — no route/snap data
                         exists in this schema, so "was he in the target
                         rotation, availability included" is the proxy
      *_yoy              S-1 minus S-2, only when both sides are present
    """
    out: dict = {"adot_s1": None, "wopr_s1": None, "tgt_games_s1": None,
                 "route_part_proxy_s1": None,
                 "target_share_s2": None, "target_share_yoy": None,
                 "air_yards_share_s2": None, "air_yards_share_yoy": None,
                 "adot_s2": None, "adot_yoy": None,
                 "wopr_s2": None, "wopr_yoy": None,
                 "role_status": "no_prior_season"}

    def _adot(u):
        if u and u["targets"] > 0 and u["rec_air_yards_seen"]:
            return round(u["rec_air_yards"] / u["targets"], 3)
        return None

    if u1 is None or u1["games"] == 0:
        return out
    out["adot_s1"] = _adot(u1)
    out["wopr_s1"] = _mean(u1["wopr"])
    out["tgt_games_s1"] = u1["tgt_games"]
    if team_games_s1:
        out["route_part_proxy_s1"] = round(u1["tgt_games"] / team_games_s1, 4)
    ts1, ay1 = _mean(u1["tgt_share"]), _mean(u1["ay_share"])
    if u2 is not None and u2["games"] > 0:
        out["target_share_s2"] = _mean(u2["tgt_share"])
        out["air_yards_share_s2"] = _mean(u2["ay_share"])
        out["adot_s2"] = _adot(u2)
        out["wopr_s2"] = _mean(u2["wopr"])
        for a, b, key, nd in ((ts1, out["target_share_s2"], "target_share_yoy", 4),
                              (ay1, out["air_yards_share_s2"], "air_yards_share_yoy", 4),
                              (out["adot_s1"], out["adot_s2"], "adot_yoy", 3),
                              (out["wopr_s1"], out["wopr_s2"], "wopr_yoy", 4)):
            if a is not None and b is not None:
                out[key] = round(a - b, nd)
    derivable = (out["adot_s1"] is not None or out["wopr_s1"] is not None)
    out["role_status"] = "ok" if derivable else "not_derivable"
    return out


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _pg(total: float, games: int):
    return round(total / games, 3) if games else None


# ---------------------------------------------------------------------------
# 2. depth charts

def s1_end_depth(rows: list[dict], s1: int, eval_season: int) -> dict[str, dict]:
    """Final-REG-week offensive depth chart of season S-1 (weekly schema).

    Returns gsis_id -> {team, pos_rank, depth_pos}. Weekly-schema rows carry
    `season`; gate: season < eval_season.
    """
    last_week: dict[str, int] = {}
    for r in rows:
        assert_in_gate(int(r["season"]) < eval_season, "depth_s1_end",
                       f"(row season {r['season']})")
        if int(r["season"]) != s1 or r.get("game_type") != "REG":
            continue
        wk = int(r["week"]) if r.get("week") else 0
        team = r["club_code"]
        last_week[team] = max(last_week.get(team, 0), wk)
    out: dict[str, dict] = {}
    for r in rows:
        if int(r["season"]) != s1 or r.get("game_type") != "REG":
            continue
        if r.get("formation") != "Offense" or not r.get("gsis_id"):
            continue
        team = r["club_code"]
        if int(r["week"] or 0) != last_week.get(team):
            continue
        rank = int(r["depth_team"]) if r.get("depth_team") else None
        cur = out.get(r["gsis_id"])
        if rank is not None and (cur is None or rank < cur["pos_rank"]):
            out[r["gsis_id"]] = {"team": team, "pos_rank": rank,
                                 "depth_pos": r.get("depth_position", "")}
    return out


def asof_snapshot_depth(rows: list[dict], season: int) -> dict[str, dict]:
    """Latest pre-Sept-1 snapshot from dated (`dt`) depth chart rows.

    Returns gsis_id -> {team, pos_rank, dt}. Rows dated at/after the gate are
    dropped BEFORE use and trip the leakage assertion if passed through.
    """
    gate = f"{as_of_date(season).isoformat()}T00:00:00Z"
    in_gate = [r for r in rows if r.get("dt", "") < gate]
    for r in in_gate:
        assert_in_gate(r["dt"] < gate, "depth_asof", f"(dt {r['dt']})")
    if not in_gate:
        return {}
    latest_dt = max(r["dt"][:10] for r in in_gate)
    out: dict[str, dict] = {}
    for r in in_gate:
        if r["dt"][:10] != latest_dt or not r.get("gsis_id"):
            continue
        if r.get("pos_grp") and "off" not in r["pos_grp"].lower():
            continue
        rank = int(r["pos_rank"]) if r.get("pos_rank") else None
        gid = r["gsis_id"].strip()
        cur = out.get(gid)
        if rank is not None and (cur is None or rank < cur["pos_rank"]):
            out[gid] = {"team": r["team"], "pos_rank": rank, "dt": r["dt"][:10]}
    return out


# ---------------------------------------------------------------------------
# 3. vacated opportunity

def vacated_opportunity(team: str, roster_ids_asof: set[str],
                        s1_player_usage: list[dict]) -> dict:
    """Sum S-1 targets/carries on `team` from players NOT on its as-of roster.

    s1_player_usage rows: {player_id, team, targets, carries} (S-1 totals).
    roster_ids_asof: gsis ids on the team's pre-Sept-1 depth chart.
    """
    vac_t = vac_c = 0.0
    for r in s1_player_usage:
        if r["team"] != team or r["player_id"] in roster_ids_asof:
            continue
        vac_t += _f(r.get("targets"))
        vac_c += _f(r.get("carries"))
    return {"vacated_targets": round(vac_t, 1), "vacated_carries": round(vac_c, 1),
            "vacated_opps": round(vac_t + vac_c, 1)}


# ---------------------------------------------------------------------------
# 4. coaching (play-caller change)

def play_caller_for(pc_rows: list[dict], team: str, season: int,
                    eval_season: int) -> str | None:
    """Primary play caller for (team, season) from the research repo's manual
    history. Play-caller appointments for season S are announced pre-season
    (Jan-Feb), so season == eval_season rows are in-gate facts. Asking about
    a season beyond the gate is leakage; file rows beyond the gate are
    filtered out before use."""
    assert_in_gate(season <= eval_season, "coaching",
                   f"(queried season {season})")
    best, best_w = None, 0.0
    for r in pc_rows:
        if int(r["season"]) > eval_season:
            continue  # future appointments exist in the file; never used
        if r["team"] == team and int(r["season"]) == season:
            w = _f(r.get("play_calling_weight"))
            if w > best_w:
                best, best_w = r["coach"], w
    return best


def coaching_change(pc_rows: list[dict], team: str, eval_season: int) -> dict:
    cur = play_caller_for(pc_rows, team, eval_season, eval_season)
    prev = play_caller_for(pc_rows, team, eval_season - 1, eval_season)
    if cur is None or prev is None:
        return {"team_changed_play_caller": None, "play_caller_s": cur,
                "play_caller_s1": prev, "coaching_status": "not_derivable"}
    return {"team_changed_play_caller": int(cur != prev), "play_caller_s": cur,
            "play_caller_s1": prev, "coaching_status": "ok"}


# ---------------------------------------------------------------------------
# 5. pedigree

def pedigree(draft_row: dict | None, birth_date: str | None,
             season: int) -> dict:
    """Age at Sept 1 + draft capital. Draft rows must be season <= S (drafts
    happen in April, before the gate)."""
    out = {"birth_date": birth_date or "", "age_at_sept1": None,
           "age_source": "", "draft_year": None, "draft_round": None,
           "draft_pick": None, "undrafted": None, "years_since_draft": None,
           "pedigree_status": "ok"}
    if draft_row is not None:
        assert_in_gate(int(draft_row["season"]) <= season, "pedigree",
                       f"(draft season {draft_row['season']})")
        out["draft_year"] = int(draft_row["season"])
        out["draft_round"] = int(draft_row["round"]) if draft_row.get("round") else None
        out["draft_pick"] = int(draft_row["pick"]) if draft_row.get("pick") else None
        out["undrafted"] = 0
        out["years_since_draft"] = season - int(draft_row["season"])
    else:
        out["undrafted"] = 1
    gate = as_of_date(season)
    if birth_date:
        try:
            b = date.fromisoformat(birth_date.strip())
            age = (gate.year - b.year) - ((gate.month, gate.day) < (b.month, b.day))
            out["age_at_sept1"] = age
            out["age_source"] = "sleeper_birth_date"
        except ValueError:
            pass
    if out["age_at_sept1"] is None and draft_row is not None and draft_row.get("age"):
        # draft_picks age = age in draft year; approximate forward
        out["age_at_sept1"] = int(float(draft_row["age"])) + (season - int(draft_row["season"]))
        out["age_source"] = "draft_age_approx"
    if out["age_at_sept1"] is None and out["undrafted"] == 1:
        out["pedigree_status"] = "no_id_match" if not birth_date else "ok"
    return out


# ---------------------------------------------------------------------------
# 6. news counts

def news_counts(items: list[dict], season: int) -> dict:
    """Keyword-flag counts over the 90 days strictly before Sept 1 of S.

    items: {published (ms epoch), text}. Window = [gate - 90d, gate); an item
    at exactly gate-90d is in, an item at the gate is leakage.
    """
    gate_ms = as_of_epoch_ms(season)
    start_ms = gate_ms - NEWS_WINDOW_DAYS * 86_400_000
    n = camp = inj = hype = 0
    for it in items:
        ts = int(it["published"])
        if ts >= gate_ms or ts < start_ms:
            continue
        assert_in_gate(ts < gate_ms, "news", f"(published {ts})")
        n += 1
        text = it.get("text", "")
        if CAMP_PROMOTION_RE.search(text):
            camp += 1
        if INJURY_RE.search(text):
            inj += 1
        if HYPE_RE.search(text):
            hype += 1
    return {"n_news_90d": n, "camp_promotion_flags": camp,
            "injury_mention_flags": inj, "hype_flags": hype}


# ---------------------------------------------------------------------------
# loaders (I/O kept out of the channel math above)

def load_weekly_stats(season: int) -> list[dict]:
    path = STATS_DIR / f"stats_player_week_{season}.csv"
    if not path.exists():
        return []
    keep = ("player_id", "player_display_name", "player_name", "position",
            "season", "week", "season_type", "team", "targets", "carries",
            "receiving_yards", "target_share", "air_yards_share",
            "receiving_air_yards", "wopr")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("season_type") != "REG" or r.get("position") not in POSITIONS:
                continue
            rows.append({k: r.get(k, "") for k in keep})
    return rows


def load_depth_chart(year: int) -> tuple[str, list[dict]]:
    """Returns (schema, rows): schema 'weekly' (2011-2024) or 'snapshot'
    (dt-dated, 2025+)."""
    path = DEPTH_DIR / f"depth_charts_{year}.csv"
    if not path.exists():
        return "missing", []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        schema = "snapshot" if "dt" in (reader.fieldnames or []) else "weekly"
        return schema, list(reader)


def load_draft_picks() -> list[dict]:
    with open(DRAFT_CSV, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("position") in
                POSITIONS | {"HB", "FB"}]


def load_adp_pool(season: int) -> list[dict]:
    """Union of FFC ADP pools across formats for season S (pool membership
    only — no ADP values are emitted as features here)."""
    players: dict[tuple, dict] = {}
    for fmt in ("ppr", "half-ppr", "standard", "2qb"):
        for teams in (12, 10, 14, 8):
            p = ADP_DIR / f"{fmt}_{teams}t_{season}.json"
            if not p.exists():
                continue
            payload = json.loads(p.read_text())
            for pl in payload.get("players", []):
                if pl.get("position") in POSITIONS:
                    players.setdefault((pl["name"], pl["position"]), pl)
            break
    return list(players.values())


def load_news_index() -> tuple[dict[str, list[dict]], int | None]:
    """sleeper_id -> [{published, text}], plus the archive's earliest
    timestamp. Identity resolution to gsis happens in Context (the sleeper
    cache only carries gsis ids for ~1/3 of players; the rest join on
    normalized name + position)."""
    if not NEWS_DIR.exists():
        return {}, None
    index: dict[str, list[dict]] = defaultdict(list)
    earliest = None
    for path in NEWS_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for it in payload.get("news", []):
            ts = it.get("published")
            if ts is None:
                continue
            ts = int(ts)
            meta = it.get("metadata") or {}
            text = " ".join(s for s in (meta.get("title"), meta.get("description"))
                            if s)
            index[path.stem].append({"published": ts, "text": text})
            earliest = ts if earliest is None else min(earliest, ts)
    return dict(index), earliest


def load_play_callers() -> list[dict]:
    if not PLAY_CALLER_CSV.exists():
        return []
    with open(PLAY_CALLER_CSV, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# assembly

class Context:
    """Shared cross-season state (loaded once)."""

    def __init__(self):
        self.draft_picks = load_draft_picks()
        self.news_index, self.news_earliest_ms = load_news_index()
        self.play_callers = load_play_callers()
        self._load_sleeper_identity()
        self.draft_by_gsis = {}
        self.draft_by_name = {}
        for r in self.draft_picks:
            gid = (r.get("gsis_id") or "").strip()
            if gid and gid not in self.draft_by_gsis:
                self.draft_by_gsis[gid] = r
            key = (norm_name(r.get("pfr_player_name", "")), r.get("position"))
            self.draft_by_name.setdefault(key, r)
        self._weekly: dict[int, list[dict]] = {}

    def _load_sleeper_identity(self) -> None:
        """gsis-first sleeper join, with a (norm_name, position) fallback —
        only ~1/3 of sleeper cache entries carry a gsis id. Ambiguous name
        keys (two sleeper players sharing name+position) are dropped."""
        self.sid_by_gsis: dict[str, str] = {}
        self.sid_by_name: dict[tuple, str | None] = {}
        self.birth_by_sid: dict[str, str] = {}
        if not SLEEPER_CACHE.exists():
            return
        cache = json.loads(SLEEPER_CACHE.read_text())
        for sid, p in cache.items():
            sid = str(sid)
            gid = (p.get("gsis_id") or "").strip()
            if gid:
                self.sid_by_gsis[gid] = sid
            if p.get("birth_date"):
                self.birth_by_sid[sid] = p["birth_date"]
            name, pos = p.get("full_name"), p.get("position")
            if name and pos in POSITIONS:
                key = (norm_name(name), pos)
                self.sid_by_name[key] = (None if key in self.sid_by_name
                                         else sid)

    def sleeper_id_for(self, gsis: str, player: str, position: str) -> str | None:
        sid = self.sid_by_gsis.get(gsis)
        if sid is None and player:
            sid = self.sid_by_name.get((norm_name(player), position))
        return sid

    def weekly(self, season: int) -> list[dict]:
        if season not in self._weekly:
            self._weekly[season] = load_weekly_stats(season)
        return self._weekly[season]


def build_season(season: int, ctx: Context) -> list[dict]:
    s1, s2 = season - 1, season - 2
    gate = as_of_date(season).isoformat()

    # --- prior-season stat aggregates, grouped per player
    by_pid_s1: dict[str, list[dict]] = defaultdict(list)
    for r in ctx.weekly(s1):
        by_pid_s1[r["player_id"]].append(r)
    by_pid_s2: dict[str, list[dict]] = defaultdict(list)
    for r in ctx.weekly(s2):
        by_pid_s2[r["player_id"]].append(r)

    # team REG game counts in S-1 (denominator for the route-participation
    # proxy): distinct weeks per team in the weekly file
    team_weeks_s1: dict[str, set] = defaultdict(set)
    for r in ctx.weekly(s1):
        if r.get("team"):
            team_weeks_s1[r["team"]].add(int(r["week"]))

    # name->gsis resolution index from all pre-gate stat seasons
    name_to_gsis: dict[tuple, str] = {}
    for yr in range(2011, season):
        for r in ctx.weekly(yr):
            name = r.get("player_display_name") or r.get("player_name") or ""
            name_to_gsis[(norm_name(name), r["position"])] = r["player_id"]

    # --- depth charts
    s1_schema, s1_rows = load_depth_chart(s1)
    s1_depth: dict[str, dict] = {}
    if s1_schema == "weekly":
        s1_depth = s1_end_depth(s1_rows, s1, season)
    elif s1_schema == "snapshot":
        # latest snapshot before March 1 of `season` = season-end state of S-1
        cutoff = f"{season}-03-01T00:00:00Z"
        rows = [r for r in s1_rows if r.get("dt", "") < cutoff]
        snap = asof_snapshot_depth(rows, season)  # dt < Sept 1 holds a fortiori
        s1_depth = {g: {"team": v["team"], "pos_rank": v["pos_rank"],
                        "depth_pos": ""} for g, v in snap.items()}

    s_schema, s_rows = load_depth_chart(season)
    asof_depth: dict[str, dict] = {}
    if s_schema == "snapshot":
        asof_depth = asof_snapshot_depth(s_rows, season)
    depth_asof_available = bool(asof_depth)

    # --- team-level S-1 usage (for vacated opportunity)
    s1_totals: dict[str, dict] = {}
    for pid, rows in by_pid_s1.items():
        u = season_usage(rows, s1, season)
        s1_totals[pid] = {"player_id": pid, "team": u["team"],
                          "targets": u["targets"], "carries": u["carries"],
                          "usage": u,
                          "player": rows[-1].get("player_display_name") or
                                    rows[-1].get("player_name") or "",
                          "position": rows[-1]["position"]}
    s1_team_usage = [{"player_id": v["player_id"], "team": v["team"],
                      "targets": v["targets"], "carries": v["carries"]}
                     for v in s1_totals.values()]
    asof_rosters: dict[str, set] = defaultdict(set)
    for gid, v in asof_depth.items():
        asof_rosters[v["team"]].add(gid)
    vacated_by_team = {}
    if depth_asof_available:
        for team in asof_rosters:
            vacated_by_team[team] = vacated_opportunity(
                team, asof_rosters[team], s1_team_usage)

    # --- row universe
    universe: dict[str, dict] = {}  # gsis (or synthetic) -> seed info
    for pid, v in s1_totals.items():
        universe[pid] = {"player": v["player"], "position": v["position"],
                         "in_prior_stats": 1, "in_adp_pool": 0, "rookie": 0}
    unresolved = 0
    for p in load_adp_pool(season):
        key = (norm_name(p["name"]), p["position"])
        gid = name_to_gsis.get(key)
        if gid is None:
            dr = ctx.draft_by_name.get(key)
            if dr is not None and int(dr["season"]) <= season:
                gid = (dr.get("gsis_id") or "").strip() or None
        if gid is None:
            gid = f"ffc:{p['player_id']}"
            unresolved += 1
        if gid in universe:
            universe[gid]["in_adp_pool"] = 1
        else:
            universe[gid] = {"player": p["name"], "position": p["position"],
                             "in_prior_stats": 0, "in_adp_pool": 1, "rookie": 0}
    for r in ctx.draft_picks:
        if int(r["season"]) != season or r.get("position") not in POSITIONS:
            continue
        gid = (r.get("gsis_id") or "").strip()
        if not gid:
            continue
        if gid in universe:
            universe[gid]["rookie"] = 1
        else:
            universe[gid] = {"player": r.get("pfr_player_name", ""),
                             "position": r["position"], "in_prior_stats": 0,
                             "in_adp_pool": 0, "rookie": 1}

    news_covered = (ctx.news_earliest_ms is not None and
                    as_of_epoch_ms(season) - NEWS_WINDOW_DAYS * 86_400_000
                    >= ctx.news_earliest_ms)

    rows_out: list[dict] = []
    for gid, seed in sorted(universe.items()):
        row: dict = {c: "" for c in COLUMNS}
        row.update({"season": season, "as_of": gate, "player_id": gid,
                    "player": seed["player"], "position": seed["position"],
                    "in_prior_stats": seed["in_prior_stats"],
                    "in_adp_pool": seed["in_adp_pool"],
                    "rookie_flag": seed["rookie"]})
        is_real_gsis = not gid.startswith("ffc:")

        # 1. usage
        if gid in s1_totals:
            u = s1_totals[gid]["usage"]
            g = u["games"]
            row.update({
                "games_s1": g,
                "targets_pg_s1": _pg(u["targets"], g),
                "carries_pg_s1": _pg(u["carries"], g),
                "opps_pg_s1": _pg(u["targets"] + u["carries"], g),
                "rec_yards_pg_s1": _pg(u["rec_yards"], g),
                "target_share_s1": round(sum(u["tgt_share"]) / len(u["tgt_share"]), 4)
                    if u["tgt_share"] else "",
                "air_yards_share_s1": round(sum(u["ay_share"]) / len(u["ay_share"]), 4)
                    if u["ay_share"] else "",
                "usage_status": "ok"})
            u2 = None
            if gid in by_pid_s2:
                u2 = season_usage(by_pid_s2[gid], s2, season)
                opps2 = _pg(u2["targets"] + u2["carries"], u2["games"])
                row["opps_pg_s2"] = opps2
                if opps2 is not None and row["opps_pg_s1"] is not None:
                    row["opps_pg_yoy"] = round(row["opps_pg_s1"] - opps2, 3)
            # 1b. role quality (same channel inputs, same gate)
            row.update(role_metrics(u, u2, len(team_weeks_s1.get(u["team"], ()))
                                    or None))
        else:
            row["usage_status"] = "no_prior_season"
            row["role_status"] = "no_prior_season"

        # 2. depth chart
        d1 = s1_depth.get(gid)
        if d1:
            row.update({"team_s1_end": d1["team"],
                        "pos_rank_s1_end": d1["pos_rank"],
                        "depth_pos_s1_end": d1.get("depth_pos", "")})
        da = asof_depth.get(gid)
        if depth_asof_available:
            if da:
                row.update({"team_asof": da["team"],
                            "pos_rank_asof": da["pos_rank"]})
                if d1:
                    row["pos_rank_delta"] = da["pos_rank"] - d1["pos_rank"]
                    row["changed_team"] = int(da["team"] != d1["team"])
                row["depth_status"] = "ok"
            else:
                row["depth_status"] = "not_derivable"
        else:
            # historical seasons: no dated pre-Sept-1 snapshot source exists;
            # S-1 season-end rank is still emitted where found
            row["depth_status"] = ("s1_end_only" if d1
                                   else "unavailable_pre_asof")

        # 3. vacated opportunity
        if depth_asof_available and da:
            row.update(vacated_by_team.get(da["team"], {}))
            row["vacated_status"] = "ok"
        else:
            row["vacated_status"] = ("not_derivable" if depth_asof_available
                                     else "unavailable_pre_asof")

        # 4. coaching (team = as-of team, else last-known S-1 team)
        team_for_coaching = (da["team"] if da else
                             (s1_totals.get(gid, {}).get("team") or None))
        if ctx.play_callers and team_for_coaching:
            row.update(coaching_change(ctx.play_callers, team_for_coaching,
                                       season))
        else:
            row["coaching_status"] = "not_derivable"

        # 5. pedigree
        sid = ctx.sleeper_id_for(gid if is_real_gsis else "",
                                 seed["player"], seed["position"])
        draft_row = ctx.draft_by_gsis.get(gid) if is_real_gsis else \
            ctx.draft_by_name.get((norm_name(seed["player"]), seed["position"]))
        if draft_row is not None and int(draft_row["season"]) > season:
            draft_row = None  # a future draft class must not identify anyone
        birth = ctx.birth_by_sid.get(sid) if sid else None
        row.update(pedigree(draft_row, birth, season))

        # 6. news
        if not news_covered:
            row["news_status"] = "pre_coverage"
        elif sid is None:
            row["news_status"] = "no_id_match"
        elif sid not in ctx.news_index:
            # missing != zero: absence from the backfill means "not collected",
            # not "no news happened" -- leave counts empty
            row["news_status"] = "not_collected"
        else:
            row.update(news_counts(ctx.news_index[sid], season))
            row["news_status"] = "ok"

        for k, v in row.items():
            if v is None:
                row[k] = ""
        rows_out.append(row)

    return rows_out


# ---------------------------------------------------------------------------
# coverage report

CHANNEL_PROBE = {
    "usage": "opps_pg_s1",
    "role": "adot_s1",
    "role_yoy": "adot_yoy",
    "route_part": "route_part_proxy_s1",
    "depth_s1_end": "pos_rank_s1_end",
    "depth_asof": "pos_rank_asof",
    "vacated": "vacated_opps",
    "coaching": "team_changed_play_caller",
    "age": "age_at_sept1",
    "draft": "undrafted",
    "news": "n_news_90d",
}


def coverage(rows: list[dict]) -> dict[str, float]:
    n = len(rows) or 1
    return {ch: round(100.0 * sum(1 for r in rows if str(r.get(col, "")) != "")
                      / n, 1)
            for ch, col in CHANNEL_PROBE.items()}


# ---------------------------------------------------------------------------
# CLI

def parse_seasons(args) -> list[int]:
    if args.seasons:
        lo, hi = args.seasons.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in args.season]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--season", action="append", default=[],
                   help="season to build (repeatable)")
    b.add_argument("--seasons", help="inclusive range, e.g. 2015-2024")
    args = ap.parse_args(argv)

    seasons = parse_seasons(args)
    if not seasons:
        ap.error("need --season or --seasons")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = Context()
    header = f"{'season':>6} " + " ".join(f"{c:>12}" for c in CHANNEL_PROBE)
    print(header)
    for s in seasons:
        rows = build_season(s, ctx)
        out = OUT_DIR / f"features_{s}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        cov = coverage(rows)
        print(f"{s:>6} " + " ".join(f"{cov[c]:>11.1f}%" for c in CHANNEL_PROBE)
              + f"   ({len(rows)} rows -> {out.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
