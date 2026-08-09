#!/usr/bin/env python3
"""BreakoutBench v0.3 — consolidated anonymized-track benchmark (volume build).

Supersedes evals/anon_demo.py (which stays intact as the original 10-slate
demo) by scaling the same discipline to every (season 2015-2024 x format
standard/ppr/half_ppr) with an ADP snapshot on disk, three question families
per season-format, and season-level bootstrap CIs so 1.x lifts are actually
resolvable (docs/breakoutbench-design.md §3-4).

Families (all probabilistic, all anonymized, all leakage-guarded):
- full_slate  — every gate-eligible candidate (anon_demo's v0.2 ADP gates):
                p(breakout)? The volume workhorse.
- bust        — candidates drafted top-12 positional: p(bust)? (bust per
                labels: finished outside top-24 at position, or absent).
- over_under  — candidates with a finish on record and adp_pos_rank 13-45:
                p(finishes BETTER than ADP-implied positional rank)?

Leakage rules (test-enforced, tests/test_breakoutbench.py):
- feature packets use seasons strictly < eval season + the target season ADP;
- question files carry NO outcome fields and NO player names — outcomes live
  only in the sequestered _KEY.json;
- 2025 is the untouched holdout (GAMEPLAN §9): SEASONS stops at 2024 and the
  CLI refuses --season 2025;
- base-rate baselines use only seasons < the scored season (no-peek,
  auditable via the returned seasons_used).

`build`   -> evals/questions/breakoutbench_<season>_<preset>.json (+ _KEY.json)
`score`   -> per-family + pooled Brier/log-loss, top-10 hit-rate/lift, 95%
             bootstrap CIs (resampling SEASONS — the independent unit), and a
             2015-2019 vs 2020-2024 era split, from answers files
             [{"question_id", "p"}] named *_<season>_<preset>*.json
`baseline`-> writes base-rate (no-peek) + coin answers files and scores them.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.names import norm_name  # noqa: E402

ADP_DIR = ROOT / "data" / "raw" / "ffc_adp"
LABELS = ROOT / "data" / "processed" / "labels"
QUESTIONS_DIR = ROOT / "evals" / "questions"
ANSWERS_DIR = ROOT / "evals" / "results" / "answers"
PACKET_FEATURES_DIR = ROOT / "data" / "processed" / "packet_features"

SEASONS = tuple(range(2015, 2025))  # 2025 is the untouched holdout
HOLDOUT_SEASON = 2025
PRESETS = ("standard", "ppr", "half_ppr")
PRESET_TO_FFC = {"standard": "standard", "ppr": "ppr", "half_ppr": "half-ppr"}
POSITIONS = ("QB", "RB", "WR", "TE")
ADP_GATE = {"QB": 18, "TE": 18, "RB": 40, "WR": 40}  # v0.2 breakout gates
BUST_ADP_GATE = 12
OU_RANK_LO, OU_RANK_HI = 13, 45
FAMILIES = ("full_slate", "bust", "over_under")
TOP_K = 10
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260808
ERAS = (("2015-2019", range(2015, 2020)), ("2020-2024", range(2020, 2025)))
BASE_RATE_FLOOR_SEASON = 2011  # labels start 2011

QUESTION_TEXT = {
    "full_slate": (
        "What is the probability this candidate BREAKS OUT this season "
        "(finishes top-12 QB/TE or top-24 RB/WR at position, by total fantasy "
        "points)? Every candidate in this family is drafted beyond the ADP "
        "gates, so each qualifies if they hit."
    ),
    "bust": (
        "This candidate is drafted top-12 at their position. What is the "
        "probability they BUST (finish outside the top-24 at position, or "
        "record no finish at all)?"
    ),
    "over_under": (
        "What is the probability this candidate finishes BETTER than their "
        "ADP-implied positional rank (realized positional finish strictly "
        "lower-numbered than adp_pos_rank)?"
    ),
}

INSTRUCTIONS = (
    "For each question, return the probability the answer is YES. "
    'Respond with a JSON list [{"question_id": ..., "p": <0..1>}] covering every question.'
)

# --- packet v2 enrichment (evals/packet_features.py output) -----------------
# Only anonymization-safe fields with real historical coverage. Deliberately
# excluded: names, teams, play callers, birth_date (identity leaks), and the
# as-of depth / vacated / coaching channels (null for historical seasons).
PACKET_VERSIONS = ("v1", "v2", "v3")
V2_FIELDS: dict[str, type] = {
    # usage (S-1 weekly stats)
    "opps_pg_s1": float, "target_share_s1": float, "opps_pg_yoy": float,
    # depth chart, S-1 season end
    "pos_rank_s1_end": int,
    # pedigree
    "age_at_sept1": int, "draft_round": int, "draft_pick": int, "undrafted": int,
    # news window counts (90d pre-Sept-1)
    "n_news_90d": int, "camp_promotion_flags": int,
    "injury_mention_flags": int, "hype_flags": int,
}
V2_STATUSES = ("usage_status", "news_status")
V2_SEASON_FLOOR = 2016  # news archive coverage starts ~2016-05

# --- packet v3 = v2 + draft-day-legal team Vegas --------------------------
# Source: data/processed/vegas/team_season_vegas.csv (evals/vegas_features.py).
# ONLY the draft-day-legal pair is ever loaded: the season-S week-1 implied
# team total (look-ahead lines post months early) and the PRIOR season's mean
# implied total. The retrospective season-S mean is never read here — that
# would be leakage (evals/results/vegas_history.md, timing table).
# Team attach follows packet_features conventions: the as-of dated-snapshot
# team when it exists (2025+), else the S-1 season-end depth-chart team.
# Depth charts carry era-native codes (SD/STL/OAK/JAC); the vegas tables use
# current franchise codes retroactively, so we normalize at the join.
VEGAS_TABLE = ROOT / "data" / "processed" / "vegas" / "team_season_vegas.csv"
VEGAS_TEAM_NORMALIZE = {"SD": "LAC", "STL": "LA", "OAK": "LV", "JAC": "JAX"}
V3_FIELDS: dict[str, type] = {
    "team_week1_implied_total": float,
    "team_prior_season_mean_implied_total": float,
}
V3_STATUS = "vegas_status"
_VEGAS_LEGAL_COLUMNS = ("week1_implied_total", "prior_season_mean_implied_total")


@lru_cache(maxsize=None)
def _team_vegas_index() -> dict[tuple[int, str], dict]:
    """(season, team) -> the two draft-day-legal vegas values (None if blank).

    Deliberately drops `mean_implied_total` (retrospective for season S)."""
    out: dict[tuple[int, str], dict] = {}
    with open(VEGAS_TABLE, newline="") as f:
        for r in csv.DictReader(f):
            out[(int(r["season"]), r["team"])] = {
                col: float(r[col]) if r[col] != "" else None
                for col in _VEGAS_LEGAL_COLUMNS
            }
    return out


def vegas_for(season: int, name: str, position: str,
              pid: str | None) -> dict:
    """The v3 team-vegas block for one candidate. Anonymization-safe: emits
    the two numeric values + a status, never the team code itself."""
    out: dict = {k: None for k in V3_FIELDS}
    by_pid, by_name = _packet_feature_index(season)
    row = (by_pid.get(pid) if pid else None) or \
        by_name.get((norm_name(name), position))
    if row is None:
        out[V3_STATUS] = "no_feature_row"
        return out
    team = row.get("team_asof") or row.get("team_s1_end")
    if not team:
        out[V3_STATUS] = "no_team"
        return out
    team = VEGAS_TEAM_NORMALIZE.get(team, team)
    v = _team_vegas_index().get((season, team))
    if v is None:
        out[V3_STATUS] = "no_vegas_row"
        return out
    out["team_week1_implied_total"] = v["week1_implied_total"]
    out["team_prior_season_mean_implied_total"] = \
        v["prior_season_mean_implied_total"]
    out[V3_STATUS] = "ok"
    return out


@lru_cache(maxsize=None)
def _packet_feature_index(season: int) -> tuple[dict, dict]:
    """features_<season>.csv -> (by player_id, by (norm_name, position)).

    Every row is as-of Sept 1 of `season` by construction (asserted)."""
    path = PACKET_FEATURES_DIR / f"features_{season}.csv"
    by_pid: dict[str, dict] = {}
    by_name: dict[tuple[str, str], dict] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            assert int(r["season"]) == season, f"wrong season row in {path.name}"
            assert r["as_of"] == f"{season}-09-01", "leakage: bad as_of gate"
            by_pid[r["player_id"]] = r
            by_name.setdefault((norm_name(r["player"]), r["position"]), r)
    return by_pid, by_name


def enrichment_for(season: int, name: str, position: str,
                   pid: str | None) -> dict:
    """The v2 enrichment block for one candidate. Missing != zero: absent
    values stay null and the *_status fields say why."""
    by_pid, by_name = _packet_feature_index(season)
    row = (by_pid.get(pid) if pid else None) or \
        by_name.get((norm_name(name), position))
    if row is None:
        out: dict = {k: None for k in V2_FIELDS}
        out.update({s: "no_feature_row" for s in V2_STATUSES})
        return out
    out = {}
    for k, cast in V2_FIELDS.items():
        v = row.get(k, "")
        out[k] = cast(float(v)) if v != "" else None
    for s in V2_STATUSES:
        out[s] = row[s]
    return out


# ---------------------------------------------------------------------------
# Data loading


@lru_cache(maxsize=None)
def _season_points() -> dict[str, dict[int, dict]]:
    by_player: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(LABELS / "season_points.csv", newline="") as f:
        for r in csv.DictReader(f):
            by_player[r["player_id"]][int(r["season"])] = r
    return dict(by_player)


@lru_cache(maxsize=None)
def _name_index() -> dict[tuple[str, str], list[str]]:
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, seasons in _season_points().items():
        any_row = next(iter(seasons.values()))
        index[(norm_name(any_row["player"]), any_row["position"])].append(pid)
    return dict(index)


@lru_cache(maxsize=None)
def _breakout_rows() -> list[dict]:
    """breakouts.csv rows, minimally typed. Drafted rows only (the synthetic
    undrafted_in_pool rows have no ADP and are out of scope for this track)."""
    rows = []
    with open(LABELS / "breakouts.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["match_status"] == "undrafted_in_pool" or r["position"] not in POSITIONS:
                continue
            rows.append({
                "season": int(r["season"]),
                "format": r["format"],
                "player": r["player"],
                "position": r["position"],
                "player_id": r["player_id"],
                "adp_pos_rank": int(r["adp_pos_rank"]),
                "finish_pos_rank": int(r["finish_pos_rank"]) if r["finish_pos_rank"] else None,
                "breakout": r["breakout"] == "True",
                "bust": r["bust"] == "True",
            })
    return rows


@lru_cache(maxsize=None)
def _outcome_index() -> dict[tuple[int, str], dict[tuple[str, str, int], dict]]:
    """(season, preset) -> (player_name, position, adp_pos_rank) -> label row.
    Rank is part of the key: it is computed by the identical sort in
    build_breakout_labels.py, so the join is exact."""
    index: dict[tuple[int, str], dict[tuple[str, str, int], dict]] = defaultdict(dict)
    for r in _breakout_rows():
        index[(r["season"], r["format"])][(r["player"], r["position"], r["adp_pos_rank"])] = r
    return dict(index)


def adp_path(preset: str, season: int) -> Path | None:
    fmt = PRESET_TO_FFC[preset]
    for teams in (12, 10, 14, 8):  # existence fallback across team counts
        p = ADP_DIR / f"{fmt}_{teams}t_{season}.json"
        if p.exists():
            return p
    return None


def season_formats(seasons: Sequence[int] = SEASONS) -> list[tuple[int, str]]:
    """Every (season, preset) with an ADP snapshot on disk."""
    return [(s, preset) for s in seasons for preset in PRESETS
            if adp_path(preset, s) is not None]


# ---------------------------------------------------------------------------
# build


def family_of(position: str, adp_pos_rank: int, finish_pos_rank: int | None) -> list[str]:
    """Which families a drafted candidate belongs to (may be several)."""
    fams = []
    if adp_pos_rank > ADP_GATE[position]:
        fams.append("full_slate")
    if adp_pos_rank <= BUST_ADP_GATE:
        fams.append("bust")
    if finish_pos_rank is not None and OU_RANK_LO <= adp_pos_rank <= OU_RANK_HI:
        fams.append("over_under")
    return fams


def outcome_for(family: str, row: Mapping) -> int:
    if family == "full_slate":
        return int(row["breakout"])
    if family == "bust":
        return int(row["bust"])
    if family == "over_under":
        return int(row["finish_pos_rank"] < row["adp_pos_rank"])
    raise ValueError(family)


def build_question_set(season: int, preset: str,
                       packet_version: str = "v1") -> tuple[dict, dict]:
    """(question_payload, key_payload) for one season-format.

    Feature packets are anon_demo's exactly: position, ADP fields, and prior
    seasons strictly < the eval season. Outcomes live only in the key.
    packet_version "v2" adds the anonymization-safe enrichment block from
    data/processed/packet_features (same candidates, same anon ids, same key).
    """
    assert season != HOLDOUT_SEASON, "2025 is the untouched holdout"
    assert packet_version in PACKET_VERSIONS, packet_version
    if packet_version != "v1":
        assert season >= V2_SEASON_FLOOR, \
            f"{packet_version} packets need news coverage (seasons {V2_SEASON_FLOOR}+)"
    path = adp_path(preset, season)
    if path is None:
        raise FileNotFoundError(f"no ADP snapshot for {season}/{preset}")
    payload = json.loads(path.read_text())
    players = [p for p in payload["players"] if p.get("position") in POSITIONS]

    by_pos: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(players, key=lambda p: (float(p["adp"]), -int(p.get("times_drafted", 0)))):
        by_pos[p["position"]].append(p)

    stats = _season_points()
    name_index = _name_index()
    outcomes = _outcome_index().get((season, preset), {})

    entries = []  # (family, packet, key_info)
    for pos in POSITIONS:
        for i, p in enumerate(by_pos.get(pos, [])):
            pos_rank = i + 1
            label = outcomes.get((p["name"], pos, pos_rank))
            if label is None:
                # labels not built for this row (should not happen) — skip
                continue
            fams = family_of(pos, pos_rank, label["finish_pos_rank"])
            if not fams:
                continue

            pids = name_index.get((norm_name(p["name"]), pos), [])
            pid = pids[0] if pids else None
            history = stats.get(pid, {}) if pid else {}
            past = {y: history[y] for y in history if y < season}
            first_year = min(past) if past else None

            def season_block(y: int) -> dict | None:
                r = past.get(y)
                if not r:
                    return None
                return {
                    "games": int(r["games"]),
                    "total": float(r[f"total_{preset}"]),
                    "ppg": float(r[f"ppg_{preset}"]),
                    "pos_rank": int(r[f"pos_season_rank_{preset}"]),
                }

            packet = {
                "position": pos,
                "adp_overall": float(p["adp"]),
                "adp_pos_rank": pos_rank,
                "adp_stdev": float(p.get("stdev") or 0),
                "seasons_of_data": len(past),
                "years_since_first_season": (season - first_year) if first_year else 0,
                "prior_season": season_block(season - 1),
                "two_seasons_ago": season_block(season - 2),
            }
            if packet_version in ("v2", "v3"):
                packet["enrichment"] = enrichment_for(season, p["name"], pos, pid)
            if packet_version == "v3":
                packet["team_vegas"] = vegas_for(season, p["name"], pos, pid)
            for family in fams:
                entries.append((family, packet, {
                    "family": family,
                    "player": p["name"],
                    "player_id": pid,
                    "adp_pos_rank": pos_rank,
                    "finish_pos_rank": label["finish_pos_rank"],
                    "outcome": outcome_for(family, label),
                }))

    rng = random.Random(f"breakoutbench:{season}:{preset}:20260808")
    order = list(range(len(entries)))
    rng.shuffle(order)  # anon ids uncorrelated with board order

    questions, key_questions = [], {}
    for i, ei in enumerate(order):
        family, packet, key_info = entries[ei]
        anon_id = f"B{i:04d}"
        questions.append({
            "question_id": anon_id,
            "family": family,
            "question": QUESTION_TEXT[family],
            "packet": packet,
        })
        key_questions[anon_id] = key_info
    questions.sort(key=lambda q: q["question_id"])

    counts = {fam: sum(1 for q in questions if q["family"] == fam) for fam in FAMILIES}
    question_payload = {
        "benchmark": "breakoutbench_v0.3",
        "variant": "anonymized",
        "packet": packet_version,
        "season_masked": True,
        "format": preset,
        "families": counts,
        "instructions": INSTRUCTIONS,
        "questions": questions,
    }
    key_payload = {
        "benchmark": "breakoutbench_v0.3",
        "season": season,
        "format": preset,
        "questions": key_questions,
    }
    return question_payload, key_payload


def build(seasons: Sequence[int], presets: Sequence[str],
          out_dir: Path = QUESTIONS_DIR,
          packet_version: str = "v1") -> list[tuple[Path, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    totals: dict[str, int] = defaultdict(int)
    for season, preset in season_formats(seasons):
        if preset not in presets:
            continue
        qpayload, kpayload = build_question_set(season, preset, packet_version)
        suffix = "" if packet_version == "v1" else f"_{packet_version}"
        stem = f"breakoutbench_{season}_{preset}"
        qpath = out_dir / f"{stem}{suffix}.json"
        kpath = out_dir / f"{stem}_KEY.json"  # key is shared across versions
        qpath.write_text(json.dumps(qpayload, indent=1) + "\n")
        if kpath.exists():
            # v2 rebuilds must produce the exact same anon-id -> outcome map
            existing = json.loads(kpath.read_text())
            assert existing["questions"] == json.loads(
                json.dumps(kpayload["questions"])), \
                f"key mismatch for {stem}: anon ids diverged"
        else:
            kpath.write_text(json.dumps(kpayload, indent=1) + "\n")
        for fam, n in qpayload["families"].items():
            totals[fam] += n
        written.append((qpath, kpath))
        print(f"built {sum(qpayload['families'].values()):>4} questions "
              f"({', '.join(f'{n} {f}' for f, n in qpayload['families'].items())}) "
              f"-> {qpath.name} (key sequestered)")
    print(f"TOTAL {sum(totals.values())} questions across {len(written)} season-formats: "
          + ", ".join(f"{n} {f}" for f, n in totals.items()))
    return written


# ---------------------------------------------------------------------------
# scoring


def brier(pairs: Sequence[tuple[float, int]]) -> float:
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def log_loss(pairs: Sequence[tuple[float, int]]) -> float:
    eps = 1e-15
    total = 0.0
    for p, y in pairs:
        p = min(max(p, eps), 1 - eps)
        total -= math.log(p) if y else math.log(1 - p)
    return total / len(pairs)


FNAME_RE = re.compile(r"_(\d{4})_(standard|ppr|half_ppr)")


def season_preset_of(path: Path) -> tuple[int, str]:
    m = FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"cannot parse (season, preset) from {path.name}; "
                         "answers files must contain _<season>_<preset>")
    return int(m.group(1)), m.group(2)


def load_answer_files(paths: Sequence[Path],
                      questions_dir: Path = QUESTIONS_DIR) -> dict:
    """Answers + keys -> per-(season, preset) scored pairs.

    Returns {"cells": {(season, preset): {"pairs": {family: [(p, y, anon_id)]},
    "slate": {...top-K inputs...}}, "skipped": int, "answered": int}.
    Duplicate ids keep the first answer; unknown ids and out-of-range p are
    skipped and counted (the model pays for them via missing coverage).
    """
    cells: dict[tuple[int, str], dict] = {}
    skipped = answered = 0
    for path in paths:
        season, preset = season_preset_of(Path(path))
        key_path = questions_dir / f"breakoutbench_{season}_{preset}_KEY.json"
        key = json.loads(key_path.read_text())["questions"]
        cell = cells.setdefault((season, preset), {
            "pairs": defaultdict(list), "n_total": len(key),
            "n_family_total": {f: sum(1 for q in key.values() if q["family"] == f)
                               for f in FAMILIES},
            "slate_base": (sum(q["outcome"] for q in key.values() if q["family"] == "full_slate")
                           / max(1, sum(1 for q in key.values() if q["family"] == "full_slate"))),
        })
        seen: set[str] = set()
        for a in json.loads(Path(path).read_text()):
            qid = str(a["question_id"])
            if qid not in key or qid in seen:
                skipped += 1
                continue
            try:
                p = float(a["p"])
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not 0.0 <= p <= 1.0:
                skipped += 1
                continue
            seen.add(qid)
            answered += 1
            q = key[qid]
            cell["pairs"][q["family"]].append((p, q["outcome"], qid))
    return {"cells": cells, "skipped": skipped, "answered": answered}


def cell_aggregates(cells: Mapping[tuple[int, str], dict], k: int = TOP_K) -> dict:
    """Per-(season, preset): brier/logloss sums per family + top-K hits/expected."""
    agg: dict[tuple[int, str], dict] = {}
    for (season, preset), cell in cells.items():
        fams = {}
        for family, triples in cell["pairs"].items():
            pairs = [(p, y) for p, y, _ in triples]
            fams[family] = {
                "n": len(pairs),
                "brier_sum": sum((p - y) ** 2 for p, y in pairs),
                "ll_sum": log_loss(pairs) * len(pairs) if pairs else 0.0,
            }
        fs = sorted(cell["pairs"].get("full_slate", []), key=lambda t: -t[0])
        top = fs[:k]
        agg[(season, preset)] = {
            "families": fams,
            "hits": sum(y for _, y, _ in top),
            "expected": cell["slate_base"] * len(top),
            "n_top": len(top),
        }
    return agg


def _pool(agg: Mapping[tuple[int, str], dict], seasons: Sequence[int]) -> dict:
    """Pool aggregates over a multiset of seasons (bootstrap resamples repeat)."""
    fam_sums: dict[str, dict] = {f: {"n": 0, "brier_sum": 0.0, "ll_sum": 0.0} for f in FAMILIES}
    hits = expected = 0.0
    for season in seasons:
        for (s, _preset), a in agg.items():
            if s != season:
                continue
            for family, v in a["families"].items():
                fam_sums[family]["n"] += v["n"]
                fam_sums[family]["brier_sum"] += v["brier_sum"]
                fam_sums[family]["ll_sum"] += v["ll_sum"]
            hits += a["hits"]
            expected += a["expected"]
    n_all = sum(v["n"] for v in fam_sums.values())
    return {
        "families": {f: {"n": v["n"],
                         "brier": v["brier_sum"] / v["n"] if v["n"] else None,
                         "log_loss": v["ll_sum"] / v["n"] if v["n"] else None}
                     for f, v in fam_sums.items()},
        "pooled_brier": (sum(v["brier_sum"] for v in fam_sums.values()) / n_all) if n_all else None,
        "pooled_log_loss": (sum(v["ll_sum"] for v in fam_sums.values()) / n_all) if n_all else None,
        "n": n_all,
        "hits": hits,
        "expected": expected,
        "lift": hits / expected if expected else None,
    }


def bootstrap_cis(agg: Mapping[tuple[int, str], dict], reps: int = BOOTSTRAP_REPS,
                  seed: int = BOOTSTRAP_SEED) -> dict:
    """95% CIs by resampling SEASONS with replacement (seasons are the
    independent unit — questions within a season share one realized world).
    Deterministic under seed."""
    seasons = sorted({s for s, _ in agg})
    rng = random.Random(seed)
    stats: dict[str, list[float]] = defaultdict(list)
    for _ in range(reps):
        draw = [seasons[rng.randrange(len(seasons))] for _ in seasons]
        pooled = _pool(agg, draw)
        if pooled["pooled_brier"] is not None:
            stats["pooled_brier"].append(pooled["pooled_brier"])
        if pooled["lift"] is not None:
            stats["lift"].append(pooled["lift"])
        for f in FAMILIES:
            b = pooled["families"][f]["brier"]
            if b is not None:
                stats[f"brier_{f}"].append(b)

    def ci(values: list[float]) -> tuple[float, float] | None:
        if not values:
            return None
        vs = sorted(values)
        lo = vs[int(0.025 * (len(vs) - 1))]
        hi = vs[int(0.975 * (len(vs) - 1))]
        return (lo, hi)

    return {name: ci(vals) for name, vals in stats.items()}


def score_run(paths: Sequence[Path], reps: int = BOOTSTRAP_REPS,
              seed: int = BOOTSTRAP_SEED) -> dict:
    loaded = load_answer_files([Path(p) for p in paths])
    agg = cell_aggregates(loaded["cells"])
    seasons = sorted({s for s, _ in agg})
    result = {
        "answered": loaded["answered"],
        "skipped": loaded["skipped"],
        "n_total": sum(c["n_total"] for c in loaded["cells"].values()),
        "seasons": seasons,
        "season_formats": len(agg),
        "pooled": _pool(agg, seasons),
        "ci": bootstrap_cis(agg, reps, seed),
        "eras": {},
    }
    for era_name, era_seasons in ERAS:
        in_era = [s for s in seasons if s in era_seasons]
        if not in_era:
            continue
        era_agg = {k: v for k, v in agg.items() if k[0] in era_seasons}
        result["eras"][era_name] = {
            "pooled": _pool(era_agg, in_era),
            "ci": bootstrap_cis(era_agg, reps, seed),
        }
    return result


def format_report(result: Mapping, label: str) -> str:
    def fmt_ci(ci: tuple[float, float] | None, digits: int = 3) -> str:
        return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else "—"

    p = result["pooled"]
    lines = [f"=== BreakoutBench v0.3 — {label} ===",
             f"answered {result['answered']}/{result['n_total']} questions "
             f"({result['skipped']} skipped) across {result['season_formats']} season-formats, "
             f"seasons {result['seasons'][0]}-{result['seasons'][-1]}",
             "",
             f"{'family':<12}{'n':>6}{'Brier':>9}{'95% CI':>18}{'logloss':>9}"]
    for f in FAMILIES:
        v = p["families"][f]
        if not v["n"]:
            continue
        lines.append(f"{f:<12}{v['n']:>6}{v['brier']:>9.4f}"
                     f"{fmt_ci(result['ci'].get(f'brier_{f}')):>18}{v['log_loss']:>9.4f}")
    lines.append(f"{'pooled':<12}{p['n']:>6}{p['pooled_brier']:>9.4f}"
                 f"{fmt_ci(result['ci'].get('pooled_brier')):>18}{p['pooled_log_loss']:>9.4f}")
    lines.append("")
    lines.append(f"full_slate top-{TOP_K}: {p['hits']:.0f} hits / {p['expected']:.1f} expected "
                 f"-> lift {p['lift']:.2f}x, 95% CI {fmt_ci(result['ci'].get('lift'), 2)}")
    for era, r in result["eras"].items():
        ep = r["pooled"]
        lines.append(f"  era {era}: Brier {ep['pooled_brier']:.4f} "
                     f"{fmt_ci(r['ci'].get('pooled_brier'))} · "
                     f"lift {ep['lift']:.2f}x {fmt_ci(r['ci'].get('lift'), 2)} (n={ep['n']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# baselines (base-rate no-peek + coin)


def family_base_rates(before_season: int) -> tuple[dict[str, float], list[int]]:
    """Per-family base rates from label rows in seasons STRICTLY < before_season
    (pooled across formats — more data, same no-peek guarantee). Returns
    (rates, seasons_used) so no-peek is auditable."""
    counts: dict[str, list[int]] = defaultdict(list)
    seasons_used: set[int] = set()
    for r in _breakout_rows():
        if not (BASE_RATE_FLOOR_SEASON <= r["season"] < before_season):
            continue
        for family in family_of(r["position"], r["adp_pos_rank"], r["finish_pos_rank"]):
            counts[family].append(outcome_for(family, r))
            seasons_used.add(r["season"])
    assert all(s < before_season for s in seasons_used), "no-peek violation"
    rates = {fam: sum(v) / len(v) for fam, v in counts.items()}
    return rates, sorted(seasons_used)


def write_baseline_answers(seasons: Sequence[int], presets: Sequence[str],
                           out_dir: Path = ANSWERS_DIR,
                           questions_dir: Path = QUESTIONS_DIR) -> dict[str, list[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {"base_rate": [], "coin": []}
    for season, preset in season_formats(seasons):
        if preset not in presets:
            continue
        key_path = questions_dir / f"breakoutbench_{season}_{preset}_KEY.json"
        if not key_path.exists():
            raise FileNotFoundError(f"{key_path} missing; run build first")
        key = json.loads(key_path.read_text())["questions"]
        rates, _ = family_base_rates(before_season=season)
        base = [{"question_id": qid, "p": round(rates[q["family"]], 4)}
                for qid, q in key.items()]
        coin = [{"question_id": qid, "p": 0.5} for qid in key]
        for name, answers in (("base_rate", base), ("coin", coin)):
            path = out_dir / f"{name}_{season}_{preset}.json"
            path.write_text(json.dumps(answers, indent=1) + "\n")
            written[name].append(path)
    return written


# ---------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BreakoutBench v0.3")
    sub = ap.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build all question sets + keys")
    p_build.add_argument("--season", type=int, default=None)
    p_build.add_argument("--format", dest="preset", default=None, choices=PRESETS)
    p_build.add_argument("--packet", default="v1", choices=PACKET_VERSIONS,
                         help="v2 = enriched packets (2016+, needs packet_features); "
                              "v3 = v2 + draft-day-legal team Vegas")

    p_score = sub.add_parser("score", help="score answers files")
    p_score.add_argument("answers", nargs="+", help="answers JSON files (*_<season>_<preset>*.json)")
    p_score.add_argument("--label", default="model run")
    p_score.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    p_score.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)

    p_base = sub.add_parser("baseline", help="write + score base-rate and coin answers")
    p_base.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    p_base.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)

    args = ap.parse_args(argv)

    if args.command == "build":
        if args.season == HOLDOUT_SEASON:
            ap.error(f"{HOLDOUT_SEASON} is the untouched eval holdout (GAMEPLAN §9)")
        if args.packet != "v1":
            default_seasons = [s for s in SEASONS if s >= V2_SEASON_FLOOR]
            if args.season and args.season < V2_SEASON_FLOOR:
                ap.error(f"--packet {args.packet} needs season >= "
                         f"{V2_SEASON_FLOOR} (news coverage)")
        else:
            default_seasons = list(SEASONS)
        seasons = [args.season] if args.season else default_seasons
        presets = [args.preset] if args.preset else list(PRESETS)
        build(seasons, presets, packet_version=args.packet)
    elif args.command == "score":
        paths = [Path(a) for a in args.answers]
        for p in paths:
            if season_preset_of(p)[0] == HOLDOUT_SEASON:
                ap.error(f"{HOLDOUT_SEASON} is the untouched eval holdout (GAMEPLAN §9)")
        print(format_report(score_run(paths, args.reps, args.seed), args.label))
    elif args.command == "baseline":
        written = write_baseline_answers(SEASONS, PRESETS)
        for name in ("base_rate", "coin"):
            print(format_report(score_run(written[name], args.reps, args.seed),
                                f"{name} baseline"))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
