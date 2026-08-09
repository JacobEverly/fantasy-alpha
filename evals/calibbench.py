#!/usr/bin/env python3
"""CalibBench v0 — dated forecast questions with realized outcomes (GAMEPLAN §4.2).

Two question families per (season, format), generated deterministically from
snapshots already on disk (FFC ADP boards + label CSVs):

1. season_threshold — "Will this player finish top-12 (QB/TE) / top-24 (RB/WR)
   at position this season?" One question per QB/RB/WR/TE on the season's ADP
   board. The information packet is anon_demo-style: prior-season stats only
   (strictly seasons < target) plus the target season's ADP — nothing from the
   season being predicted.
2. weekly_h2h — "Will player A outscore player B in week W?" ~50 pairs per
   season: same position, both actually played week W (weeks 5-16), and
   season-to-date ppg entering W within 20% of each other (a real coin-flip-ish
   matchup, so calibration is what separates models). Packet per side: games,
   total, ppg strictly from weeks < W, plus position and week.

`build` writes three files to evals/questions/:
- calibbench_<season>_<preset>.json        named variant (internal regression
  only — contaminated by pretraining, never published; see
  docs/breakoutbench-design.md §2)
- calibbench_<season>_<preset>_anon.json   anonymized variant: shuffled anon
  ids, no names, season masked (the structure track)
- calibbench_<season>_<preset>_KEY.json    outcomes + anon-id mapping. The
  scorer uses it; the model must never see it. Question files contain NO
  outcome fields.

`score` reads an answers JSON [{"question_id": ..., "p": ...}] (named or anon
ids, mixable) and reports Brier, log-loss, a 10-bin reliability table, and ECE
per family. `baseline` scores two reference forecasters: coin (p=0.5) and
base-rate (per-family base rate computed from 2015-2023 excluding the scored
season — it never peeks at the season it is scored on).

IMPORTANT: 2025 is the untouched eval holdout (GAMEPLAN §4/§9). Every
subcommand refuses --season 2025 without an explicit --include-holdout, which
is for gate-time evaluation only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from evals.names import norm_name

ROOT = Path(__file__).resolve().parent.parent
ADP_DIR = ROOT / "data" / "raw" / "ffc_adp"
LABELS = ROOT / "data" / "processed" / "labels"
QUESTIONS_DIR = ROOT / "evals" / "questions"

POSITIONS = ("QB", "RB", "WR", "TE")
PRESET_TO_FFC = {"standard": "standard", "half_ppr": "half-ppr", "ppr": "ppr"}
FINISH_THRESHOLD = {"QB": 12, "TE": 12, "RB": 24, "WR": 24}

# 2025 is the untouched holdout (GAMEPLAN §4/§9): base rates additionally stop
# at 2023 so the base-rate baseline can never peek at 2024-2025 outcomes when
# scoring those seasons either.
HOLDOUT_SEASON = 2025
BASE_RATE_SEASONS = tuple(range(2015, 2024))

# weekly_h2h generation knobs (pinned; changing them is a new benchmark version)
H2H_WEEKS = range(5, 17)  # weeks 5-16: enough season-to-date signal, no week-17 rest games
H2H_MIN_GAMES = 3  # games entering W required for a meaningful ppg
H2H_MIN_PPG = 5.0  # both sides fantasy-relevant, not scrub-vs-scrub noise
H2H_PPG_TOLERANCE = 0.20  # |ppg_a - ppg_b| <= 20% of the larger
H2H_DEFAULT_PAIRS = 300  # v0.1: 50 -> 300 (statistical power; ~1000 available per season)

SEASON_QUESTION = (
    "Will this player finish top-{threshold} at {position} this season "
    "(positional rank by total fantasy points)?"
)
H2H_QUESTION = "Will player A outscore player B in week {week}?"


def _rng(season: int, preset: str) -> random.Random:
    return random.Random(f"calibbench:{season}:{preset}:20260808")


# ---------------------------------------------------------------------------
# Label loading (cached; shared across seasons)


@lru_cache(maxsize=None)
def _season_points() -> dict[str, dict[int, dict]]:
    """nflverse player_id -> season -> season_points.csv row."""
    by_player: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(LABELS / "season_points.csv", newline="") as f:
        for r in csv.DictReader(f):
            by_player[r["player_id"]][int(r["season"])] = r
    return dict(by_player)


@lru_cache(maxsize=None)
def _name_index() -> dict[tuple[str, str], list[str]]:
    """(norm_name, position) -> nflverse player_ids (anon_demo's join)."""
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, seasons in _season_points().items():
        any_row = next(iter(seasons.values()))
        index[(norm_name(any_row["player"]), any_row["position"])].append(pid)
    return dict(index)


@lru_cache(maxsize=None)
def _weekly_by_season() -> dict[int, dict[str, dict]]:
    """season -> player_id -> {name, position, weeks: {week: pts-per-preset}}."""
    out: dict[int, dict[str, dict]] = defaultdict(dict)
    with open(LABELS / "weekly_points.csv", newline="") as f:
        for r in csv.DictReader(f):
            player = out[int(r["season"])].setdefault(r["player_id"], {
                "name": r["player"], "position": r["position"], "weeks": {},
            })
            player["weeks"][int(r["week"])] = {
                p: float(r[f"pts_{p}"]) for p in PRESET_TO_FFC
            }
    return dict(out)


def _adp_path(preset: str, season: int) -> Path | None:
    fmt = PRESET_TO_FFC[preset]
    for teams in (12, 10, 14, 8):  # historical years only have *_8t_* (coerced to 12-team data)
        p = ADP_DIR / f"{fmt}_{teams}t_{season}.json"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Family 1: season_threshold


def season_to_date(weeks: Mapping[int, float], upto_week: int) -> tuple[int, float, float]:
    """(games, total, ppg) from weeks strictly before upto_week. The only
    aggregation the weekly_h2h packets are allowed to use."""
    played = [pts for w, pts in weeks.items() if w < upto_week]
    games = len(played)
    total = round(sum(played), 2)
    ppg = round(total / games, 2) if games else 0.0
    return games, total, ppg


def build_season_questions(season: int, preset: str) -> list[dict]:
    """One question per QB/RB/WR/TE on the season's ADP board.

    Packet features come from seasons < target plus the target season's ADP
    (anon_demo's as-of discipline). Outcome: realized positional finish inside
    the position's threshold; board players who never appeared score 0.
    """
    path = _adp_path(preset, season)
    if path is None:
        raise FileNotFoundError(f"no ADP snapshot for {season}/{preset}")
    payload = json.loads(path.read_text())
    players = [p for p in payload["players"] if p.get("position") in POSITIONS]

    by_pos: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(players, key=lambda p: (float(p["adp"]), -int(p.get("times_drafted", 0)))):
        by_pos[p["position"]].append(p)

    stats = _season_points()
    name_index = _name_index()

    questions = []
    for pos in POSITIONS:
        for i, p in enumerate(by_pos.get(pos, [])):
            pos_rank = i + 1
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

            this_year = history.get(season) if pid else None
            threshold = FINISH_THRESHOLD[pos]
            outcome = int(
                this_year is not None
                and int(this_year[f"pos_season_rank_{preset}"]) <= threshold
            )
            questions.append({
                "question_id": f"S{season}:{preset}:{p['player_id']}",
                "family": "season_threshold",
                "question": SEASON_QUESTION.format(threshold=threshold, position=pos),
                "player": p["name"],
                "nflverse_id": pid,
                "packet": {
                    "position": pos,
                    "threshold": threshold,
                    "adp_overall": float(p["adp"]),
                    "adp_pos_rank": pos_rank,
                    "adp_stdev": float(p.get("stdev") or 0),
                    "seasons_of_data": len(past),
                    "years_since_first_season": (season - first_year) if first_year else 0,
                    "prior_season": season_block(season - 1),
                    "two_seasons_ago": season_block(season - 2),
                },
                "outcome": outcome,
            })
    return questions


# ---------------------------------------------------------------------------
# Family 2: weekly_h2h


def build_h2h_questions(season: int, preset: str, n_pairs: int = H2H_DEFAULT_PAIRS) -> list[dict]:
    """~n_pairs same-position head-to-heads across weeks 5-16.

    Eligibility per (week, position): played week W, >= H2H_MIN_GAMES games
    entering W, season-to-date ppg >= H2H_MIN_PPG, and the pair's ppg within
    20% of each other. Pairing and the final draw are seeded — same season,
    same file, every run. Outcome: pts_a > pts_b in week W (a tie scores 0).
    """
    players = _weekly_by_season().get(season, {})
    rng = _rng(season, preset)

    all_pairs: list[dict] = []
    for week in H2H_WEEKS:
        by_pos: dict[str, list[tuple[str, dict, tuple]]] = defaultdict(list)
        for pid in sorted(players):  # sorted -> deterministic before shuffling
            info = players[pid]
            weeks = {w: pts[preset] for w, pts in info["weeks"].items()}
            if week not in weeks:
                continue
            games, total, ppg = season_to_date(weeks, week)
            if games < H2H_MIN_GAMES or ppg < H2H_MIN_PPG:
                continue
            by_pos[info["position"]].append((pid, info, (games, total, ppg)))

        for pos in POSITIONS:
            group = by_pos.get(pos, [])
            rng.shuffle(group)
            used: set[str] = set()
            for i, (pid_a, info_a, (g_a, t_a, ppg_a)) in enumerate(group):
                if pid_a in used:
                    continue
                for pid_b, info_b, (g_b, t_b, ppg_b) in group[i + 1:]:
                    if pid_b in used:
                        continue
                    if abs(ppg_a - ppg_b) > H2H_PPG_TOLERANCE * max(ppg_a, ppg_b):
                        continue
                    used.update((pid_a, pid_b))
                    pts_a = players[pid_a]["weeks"][week][preset]
                    pts_b = players[pid_b]["weeks"][week][preset]
                    all_pairs.append({
                        "question_id": f"W{season}:{preset}:w{week:02d}:{pid_a}:{pid_b}",
                        "family": "weekly_h2h",
                        "question": H2H_QUESTION.format(week=week),
                        "player_a": info_a["name"],
                        "player_b": info_b["name"],
                        "packet": {
                            "position": pos,
                            "week": week,
                            "a": {"games": g_a, "total": t_a, "ppg": ppg_a},
                            "b": {"games": g_b, "total": t_b, "ppg": ppg_b},
                        },
                        "outcome": int(pts_a > pts_b),
                    })
                    break

    rng.shuffle(all_pairs)
    return all_pairs[:n_pairs]


# ---------------------------------------------------------------------------
# Question-set assembly + serialization


def build_question_set(season: int, preset: str, n_pairs: int = H2H_DEFAULT_PAIRS) -> tuple[dict, dict, dict]:
    """(named_file_payload, anon_file_payload, key_payload) for one season+format.

    Outcomes live ONLY in the key payload; both question payloads are
    outcome-free. The anon payload additionally strips names and the season.
    """
    questions = build_season_questions(season, preset) + build_h2h_questions(season, preset, n_pairs)

    order = list(range(len(questions)))
    _rng(season, preset).shuffle(order)  # anon ids uncorrelated with board order
    anon_ids = {questions[qi]["question_id"]: f"Q{i:04d}" for i, qi in enumerate(order)}

    instructions = (
        "For each question, return the probability the answer is YES. "
        'Respond with a JSON list [{"question_id": ..., "p": <0..1>}] covering every question.'
    )
    named_questions, anon_questions, key_questions = [], [], {}
    for q in questions:
        qid = q["question_id"]
        named = {k: v for k, v in q.items() if k not in ("outcome", "nflverse_id")}
        anon = {
            "question_id": anon_ids[qid],
            "family": q["family"],
            "question": q["question"],
            "packet": q["packet"],
        }
        named_questions.append(named)
        anon_questions.append(anon)
        key_questions[qid] = {
            "anon_id": anon_ids[qid],
            "family": q["family"],
            "outcome": q["outcome"],
            **({"player": q["player"], "nflverse_id": q["nflverse_id"]}
               if q["family"] == "season_threshold"
               else {"player_a": q["player_a"], "player_b": q["player_b"]}),
        }
    anon_questions.sort(key=lambda q: q["question_id"])

    counts = {fam: sum(1 for q in questions if q["family"] == fam)
              for fam in ("season_threshold", "weekly_h2h")}
    named_payload = {
        "benchmark": "calibbench_v0",
        "variant": "named",
        "season": season,
        "format": preset,
        "families": counts,
        "instructions": instructions,
        "questions": named_questions,
    }
    anon_payload = {
        "benchmark": "calibbench_v0",
        "variant": "anonymized",
        "season_masked": True,
        "format": preset,
        "families": counts,
        "instructions": instructions,
        "questions": anon_questions,
    }
    key_payload = {
        "benchmark": "calibbench_v0",
        "season": season,
        "format": preset,
        "questions": key_questions,
    }
    return named_payload, anon_payload, key_payload


def build(season: int, preset: str, n_pairs: int = H2H_DEFAULT_PAIRS,
          out_dir: Path = QUESTIONS_DIR) -> tuple[Path, Path, Path]:
    named, anon, key = build_question_set(season, preset, n_pairs)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"calibbench_{season}_{preset}"
    paths = (out_dir / f"{stem}.json", out_dir / f"{stem}_anon.json", out_dir / f"{stem}_KEY.json")
    for path, payload in zip(paths, (named, anon, key)):
        path.write_text(json.dumps(payload, indent=1) + "\n")
    counts = named["families"]
    shown = paths[0].relative_to(ROOT) if paths[0].is_relative_to(ROOT) else paths[0]
    print(f"built {sum(counts.values())} questions "
          f"({counts['season_threshold']} season_threshold, {counts['weekly_h2h']} weekly_h2h) "
          f"-> {shown} (+ _anon, key sequestered)")
    return paths


# ---------------------------------------------------------------------------
# Scoring: proper scoring rules + reliability


def brier(pairs: Sequence[tuple[float, int]]) -> float:
    """Mean squared error of probability vs 0/1 outcome. 0.25 = coin on a
    50% base rate; lower is better."""
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def log_loss(pairs: Sequence[tuple[float, int]]) -> float:
    eps = 1e-15
    total = 0.0
    for p, y in pairs:
        p = min(max(p, eps), 1 - eps)
        total -= math.log(p) if y else math.log(1 - p)
    return total / len(pairs)


def reliability_table(pairs: Sequence[tuple[float, int]], n_bins: int = 10) -> list[dict]:
    """Per-bin (n, mean forecast, realized frequency); p=1.0 lands in the top bin."""
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    table = []
    for i, members in enumerate(bins):
        table.append({
            "bin": f"[{i / n_bins:.1f},{(i + 1) / n_bins:.1f})",
            "n": len(members),
            "mean_p": round(sum(p for p, _ in members) / len(members), 4) if members else None,
            "freq": round(sum(y for _, y in members) / len(members), 4) if members else None,
        })
    return table


def ece(pairs: Sequence[tuple[float, int]], n_bins: int = 10) -> float:
    """Expected calibration error: bin-weight-averaged |mean_p - freq|."""
    total = 0.0
    for row in reliability_table(pairs, n_bins):
        if row["n"]:
            total += row["n"] / len(pairs) * abs(row["mean_p"] - row["freq"])
    return total


def score_answers(answers: Sequence[Mapping], key: Mapping) -> dict[str, dict]:
    """Answers [{question_id, p}] (named or anon ids) vs a key payload.

    Returns {family: {n, brier, log_loss, ece, reliability}} plus 'overall'.
    Unknown question ids raise; every p must be in [0, 1].
    """
    key_questions = key["questions"]
    by_anon = {q["anon_id"]: qid for qid, q in key_questions.items()}
    pairs_by_family: dict[str, list[tuple[float, int]]] = defaultdict(list)
    seen: set[str] = set()
    for a in answers:
        qid = a["question_id"]
        qid = by_anon.get(qid, qid)
        if qid not in key_questions:
            raise KeyError(f"unknown question_id {a['question_id']!r}")
        if qid in seen:
            raise ValueError(f"duplicate answer for {a['question_id']!r}")
        seen.add(qid)
        p = float(a["p"])
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p={p} out of [0,1] for {a['question_id']!r}")
        pairs_by_family[key_questions[qid]["family"]].append((p, key_questions[qid]["outcome"]))

    results = {}
    all_pairs: list[tuple[float, int]] = []
    for family, pairs in sorted(pairs_by_family.items()):
        all_pairs.extend(pairs)
        results[family] = {
            "n": len(pairs),
            "brier": round(brier(pairs), 4),
            "log_loss": round(log_loss(pairs), 4),
            "ece": round(ece(pairs), 4),
            "reliability": reliability_table(pairs),
        }
    results["overall"] = {
        "n": len(all_pairs),
        "brier": round(brier(all_pairs), 4),
        "log_loss": round(log_loss(all_pairs), 4),
        "ece": round(ece(all_pairs), 4),
        "reliability": reliability_table(all_pairs),
    }
    return results


def format_report(results: Mapping[str, Mapping], label: str) -> str:
    lines = [f"=== CalibBench v0 — {label} ==="]
    lines.append(f"{'family':<18}{'n':>6}{'Brier':>9}{'logloss':>9}{'ECE':>8}")
    for family, r in results.items():
        lines.append(f"{family:<18}{r['n']:>6}{r['brier']:>9.4f}{r['log_loss']:>9.4f}{r['ece']:>8.4f}")
    lines.append("")
    lines.append("reliability (overall, 10 bins): bin  n  mean_p  freq")
    for row in results["overall"]["reliability"]:
        if row["n"]:
            lines.append(f"  {row['bin']:<10}{row['n']:>5}  {row['mean_p']:.3f}  {row['freq']:.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baselines: coin + family base rate (no-peek)


def family_base_rates(preset: str, exclude_season: int) -> tuple[dict[str, float], list[int]]:
    """Per-family outcome base rates over BASE_RATE_SEASONS minus the scored
    season. Seasons without an ADP snapshot for this preset are skipped.
    Returns (rates, seasons_used) so no-peek is auditable."""
    counts: dict[str, list[int]] = defaultdict(list)
    seasons_used: list[int] = []
    for season in BASE_RATE_SEASONS:
        if season == exclude_season or _adp_path(preset, season) is None:
            continue
        seasons_used.append(season)
        for q in build_season_questions(season, preset):
            counts["season_threshold"].append(q["outcome"])
        for q in build_h2h_questions(season, preset):
            counts["weekly_h2h"].append(q["outcome"])
    rates = {fam: sum(outcomes) / len(outcomes) for fam, outcomes in counts.items()}
    return rates, seasons_used


def run_baselines(season: int, preset: str, n_pairs: int = H2H_DEFAULT_PAIRS,
                  verbose: bool = True) -> dict[str, dict]:
    """Score coin (p=0.5) and family-base-rate forecasters on the season's
    question set (regenerated in memory — build not required first)."""
    _, _, key = build_question_set(season, preset, n_pairs)
    rates, seasons_used = family_base_rates(preset, exclude_season=season)

    coin = [{"question_id": qid, "p": 0.5} for qid in key["questions"]]
    base = [{"question_id": qid, "p": rates[q["family"]]}
            for qid, q in key["questions"].items()]
    results = {
        "coin": score_answers(coin, key),
        "base_rate": score_answers(base, key),
    }
    if verbose:
        print(f"base rates from seasons {seasons_used[0]}-{seasons_used[-1]} "
              f"excluding {season} (n={len(seasons_used)}): "
              + ", ".join(f"{fam}={rate:.4f}" for fam, rate in sorted(rates.items())))
        print()
        print(f"=== CalibBench v0 baselines — season {season} ({preset}) ===")
        header = f"{'family':<18}{'n':>6}" + "".join(
            f"{name + ' ' + m:>18}" for name in ("coin", "base_rate") for m in ("Brier", "ECE"))
        print(header)
        for family in (*sorted(k for k in results["coin"] if k != "overall"), "overall"):
            row = f"{family:<18}{results['coin'][family]['n']:>6}"
            for name in ("coin", "base_rate"):
                r = results[name][family]
                row += f"{r['brier']:>18.4f}{r['ece']:>18.4f}"
            print(row)
        print("(model answers score the same way: "
              f".venv/bin/python -m evals.calibbench score --season {season} "
              f"--format {preset} --answers <file>)")
    return results


def run_baselines_pooled(seasons: Sequence[int], preset: str,
                         n_pairs: int = H2H_DEFAULT_PAIRS,
                         verbose: bool = True) -> dict[str, dict]:
    """Coin + base-rate forecasters scored on the POOLED question set across
    seasons (each season's base rate still excludes that season — no-peek)."""
    pooled_answers: dict[str, list[dict]] = {"coin": [], "base_rate": []}
    pooled_key: dict[str, dict] = {}
    for season in seasons:
        if _adp_path(preset, season) is None:
            continue
        _, _, key = build_question_set(season, preset, n_pairs)
        rates, _ = family_base_rates(preset, exclude_season=season)
        for qid, q in key["questions"].items():
            pooled_key[qid] = q
            pooled_answers["coin"].append({"question_id": qid, "p": 0.5})
            pooled_answers["base_rate"].append({"question_id": qid, "p": rates[q["family"]]})
    results = {name: score_answers(answers, {"questions": pooled_key})
               for name, answers in pooled_answers.items()}
    if verbose:
        for name in ("coin", "base_rate"):
            print(format_report(
                results[name],
                f"{name} baseline pooled {seasons[0]}-{seasons[-1]} ({preset}, "
                f"{n_pairs} h2h pairs/season)"))
            print()
    return results


# ---------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CalibBench v0 — calibration benchmark")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--season", type=int, default=None)
        p.add_argument("--all-seasons", action="store_true",
                       help="run over every non-holdout season 2015-2024 with an ADP snapshot")
        p.add_argument("--format", dest="preset", default="ppr", choices=sorted(PRESET_TO_FFC))
        p.add_argument(
            "--include-holdout", action="store_true",
            help="allow the 2025 holdout season (gate-time evaluation ONLY — see GAMEPLAN §9)",
        )

    p_build = sub.add_parser("build", help="write question set + key to evals/questions/")
    common(p_build)
    p_build.add_argument("--h2h-pairs", type=int, default=H2H_DEFAULT_PAIRS)

    p_score = sub.add_parser("score", help="score an answers JSON against the key")
    common(p_score)
    p_score.add_argument("--answers", required=True, help='JSON [{"question_id": ..., "p": ...}]')

    p_base = sub.add_parser("baseline", help="score coin and base-rate forecasters")
    common(p_base)
    p_base.add_argument("--h2h-pairs", type=int, default=H2H_DEFAULT_PAIRS)

    args = ap.parse_args(argv)
    if args.season == HOLDOUT_SEASON and not args.include_holdout:
        ap.error(f"{HOLDOUT_SEASON} is the untouched eval holdout (GAMEPLAN §9); "
                 "pass --include-holdout only at gate time")
    if args.season is None and not getattr(args, "all_seasons", False):
        ap.error("--season or --all-seasons is required")
    all_seasons = [s for s in range(2015, HOLDOUT_SEASON)
                   if _adp_path(args.preset, s) is not None]

    if args.command == "build":
        if args.all_seasons:
            for season in all_seasons:
                build(season, args.preset, args.h2h_pairs)
        else:
            build(args.season, args.preset, args.h2h_pairs)
    elif args.command == "score":
        if args.all_seasons:
            ap.error("score needs a single --season")
        key_path = QUESTIONS_DIR / f"calibbench_{args.season}_{args.preset}_KEY.json"
        if not key_path.exists():
            ap.error(f"no key at {key_path}; run build first")
        key = json.loads(key_path.read_text())
        answers = json.loads(Path(args.answers).read_text())
        results = score_answers(answers, key)
        print(format_report(results, f"season {args.season} ({args.preset}) — {args.answers}"))
    elif args.command == "baseline":
        if args.all_seasons:
            run_baselines_pooled(all_seasons, args.preset, args.h2h_pairs)
        else:
            run_baselines(args.season, args.preset, args.h2h_pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
