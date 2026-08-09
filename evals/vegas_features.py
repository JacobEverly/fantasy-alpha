#!/usr/bin/env python3
"""Vegas game-line features from the nflverse games.csv snapshot.

Source: dated immutable snapshots in data/raw/vegas/ captured by
scripts/collect_vegas_lines.py (Lee Sharpe's games.csv; per-game closing
spread_line + total_line back to 1999).

TIMING — read this before using anything here in a draft-day packet
--------------------------------------------------------------------
These are *closing game lines*, set per-week a few days before each game.
There is no such thing as a "preseason aggregate of closing lines" that is
knowable on draft day: the season-mean implied total for season S only exists
after season S finishes. Honest knowability per emitted feature:

- per (team, week) implied total          -> knowable a few days BEFORE that
  week's game. Fine for in-season features and weekly sit/start context.
- per (team, season) mean implied total   -> RETROSPECTIVE descriptor. Fine
  as a season-level environment score for evals/labels; never in a
  draft-day packet for that same season.
- draft-day-legal feature                 -> week-1 implied total of season S
  (week-1 look-ahead lines post months early; certainly by late August) plus
  the PRIOR season's mean implied total. Nothing else from this file is
  draft-legal for season S.

Implied team total convention: games.csv `spread_line` is the home team's
expected margin (positive = home favored), `total_line` the game total, so
    home implied = total/2 + spread/2,   away implied = total/2 - spread/2.

Team codes: games.csv uses era-native codes (SD, STL, OAK, JAC), but our
labels / stats_player use current franchise codes retroactively (LAC, LA, LV,
JAX for all seasons). This module normalizes to current codes at load time
(NORMALIZE_TEAM), so the emitted tables join directly against
data/processed/labels/*.csv and data/processed/packet_features/ on
(season, team) / (season, week, team) with no further mapping.

Emitted tables (data/processed/vegas/):
- implied_team_totals.csv   keys (season, week, team); pre-game knowable.
- team_season_vegas.csv     keys (season, team); mean_implied_total is
  retrospective, week1_implied_total + prior_season_mean_implied_total are
  the draft-day-legal pair.

CLI:
    python -m evals.vegas_features --write-tables   # emit processed tables
    python -m evals.vegas_features --study          # walk-forward validation
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_VEGAS = ROOT / "data" / "raw" / "vegas"
PROCESSED = ROOT / "data" / "processed" / "vegas"
LABELS = ROOT / "data" / "processed" / "labels"
STATS_DIR = ROOT / "data" / "raw" / "nflverse" / "stats_player"

# Era-native code -> current franchise code (matches how nflverse
# stats_player and our labels code historical seasons retroactively).
NORMALIZE_TEAM = {"SD": "LAC", "STL": "LA", "OAK": "LV", "JAC": "JAX"}

HOLDOUT_SEASON = 2025  # untouched eval holdout: never an outcome in studies


# ------------------------------------------------------------------ loading

def latest_snapshot(raw_dir: Path = RAW_VEGAS) -> Path:
    snaps = sorted(raw_dir.glob("games_*.csv"))
    if not snaps:
        raise FileNotFoundError(
            f"no games_*.csv snapshot in {raw_dir}; run scripts/collect_vegas_lines.py")
    return snaps[-1]


def load_games(path: Path | None = None) -> list[dict]:
    """Regular-season games with both spread_line and total_line present."""
    path = path or latest_snapshot()
    games = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("game_type") != "REG":
                continue
            if not row.get("spread_line") or not row.get("total_line"):
                continue
            games.append({
                "game_id": row["game_id"],
                "season": int(row["season"]),
                "week": int(row["week"]),
                "gameday": row.get("gameday", ""),
                "home_team": NORMALIZE_TEAM.get(row["home_team"], row["home_team"]),
                "away_team": NORMALIZE_TEAM.get(row["away_team"], row["away_team"]),
                "spread_line": float(row["spread_line"]),
                "total_line": float(row["total_line"]),
            })
    return games


# ----------------------------------------------------------------- features

def implied_team_totals(games: list[dict]) -> list[dict]:
    """Per (season, week, team) implied team total. Knowable pre-game."""
    out = []
    for g in games:
        half_total = g["total_line"] / 2.0
        half_spread = g["spread_line"] / 2.0
        for team, opp, is_home, edge in (
            (g["home_team"], g["away_team"], 1, half_spread),
            (g["away_team"], g["home_team"], 0, -half_spread),
        ):
            out.append({
                "season": g["season"],
                "week": g["week"],
                "team": team,
                "opponent": opp,
                "is_home": is_home,
                "gameday": g["gameday"],
                "total_line": g["total_line"],
                "team_spread": -2 * edge,  # points team is laying (neg = favored)... see note
                "implied_total": round(half_total + edge, 2),
            })
    # team_spread sign convention: negative means the team is favored by that
    # many points (standard sportsbook display for that team).
    return out


def season_mean_implied(weekly: list[dict]) -> dict[tuple[int, str], dict]:
    """Per (season, team) mean implied total. RETROSPECTIVE descriptor."""
    acc: dict[tuple[int, str], list[float]] = defaultdict(list)
    for r in weekly:
        acc[(r["season"], r["team"])].append(r["implied_total"])
    return {
        key: {
            "games_with_lines": len(vals),
            "mean_implied_total": round(statistics.fmean(vals), 3),
        }
        for key, vals in acc.items()
    }


def draft_day_features(weekly: list[dict],
                       season_means: dict[tuple[int, str], dict]
                       ) -> dict[tuple[int, str], dict]:
    """Per (season, team): the draft-day-legal pair for that season.

    week1_implied_total: season-S week-1 line (posted well before drafts).
    prior_season_mean_implied_total: season S-1 mean (codes already
    franchise-normalized). Either may be None (expansion teams, lines not
    yet posted for a future season).
    """
    week1: dict[tuple[int, str], float] = {}
    for r in weekly:
        if r["week"] == 1:
            week1[(r["season"], r["team"])] = r["implied_total"]

    keys = set(week1) | {(s + 1, t) for (s, t) in season_means}
    # keep only (season, team) pairs that exist in the weekly data's seasons
    seasons_seen = {r["season"] for r in weekly}
    out = {}
    for season, team in sorted(keys):
        if season not in seasons_seen:
            continue
        prior_hit = season_means.get((season - 1, team))
        out[(season, team)] = {
            "week1_implied_total": week1.get((season, team)),
            "prior_season_mean_implied_total": (
                prior_hit["mean_implied_total"] if prior_hit else None),
        }
    return out


# ------------------------------------------------------------------- tables

def write_tables(out_dir: Path = PROCESSED, snapshot: Path | None = None) -> dict:
    games = load_games(snapshot)
    weekly = implied_team_totals(games)
    means = season_mean_implied(weekly)
    draft = draft_day_features(weekly, means)

    out_dir.mkdir(parents=True, exist_ok=True)

    weekly_path = out_dir / "implied_team_totals.csv"
    with open(weekly_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "season", "week", "team", "opponent", "is_home", "gameday",
            "total_line", "team_spread", "implied_total"])
        w.writeheader()
        for r in sorted(weekly, key=lambda r: (r["season"], r["week"], r["team"])):
            w.writerow(r)

    season_path = out_dir / "team_season_vegas.csv"
    all_keys = sorted(set(means) | set(draft))
    with open(season_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["season", "team", "games_with_lines", "mean_implied_total",
                    "week1_implied_total", "prior_season_mean_implied_total"])
        for season, team in all_keys:
            m = means.get((season, team), {})
            d = draft.get((season, team), {})
            w.writerow([
                season, team,
                m.get("games_with_lines", 0),
                m.get("mean_implied_total", ""),
                d.get("week1_implied_total", "") if d.get("week1_implied_total") is not None else "",
                d.get("prior_season_mean_implied_total", "") if d.get("prior_season_mean_implied_total") is not None else "",
            ])

    return {"weekly_rows": len(weekly), "season_rows": len(all_keys),
            "weekly_path": str(weekly_path), "season_path": str(season_path)}


# -------------------------------------------------------------------- study

def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _residualize(ys: list[float], xs: list[float]) -> list[float]:
    """Residuals of ys after simple OLS on xs (intercept + slope)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    beta = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx) if sxx else 0.0
    return [y - (my + beta * (x - mx)) for x, y in zip(xs, ys)]


def _bootstrap_ci(pairs: list[tuple[float, float]], n_boot: int = 2000,
                  seed: int = 20260808) -> tuple[float, float, float]:
    """(point, lo95, hi95) for Pearson r over resampled pairs."""
    point = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
    rng = random.Random(seed)
    stats = []
    n = len(pairs)
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        r = _pearson([p[0] for p in sample], [p[1] for p in sample])
        if not math.isnan(r):
            stats.append(r)
    if not stats:  # degenerate inputs (e.g. zero-variance residuals)
        return point, float("nan"), float("nan")
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(int(0.975 * len(stats)), len(stats) - 1)]
    return point, lo, hi


def load_team_fantasy_points(labels_path: Path | None = None,
                             max_week: int = 18) -> dict[tuple[int, str], float]:
    """Realized team output: summed player half-PPR points per (season, team)."""
    labels_path = labels_path or (LABELS / "weekly_points.csv")
    acc: dict[tuple[int, str], float] = defaultdict(float)
    with open(labels_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            week = int(row["week"])
            if week > max_week:
                continue
            team = NORMALIZE_TEAM.get(row["team"], row["team"])
            if not team:
                continue
            acc[(int(row["season"]), team)] += float(row["pts_half_ppr"])
    return dict(acc)


def load_team_offensive_tds(stats_dir: Path = STATS_DIR
                            ) -> dict[tuple[int, str], int]:
    """Team offensive TDs per (season, team): sum of player rushing_tds +
    receiving_tds (receiving counts each passing TD exactly once)."""
    acc: dict[tuple[int, str], int] = defaultdict(int)
    for path in sorted(stats_dir.glob("stats_player_week_*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("season_type") != "REG":
                    continue
                tds = 0
                for col in ("rushing_tds", "receiving_tds"):
                    v = row.get(col)
                    if v:
                        tds += int(float(v))
                if tds:
                    team = NORMALIZE_TEAM.get(row["team"], row["team"])
                    acc[(int(row["season"]), team)] += tds
    return dict(acc)


ERAS = [(2000, 2007), (2008, 2015), (2016, 2024)]


def build_study_rows(weekly: list[dict] | None = None,
                     fantasy: dict | None = None,
                     tds: dict | None = None,
                     last_season: int = HOLDOUT_SEASON - 1) -> list[dict]:
    """Walk-forward rows: one per (season S, team), features knowable before
    season S games, outcomes realized in season S. 2025 (holdout) excluded."""
    weekly = weekly if weekly is not None else implied_team_totals(load_games())
    means = season_mean_implied(weekly)
    draft = draft_day_features(weekly, means)
    fantasy = fantasy if fantasy is not None else load_team_fantasy_points()
    tds = tds if tds is not None else load_team_offensive_tds()

    rows = []
    for (season, team), feats in sorted(draft.items()):
        if season > last_season or season == HOLDOUT_SEASON:
            continue
        naive = fantasy.get((season - 1, team))
        out_pts = fantasy.get((season, team))
        out_tds = tds.get((season, team))
        if (feats["week1_implied_total"] is None
                or feats["prior_season_mean_implied_total"] is None
                or naive is None or out_pts is None or out_tds is None):
            continue
        rows.append({
            "season": season, "team": team,
            "week1_implied": feats["week1_implied_total"],
            "prior_mean_implied": feats["prior_season_mean_implied_total"],
            "naive_prior_points": naive,
            "out_points": out_pts,
            "out_tds": float(out_tds),
        })
    return rows


def run_study(rows: list[dict] | None = None, n_boot: int = 2000) -> dict:
    """Correlations of Vegas features vs realized team output, walk-forward,
    per era, against the naive prior-season-realized-points baseline; plus an
    incremental test (corr of Vegas feature with the naive model's residual)."""
    rows = rows if rows is not None else build_study_rows()
    features = ["prior_mean_implied", "week1_implied", "naive_prior_points"]
    outcomes = ["out_points", "out_tds"]
    results: dict = {"n_rows": len(rows), "eras": {}}

    for lo, hi in ERAS + [(ERAS[0][0], ERAS[-1][1])]:
        era_rows = [r for r in rows if lo <= r["season"] <= hi]
        label = f"{lo}-{hi}"
        era_out: dict = {"n": len(era_rows)}
        for outcome in outcomes:
            block: dict = {}
            for feat in features:
                pairs = [(r[feat], r[outcome]) for r in era_rows]
                block[feat] = _bootstrap_ci(pairs, n_boot=n_boot)
            # incremental value of each Vegas feature over the naive baseline
            ys = [r[outcome] for r in era_rows]
            xs = [r["naive_prior_points"] for r in era_rows]
            resid = _residualize(ys, xs)
            for feat in ("prior_mean_implied", "week1_implied"):
                pairs = list(zip([r[feat] for r in era_rows], resid))
                block[f"{feat}_incremental"] = _bootstrap_ci(pairs, n_boot=n_boot)
            era_out[outcome] = block
        results["eras"][label] = era_out
    return results


def _fmt(ci: tuple[float, float, float]) -> str:
    return f"{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-tables", action="store_true")
    ap.add_argument("--study", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)

    if args.write_tables:
        info = write_tables()
        print(f"wrote {info['weekly_rows']} weekly rows -> {info['weekly_path']}")
        print(f"wrote {info['season_rows']} season rows -> {info['season_path']}")
    if args.study:
        res = run_study(n_boot=args.n_boot)
        print(f"study rows: {res['n_rows']} (team-seasons, walk-forward, holdout 2025 excluded)")
        for era, block in res["eras"].items():
            print(f"\n=== {era} (n={block['n']}) ===")
            for outcome in ("out_points", "out_tds"):
                print(f"  outcome {outcome}:")
                for feat, ci in block[outcome].items():
                    print(f"    {feat:38s} r = {_fmt(ci)}")
    if not (args.write_tables or args.study):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
