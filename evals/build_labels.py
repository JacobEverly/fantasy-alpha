#!/usr/bin/env python3
"""Derive outcome labels from nflverse weekly stats using the league engine.

Outputs (data/processed/labels/):
- weekly_points.csv: per player-week fantasy points + within-(season,week,position)
  finish rank, under each scoring preset. Feeds CalibBench and DraftGym rewards.
- season_points.csv: per player-season totals, games, ppg + within-(season,position)
  finish rank per preset. Feeds DraftBench and breakout labels (after ADP join).

Regular season only. Positions QB/RB/WR/TE (no K/DST in v0 product scope).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from harness.scoring import PRESETS

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "nflverse" / "stats_player"
OUT = ROOT / "data" / "processed" / "labels"
SEASONS = range(1999, 2026)
POSITIONS = {"QB", "RB", "WR", "TE"}
PRESET_NAMES = ["standard", "half_ppr", "ppr"]


def rank_desc(values: list[tuple[str, float]]) -> dict[str, int]:
    ordered = sorted(values, key=lambda kv: kv[1], reverse=True)
    return {key: i + 1 for i, (key, _) in enumerate(ordered)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    weekly_out = open(OUT / "weekly_points.csv", "w", newline="")
    weekly_cols = [
        "season", "week", "player_id", "player", "position", "team",
        *(f"pts_{p}" for p in PRESET_NAMES),
        *(f"pos_week_rank_{p}" for p in PRESET_NAMES),
    ]
    weekly_writer = csv.DictWriter(weekly_out, fieldnames=weekly_cols)
    weekly_writer.writeheader()

    season_rows: list[dict] = []

    for season in SEASONS:
        path = RAW / f"stats_player_week_{season}.csv"
        rows = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("season_type") != "REG" or r.get("position") not in POSITIONS:
                    continue
                pts = {p: PRESETS[p].score(r) for p in PRESET_NAMES}
                rows.append({
                    "season": season,
                    "week": int(r["week"]),
                    "player_id": r["player_id"],
                    "player": r.get("player_display_name") or r.get("player_name"),
                    "position": r["position"],
                    "team": r.get("team", ""),
                    "pts": pts,
                })

        # weekly finish ranks within (week, position)
        by_week_pos: dict[tuple, list] = defaultdict(list)
        for row in rows:
            by_week_pos[(row["week"], row["position"])].append(row)
        for (week, pos), group in by_week_pos.items():
            for preset in PRESET_NAMES:
                ranks = rank_desc([(r["player_id"], r["pts"][preset]) for r in group])
                for r in group:
                    r[f"pos_week_rank_{preset}"] = ranks[r["player_id"]]

        for r in rows:
            weekly_writer.writerow({
                "season": r["season"], "week": r["week"], "player_id": r["player_id"],
                "player": r["player"], "position": r["position"], "team": r["team"],
                **{f"pts_{p}": r["pts"][p] for p in PRESET_NAMES},
                **{f"pos_week_rank_{p}": r[f"pos_week_rank_{p}"] for p in PRESET_NAMES},
            })

        # season aggregates
        agg: dict[str, dict] = {}
        for r in rows:
            a = agg.setdefault(r["player_id"], {
                "season": season, "player_id": r["player_id"], "player": r["player"],
                "position": r["position"], "teams": set(), "games": 0,
                **{f"total_{p}": 0.0 for p in PRESET_NAMES},
            })
            a["teams"].add(r["team"])
            a["games"] += 1
            a["player"] = r["player"]
            for p in PRESET_NAMES:
                a[f"total_{p}"] = round(a[f"total_{p}"] + r["pts"][p], 2)

        by_pos: dict[str, list] = defaultdict(list)
        for a in agg.values():
            by_pos[a["position"]].append(a)
        for pos, group in by_pos.items():
            for preset in PRESET_NAMES:
                ranks = rank_desc([(a["player_id"], a[f"total_{preset}"]) for a in group])
                for a in group:
                    a[f"pos_season_rank_{preset}"] = ranks[a["player_id"]]

        for a in agg.values():
            a["team"] = "/".join(sorted(t for t in a.pop("teams") if t))
            for p in PRESET_NAMES:
                a[f"ppg_{p}"] = round(a[f"total_{p}"] / a["games"], 2) if a["games"] else 0.0
            season_rows.append(a)

        print(f"{season}: {len(rows)} player-weeks, {len(agg)} player-seasons", flush=True)

    weekly_out.close()

    season_cols = [
        "season", "player_id", "player", "position", "team", "games",
        *(f"total_{p}" for p in PRESET_NAMES),
        *(f"ppg_{p}" for p in PRESET_NAMES),
        *(f"pos_season_rank_{p}" for p in PRESET_NAMES),
    ]
    with open(OUT / "season_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=season_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(season_rows)

    # sanity: recent top finishers, half-PPR
    for season in (2023, 2024, 2025):
        top = [a for a in season_rows if a["season"] == season]
        for pos in ("QB", "RB", "WR", "TE"):
            best = sorted((a for a in top if a["position"] == pos),
                          key=lambda a: a["total_half_ppr"], reverse=True)[:3]
            names = ", ".join(f"{a['player']} {a['total_half_ppr']}" for a in best)
            print(f"sanity {season} {pos}: {names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
