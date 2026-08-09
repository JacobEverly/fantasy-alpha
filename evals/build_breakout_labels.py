#!/usr/bin/env python3
"""Join FFC ADP history to realized season outcomes -> breakout/bust/alpha labels.

Per (season, format): ADP positional rank vs realized positional finish
(season_points.csv), using the preregistered BreakoutBench definitions
(docs/breakoutbench-design.md):
  alpha    = adp_pos_rank - finish_pos_rank
  breakout = finished top-12 (QB/TE) / top-24 (RB/WR) with ADP outside
             top-24 (QB/TE) / top-40 (RB/WR) at position
  bust     = drafted top-12 positional, finished outside top-24
             (games < 8 -> flagged injury_bust)

Name matching is normalized name+position with team tiebreak; unmatched ADP
players (mostly non-appearing rookies/retirees) get finish=None and count as
busts only when drafted top-12 positional.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from evals.names import norm_name

ROOT = Path(__file__).resolve().parent.parent
ADP_DIR = ROOT / "data" / "raw" / "ffc_adp"
LABELS = ROOT / "data" / "processed" / "labels"

SEASONS = range(2007, 2026)  # ADP-bound; seasons without ADP files skip gracefully
FORMATS = {"standard": "standard", "ppr": "ppr", "half-ppr": "half_ppr"}  # ffc -> preset
POSITIONS = {"QB", "RB", "WR", "TE"}
FINISH_GATE = {"QB": 12, "TE": 12, "RB": 24, "WR": 24}
ADP_GATE = {"QB": 18, "TE": 18, "RB": 40, "WR": 40}  # v0.2: QB/TE 24->18 (pools rarely go 24 deep)
BUST_ADP_GATE = 12
BUST_FINISH_GATE = 24


def load_season_points() -> dict[int, list[dict]]:
    by_season: dict[int, list[dict]] = defaultdict(list)
    with open(LABELS / "season_points.csv", newline="") as f:
        for r in csv.DictReader(f):
            r["season"] = int(r["season"])
            r["games"] = int(r["games"])
            by_season[r["season"]].append(r)
    return by_season


def adp_file(fmt: str, year: int) -> Path | None:
    for teams in (12, 10, 14, 8):
        p = ADP_DIR / f"{fmt}_{teams}t_{year}.json"
        if p.exists():
            return p
    return None


def main() -> int:
    season_points = load_season_points()
    out_rows: list[dict] = []
    unmatched_log: list[str] = []

    for year in SEASONS:
        stats = season_points.get(year, [])
        by_key: dict[tuple, list[dict]] = defaultdict(list)
        for r in stats:
            by_key[(norm_name(r["player"]), r["position"])].append(r)

        for fmt, preset in FORMATS.items():
            path = adp_file(fmt, year)
            if path is None:
                continue
            payload = json.loads(path.read_text())
            players = [p for p in payload["players"] if p.get("position") in POSITIONS]

            # ADP positional ranks
            by_pos: dict[str, list[dict]] = defaultdict(list)
            for p in sorted(players, key=lambda p: (float(p["adp"]), -int(p.get("times_drafted", 0)))):
                by_pos[p["position"]].append(p)
            for pos, group in by_pos.items():
                for i, p in enumerate(group):
                    p["_adp_pos_rank"] = i + 1

            for p in players:
                pos = p["position"]
                candidates = by_key.get((norm_name(p["name"]), pos), [])
                match = None
                if len(candidates) == 1:
                    match = candidates[0]
                elif len(candidates) > 1:
                    team_hits = [c for c in candidates if p.get("team") and p["team"] in c["team"].split("/")]
                    match = team_hits[0] if team_hits else candidates[0]
                else:
                    unmatched_log.append(f"{year} {fmt} {p['name']} ({pos}, {p.get('team')})")

                finish = int(match[f"pos_season_rank_{preset}"]) if match else None
                games = match["games"] if match else 0
                adp_rank = p["_adp_pos_rank"]
                alpha = (adp_rank - finish) if finish else None
                breakout = bool(finish and finish <= FINISH_GATE[pos] and adp_rank > ADP_GATE[pos])
                bust = bool(adp_rank <= BUST_ADP_GATE and (finish is None or finish > BUST_FINISH_GATE))
                out_rows.append({
                    "season": year,
                    "format": preset,
                    "player": p["name"],
                    "position": pos,
                    "adp_team": p.get("team", ""),
                    "player_id": match["player_id"] if match else "",
                    "match_status": "matched" if match else "unmatched",
                    "adp": p["adp"],
                    "adp_pos_rank": adp_rank,
                    "finish_pos_rank": finish if finish is not None else "",
                    "games": games,
                    "alpha": alpha if alpha is not None else "",
                    "breakout": breakout,
                    "bust": bust,
                    "injury_bust": bool(bust and match and games < 8),
                })

            # v0.2: top finishers absent from the ADP pool entirely (e.g. Puka
            # Nacua 2023) are the deepest breakouts — synthesize their rows.
            matched_ids = {r["player_id"] for r in out_rows
                           if r["season"] == year and r["format"] == preset and r["player_id"]}
            pool_depth = {pos: len(group) for pos, group in by_pos.items()}
            for s in stats:
                pos = s["position"]
                finish = int(s[f"pos_season_rank_{preset}"])
                if pos not in POSITIONS or finish > FINISH_GATE[pos] or s["player_id"] in matched_ids:
                    continue
                out_rows.append({
                    "season": year, "format": preset, "player": s["player"],
                    "position": pos, "adp_team": s["team"], "player_id": s["player_id"],
                    "match_status": "undrafted_in_pool", "adp": "",
                    "adp_pos_rank": pool_depth.get(pos, 0) + 1,
                    "finish_pos_rank": finish, "games": s["games"], "alpha": "",
                    "breakout": True, "bust": False, "injury_bust": False,
                })

    LABELS.mkdir(parents=True, exist_ok=True)
    with open(LABELS / "breakouts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    (LABELS / "breakouts_unmatched.txt").write_text("\n".join(unmatched_log))

    matched = sum(1 for r in out_rows if r["match_status"] == "matched")
    print(f"rows={len(out_rows)} matched={matched} ({matched/len(out_rows):.1%}) unmatched={len(unmatched_log)}")
    for year in (2022, 2023, 2024):
        names = sorted({f"{r['player']} ({r['position']}, ADP {r['position']}{r['adp_pos_rank']} -> fin {r['position']}{r['finish_pos_rank']})"
                        for r in out_rows if r["season"] == year and r["breakout"] and r["format"] == "ppr"})
        print(f"sanity {year} ppr breakouts ({len(names)}): {'; '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
