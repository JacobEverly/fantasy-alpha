#!/usr/bin/env python3
"""BreakoutBench anonymized-track demo: build a blind slate, score picks.

`build`: emits an anonymized candidate slate for (season, format) — drafted
players beyond the v0.2 ADP gates, features strictly from seasons < target
plus target-season ADP. No outcome fields. A separate key file maps anon ids
back to players; the scorer uses it, the model must not.

`score`: reads a picks file [{"anon_id": ..., "p_breakout": ...}] and scores
against data/processed/labels/breakouts.csv: hits@K vs base rate, Brier.

Boundary note: undrafted breakouts (Puka-2023 class) are out of scope for the
structure-only track — no ADP, and rookies have no NFL priors. That class
needs the text-evidence channel (docs/training-data-design.md).
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "processed" / "labels"
DEMO = ROOT / "evals" / "demo"
ADP_GATE = {"QB": 18, "TE": 18, "RB": 40, "WR": 40}
POSITIONS = {"QB", "RB", "WR", "TE"}


def load_season_points() -> dict[str, dict[int, dict]]:
    by_player: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(LABELS / "season_points.csv", newline="") as f:
        for r in csv.DictReader(f):
            by_player[r["player_id"]][int(r["season"])] = r
    return by_player


def build(season: int, fmt: str, preset: str) -> None:
    adp_dir = ROOT / "data" / "raw" / "ffc_adp"
    adp_path = next(p for t in (12, 10, 14, 8) if (p := adp_dir / f"{fmt}_{t}t_{season}.json").exists())
    payload = json.loads(adp_path.read_text())
    players = [p for p in payload["players"] if p.get("position") in POSITIONS]

    by_pos: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(players, key=lambda p: (float(p["adp"]), -int(p.get("times_drafted", 0)))):
        by_pos[p["position"]].append(p)

    sys.path.insert(0, str(ROOT))
    from evals.names import norm_name

    stats = load_season_points()
    name_index: dict[tuple, list[str]] = defaultdict(list)
    for pid, seasons in stats.items():
        any_row = next(iter(seasons.values()))
        name_index[(norm_name(any_row["player"]), any_row["position"])].append(pid)

    candidates = []
    for pos, group in by_pos.items():
        for i, p in enumerate(group):
            pos_rank = i + 1
            if pos_rank <= ADP_GATE[pos]:
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

            candidates.append({
                "player": p["name"],
                "player_id": pid,
                "features": {
                    "position": pos,
                    "adp_overall": float(p["adp"]),
                    "adp_pos_rank": pos_rank,
                    "adp_stdev": float(p.get("stdev") or 0),
                    "seasons_of_data": len(past),
                    "years_since_first_season": (season - first_year) if first_year else 0,
                    "prior_season": season_block(season - 1),
                    "two_seasons_ago": season_block(season - 2),
                },
            })

    rng = random.Random(20260808)
    rng.shuffle(candidates)
    slate, key = [], {}
    for i, c in enumerate(candidates):
        anon = f"P{i:03d}"
        key[anon] = {"player": c["player"], "player_id": c["player_id"]}
        slate.append({"anon_id": anon, **c["features"]})

    DEMO.mkdir(parents=True, exist_ok=True)
    (DEMO / f"slate_{season}_{preset}.json").write_text(json.dumps({
        "task": "Pick the 10 candidates most likely to BREAK OUT this season "
                "(finish top-12 QB/TE or top-24 RB/WR at position). All candidates "
                "are drafted beyond the ADP gates, so every one qualifies if they hit. "
                "Return [{anon_id, p_breakout}] for exactly 10.",
        "season_masked": True,
        "format": preset,
        "n_candidates": len(slate),
        "candidates": slate,
    }, indent=1))
    (DEMO / f"slate_{season}_{preset}_KEY.json").write_text(json.dumps(key, indent=1))
    print(f"slate built: {len(slate)} candidates -> evals/demo/slate_{season}_{preset}.json (key sequestered)")


def score(season: int, preset: str, picks_file: str) -> None:
    picks = json.loads(Path(picks_file).read_text())
    key = json.loads((DEMO / f"slate_{season}_{preset}_KEY.json").read_text())

    outcomes: dict[str, dict] = {}
    with open(LABELS / "breakouts.csv", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["season"]) == season and r["format"] == preset and r["player_id"]:
                outcomes[r["player_id"]] = r

    slate = json.loads((DEMO / f"slate_{season}_{preset}.json").read_text())["candidates"]
    eligible_ids = [key[c["anon_id"]]["player_id"] for c in slate]
    eligible_breakouts = [pid for pid in eligible_ids if pid and outcomes.get(pid, {}).get("breakout") == "True"]
    base_rate = len(eligible_breakouts) / len(eligible_ids)

    hits, brier_terms, rows = 0, [], []
    for pick in picks:
        info = key[pick["anon_id"]]
        pid = info["player_id"]
        out = outcomes.get(pid, {}) if pid else {}
        hit = out.get("breakout") == "True"
        hits += hit
        p = float(pick["p_breakout"])
        brier_terms.append((p - (1.0 if hit else 0.0)) ** 2)
        rows.append((pick["anon_id"], info["player"], p, "HIT" if hit else "miss",
                     f"fin {out.get('finish_pos_rank','—')}" if out else "no data"))

    k = len(picks)
    print(f"\n=== BreakoutBench anonymized demo — season {season} ({preset}) ===")
    print(f"slate: {len(eligible_ids)} gate-eligible candidates; true breakouts among them: "
          f"{len(eligible_breakouts)} (base rate {base_rate:.1%}; random@{k} ≈ {base_rate*k:.1f} hits)")
    print(f"model hits@{k}: {hits}  |  lift vs random: {hits/(base_rate*k):.1f}x  |  Brier(picks): {sum(brier_terms)/k:.3f}")
    for r in rows:
        print(f"  {r[0]}  {r[1]:<24} p={r[2]:.2f}  {r[3]:>4}  {r[4]}")
    truth = sorted(({"player": key_v["player"] for a, key_v in key.items()
                     if key_v["player_id"] in set(eligible_breakouts)}.get("player", kv["player"]) if False else kv["player"])
                   for kv in key.values() if kv["player_id"] in set(eligible_breakouts))
    print(f"actual breakouts in slate: {', '.join(truth)}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "score":
        score(int(sys.argv[2]), sys.argv[3], sys.argv[4])
