#!/usr/bin/env python3
"""Current-season platform-ranking pilot.

Compare a host's default draft-room ordering with a deterministic roster
optimizer in matched simulated drafts. Both arms face opponents anchored to
the same host ordering. The resulting rosters are scored with ESPN's weekly
PPR projections, so this is a product-mechanics check rather than evidence of
realized forecasting skill.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from evals.draftbench import (
    AutopickADP,
    DraftPlayer,
    DraftSimulator,
    GreedyVORP,
    optimal_lineup_points,
)
from evals.names import norm_name
from harness.league import LeagueConfig

ROOT = Path(__file__).resolve().parent.parent
POSITIONS = {"QB", "RB", "WR", "TE"}
LEAGUE = LeagueConfig(
    teams=12,
    scoring="ppr",
    roster={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2, "BENCH": 6},
)


def _key(name: str, position: str) -> tuple[str, str]:
    return norm_name(name), position.upper()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_weekly_projections(path: Path, season: int) -> dict[tuple[str, str], dict[int, float]]:
    payload = json.loads(path.read_text())
    projections: dict[tuple[str, str], dict[int, float]] = {}
    position_by_id = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}
    for item in payload["players"]:
        player = item.get("player", {})
        position = position_by_id.get(player.get("defaultPositionId"))
        if position is None:
            continue
        weeks = {
            int(row["scoringPeriodId"]): float(row.get("appliedTotal") or 0.0)
            for row in player.get("stats", [])
            if row.get("seasonId") == season
            and row.get("statSourceId") == 1
            and row.get("statSplitTypeId") == 1
            and 1 <= int(row.get("scoringPeriodId", 0)) <= 17
        }
        if weeks:
            projections[_key(player.get("fullName", ""), position)] = weeks
    return projections


def build_pool(
    host: str,
    rows: Sequence[Mapping],
    projections: Mapping[tuple[str, str], Mapping[int, float]],
) -> tuple[list[DraftPlayer], dict[str, Mapping[int, float]]]:
    pool: list[DraftPlayer] = []
    weekly: dict[str, Mapping[int, float]] = {}
    for index, row in enumerate(rows):
        position = str(row["pos"]).upper()
        rank = row.get("host_rank_ppr") if host == "espn" else row.get("host_rank")
        player_weeks = projections.get(_key(str(row["name"]), position))
        if position not in POSITIONS or not rank or not player_weeks:
            continue
        player_id = f"{host}:{index}"
        rank_number = float(rank)
        pool.append(
            DraftPlayer(
                player_id=player_id,
                name=str(row["name"]),
                position=position,
                team=str(row.get("team", "")),
                adp=rank_number,
                stdev=max(1.0, 0.12 * rank_number),
                times_drafted=1,
                prev_points=sum(player_weeks.values()),
                nflverse_id=None,
            )
        )
        weekly[player_id] = player_weeks
    return pool, weekly


def projected_roster_points(
    roster: Sequence[DraftPlayer], weekly: Mapping[str, Mapping[int, float]]
) -> float:
    return sum(
        optimal_lineup_points(
            [(player.position, weekly[player.player_id].get(week, 0.0)) for player in roster],
            LEAGUE,
        )
        for week in range(1, 18)
    )


def bootstrap_mean_ci(values: Sequence[float], seed: int = 20260907) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(10_000)
    )
    return means[250], means[9750]


def run_platform(
    host: str,
    rows: Sequence[Mapping],
    projections: Mapping[tuple[str, str], Mapping[int, float]],
    seeds_per_slot: int,
) -> dict:
    pool, weekly = build_pool(host, rows, projections)
    simulator = DraftSimulator(pool, LEAGUE)
    pairs = []
    for slot in (1, 6, 12):
        for seed_index in range(seeds_per_slot):
            seed = 20260907 + slot * 10_000 + seed_index
            baseline = simulator.simulate(AutopickADP(), slot, seed)
            assisted = simulator.simulate(GreedyVORP(), slot, seed)
            baseline_points = projected_roster_points(baseline.agent_roster, weekly)
            assisted_points = projected_roster_points(assisted.agent_roster, weekly)
            pairs.append(
                {
                    "slot": slot,
                    "seed": seed,
                    "platform_default_points": round(baseline_points, 4),
                    "agent_assisted_points": round(assisted_points, 4),
                    "difference": round(assisted_points - baseline_points, 4),
                }
            )
    baseline_values = [row["platform_default_points"] for row in pairs]
    assisted_values = [row["agent_assisted_points"] for row in pairs]
    differences = [row["difference"] for row in pairs]
    baseline_mean = statistics.mean(baseline_values)
    difference_mean = statistics.mean(differences)
    ci_low, ci_high = bootstrap_mean_ci(differences)
    return {
        "platform": host,
        "matched_players": len(pool),
        "drafts": len(pairs),
        "platform_default_mean": round(baseline_mean, 4),
        "agent_assisted_mean": round(statistics.mean(assisted_values), 4),
        "mean_difference": round(difference_mean, 4),
        "mean_improvement_percent": round(100.0 * difference_mean / baseline_mean, 4),
        "difference_bootstrap_95_ci": [round(ci_low, 4), round(ci_high, 4)],
        "wins": sum(value > 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "pairs": pairs,
    }


def write_svg(results: Sequence[Mapping], path: Path) -> None:
    width, height = 1040, 620
    left, right, top, bottom = 115, 55, 130, 100
    plot_width, plot_height = width - left - right, height - top - bottom
    means = [
        float(value)
        for result in results
        for value in (result["platform_default_mean"], result["agent_assisted_mean"])
    ]
    low = 50.0 * int((min(means) - 25) / 50)
    high = 50.0 * (int((max(means) + 25) / 50) + 1)

    def sy(value: float) -> float:
        return top + (high - value) / (high - low) * plot_height

    centers = [left + plot_width * 0.28, left + plot_width * 0.72]
    platform_color = "#94A3B8"
    agent_color = "#059669"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Decision engine compared with ESPN and Yahoo defaults</title>',
        '<desc id="desc">Grouped comparison of average projected season points from platform default rankings and our decision engine.</desc>',
        '<rect width="100%" height="100%" rx="18" fill="#FAFAF9"/>',
        '<text x="115" y="42" font-family="Inter,Arial,sans-serif" font-size="25" font-weight="700" fill="#111827">Decision engine vs. platform default rankings</text>',
        '<text x="115" y="68" font-family="Inter,Arial,sans-serif" font-size="14" fill="#6B7280">Average projected season points · 60 matched 12-team PPR drafts per room</text>',
        f'<circle cx="700" cy="40" r="7" fill="{platform_color}"/>',
        '<text x="716" y="45" font-family="Inter,Arial,sans-serif" font-size="13" fill="#374151">Platform default</text>',
        f'<circle cx="855" cy="40" r="7" fill="{agent_color}"/>',
        '<text x="871" y="45" font-family="Inter,Arial,sans-serif" font-size="13" fill="#374151">Our engine</text>',
    ]
    value = low
    while value <= high:
        y = sy(value)
        svg += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>',
            f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="12" fill="#6B7280">{value:.0f}</text>',
        ]
        value += 50.0
    for index, result in enumerate(results):
        center = centers[index]
        default_x, agent_x = center - 70, center + 70
        default_y = sy(float(result["platform_default_mean"]))
        agent_y = sy(float(result["agent_assisted_mean"]))
        svg += [
            f'<line x1="{default_x:.1f}" y1="{default_y:.1f}" x2="{agent_x:.1f}" y2="{agent_y:.1f}" stroke="#CBD5E1" stroke-width="4"/>',
            f'<circle cx="{default_x:.1f}" cy="{default_y:.1f}" r="13" fill="{platform_color}" stroke="#FAFAF9" stroke-width="4"/>',
            f'<circle cx="{agent_x:.1f}" cy="{agent_y:.1f}" r="13" fill="{agent_color}" stroke="#FAFAF9" stroke-width="4"/>',
            f'<text x="{default_x:.1f}" y="{default_y-22:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="600" fill="#475569">{result["platform_default_mean"]:.0f}</text>',
            f'<text x="{agent_x:.1f}" y="{agent_y-22:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="700" fill="#047857">{result["agent_assisted_mean"]:.0f}</text>',
            f'<text x="{center:.1f}" y="{min(default_y, agent_y)-48:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#047857">+{result["mean_improvement_percent"]:.1f}%</text>',
            f'<text x="{center:.1f}" y="{height-bottom+34}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#111827">{str(result["platform"]).upper()} draft room</text>',
            f'<text x="{center:.1f}" y="{height-bottom+58}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#64748B">Our engine won {result["wins"]} of {result["drafts"]} matched drafts</text>',
        ]
    svg += [
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#374151"/>',
        f'<text x="29" y="{top+plot_height/2:.1f}" text-anchor="middle" transform="rotate(-90 29 {top+plot_height/2:.1f})" font-family="Inter,Arial,sans-serif" font-size="15" font-weight="600" fill="#374151">Average projected season points</text>',
        f'<text x="{left}" y="{height-18}" font-family="Inter,Arial,sans-serif" font-size="11" fill="#6B7280">September 7, 2026 pilot · ESPN weekly PPR projections · simulated room behavior · not realized season performance</text>',
        '</svg>',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-09-07")
    parser.add_argument("--seeds-per-slot", type=int, default=20)
    parser.add_argument(
        "--scorecard", default="artifacts/current-platform-pilot-v1/scorecard.json"
    )
    parser.add_argument(
        "--figure", default="docs/assets/current-platform-pilot.svg"
    )
    args = parser.parse_args()

    host_dir = ROOT / "data" / "raw" / "host_ranks" / args.date
    host_ranks_path = host_dir / "host_ranks_parsed.json"
    espn_projection_path = host_dir / "espn_kona_player_info.json"
    hosts = json.loads(host_ranks_path.read_text())
    projections = load_weekly_projections(espn_projection_path, int(args.date[:4]))
    results = [
        run_platform(host, hosts[host], projections, args.seeds_per_slot)
        for host in ("espn", "yahoo")
    ]
    scorecard = {
        "protocol": "current-platform-pilot-v1",
        "snapshot_date": args.date,
        "source_captured_at": hosts.get("_captured_at"),
        "source_hashes": {
            str(host_ranks_path.relative_to(ROOT)): _sha256(host_ranks_path),
            str(espn_projection_path.relative_to(ROOT)): _sha256(espn_projection_path),
        },
        "league": "12-team PPR, 15 rounds, slots 1/6/12",
        "evaluation": "ESPN weekly PPR projections",
        "interpretation": "product-mechanics pilot; not realized forecasting performance",
        "results": results,
    }
    scorecard_path = ROOT / args.scorecard
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_path.write_text(json.dumps(scorecard, indent=2) + "\n")
    write_svg(results, ROOT / args.figure)
    for result in results:
        print(
            f'{result["platform"]}: {result["mean_improvement_percent"]:+.2f}% '
            f'({result["wins"]}/{result["drafts"]} wins)'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
