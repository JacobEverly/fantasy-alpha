#!/usr/bin/env python3
"""Historical slot-value curve: expected realized VORP by overall draft slot.

This is the price axis for the pick-timing scoring family
(docs/breakoutbench-design.md section 3, item 5):

    timing_score = survival(act_pick | adp, stdev) * realized_VORP
                   - slot_expected_VORP(act_pick)

For each (format preset, season 2011-2024) we join the FFC ADP pool to
realized season outcomes (norm_name+position, team tiebreak), compute each
drafted player's realized VORP for a canonical 12-team QB1/RB2/WR3/TE1/FLEX2
league (replacement levels via harness.valuation on that season's realized
totals; the 2qb format uses QB2 and standard points), then aggregate expected
realized VORP by overall draft-slot bucket and fit a monotone-decreasing
smoothed per-pick curve (weighted pool-adjacent-violators, stdlib only).

DELIBERATE SAME-SEASON JOIN -- NOT A FEATURE, NOT PACKET DATA
-------------------------------------------------------------
This module intentionally joins season-N ADP to season-N realized outcomes.
That is not leakage: the output is a *scoring curve* -- the historical market
price of a draft slot in realized-VORP units -- used only to score conviction
calls after outcomes are known. It must NEVER be fed to a model as a feature
or included in an evidence packet for a season whose outcomes it contains.
The packet/feature side of the house (evals/packet_features.py) is where the
strictly-prior-seasons discipline applies; this file is on the scoring side
of that wall.

Usage:
    python -m evals.slot_values          # rebuild CSV + report
Importable API (what the timing family uses):
    slot_expected_vorp(pick, fmt) -> float
    survival_probability(act_pick, adp, stdev) -> float
    timing_score(act_pick, adp, stdev, realized_vorp, fmt) -> float
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.names import norm_name  # noqa: E402
from harness.league import LeagueConfig  # noqa: E402
from harness.scoring import PRESETS  # noqa: E402
from harness.valuation import ProjectedPlayer, assign_starters  # noqa: E402

ADP_DIR = ROOT / "data" / "raw" / "ffc_adp"
LABELS = ROOT / "data" / "processed" / "labels"
CSV_PATH = LABELS / "slot_values.csv"
REPORT_PATH = ROOT / "evals" / "results" / "slot_values.md"

SEASONS = range(2011, 2025)  # 2025 is the untouched eval holdout
CORE_POSITIONS = {"QB", "RB", "WR", "TE"}
MAX_PICK = 180  # 12 teams x 15 rounds
TEAMS = 12

# format -> (season_points column suffix, QB starter slots)
FORMATS: dict[str, tuple[str, int]] = {
    "ppr": ("ppr", 1),
    "half-ppr": ("half_ppr", 1),
    "standard": ("standard", 1),
    "2qb": ("standard", 2),  # FFC 2QB drafts score standard; market prices 2 QB slots
}

# picks 1-3, 4-6, 7-9, 10-12, then round-sized buckets through 180
BUCKETS: list[tuple[int, int]] = [(1, 3), (4, 6), (7, 9), (10, 12)] + [
    (lo, lo + 11) for lo in range(13, MAX_PICK, 12)
]


# ---------------------------------------------------------------------------
# Data loading + join (norm_name+position, team tiebreak -- the established
# pattern from evals/build_breakout_labels.py)


def adp_path(fmt: str, season: int) -> Path | None:
    """Existence fallback across team-count suffixes (evals/anon_demo.py pattern)."""
    return next(
        (p for t in (12, 10, 14, 8) if (p := ADP_DIR / f"{fmt}_{t}t_{season}.json").exists()),
        None,
    )


def load_season_points() -> dict[int, list[dict]]:
    by_season: dict[int, list[dict]] = defaultdict(list)
    with open(LABELS / "season_points.csv", newline="") as f:
        for r in csv.DictReader(f):
            by_season[int(r["season"])].append(r)
    return by_season


def replacement_levels(rows: list[dict], preset: str, qb_slots: int) -> dict[str, float]:
    """Replacement points per position for the canonical league on realized totals."""
    league = LeagueConfig(
        teams=TEAMS,
        roster={"QB": qb_slots, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2, "BENCH": 6},
        scoring=PRESETS[preset],
    )
    pool = [
        ProjectedPlayer(
            player_id=r["player_id"],
            name=r["player"],
            position=r["position"],
            points=float(r[f"total_{preset}"]),
        )
        for r in rows
        if r["position"] in CORE_POSITIONS
    ]
    return assign_starters(pool, league).replacement_points


def season_observations(
    fmt: str, season: int, season_rows: list[dict]
) -> tuple[list[tuple[int, float]], int, int]:
    """(overall_pick, realized_vorp) observations for one (format, season).

    Overall pick = ordinal ADP rank over the full drafted pool (K/DEF hold
    their slots but contribute no VORP observation). Drafted core-position
    players with no realized stat line scored 0 points -- their negative VORP
    is the real cost of a busted pick, so they stay in.
    """
    path = adp_path(fmt, season)
    if path is None:
        return [], 0, 0
    payload = json.loads(path.read_text())
    drafted = sorted(
        payload["players"],
        key=lambda p: (float(p["adp"]), -int(p.get("times_drafted", 0))),
    )

    preset, qb_slots = FORMATS[fmt]
    replacement = replacement_levels(season_rows, preset, qb_slots)

    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in season_rows:
        if r["position"] in CORE_POSITIONS:
            by_key[(norm_name(r["player"]), r["position"])].append(r)

    obs: list[tuple[int, float]] = []
    matched = unmatched = 0
    for slot, p in enumerate(drafted, start=1):
        if slot > MAX_PICK:
            break
        pos = p.get("position")
        if pos not in CORE_POSITIONS:
            continue
        candidates = by_key.get((norm_name(p["name"]), pos), [])
        if len(candidates) > 1 and p.get("team"):
            team_hits = [c for c in candidates if p["team"] in c["team"].split("/")]
            candidates = team_hits or candidates
        if candidates:
            points = float(candidates[0][f"total_{preset}"])
            matched += 1
        else:
            points = 0.0  # drafted, never produced a stat line -> full bust cost
            unmatched += 1
        obs.append((slot, round(points - replacement.get(pos, 0.0), 2)))
    return obs, matched, unmatched


# ---------------------------------------------------------------------------
# Aggregation: bucket stats + monotone-decreasing smoothed per-pick curve


def _pava_nonincreasing(values: list[float], weights: list[float]) -> list[float]:
    """Weighted pool-adjacent-violators, non-increasing fit (stdlib)."""
    # fit non-decreasing on the negated series, negate back
    blocks: list[list[float]] = []  # [value, weight, count]
    for v, w in zip(values, weights):
        blocks.append([-v, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out: list[float] = []
    for v, _w, c in blocks:
        out.extend([-v] * c)
    return out


def smoothed_curve(obs: list[tuple[int, float]]) -> dict[int, float]:
    """Monotone-decreasing per-pick expected VORP for picks 1..MAX_PICK.

    Per-pick means (weighted by observation count) are PAVA-pooled; picks
    with no observations inherit the previous fitted value. The price curve
    is floored at 0: a slot's opportunity cost is non-negative (you can
    always take best available), even though raw late-slot means go negative
    (bust backup QBs carry large negative signed VORP). This matches the
    design doc's economics -- last-round slots cost ~zero, so hits there are
    pure capture. Bucket stats stay raw/empirical.
    """
    per_pick: dict[int, list[float]] = defaultdict(list)
    for pick, vorp in obs:
        per_pick[pick].append(vorp)
    observed = sorted(per_pick)
    if not observed:
        return {p: 0.0 for p in range(1, MAX_PICK + 1)}
    means = [statistics.fmean(per_pick[p]) for p in observed]
    weights = [float(len(per_pick[p])) for p in observed]
    fitted = _pava_nonincreasing(means, weights)
    by_pick = dict(zip(observed, fitted))
    curve: dict[int, float] = {}
    last = by_pick[observed[0]]
    for p in range(1, MAX_PICK + 1):
        last = by_pick.get(p, last)
        curve[p] = round(max(last, 0.0), 2)
    return curve


def bucket_stats(obs: list[tuple[int, float]]) -> list[dict]:
    rows = []
    for lo, hi in BUCKETS:
        vals = sorted(v for p, v in obs if lo <= p <= hi)
        if not vals:
            rows.append({"bucket": f"{lo}-{hi}", "n": 0, "mean_vorp": "",
                         "median_vorp": "", "p25": "", "p75": ""})
            continue
        q = statistics.quantiles(vals, n=4, method="inclusive") if len(vals) > 1 else [vals[0]] * 3
        rows.append({
            "bucket": f"{lo}-{hi}",
            "n": len(vals),
            "mean_vorp": round(statistics.fmean(vals), 2),
            "median_vorp": round(statistics.median(vals), 2),
            "p25": round(q[0], 2),
            "p75": round(q[2], 2),
        })
    return rows


def build_all() -> dict[str, dict]:
    """Compute {fmt: {"buckets": [...], "curve": {pick: vorp}, "seasons": [...],
    "matched": int, "unmatched": int}} across SEASONS."""
    by_season = load_season_points()
    out: dict[str, dict] = {}
    for fmt in FORMATS:
        pooled: list[tuple[int, float]] = []
        seasons_used: list[int] = []
        matched = unmatched = 0
        for season in SEASONS:
            rows = by_season.get(season)
            if not rows:
                continue
            obs, m, u = season_observations(fmt, season, rows)
            if obs:
                pooled.extend(obs)
                seasons_used.append(season)
                matched += m
                unmatched += u
        out[fmt] = {
            "buckets": bucket_stats(pooled),
            "curve": smoothed_curve(pooled),
            "seasons": seasons_used,
            "matched": matched,
            "unmatched": unmatched,
        }
    return out


def write_csv(results: dict[str, dict], path: Path = CSV_PATH) -> None:
    fields = ["format", "row_type", "bucket", "pick", "n",
              "mean_vorp", "median_vorp", "p25", "p75", "smoothed_vorp"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fmt, res in results.items():
            for b in res["buckets"]:
                w.writerow({"format": fmt, "row_type": "bucket", "pick": "",
                            "smoothed_vorp": "", **b})
            for pick in range(1, MAX_PICK + 1):
                w.writerow({"format": fmt, "row_type": "pick", "bucket": "",
                            "pick": pick, "n": "", "mean_vorp": "", "median_vorp": "",
                            "p25": "", "p75": "",
                            "smoothed_vorp": res["curve"][pick]})


# ---------------------------------------------------------------------------
# Public API for the timing family


_CURVES: dict[str, dict[int, float]] | None = None


def _load_curves() -> dict[str, dict[int, float]]:
    global _CURVES
    if _CURVES is None:
        curves: dict[str, dict[int, float]] = defaultdict(dict)
        with open(CSV_PATH, newline="") as f:
            for r in csv.DictReader(f):
                if r["row_type"] == "pick":
                    curves[r["format"]][int(r["pick"])] = float(r["smoothed_vorp"])
        _CURVES = dict(curves)
    return _CURVES


def slot_expected_vorp(pick: int, fmt: str) -> float:
    """Expected realized VORP of an overall draft slot (smoothed historical curve).

    Picks past MAX_PICK price at the last-round value (~replacement, near zero):
    undrafted fliers cost essentially nothing.
    """
    curves = _load_curves()
    if fmt not in curves:
        raise KeyError(f"unknown format {fmt!r}; have {sorted(curves)}")
    return curves[fmt][min(max(int(pick), 1), MAX_PICK)]


def survival_probability(act_pick: float, adp: float, stdev: float) -> float:
    """P(player still on the board at act_pick), draft slot ~ N(adp, stdev).

    Normal CDF via math.erf; stdev floored at 1.0. At act_pick == adp this is
    0.5; acting after ADP collapses toward 0, well before ADP toward 1.
    """
    stdev = max(float(stdev), 1.0)
    z = (float(adp) - float(act_pick)) / stdev
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def timing_score(act_pick: float, adp: float, stdev: float,
                 realized_vorp: float, fmt: str) -> float:
    """Design-doc formula: survival x realized_VORP - slot_expected_VORP(act_pick)."""
    return round(
        survival_probability(act_pick, adp, stdev) * float(realized_vorp)
        - slot_expected_vorp(int(round(act_pick)), fmt),
        2,
    )


# ---------------------------------------------------------------------------
# Report


def _worked_examples() -> list[str]:
    lines = ["## Worked timing-score examples (ppr curve)", ""]

    def ex(label: str, act: int, adp: float, stdev: float, rv: float) -> str:
        s = survival_probability(act, adp, stdev)
        sev = slot_expected_vorp(act, "ppr")
        t = timing_score(act, adp, stdev, rv, "ppr")
        return (f"- {label}: act_pick={act}, adp={adp}, stdev={stdev}, "
                f"realized_vorp={rv} -> survival={s:.3f}, "
                f"slot_expected_vorp={sev:.2f}, **timing_score={t:.2f}**")

    lines.append("### (a) 8th-round-ADP conviction call (adp 90, stdev 8, realized VORP 120)")
    for act in (1, 78, 100):
        lines.append(ex(f"acted at pick {act}", act, 90, 8, 120))
    lines.append("")
    lines.append("The round-7 action (pick 78) wins: nearly full survival odds at a slot "
                 "priced far below the player's realized value. Pick 1 pays a first-round "
                 "opportunity cost for the same player; pick 100 is after the market -- "
                 "survival collapses and the credit evaporates.")
    lines.append("")
    lines.append("### (b) Undrafted flier acted at pick 175 (adp 200, stdev 15, realized VORP 80)")
    lines.append(ex("acted at pick 175", 175, 200, 15, 80))
    lines.append("")
    lines.append("A last-round slot prices near replacement, so a hit there is almost pure capture.")
    lines.append("")
    lines.append("### (c) First-rounder taken at his ADP (adp 5, stdev 2, realized VORP 150)")
    lines.append(ex("acted at pick 5", 5, 5, 2, 150))
    lines.append("")
    lines.append("At your own ADP survival is a coin flip and the slot price is steep -- "
                 "taking the market's pick at the market's price earns roughly nothing, "
                 "exactly the intended economics.")
    return lines


def write_report(results: dict[str, dict], path: Path = REPORT_PATH) -> None:
    from datetime import date

    lines = [
        "# Slot-value curves (expected realized VORP by overall draft slot)",
        "",
        f"Generated {date.today().isoformat()} by `evals/slot_values.py`. "
        f"Canonical league: 12-team QB1/RB2/WR3/TE1/FLEX2 (2qb format: QB2, standard points). "
        f"Seasons pooled per format below; 2025 holdout untouched.",
        "",
        "Same-season join is deliberate: this is a scoring curve (slot price axis "
        "for the pick-timing family), never packet/feature data.",
        "",
    ]
    for fmt, res in results.items():
        seasons = res["seasons"]
        lines.append(f"## {fmt}")
        lines.append("")
        lines.append(f"Seasons {min(seasons)}-{max(seasons)} ({len(seasons)} pooled), "
                     f"{res['matched']} drafted players matched, {res['unmatched']} "
                     f"unmatched (scored as 0-point busts).")
        lines.append("")
        lines.append("| bucket | n | mean VORP | median | p25 | p75 | smoothed @ bucket start |")
        lines.append("|---|---|---|---|---|---|---|")
        for b in res["buckets"]:
            lo = int(b["bucket"].split("-")[0])
            lines.append(f"| {b['bucket']} | {b['n']} | {b['mean_vorp']} | "
                         f"{b['median_vorp']} | {b['p25']} | {b['p75']} | "
                         f"{res['curve'][lo]} |")
        lines.append("")
        # face validity
        early = next(b for b in res["buckets"] if b["bucket"] == "1-3")
        r8 = next(b for b in res["buckets"] if b["bucket"] == "85-96")
        verdict = "PASS" if early["mean_vorp"] and r8["mean_vorp"] != "" and early["mean_vorp"] > r8["mean_vorp"] + 30 else "CHECK"
        lines.append(f"Face validity: picks 1-3 mean VORP {early['mean_vorp']} vs "
                     f"round-8 (85-96) mean {r8['mean_vorp']} -> {verdict}. "
                     f"Smoothed curve is monotone non-increasing by construction.")
        lines.append("")
    lines.extend(_worked_examples())
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    results = build_all()
    write_csv(results)
    global _CURVES
    _CURVES = None  # reload from the CSV just written
    write_report(results)
    print(f"wrote {CSV_PATH} and {REPORT_PATH}")
    for fmt, res in results.items():
        early = next(b for b in res["buckets"] if b["bucket"] == "1-3")
        r8 = next(b for b in res["buckets"] if b["bucket"] == "85-96")
        print(f"{fmt}: picks 1-3 mean VORP {early['mean_vorp']} "
              f"(n={early['n']}), round-8 mean {r8['mean_vorp']} (n={r8['n']})")


if __name__ == "__main__":
    main()
