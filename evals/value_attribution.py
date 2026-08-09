#!/usr/bin/env python3
"""Value ledger — per-pick value attribution for drafting policies.

For every pick in a simulated draft:

    capture = realized_season_vorp(player, season, league)
              - slot_expected_vorp(pick_number, format)

i.e. how much realized value the pick returned over its historical market
price (the slot-value curve, evals/slot_values.py). Jacob's question: which
players/picks does a policy consistently harvest value from — picks that
return +50/+60 points over their draft cost — and does it find MORE of them
than the market's own base rate (autopick gets +50 picks by luck too)?

ATTRIBUTION / TELEMETRY ONLY — NEVER REWARD
-------------------------------------------
This layer must never touch the RL reward. DraftGym's terminal reward
(realized roster points minus the same-seed autopick control) already prices
per-pick value capture implicitly and end-to-end: a +50 pick shows up as
roster points, a reach shows up as the displaced better player. Adding a
per-pick capture term to the reward would double-count the exact same signal
and reintroduce the hindsight-variance reward-hacking vector the terminal
design was chosen to avoid (docs/training-risk-register.md risk 7). This
module is the accounting department, not the incentive scheme: it explains
WHERE a policy's terminal reward came from. DraftGym exposes it only as
``info["value_ledger"]`` telemetry; the reward math is untouched.

Sibling of the pick-timing family (docs/breakoutbench-design.md §3 item 5):
same cost axis (slot_expected_vorp), applied to whole simulated drafts
instead of individual conviction calls.

Measurement notes
-----------------
- Realized VORP uses the same canonical-league replacement machinery as
  slot_values.py: season_points totals at the board's preset minus the
  replacement level of a 12-team QB1/RB2/WR3/TE1/FLEX2 league computed on
  that season's realized totals (harness.valuation.assign_starters). Board
  players with no realized stat line (nflverse_id is None) score 0 points —
  a bust's negative VORP is a real cost, same convention as slot_values.
- Basis is HINDSIGHT full-season totals (like the slot curve itself), not
  the realistic-manager weekly policy that sets the RL reward. Every ledger
  row carries ``basis="hindsight_season_total"`` to make that explicit: a
  +50-capture player is value that existed on the roster; whether a
  no-future-knowledge manager harvested all of it is the realistic-lineup
  question, answered by the reward, not by this ledger.
- Freak-injury caveat: rows join data/processed/labels/injury_context.csv
  and flag ``season_ending_acute`` players, so captures/losses driven by
  luck (a season-ending acute injury) stay visible separately from skill.
- The slot curve is built for the canonical 12-team league; ledgers over
  non-standard leagues (synthetic test pools) use it as an approximate
  price axis — fine for telemetry, not for cross-league claims.

Usage:
    python -m evals.value_attribution        # full grid + LLM replays ->
                                             # evals/results/value_attribution.md
Importable API:
    build_ledger(...)          pure per-pick capture rows (injectable inputs)
    value_ledger_rows(...)     disk-backed rows for one seat's picks
    ledger_from_result(...)    rows for a DraftResult's agent seat
    aggregate_ledgers(...)     distribution + tier rates + consistency
    market_delta(...)          paired +50-rate delta vs autopick, bootstrap CI

2025 stays the untouched holdout (GAMEPLAN §9): nothing here evaluates it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence

from evals.agents_execution import EXECUTION_AGENTS, bootstrap_ci
from evals.draftbench import (
    DEFAULT_TEAMS,
    LABELS,
    PRESET_TO_FFC,
    AutopickADP,
    BASELINE_AGENTS,
    DraftPlayer,
    DraftResult,
    DraftSimulator,
    _season_rows,
    run,
)
from evals.slot_values import replacement_levels, slot_expected_vorp
from harness.league import LeagueConfig

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evals" / "results"
REPORT_PATH = RESULTS_DIR / "value_attribution.md"
QWEN_EPISODES = RESULTS_DIR / "draftgym_qwen_baseline.json"
FRONTIER_EPISODES = RESULTS_DIR / "frontier_draftgym.json"

BASIS = "hindsight_season_total"
CAPTURE_TIERS = (25.0, 50.0, 100.0)  # +50 is Jacob's headline number
ROUND_BUCKETS = (("1-3", 1, 3), ("4-6", 4, 6), ("7-9", 7, 9),
                 ("10-12", 10, 12), ("13+", 13, 10**9))


# ---------------------------------------------------------------------------
# Realized-VORP inputs (same canonical-league machinery as slot_values.py)


@lru_cache(maxsize=None)
def _replacement(season: int, preset: str) -> dict[str, float]:
    """Canonical-league replacement points on the season's realized totals."""
    rows = _season_rows().get(season, [])
    if not rows:
        return {}
    return replacement_levels(rows, preset, qb_slots=1)


@lru_cache(maxsize=None)
def _season_totals(season: int, preset: str) -> dict[str, float]:
    """nflverse player_id -> realized season total at this preset."""
    return {
        r["player_id"]: float(r[f"total_{preset}"])
        for r in _season_rows().get(season, [])
    }


@lru_cache(maxsize=None)
def _injury_flags() -> dict[tuple[int, str], bool]:
    """(season, nflverse player_id) -> season_ending_acute."""
    path = LABELS / "injury_context.csv"
    if not path.exists():
        return {}
    out: dict[tuple[int, str], bool] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(int(r["season"]), r["player_id"])] = r["season_ending_acute"] == "True"
    return out


def realized_season_vorp(player: DraftPlayer, season: int, preset: str) -> float:
    """Hindsight realized VORP: season total minus canonical replacement.

    Unmatched board players (nflverse_id None — never appeared) score 0
    total, so their negative VORP is the realized cost of the busted pick.
    """
    total = _season_totals(season, preset).get(player.nflverse_id or "", 0.0)
    return round(total - _replacement(season, preset).get(player.position, 0.0), 2)


# ---------------------------------------------------------------------------
# Per-pick capture rows


def build_ledger(
    picks: Sequence[tuple[int, DraftPlayer]],
    *,
    teams: int,
    realized_vorp: Callable[[DraftPlayer], float],
    slot_cost: Callable[[int], float],
    injury_flag: Callable[[DraftPlayer], bool] | None = None,
) -> list[dict]:
    """Pure ledger builder: capture = realized_vorp(player) - slot_cost(pick).

    picks: (overall pick_number, DraftPlayer) for ONE seat. Inputs are
    injectable so the math is testable without disk; the disk-backed
    wrappers below supply the slot_values machinery.
    """
    rows: list[dict] = []
    for pick_number, player in sorted(picks, key=lambda t: t[0]):
        vorp = round(float(realized_vorp(player)), 2)
        cost = round(float(slot_cost(pick_number)), 2)
        rows.append({
            "pick_number": int(pick_number),
            "round": (int(pick_number) - 1) // int(teams) + 1,
            "player_id": player.player_id,
            "name": player.name,
            "position": player.position,
            "adp": player.adp,
            "realized_vorp": vorp,
            "slot_cost": cost,
            "capture": round(vorp - cost, 2),
            "season_ending_acute": bool(injury_flag(player)) if injury_flag else False,
            "basis": BASIS,
        })
    return rows


def value_ledger_rows(
    picks: Sequence[tuple[int, DraftPlayer]],
    season: int,
    preset: str,
    teams: int = DEFAULT_TEAMS,
) -> list[dict]:
    """Disk-backed ledger for one seat's picks in a (season, preset) draft."""
    fmt = PRESET_TO_FFC[preset]  # slot_values curve key
    flags = _injury_flags()
    return build_ledger(
        picks,
        teams=teams,
        realized_vorp=lambda p: realized_season_vorp(p, season, preset),
        slot_cost=lambda n: slot_expected_vorp(n, fmt),
        injury_flag=lambda p: flags.get((season, p.nflverse_id or ""), False),
    )


def ledger_from_result(
    result: DraftResult,
    season: int,
    preset: str,
    league: LeagueConfig,
    pool_by_id: Mapping[str, DraftPlayer],
) -> list[dict]:
    """Ledger for a DraftResult's agent seat."""
    picks = [
        (r.pick_number, pool_by_id[r.player_id])
        for r in result.picks
        if r.team == result.slot
    ]
    return value_ledger_rows(picks, season, preset, teams=league.teams)


# ---------------------------------------------------------------------------
# Aggregation: distribution, tier rates, provenance, consistency


def _percentile(values: Sequence[float], q: float) -> float:
    """Inclusive linear-interpolation percentile (q in 0..1)."""
    vals = sorted(values)
    if not vals:
        raise ValueError("no values")
    if len(vals) == 1:
        return vals[0]
    k = q * (len(vals) - 1)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (k - lo) * (vals[hi] - vals[lo])


def _tier_count(rows: Sequence[Mapping], tier: float) -> int:
    return sum(1 for r in rows if r["capture"] >= tier)


def aggregate_ledgers(drafts: Sequence[Sequence[Mapping]]) -> dict:
    """Aggregate per-draft ledgers into one policy's value profile.

    drafts: one ledger (list of pick rows) per simulated draft.
    """
    all_rows = [r for d in drafts for r in d]
    captures = [r["capture"] for r in all_rows]
    n_drafts = len(drafts)
    if not all_rows or not n_drafts:
        raise ValueError("aggregate_ledgers needs at least one non-empty draft")

    tier_rates = {
        int(t): round(sum(_tier_count(d, t) for d in drafts) / n_drafts, 3)
        for t in CAPTURE_TIERS
    }
    plus50_counts = [_tier_count(d, 50.0) for d in drafts]

    by_bucket: dict[str, dict] = {}
    for label, lo, hi in ROUND_BUCKETS:
        rows = [r for r in all_rows if lo <= r["round"] <= hi]
        if rows:
            by_bucket[label] = {
                "n_picks": len(rows),
                "mean_capture": round(statistics.fmean(r["capture"] for r in rows), 2),
                "plus50": sum(1 for r in rows if r["capture"] >= 50.0),
            }
    by_position: dict[str, dict] = {}
    for pos in sorted({r["position"] for r in all_rows}):
        rows = [r for r in all_rows if r["position"] == pos]
        by_position[pos] = {
            "n_picks": len(rows),
            "mean_capture": round(statistics.fmean(r["capture"] for r in rows), 2),
            "plus50": sum(1 for r in rows if r["capture"] >= 50.0),
        }

    plus50_rows = [r for r in all_rows if r["capture"] >= 50.0]
    loss50_rows = [r for r in all_rows if r["capture"] <= -50.0]
    return {
        "n_drafts": n_drafts,
        "n_picks": len(all_rows),
        "capture_mean": round(statistics.fmean(captures), 2),
        "capture_median": round(statistics.median(captures), 2),
        "capture_p10": round(_percentile(captures, 0.10), 2),
        "capture_p90": round(_percentile(captures, 0.90), 2),
        "tier_rates_per_draft": tier_rates,  # {25: x, 50: y, 100: z} picks/draft
        "consistency": {
            "ge1_plus50": round(sum(1 for c in plus50_counts if c >= 1) / n_drafts, 3),
            "ge2_plus50": round(sum(1 for c in plus50_counts if c >= 2) / n_drafts, 3),
        },
        "by_round_bucket": by_bucket,
        "by_position": by_position,
        "injury_caveat": {
            "plus50_on_season_ending_acute": sum(
                1 for r in plus50_rows if r["season_ending_acute"]),
            "minus50_on_season_ending_acute": sum(
                1 for r in loss50_rows if r["season_ending_acute"]),
            "plus50_total": len(plus50_rows),
            "minus50_total": len(loss50_rows),
        },
    }


def market_delta(
    agent_drafts: Sequence[Sequence[Mapping]],
    control_drafts: Sequence[Sequence[Mapping]],
    tier: float = 50.0,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired per-draft +tier-count delta vs the same-seed autopick control.

    Autopick IS the market's base rate of value capture (everyone gets some
    +50 picks by luck); a policy's claim is "finds MORE +50 players than the
    market does", stated as the mean paired delta with a bootstrap 95% CI.
    """
    if len(agent_drafts) != len(control_drafts):
        raise ValueError("agent/control draft lists must pair 1:1 (same cells)")
    deltas = [
        float(_tier_count(a, tier) - _tier_count(c, tier))
        for a, c in zip(agent_drafts, control_drafts)
    ]
    if not deltas:
        raise ValueError("need at least one paired draft")
    lo, hi = bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
    return {
        "tier": int(tier),
        "n": len(deltas),
        "mean_delta": round(statistics.fmean(deltas), 3),
        "ci95": (round(lo, 3), round(hi, 3)),
        "verdict": "more" if lo > 0 else ("fewer" if hi < 0 else "unresolved"),
    }


# ---------------------------------------------------------------------------
# Collection: heuristic grid (exact re-simulation) + stored LLM episodes


def collect_grid_ledgers(n_seeds: int = 5, verbose: bool = False) -> dict[str, dict]:
    """Run the standard DraftBench grid (2015-2024 x 3 formats x slots 1/6/12
    x n_seeds) for autopick + baselines + execution agents, collecting the
    per-draft value ledger of every agent seat and its same-seed autopick
    control via draftbench.run's reporting-only on_result hook.

    Returns {agent: {"drafts": [ledger...], "controls": [ledger...],
    "cells": [(season, preset, slot, seed)...]}} with drafts/controls paired.
    """
    out: dict[str, dict] = defaultdict(lambda: {"drafts": [], "controls": [], "cells": []})
    control_cache: dict[tuple, list[dict]] = {}

    def hook(agent_name, season, preset, slot, seed, result, control, pool_by_id, league):
        cell = (season, preset, slot, seed)
        if cell not in control_cache:
            control_cache[cell] = ledger_from_result(control, season, preset, league, pool_by_id)
        ctrl_rows = control_cache[cell]
        rows = (
            ctrl_rows
            if result is control
            else ledger_from_result(result, season, preset, league, pool_by_id)
        )
        entry = out[agent_name]
        entry["drafts"].append(rows)
        entry["controls"].append(ctrl_rows)
        entry["cells"].append(cell)

    # champ_sims=1: the normalized-odds columns are irrelevant here — the
    # simulator itself is exactly the standard grid (same seeds, same RNG).
    run(n_seeds=n_seeds, agents=BASELINE_AGENTS, out_path=None, verbose=verbose,
        champ_sims=1, on_result=hook)
    run(n_seeds=n_seeds, agents=EXECUTION_AGENTS, out_path=None, verbose=verbose,
        champ_sims=1, on_result=hook)
    return dict(out)


def _replay_episode_ledger(ep: dict, mask_names: bool) -> dict:
    """Replay one stored LLM DraftGym episode from its recorded picks (the
    evals/normalize.py rescore pattern — seed-deterministic opponents make
    the replay exact; verified against the stored reward) and take the
    agent's value ledger straight from DraftGym's terminal telemetry."""
    from envs.draftgym import DraftGym

    gym = DraftGym(
        season=ep["season"], preset=ep["preset"], agent_slot=ep["agent_slot"],
        seed=ep["seed"], enable_evidence=False, mask_names=mask_names,
    )
    gym.reset()
    done, reward, info = False, 0.0, {}
    for pick in ep["picks"]:
        _, reward, done, info = gym.step({"pick": pick["action"]["pick"]})
    if not done:
        raise RuntimeError("replay did not finish the episode")

    control = DraftSimulator(gym.pool, gym.league).simulate(
        AutopickADP(), ep["agent_slot"], ep["seed"]
    )
    return {
        "rows": info["value_ledger"],
        "control_rows": ledger_from_result(
            control, ep["season"], ep["preset"], gym.league, gym._by_id
        ),
        "cell": (ep["season"], ep["preset"], ep["agent_slot"], ep["seed"]),
        "exact": abs(reward - ep["reward"]) < 0.01,
    }


def collect_llm_ledgers() -> dict[str, dict]:
    """Reconstruct every stored LLM episode (named + masked) into ledgers.

    Returns the same {label: {"drafts", "controls", "cells", "inexact"}}
    shape as collect_grid_ledgers.
    """
    groups: dict[str, dict] = defaultdict(
        lambda: {"drafts": [], "controls": [], "cells": [], "inexact": []}
    )

    def add(label: str, ep: dict) -> None:
        masked = bool(ep.get("mask_names"))
        rep = _replay_episode_ledger(ep, masked)
        g = groups[f"{label}_{'masked' if masked else 'named'}"]
        g["drafts"].append(rep["rows"])
        g["controls"].append(rep["control_rows"])
        g["cells"].append(rep["cell"])
        if not rep["exact"]:
            g["inexact"].append(rep["cell"])

    if QWEN_EPISODES.exists():
        for ep in json.loads(QWEN_EPISODES.read_text())["episodes"]:
            add("qwen3.5-9b", ep)
    if FRONTIER_EPISODES.exists():
        frontier = json.loads(FRONTIER_EPISODES.read_text())
        for model_key, blob in frontier.items():
            if isinstance(blob, dict) and "episodes" in blob:
                for ep in blob["episodes"]:
                    add(blob.get("model", model_key).split("/")[-1], ep)
    return dict(groups)


# ---------------------------------------------------------------------------
# Report


def _top_captures(policies: Mapping[str, dict], k: int = 3) -> list[dict]:
    """Best distinct captured players across all policies' drafts."""
    best: dict[str, dict] = {}
    for label, entry in policies.items():
        for rows, cell in zip(entry["drafts"], entry["cells"]):
            for r in rows:
                key = f"{r['name']}|{cell[0]}"
                if key not in best or r["capture"] > best[key]["capture"]:
                    best[key] = {**r, "policy": label,
                                 "season": cell[0], "format": cell[1]}
    return sorted(best.values(), key=lambda r: -r["capture"])[:k]


def write_report(
    grid: Mapping[str, dict], llm: Mapping[str, dict], path: Path = REPORT_PATH
) -> str:
    from datetime import date

    policies = {**grid, **llm}
    aggs = {label: aggregate_ledgers(e["drafts"]) for label, e in policies.items()}
    deltas = {
        label: market_delta(e["drafts"], e["controls"])
        for label, e in policies.items()
    }

    lines = [
        "# Value ledger — per-pick value capture vs the slot-price curve",
        "",
        f"Generated {date.today().isoformat()} by `evals/value_attribution.py`.",
        "",
        "Per pick: `capture = realized_season_vorp(player) - "
        "slot_expected_vorp(pick, fmt)` (cost axis: evals/slot_values.py; "
        "realized VORP via the same canonical 12-team QB1/RB2/WR3/TE1/FLEX2 "
        "replacement machinery, hindsight full-season totals — see the module "
        "docstring's hindsight-vs-realistic note). **Attribution/telemetry "
        "only — never reward**: DraftGym's terminal reward already prices "
        "value capture implicitly; shaping with this signal would "
        "double-count it.",
        "",
        "Autopick IS the market's base rate of value capture (everyone lands "
        "some +50 picks by luck). Every policy is therefore reported relative "
        "to its same-seed autopick control: the claim format is \"finds MORE "
        "+50 players than the market does\", mean paired per-draft delta with "
        "a 10k-resample bootstrap 95% CI.",
        "",
        "Grid policies: 2015-2024 x standard/half_ppr/ppr x slots 1/6/12 x 5 "
        "seeds (half_ppr ADP exists 2018+; n=405 drafts). LLM policies: the "
        "stored DraftGym episodes replayed exactly from seeds + recorded "
        "picks (evals/normalize.py rescore pattern); small n — read the CIs.",
        "",
        "## Policy comparison — value capture per draft (<=15 picks; thin "
        "QB/RB/WR/TE boards can end drafts a pick early)",
        "",
        "| policy | drafts | +25/draft | **+50/draft** | +100/draft | "
        "Δ+50 vs market [95% CI] | ≥1 +50 | ≥2 +50 | capture mean | median | p10 | p90 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, agg in aggs.items():
        d = deltas[label]
        tiers = agg["tier_rates_per_draft"]
        cons = agg["consistency"]
        ci = f"{d['mean_delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
        if label == "autopick_adp":
            ci = "0 (is the market)"
        lines.append(
            f"| {label} | {agg['n_drafts']} | {tiers[25]:.2f} | **{tiers[50]:.2f}** "
            f"| {tiers[100]:.2f} | {ci} | {100*cons['ge1_plus50']:.0f}% "
            f"| {100*cons['ge2_plus50']:.0f}% | {agg['capture_mean']:+.1f} "
            f"| {agg['capture_median']:+.1f} | {agg['capture_p10']:+.1f} "
            f"| {agg['capture_p90']:+.1f} |"
        )

    lines += [
        "",
        "Named-vs-masked LLM rows are the memorization story told per player: "
        "a named model can \"find\" +50 players by remembering how the season "
        "went; masked boards force the capture rate back toward what features "
        "alone support.",
        "",
        "## Where captures come from",
        "",
        "+50 captures by round bucket (counts across each policy's drafts; "
        "grid policies n=405 drafts, LLM policies far fewer — rates not "
        "directly comparable across different n):",
        "",
        "| policy | " + " | ".join(b[0] for b in ROUND_BUCKETS) + " |",
        "|---|" + "---|" * len(ROUND_BUCKETS),
    ]
    for label, agg in aggs.items():
        cells = [
            str(agg["by_round_bucket"].get(b[0], {}).get("plus50", 0))
            for b in ROUND_BUCKETS
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "+50 captures by position:",
        "",
        "| policy | QB | RB | WR | TE |",
        "|---|---|---|---|---|",
    ]
    for label, agg in aggs.items():
        cells = [
            str(agg["by_position"].get(pos, {}).get("plus50", 0))
            for pos in ("QB", "RB", "WR", "TE")
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Freak-injury caveat (luck, visible separately)",
        "",
        "Rows join `data/processed/labels/injury_context.csv`; "
        "`season_ending_acute` players are flagged so a -50 'loss' that was a "
        "torn ACL in week 2 (luck) is never silently read as drafting skill, "
        "and a +50 on a flagged player (value banked before the injury) is "
        "visible too:",
        "",
        "| policy | +50 on flagged | of +50 total | -50 on flagged | of -50 total |",
        "|---|---|---|---|---|",
    ]
    for label, agg in aggs.items():
        c = agg["injury_caveat"]
        lines.append(
            f"| {label} | {c['plus50_on_season_ending_acute']} | {c['plus50_total']} "
            f"| {c['minus50_on_season_ending_acute']} | {c['minus50_total']} |"
        )

    lines += ["", "## Example top captures", ""]
    for r in _top_captures(policies):
        flag = " (season-ending acute injury season)" if r["season_ending_acute"] else ""
        lines.append(
            f"- **{r['name']}** ({r['position']}, {r['season']} {r['format']}): "
            f"pick {r['pick_number']} (round {r['round']}, ADP {r['adp']}) -> "
            f"realized VORP {r['realized_vorp']:+.1f} vs slot cost "
            f"{r['slot_cost']:.1f} = **capture {r['capture']:+.1f}** "
            f"[{r['policy']}]{flag}"
        )

    inexact = [
        f"{label}: {cell}" for label, e in llm.items() for cell in e.get("inexact", [])
    ]
    lines += [
        "",
        "## Replay exactness",
        "",
        ("All LLM episode replays reproduced their stored rewards exactly."
         if not inexact else
         "Rows replayed approximately (stored reward not reproduced):"),
    ]
    lines += [f"- {n}" for n in inexact]

    lines += [
        "",
        "## Product note",
        "",
        "DraftGym now emits this table as terminal telemetry "
        "(`info[\"value_ledger\"]`, reward math untouched). This is the "
        "post-draft user receipt: \"your best value: X at pick 68, +87 vs "
        "slot\" — per-pick rows with name, slot cost, realized VORP, capture, "
        "and the injury-luck flag, ready for the app's draft recap surface "
        "(timestamps/freshness per DESIGN.md when live).",
        "",
        "Reproduce: `python -m evals.value_attribution`",
        "",
    ]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Value ledger (per-pick capture attribution)")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per grid cell (default 5)")
    ap.add_argument("--out", default=str(REPORT_PATH), help="report path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    grid = collect_grid_ledgers(n_seeds=args.seeds, verbose=args.verbose)
    llm = collect_llm_ledgers()
    print(write_report(grid, llm, Path(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
