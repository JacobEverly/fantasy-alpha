#!/usr/bin/env python3
"""Normalized season scoring — the league-anchored index + championship odds.

Raw season point totals ("Fable 2900 vs ours 3050") are unrelatable and not
comparable across eras or scoring formats. This module supplies the
human-readable layer on top of the existing paired-control points metrics
(which remain the statistical backbone everywhere):

1. **League Index** — for any evaluated roster,
       index = 100 * roster_season_points / best_OPPOSING_team_season_points
   within the same simulated league (same seed, same lineup policy).
   100 = tied with the league-best opponent; >100 = outscored everyone.
   Also emitted: league_rank (1..teams) and percentile.

2. **Championship odds** — Monte Carlo over hypothetical H2H schedules using
   the teams' REALIZED weekly scores. Documented convention:
     - regular season = weeks 1-14, each week a uniform random pairing of all
       teams (round-robin-style; every team plays exactly once per week);
     - a tie in a weekly matchup counts half a win to each side;
     - standings by W-L, tiebreak regular-season points-for, then lower team
       index (deterministic);
     - top-4 playoffs, seeded 1-4: semifinal = week 15 score (1v4, 2v3),
       final = weeks 16+17 summed;
     - playoff-game ties break by regular-season points-for, then better seed;
     - pre-2021 seasons whose data lacks a week 17 (16 weekly scores): the
       final is weeks 15+16 summed — week 15 is shared with the semifinal by
       convention, documented here and pinned in tests.
   Deterministic under (inputs, n_sims, seed). Output per team: expected
   regular-season wins, playoff probability, title probability.

Zero third-party dependencies (stdlib only). The reporting wiring lives in
evals/draftbench.py and envs/draftgym.py; the record translation is
`python -m evals.normalize --rescore` (writes evals/results/normalized_rescore.md).
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence

REGULAR_SEASON_WEEKS = 14
PLAYOFF_TEAMS = 4
SEMIFINAL_WEEK = 15  # 1-based
FINAL_WEEKS = (16, 17)  # 1-based; pre-2021 fallback = (15, 16), see module doc
MIN_WEEKS = 16
DEFAULT_N_SIMS = 2000


# ---------------------------------------------------------------------------
# League Index


def league_index(team_points: float, opponent_points: Sequence[float]) -> float:
    """100 * team season points / best OPPOSING team's season points.

    100 = tied with the league-best opponent; >100 = outscored every opponent.
    """
    if not opponent_points:
        raise ValueError("league_index needs at least one opponent")
    best = max(opponent_points)
    if best <= 0:
        raise ValueError(f"best opponent points must be positive, got {best}")
    return round(100.0 * team_points / best, 1)


def league_rank(team_points: float, opponent_points: Sequence[float]) -> int:
    """1-based rank within the league (1 = highest total; ties share the
    better rank)."""
    return 1 + sum(1 for p in opponent_points if p > team_points)


def league_percentile(team_points: float, opponent_points: Sequence[float]) -> float:
    """Percentile within the league: 100 = first place, 0 = last place."""
    teams = len(opponent_points) + 1
    rank = league_rank(team_points, opponent_points)
    return round(100.0 * (teams - rank) / (teams - 1), 1)


# ---------------------------------------------------------------------------
# Championship odds (schedule Monte Carlo)


def championship_odds(
    weekly_scores_by_team: Sequence[Sequence[float]],
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = 0,
) -> list[dict]:
    """Simulate H2H fantasy seasons over the teams' realized weekly scores.

    See the module docstring for the full schedule/playoff convention. The
    weekly scores are the teams' REALIZED ones — only the schedule (who plays
    whom each regular-season week) is random. Returns one dict per team:
    {"expected_wins", "playoff_pct", "title_pct"} (probabilities in 0..1).
    Deterministic under (inputs, n_sims, seed).
    """
    teams = len(weekly_scores_by_team)
    if teams < PLAYOFF_TEAMS:
        raise ValueError(f"need at least {PLAYOFF_TEAMS} teams, got {teams}")
    if teams % 2:
        raise ValueError(f"need an even number of teams for weekly pairings, got {teams}")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    scores = [[float(x) for x in w] for w in weekly_scores_by_team]
    n_weeks = min(len(s) for s in scores)
    if n_weeks < MIN_WEEKS:
        raise ValueError(f"need at least {MIN_WEEKS} weekly scores per team, got {n_weeks}")
    if any(len(s) != n_weeks for s in scores):
        raise ValueError("all teams must have the same number of weekly scores")

    semi = [s[SEMIFINAL_WEEK - 1] for s in scores]
    if n_weeks >= FINAL_WEEKS[-1]:
        final = [s[FINAL_WEEKS[0] - 1] + s[FINAL_WEEKS[1] - 1] for s in scores]
    else:
        # Pre-2021 convention: no week 17 — final = weeks 15+16 summed
        # (week 15 shared with the semifinal; documented, pinned in tests).
        final = [s[SEMIFINAL_WEEK - 1] + s[SEMIFINAL_WEEK] for s in scores]
    points_for = [sum(s[:REGULAR_SEASON_WEEKS]) for s in scores]

    rng = random.Random(seed)
    wins_sum = [0.0] * teams
    playoff_ct = [0] * teams
    title_ct = [0] * teams
    order = list(range(teams))

    def playoff_game(a: int, b: int, game_points: Sequence[float]) -> int:
        """a is the better seed; ties break points-for then better seed."""
        if game_points[a] != game_points[b]:
            return a if game_points[a] > game_points[b] else b
        if points_for[a] != points_for[b]:
            return a if points_for[a] > points_for[b] else b
        return a

    for _ in range(n_sims):
        wins = [0.0] * teams
        for wk in range(REGULAR_SEASON_WEEKS):
            rng.shuffle(order)
            for i in range(0, teams, 2):
                a, b = order[i], order[i + 1]
                if scores[a][wk] > scores[b][wk]:
                    wins[a] += 1.0
                elif scores[b][wk] > scores[a][wk]:
                    wins[b] += 1.0
                else:
                    wins[a] += 0.5
                    wins[b] += 0.5
        seeds = sorted(range(teams), key=lambda t: (-wins[t], -points_for[t], t))
        seeds = seeds[:PLAYOFF_TEAMS]
        for t in seeds:
            playoff_ct[t] += 1
        finalist_a = playoff_game(seeds[0], seeds[3], semi)  # 1v4
        finalist_b = playoff_game(seeds[1], seeds[2], semi)  # 2v3
        hi, lo = (
            (finalist_a, finalist_b)
            if seeds.index(finalist_a) < seeds.index(finalist_b)
            else (finalist_b, finalist_a)
        )
        title_ct[playoff_game(hi, lo, final)] += 1
        for t in range(teams):
            wins_sum[t] += wins[t]

    return [
        {
            "expected_wins": round(wins_sum[t] / n_sims, 2),
            "playoff_pct": round(playoff_ct[t] / n_sims, 4),
            "title_pct": round(title_ct[t] / n_sims, 4),
        }
        for t in range(teams)
    ]


# ---------------------------------------------------------------------------
# Combined per-roster report fields


def normalized_metrics(
    weekly_scores_by_team: Sequence[Sequence[float]],
    team_index: int,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = 0,
) -> dict:
    """League Index + rank/percentile + championship odds for one team.

    weekly_scores_by_team[i] = team i's realized weekly points under ONE
    lineup policy applied to every team in the same simulated league.
    """
    teams = len(weekly_scores_by_team)
    if not 0 <= team_index < teams:
        raise ValueError(f"team_index {team_index} out of range for {teams} teams")
    totals = [sum(w) for w in weekly_scores_by_team]
    mine = totals[team_index]
    opponents = [t for i, t in enumerate(totals) if i != team_index]
    odds = championship_odds(weekly_scores_by_team, n_sims=n_sims, seed=seed)[team_index]
    return {
        "league_index": league_index(mine, opponents),
        "league_rank": league_rank(mine, opponents),
        "league_percentile": league_percentile(mine, opponents),
        "expected_wins": odds["expected_wins"],
        "playoff_pct": odds["playoff_pct"],
        "title_pct": odds["title_pct"],
    }


# ---------------------------------------------------------------------------
# Record translation: `python -m evals.normalize --rescore`
# (heavy imports kept inside functions — the metric layer above is stdlib-only)


def _fmt_pct(p: float) -> str:
    return f"{100.0 * p:.0f}%"


def _grid_rescore(champ_sims: int) -> list[dict]:
    """Re-run the heuristic DraftBench grids (deterministic, simulation-local)
    with normalized reporting on. Exact: same seeds, same simulator."""
    import statistics

    from evals.agents_execution import EXECUTION_AGENTS
    from evals.draftbench import BASELINE_AGENTS, run

    rows = run(out_path=None, verbose=False, agents=BASELINE_AGENTS, champ_sims=champ_sims)
    rows += run(out_path=None, verbose=False, agents=EXECUTION_AGENTS, champ_sims=champ_sims)

    control = {}
    for r in rows:
        if r["agent"] == "autopick_adp":
            control[(r["season"], r["format"], r["slot"], r["seed"])] = r

    out = []
    order = list(dict.fromkeys(r["agent"] for r in rows))
    for agent in order:
        ars = [r for r in rows if r["agent"] == agent]
        ctrls = [control[(r["season"], r["format"], r["slot"], r["seed"])] for r in ars]
        out.append(
            {
                "row": agent,
                "n": len(ars),
                "pac": statistics.mean(r["points_above_control"] for r in ars),
                "index": statistics.mean(r["league_index"] for r in ars),
                "title": statistics.mean(r["title_pct"] for r in ars),
                "playoff": statistics.mean(r["playoff_pct"] for r in ars),
                "ctrl_index": statistics.mean(c["league_index"] for c in ctrls),
                "ctrl_title": statistics.mean(c["title_pct"] for c in ctrls),
                "exact": True,
            }
        )
    return out


def _replay_episode(ep: dict, mask_names: bool, champ_sims: int) -> dict:
    """Replay one stored LLM DraftGym episode from its recorded picks.

    Opponents are fully seed-deterministic, so replaying the recorded picks
    reconstructs the entire league exactly; we verify by matching the stored
    reward. Also rebuilds the same-seed AutopickADP control league for the
    control's normalized numbers."""
    from envs.draftgym import DraftGym, realistic_weekly_lineups
    from evals.draftbench import AutopickADP, DraftSimulator

    gym = DraftGym(
        season=ep["season"],
        preset=ep["preset"],
        agent_slot=ep["agent_slot"],
        seed=ep["seed"],
        enable_evidence=False,
        mask_names=mask_names,
    )
    gym.reset()
    done = False
    reward = 0.0
    for pick in ep["picks"]:
        _, reward, done, info = gym.step({"pick": pick["action"]["pick"]})
    if not done:
        raise RuntimeError("replay did not finish the episode")
    exact = abs(reward - ep["reward"]) < 0.01

    control = DraftSimulator(gym.pool, gym.league).simulate(
        AutopickADP(), ep["agent_slot"], ep["seed"]
    )
    ctrl_weekly = [
        realistic_weekly_lineups(r, gym.weekly, gym.league) for r in control.rosters
    ]
    ctrl = normalized_metrics(
        ctrl_weekly, ep["agent_slot"] - 1, n_sims=champ_sims, seed=ep["seed"]
    )
    return {
        "reward_replayed": reward,
        "exact": exact,
        "index": info["league_index"],
        "rank": info["league_rank"],
        "playoff": info["playoff_pct"],
        "title": info["title_pct"],
        "ctrl_index": ctrl["league_index"],
        "ctrl_title": ctrl["title_pct"],
        "ctrl_playoff": ctrl["playoff_pct"],
    }


def rescore(out_path=None, champ_sims: int = DEFAULT_N_SIMS) -> str:
    """Recompute every headline result on the normalized scale and write
    evals/results/normalized_rescore.md."""
    import json
    import statistics
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    out_path = Path(out_path) if out_path else root / "evals" / "results" / "normalized_rescore.md"

    lines = [
        "# Normalized rescore — League Index + championship odds",
        "",
        "Every headline points result translated onto the league-anchored scale",
        "(evals/normalize.py). League Index: 100 = tied with the best OPPOSING",
        "team in the same simulated league (same seed, same lineup policy);",
        ">100 = outscored everyone. Title/playoff odds: schedule Monte Carlo",
        f"({champ_sims} sims/league, weeks 1-14 regular season, top-4 playoff,",
        "semi wk 15, final wks 16+17; deterministic under seed). Raw",
        "points-above-control stays the statistical backbone; this is the",
        "human-readable layer.",
        "",
        "## DraftBench heuristic grids (hindsight-optimal lineups, 2015-2024, n=405 each)",
        "",
        "Exact re-runs: same simulator, same seeds — simulation-local, no API calls.",
        "Cells are grid means; the control's own means sit in the last columns",
        "(the autopick control is NOT index 100 on average — even the league's",
        "best drafter usually trails somebody in an 11-opponent league).",
        "",
        "| agent | pts vs control | League Index | playoff% | title% | control index | control title% |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in _grid_rescore(champ_sims):
        lines.append(
            f"| {g['row']} | {g['pac']:+.1f} | {g['index']:.1f} | {_fmt_pct(g['playoff'])} "
            f"| {_fmt_pct(g['title'])} | {g['ctrl_index']:.1f} | {_fmt_pct(g['ctrl_title'])} |"
        )

    lines += [
        "",
        "## DraftGym LLM episodes (realistic-manager lineups)",
        "",
        "Reconstructed by replaying each stored episode's recorded picks against",
        "the seed-deterministic opponents (envs/draftgym.py) — the full league is",
        "recovered exactly; a row is 'exact' when the replayed reward matches the",
        "stored one to 0.01. Control = same-seed AutopickADP league, same seat.",
        "",
        "| contender | episode | reward (pts) | League Index | rank | playoff% | title% | control index | control title% | exact |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def ep_key(ep: dict) -> str:
        return f"{ep['season']} {ep['preset']} slot {ep['agent_slot']} seed {ep['seed']}"

    approx_notes: list[str] = []
    qwen = json.loads((root / "evals" / "results" / "draftgym_qwen_baseline.json").read_text())
    qwen_named, qwen_masked = [], []
    for ep in qwen["episodes"]:
        masked = bool(ep.get("mask_names"))
        r = _replay_episode(ep, masked, champ_sims)
        (qwen_masked if masked else qwen_named).append((ep, r))
        if not r["exact"]:
            approx_notes.append(f"qwen {ep_key(ep)} masked={masked}")

    def emit(label: str, ep: dict, r: dict) -> None:
        lines.append(
            f"| {label} | {ep_key(ep)} | {ep['reward']:+.1f} | {r['index']:.1f} | {r['rank']}/12 "
            f"| {_fmt_pct(r['playoff'])} | {_fmt_pct(r['title'])} | {r['ctrl_index']:.1f} "
            f"| {_fmt_pct(r['ctrl_title'])} | {'yes' if r['exact'] else 'NO — approximated'} |"
        )

    for ep, r in qwen_named:
        emit("Qwen3.5-9B named", ep, r)
    for ep, r in qwen_masked:
        emit("Qwen3.5-9B masked", ep, r)

    frontier = json.loads((root / "evals" / "results" / "frontier_draftgym.json").read_text())
    frontier_rows = []
    for model_key, blob in frontier.items():
        if not isinstance(blob, dict) or "episodes" not in blob:
            continue
        for ep in blob["episodes"]:
            masked = bool(ep.get("mask_names"))
            r = _replay_episode(ep, masked, champ_sims)
            label = f"{blob.get('model', model_key)} {'masked' if masked else 'named'}"
            emit(label, ep, r)
            frontier_rows.append((label, ep, r))
            if not r["exact"]:
                approx_notes.append(f"{model_key} {ep_key(ep)} masked={masked}")

    def mean_line(label: str, pairs: list) -> str:
        if not pairs:
            return ""
        rw = statistics.mean(ep["reward"] for ep, _ in pairs)
        ix = statistics.mean(r["index"] for _, r in pairs)
        tt = statistics.mean(r["title"] for _, r in pairs)
        pl = statistics.mean(r["playoff"] for _, r in pairs)
        ci = statistics.mean(r["ctrl_index"] for _, r in pairs)
        ct = statistics.mean(r["ctrl_title"] for _, r in pairs)
        return (
            f"| {label} (mean, n={len(pairs)}) | — | {rw:+.1f} | {ix:.1f} | — "
            f"| {_fmt_pct(pl)} | {_fmt_pct(tt)} | {ci:.1f} | {_fmt_pct(ct)} | — |"
        )

    lines.append(mean_line("Qwen3.5-9B named", qwen_named))
    lines.append(mean_line("Qwen3.5-9B masked", qwen_masked))

    lines += [
        "",
        "## Approximation ledger",
        "",
    ]
    if approx_notes:
        lines.append("Rows whose replayed reward did NOT match the stored artifact (approximated):")
        lines += [f"- {n}" for n in approx_notes]
    else:
        lines.append(
            "None — every LLM episode replay reproduced its stored reward exactly, and"
        )
        lines.append(
            "the heuristic grids are exact deterministic re-runs. No approximated rows."
        )
    lines.append("")

    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Normalized season scoring (league index + title odds)")
    ap.add_argument("--rescore", action="store_true", help="recompute the record onto the normalized scale")
    ap.add_argument("--sims", type=int, default=DEFAULT_N_SIMS, help="Monte Carlo sims per league")
    ap.add_argument("--out", default=None, help="rescore output path")
    args = ap.parse_args(argv)
    if args.rescore:
        print(rescore(out_path=args.out, champ_sims=args.sims))
        return 0
    ap.error("nothing to do (pass --rescore)")
    return 2


if __name__ == "__main__":
    import sys

    sys.exit(main())
