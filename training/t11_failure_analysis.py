"""Reproducible, anonymized diagnosis of the first Tinker SFT evaluation."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.tinker_sft_scorecard import _prediction_rows
from envs.draftgym import DraftGym
from training.tinker_backend import ROOT

V1_ROOT = ROOT / "artifacts/tinker-sft-v1/evaluation"
SCORECARD_PATH = V1_ROOT / "scorecard.json"
OUTPUT_PATH = ROOT / "artifacts/tinker-sft-t11/t1-failure-analysis.json"
REPORT_PATH = ROOT / "docs/tinker-sft-t11-failure-analysis.md"


def prediction_diagnostics(
    base: Mapping[str, Mapping[str, Any]],
    adapter: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    common = sorted(set(base) & set(adapter))
    rows = []
    by_family: dict[str, list[float]] = defaultdict(list)
    for key in common:
        b, a = base[key], adapter[key]
        y = int(b["y"])
        bp, ap = float(b["p"]), float(a["p"])
        improvement = (bp - y) ** 2 - (ap - y) ** 2
        family = f"{b['benchmark']}:{b['family']}"
        by_family[family].append(improvement)
        rows.append({
            "key": key, "benchmark": b["benchmark"], "family": b["family"],
            "season": int(b["season"]), "question_id": b["question_id"],
            "y": y, "base_p": bp, "adapter_p": ap,
            "brier_improvement": improvement,
            "adapter_high_confidence_wrong": (
                (y == 1 and ap <= 0.15) or (y == 0 and ap >= 0.85)
            ),
            "adapter_more_extreme": abs(ap - 0.5) > abs(bp - 0.5),
        })
    high_wrong = [r for r in rows if r["adapter_high_confidence_wrong"]]
    base_high_wrong = [
        r for r in rows
        if (r["y"] == 1 and r["base_p"] <= 0.15)
        or (r["y"] == 0 and r["base_p"] >= 0.85)
    ]
    more_extreme = [r for r in rows if r["adapter_more_extreme"]]
    return {
        "matched_examples": len(rows),
        "mean_brier_improvement": statistics.fmean(r["brier_improvement"] for r in rows),
        "adapter_high_confidence_wrong": len(high_wrong),
        "adapter_high_confidence_wrong_rate": len(high_wrong) / len(rows),
        "base_high_confidence_wrong": len(base_high_wrong),
        "base_high_confidence_wrong_rate": len(base_high_wrong) / len(rows),
        "adapter_more_extreme": len(more_extreme),
        "adapter_more_extreme_rate": len(more_extreme) / len(rows),
        "more_extreme_and_worse": sum(
            r["adapter_more_extreme"] and r["brier_improvement"] < 0 for r in rows
        ),
        "by_family_mean_brier_improvement": {
            key: statistics.fmean(values) for key, values in sorted(by_family.items())
        },
        "largest_overconfident_misses": sorted(
            high_wrong, key=lambda r: r["brier_improvement"]
        )[:15],
    }


def _replay(path: Path) -> dict[str, Any]:
    episode = json.loads(path.read_text())
    gym = DraftGym(
        season=episode["season"], preset=episode["preset"],
        agent_slot=episode["agent_slot"], seed=episode["seed"],
        mask_names=True, enable_evidence=False,
    )
    obs = gym.reset()
    picks = []
    info: dict[str, Any] = {}
    for stored in episode["picks"]:
        masked = stored["action"]["pick"]
        chosen = next(x for x in obs["top_available"] if x["player_id"] == masked)
        picks.append({
            "pick_number": int(obs["pick_number"]), "round": int(obs["round"]),
            "player_id": masked, "position": chosen["position"],
            "adp": chosen["adp"], "adp_stdev": chosen["adp_stdev"],
            "prev_season_points": chosen["prev_season_points"],
        })
        obs, reward, done, info = gym.step(stored["action"])
    ledger = []
    for row in info.get("value_ledger") or []:
        ledger.append({
            "player_id": gym._to_masked[row["player_id"]],
            "pick_number": row["pick_number"], "round": row["round"],
            "position": row["position"], "capture": row["capture"],
            "season_ending_acute": row["season_ending_acute"],
            "basis": row["basis"],
        })
    by_id = {row["player_id"]: row for row in ledger}
    for pick in picks:
        pick.update({
            "realized_capture_diagnostic": by_id[pick["player_id"]]["capture"],
            "season_ending_acute": by_id[pick["player_id"]]["season_ending_acute"],
        })
    positions = Counter(p["position"] for p in picks)
    return {
        "season": episode["season"], "agent_slot": episode["agent_slot"],
        "seed": episode["seed"], "reward": float(episode["reward"]),
        "positions": dict(sorted(positions.items())), "picks": picks,
        "tool_calls": int(episode["tool_calls"]),
    }


def _episode_diagnostics(base: Mapping[str, Any], adapter: Mapping[str, Any]) -> dict[str, Any]:
    first = next(
        (i for i, (b, a) in enumerate(zip(base["picks"], adapter["picks"], strict=True))
         if b["player_id"] != a["player_id"]),
        None,
    )
    base_ids = {p["player_id"] for p in base["picks"]}
    adapter_only = [p for p in adapter["picks"] if p["player_id"] not in base_ids]
    return {
        "season": base["season"], "agent_slot": base["agent_slot"], "seed": base["seed"],
        "base_reward": base["reward"], "adapter_reward": adapter["reward"],
        "delta": adapter["reward"] - base["reward"],
        "first_divergence": None if first is None else {
            "round": base["picks"][first]["round"],
            "pick_number": base["picks"][first]["pick_number"],
            "base_choice": base["picks"][first],
            "adapter_choice": adapter["picks"][first],
            "matched_state": True,
        },
        "base_positions": base["positions"], "adapter_positions": adapter["positions"],
        "adapter_only_best_capture_diagnostic": (
            max(adapter_only, key=lambda p: p["realized_capture_diagnostic"])
            if adapter_only else None
        ),
        "adapter_only_worst_capture_diagnostic": (
            min(adapter_only, key=lambda p: p["realized_capture_diagnostic"])
            if adapter_only else None
        ),
    }


def _draft_counts(episodes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    picks = [p for episode in episodes for p in episode["picks"]]
    return {
        "early_qb_or_te_before_round_5": sum(
            p["round"] < 5 and p["position"] in {"QB", "TE"} for p in picks
        ),
        "late_negative_capture_diagnostic": sum(
            p["round"] >= 11 and p["realized_capture_diagnostic"] < 0 for p in picks
        ),
        "zero_history_picks": sum(float(p["prev_season_points"]) <= 0 for p in picks),
        "acute_injury_outcomes": sum(bool(p["season_ending_acute"]) for p in picks),
        "episodes_with_more_than_two_qbs": sum(
            int(episode["positions"].get("QB", 0)) > 2 for episode in episodes
        ),
        "episodes_with_more_than_two_tes": sum(
            int(episode["positions"].get("TE", 0)) > 2 for episode in episodes
        ),
        "tool_calls": sum(int(episode["tool_calls"]) for episode in episodes),
    }


def analyze() -> dict[str, Any]:
    scorecard = json.loads(SCORECARD_PATH.read_text())
    prediction = prediction_diagnostics(
        _prediction_rows(V1_ROOT / "base"), _prediction_rows(V1_ROOT / "adapter")
    )
    base_episodes, adapter_episodes, pairs = [], [], []
    for entry in scorecard["paired_draftgym"]["episodes"]:
        stem = (
            f"draftgym_{entry['season']}_slot{entry['agent_slot']}_seed{entry['seed']}.json"
        )
        base = _replay(V1_ROOT / "base" / stem)
        adapter = _replay(V1_ROOT / "adapter" / stem)
        base_episodes.append(base)
        adapter_episodes.append(adapter)
        pairs.append(_episode_diagnostics(base, adapter))

    from training.tinker_backend import load_corpus

    corpus = load_corpus()
    authored_actions = 0
    authored_tools = 0
    for row in corpus:
        text = row["messages"][-1]["content"]
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("pick" in obj or "tool" in obj):
            authored_actions += 1
            authored_tools += int("tool" in obj)

    result = {
        "protocol": "fantasy-alpha-t1-diagnosis-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "scorecard": str(SCORECARD_PATH.relative_to(ROOT)),
            "paired_prediction_examples": prediction["matched_examples"],
            "paired_draft_episodes": len(pairs),
        },
        "training_data_gap": {
            "t1_rows": len(corpus),
            "model_authored_draft_actions": authored_actions,
            "model_authored_tool_calls": authored_tools,
            "evidence_appeared_only_as_prefetched_user_context": True,
        },
        "probability": {
            **prediction,
            "base_log_loss": scorecard["results"]["base"]["calib"]["overall"]["log_loss"],
            "adapter_log_loss": scorecard["results"]["adapter"]["calib"]["overall"]["log_loss"],
        },
        "draftgym": {
            "mean_delta": scorecard["paired_draftgym"]["mean_delta"],
            "median_delta": scorecard["paired_draftgym"]["median_delta"],
            "wins": scorecard["paired_draftgym"]["wins"],
            "losses": scorecard["paired_draftgym"]["losses"],
            "base_error_counts": _draft_counts(base_episodes),
            "adapter_error_counts": _draft_counts(adapter_episodes),
            "episodes": pairs,
            "capture_warning": (
                "Realized capture is post-outcome attribution only, not a causal label or input."
            ),
        },
        "evidence_strength": {
            "observed": [
                "T1 contains no model-authored draft or tool actions.",
                "Neither evaluated arm called a tool.",
                "T1 won only two of six paired drafts and had a negative median delta.",
                "Both arms repeatedly over-drafted QB/TE bench depth.",
                "T1 produced more extreme high-confidence misses and worse log loss.",
            ],
            "hypotheses": [
                "Explicit roster/timing demonstrations may reduce avoidable bench imbalance.",
                "Conservative outcome-blind shrinkage may reduce damaging probability tails.",
                "Paired tool-call and post-result demonstrations may teach selective evidence use.",
            ],
        },
    }
    return result


def write_report(result: Mapping[str, Any]) -> None:
    p = result["probability"]
    d = result["draftgym"]
    gap = result["training_data_gap"]
    lines = [
        "# T1 failure analysis — what T1.1 must fix",
        "",
        "This analysis uses only the already-opened 2018/2023/2024 development",
        "results. Player examples remain anonymized. The 2025 gate was not opened.",
        "",
        "## Root finding",
        "",
        f"The clearest issue is supervision mismatch: T1 had **{gap['t1_rows']}**",
        f"forecast traces but **{gap['model_authored_draft_actions']} model-authored",
        f"draft actions** and **{gap['model_authored_tool_calls']} tool calls**. Evidence",
        "was pre-fetched into the user message, so the model never learned when to ask",
        "for it. The draft improvement therefore cannot be credited to learned draft",
        "process from this corpus.",
        "",
        "## Error taxonomy",
        "",
        "| observed failure | evidence | T1.1 intervention |",
        "|---|---|---|",
        f"| Unstable draft reward | mean delta {d['mean_delta']:+.2f}, median {d['median_delta']:+.2f}; {d['wins']}/6 wins | more matched drafts plus explicit action supervision |",
        f"| QB/TE bench overinvestment | T1 had {d['adapter_error_counts']['episodes_with_more_than_two_qbs']} episodes with >2 QBs and {d['adapter_error_counts']['episodes_with_more_than_two_tes']} with >2 TEs | roster guardrail and recovery traces |",
        f"| Late-round misses | {d['adapter_error_counts']['late_negative_capture_diagnostic']} T1 late picks had negative realized-capture diagnostics | late-round opportunity-cost examples |",
        f"| Unsupported confidence tails | high-confidence misses {p['base_high_confidence_wrong']}→{p['adapter_high_confidence_wrong']}; log loss {p['base_log_loss']:.4f}→{p['adapter_log_loss']:.4f} | outcome-blind shrinkage and confidence correction |",
        f"| No evidence seeking | {d['adapter_error_counts']['tool_calls']} tool calls in T1 evaluation | paired selective tool-call/post-result traces |",
        "",
        "Realized capture is used below only to explain completed drafts. It is hindsight",
        "attribution, not a training input and not proof that an earlier choice caused the",
        "terminal reward difference.",
        "",
        "## Six paired drafts",
        "",
        "| season/slot | delta | first matched-state divergence | T1 position mix | best T1-only capture diagnostic |",
        "|---|---:|---|---|---|",
    ]
    for episode in d["episodes"]:
        first = episode["first_divergence"]
        divergence = "none" if first is None else (
            f"round {first['round']}: {first['base_choice']['player_id']} "
            f"→ {first['adapter_choice']['player_id']}"
        )
        best = episode["adapter_only_best_capture_diagnostic"]
        best_text = "—" if best is None else (
            f"{best['player_id']} (round {best['round']}, "
            f"{best['realized_capture_diagnostic']:+.1f})"
        )
        mix = ", ".join(f"{k}{v}" for k, v in episode["adapter_positions"].items())
        lines.append(
            f"| {episode['season']}/slot {episode['agent_slot']} | {episode['delta']:+.2f} | "
            f"{divergence} | {mix} | {best_text} |"
        )
    lines += [
        "",
        "The two large wins contain valuable examples, but they do not establish a",
        "repeatable policy: one first diverged immediately and the other only in round",
        "eight, while four paired drafts regressed. T1.1 therefore teaches the process",
        "directly and precommits to median and paired-win criteria rather than relying on",
        "another favorable mean.",
        "",
        "## Evidence versus hypotheses",
        "",
        "Observed facts:",
        "",
    ]
    lines.extend(f"- {x}" for x in result["evidence_strength"]["observed"])
    lines += ["", "Testable hypotheses for T1.1:", ""]
    lines.extend(f"- {x}" for x in result["evidence_strength"]["hypotheses"])
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = analyze()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report(result)
    print(json.dumps({
        "output": str(OUTPUT_PATH.relative_to(ROOT)),
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "draft_wins": result["draftgym"]["wins"],
        "high_confidence_misses": result["probability"]["adapter_high_confidence_wrong"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
