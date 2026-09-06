"""Quantify why T1.1 failed before defining the T1.2 recipe."""
from __future__ import annotations

import argparse
import difflib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from training.t11_dataset import _extract_observation, load_t11_corpus
from training.tinker_backend import ROOT, RunConfig, load_corpus, render_rows

OUTPUT_PATH = ROOT / "artifacts/tinker-sft-t12/t11-collapse-analysis.json"
REPORT_PATH = ROOT / "docs/tinker-sft-t12-failure-analysis.md"
T11_SCORECARD = ROOT / "artifacts/tinker-sft-t11/evaluation/scorecard.json"
T1_SUMMARY = ROOT / "artifacts/tinker-sft-v1/t1-full/run-summary.json"
T11_SUMMARY = ROOT / "artifacts/tinker-sft-t11/t11-full/run-summary.json"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _target_tokens(rows: Sequence[dict[str, Any]]) -> list[int]:
    datums, _, _ = render_rows(rows, RunConfig(run_name="t12-failure-analysis"))
    return [
        sum(float(value) > 0 for value in datum.loss_fn_inputs["weights"].data)
        for datum in datums
    ]


def _response_diversity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_schema: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        schema = str(row["meta"].get("t11_subtype", row["meta"]["family"]))
        by_schema[schema].append(_normalize(row["messages"][-1]["content"]))
    result: dict[str, Any] = {}
    for schema, texts in sorted(by_schema.items()):
        counts = Counter(texts)
        near_pairs = 0
        if len(texts) <= 150:
            unique = sorted(counts)
            for index, left in enumerate(unique):
                near_pairs += sum(
                    difflib.SequenceMatcher(None, left, right).ratio() >= 0.90
                    for right in unique[index + 1:]
                )
        result[schema] = {
            "rows": len(texts),
            "unique_responses": len(counts),
            "exact_duplicate_rows": len(texts) - len(counts),
            "largest_exact_duplicate_group": max(counts.values()),
            "near_duplicate_unique_pairs_at_0_90": near_pairs,
        }
    return result


def _draft_label_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    picks = []
    tools = []
    for row in rows:
        subtype = row["meta"].get("t11_subtype")
        if subtype not in {"direct_pick", "post_tool_pick", "tool_call"}:
            continue
        action = json.loads(row["messages"][-1]["content"])
        obs = _extract_observation(row["messages"][1]["content"])
        visible = list(obs["top_available"])
        if "pick" in action:
            adp_leader = min(visible, key=lambda candidate: float(candidate["adp"]))
            selected = next(
                candidate for candidate in visible
                if str(candidate["player_id"]) == str(action["pick"])
            )
            picks.append({
                "subtype": subtype,
                "round": int(obs["round"]),
                "pick": str(action["pick"]),
                "position": str(selected["position"]),
                "equals_visible_adp_leader": str(action["pick"]) == str(adp_leader["player_id"]),
                "adp_rank_among_visible": 1 + sorted(
                    visible, key=lambda candidate: float(candidate["adp"])
                ).index(selected),
            })
        else:
            tool = action["tool"]
            tools.append({
                "name": tool["name"],
                "target": tool["arguments"]["team_or_player"],
                "round": int(obs["round"]),
            })
    position_counts = Counter(pick["position"] for pick in picks)
    return {
        "pick_labels": len(picks),
        "unique_pick_action_strings": len({pick["pick"] for pick in picks}),
        "position_labels": dict(sorted(position_counts.items())),
        "rounds": {
            "early_1_4": sum(pick["round"] <= 4 for pick in picks),
            "middle_5_10": sum(5 <= pick["round"] <= 10 for pick in picks),
            "late_11_plus": sum(pick["round"] >= 11 for pick in picks),
        },
        "equals_visible_adp_leader": sum(pick["equals_visible_adp_leader"] for pick in picks),
        "equals_visible_adp_leader_rate": statistics.fmean(
            pick["equals_visible_adp_leader"] for pick in picks
        ),
        "median_visible_adp_rank": statistics.median(
            pick["adp_rank_among_visible"] for pick in picks
        ),
        "tool_labels": len(tools),
        "unique_tool_targets": len({tool["target"] for tool in tools}),
        "tool_round_range": [min(tool["round"] for tool in tools), max(tool["round"] for tool in tools)],
    }


def analyze() -> dict[str, Any]:
    broad = load_corpus()
    targeted = load_t11_corpus()
    broad_tokens = _target_tokens(broad)
    targeted_tokens = _target_tokens(targeted)
    scorecard = json.loads(T11_SCORECARD.read_text())
    t1_summary = json.loads(T1_SUMMARY.read_text())
    t11_summary = json.loads(T11_SUMMARY.read_text())

    broad_prompt = {
        _normalize("\n".join(message["content"] for message in row["messages"][:-1])): row
        for row in broad
    }
    targeted_prompt = {
        _normalize("\n".join(message["content"] for message in row["messages"][:-1])): row
        for row in targeted
    }
    overlaps = sorted(set(broad_prompt) & set(targeted_prompt))
    conflicting = sum(
        _normalize(broad_prompt[key]["messages"][-1]["content"])
        != _normalize(targeted_prompt[key]["messages"][-1]["content"])
        for key in overlaps
    )
    target_by_schema: Counter[str] = Counter()
    input_by_schema: Counter[str] = Counter()
    datums, _, _ = render_rows(targeted, RunConfig(run_name="t12-failure-analysis"))
    for row, datum, target in zip(targeted, datums, targeted_tokens, strict=True):
        subtype = str(row["meta"]["t11_subtype"])
        target_by_schema[subtype] += target
        input_by_schema[subtype] += datum.model_input.length

    t11_result = scorecard["results"]["t11"]
    draft = t11_result["draftgym"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uses_existing_artifacts_only": True,
        "sealed_2025_untouched": True,
        "supervision_balance": {
            "t1_rows": len(broad),
            "t11_rows": len(targeted),
            "t1_assistant_target_tokens": sum(broad_tokens),
            "t11_assistant_target_tokens": sum(targeted_tokens),
            "unweighted_t11_target_share": sum(targeted_tokens) / (
                sum(broad_tokens) + sum(targeted_tokens)
            ),
            "t11_rendered_input_tokens_by_subtype": dict(sorted(input_by_schema.items())),
            "t11_assistant_target_tokens_by_subtype": dict(sorted(target_by_schema.items())),
        },
        "response_diversity": {
            "t1": _response_diversity(broad),
            "t11": _response_diversity(targeted),
        },
        "prompt_conflicts": {
            "overlapping_prompts": len(overlaps),
            "different_targets": conflicting,
            "resolution": "T1.2 supersedes the old T1 answer with the T1.1 correction",
        },
        "draft_label_policy": _draft_label_analysis(targeted),
        "training_strength": {
            "t1": {
                "learning_rate": t1_summary["config"]["learning_rate"],
                "epochs": t1_summary["config"]["epochs"],
                "development_nll": t1_summary["development_nll_history"],
            },
            "t11": {
                "learning_rate": t11_summary["config"]["learning_rate"],
                "epochs": t11_summary["config"]["epochs"],
                "development_nll": t11_summary["development_nll_history"],
                "relative_nll_drop": 1 - (
                    t11_summary["final_development_nll"]
                    / t11_summary["initial_development_nll"]
                ),
            },
        },
        "observed_product_failures": {
            "structured_coverage": t11_result["structured_coverage"],
            "prediction_families_failed_after_repair": sum(
                not record.get("parse_success", True)
                for record in json.loads(
                    (ROOT / "artifacts/tinker-sft-t11/evaluation/t11/run-state.json").read_text()
                )["records"]
                if record["benchmark"] != "draftgym"
            ),
            "tool_calls": draft["tool_calls"],
            "tool_opportunities": draft["tool_opportunities"],
            "illegal_picks": draft["illegal_picks"],
            "zero_reward_draft_episodes": sum(
                float(episode["reward"]) == 0.0 for episode in draft["episodes_detail"]
            ),
            "draft_episodes": draft["episodes"],
            "wins_vs_base": scorecard["comparisons"]["draftgym"]["base"]["wins"],
        },
        "evidence": [
            "Short targeted labels supplied only about 3% of unweighted assistant-token loss.",
            "The combined sources contain 120 same-prompt/different-answer forecast conflicts.",
            "T1.1 development NLL fell by more than 90% while frozen behavior regressed.",
            "T1.1 emitted no tools and mostly reproduced the control reward in DraftGym.",
        ],
        "hypotheses": [
            "Two epochs at 3e-4 on the narrow corpus likely over-specialized the adapter.",
            "Row-count balancing hid the much smaller targeted assistant-token signal.",
            "Selecting only by NLL favored memorization and ignored retained format behavior.",
            "Rule-policy imitation can teach discipline but cannot guarantee reward improvement.",
        ],
    }


def write_report(payload: Mapping[str, Any]) -> None:
    balance = payload["supervision_balance"]
    policy = payload["draft_label_policy"]
    failures = payload["observed_product_failures"]
    strength = payload["training_strength"]["t11"]
    lines = [
        "# T1.1 collapse analysis — inputs to T1.2",
        "",
        "This analysis uses only existing development artifacts. The 2025 holdout remains sealed.",
        "",
        "## What the data proves",
        "",
        (
            f"- T1 supplied {balance['t1_assistant_target_tokens']:,} supervised answer tokens; "
            f"T1.1 supplied only {balance['t11_assistant_target_tokens']:,}. Despite 300 of 1,111 "
            f"source rows being targeted, they represented just {balance['unweighted_t11_target_share']:.1%} "
            "of the unweighted answer-token signal."
        ),
        (
            f"- There are {payload['prompt_conflicts']['different_targets']} same-prompt/different-answer "
            "forecast pairs. Keeping both would train contradictory targets."
        ),
        (
            f"- T1.1 NLL fell {strength['relative_nll_drop']:.1%}, yet structured coverage was "
            f"{failures['structured_coverage']:.1%}, {failures['prediction_families_failed_after_repair']} "
            f"prediction families failed, and {failures['illegal_picks']} illegal picks appeared."
        ),
        (
            f"- T1.1 made {failures['tool_calls']} tool calls over {failures['tool_opportunities']} "
            f"opportunities; {failures['zero_reward_draft_episodes']}/{failures['draft_episodes']} "
            "drafts exactly matched the control reward."
        ),
        (
            f"- Of {policy['pick_labels']} draft labels, {policy['equals_visible_adp_leader_rate']:.1%} "
            "select the visible ADP leader. This is a direct descriptive proxy, not proof that the "
            "rule teacher equals the environment's stochastic control policy."
        ),
        "",
        "## Evidence-backed T1.2 response",
        "",
        "T1.2 removes conflicting old forecast targets, weights by assistant-token loss rather",
        "than row count, lowers the learning rate from 3e-4 to 1e-4, uses one epoch, and selects",
        "checkpoints by format, legality, tool behavior, and task quality before NLL.",
        "",
        "## Hypotheses, not established facts",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["hypotheses"])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    payload = analyze()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
