"""Fit-free held-out and matched-arm evaluation for value of information."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from envs.play_llm_supervised import run_episode
from evals.value_of_information import (
    ARTIFACT_DIR,
    HARD_CAP_USD,
    PROTOCOL as SOURCE_PROTOCOL,
    BudgetedSampler,
    arm_evaluation_panel,
    collected_cost,
    load_collected_episodes,
    usage_cost,
    verify_frozen_protocol,
)
from harness.tool_supervisor import (
    AlwaysActSupervisor,
    HardCodedToolSupervisor,
    LearnedToolSupervisor,
    SupervisorDecision,
    relevant_tool_result,
    visible_candidate,
)
from harness.value_supervisor import (
    AlwaysEvidenceOnceSupervisor,
    OutcomeValueRuleSupervisor,
    OutcomeValueSupervisor,
)
from training.tinker_backend import BASE_MODEL, ROOT, TinkerChatSampler, sha256_file, utc_now
from training.value_of_information_supervisor import (
    ARTIFACT_PATH,
    CANDIDATE_FREEZE_PATH,
    DATASET_PATH,
    EXISTING_ARTIFACT_PATH,
    PROTOCOL,
    score_policy,
)

HELDOUT_PATH = ROOT / "evals/frozen/value_of_information_v1/heldout.jsonl"
HELDOUT_RESULT = ARTIFACT_DIR / "heldout-counterfactual-scorecard.json"
ARM_RECORDS = ARTIFACT_DIR / "matched-arm-records.jsonl"
ARM_RESULT = ARTIFACT_DIR / "matched-arm-scorecard.json"
FINAL_SCORECARD = ARTIFACT_DIR / "scorecard.json"
SPEND_AUDIT = ARTIFACT_DIR / "spend-audit.json"
ARM_NAMES = (
    "base_qwen", "existing_supervisor", "outcome_value_rule",
    "learned_outcome_value",
)


class PostEvidenceCompletionSupervisor:
    """Approve a reconsidered pick after one successful relevant lookup."""

    def __init__(self, inner: Any):
        self.inner = inner

    def decide(
        self, observation: Mapping[str, Any], proposed_action: Mapping[str, Any],
    ) -> SupervisorDecision:
        candidate = visible_candidate(observation, proposed_action)
        if candidate is not None:
            status, _ = relevant_tool_result(
                observation, str(candidate["player_id"]),
            )
            if status == "success":
                return SupervisorDecision(
                    "ACT_NOW", 1.0, "post_evidence_reconsideration_complete",
                )
        return self.inner.decide(observation, proposed_action)


def evaluation_panel() -> list[dict[str, Any]]:
    return arm_evaluation_panel()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def verify_candidates() -> dict[str, Any]:
    verify_frozen_protocol()
    freeze = json.loads(CANDIDATE_FREEZE_PATH.read_text())
    if freeze.get("protocol") != PROTOCOL:
        raise ValueError("unexpected candidate-freeze protocol")
    if freeze.get("heldout_labels_seen") is not False:
        raise ValueError("candidate freeze must predate heldout inspection")
    checks = {
        ARTIFACT_PATH: freeze["artifact_sha256"],
        DATASET_PATH: freeze["development_dataset_sha256"],
        EXISTING_ARTIFACT_PATH: freeze["existing_supervisor_artifact_sha256"],
    }
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen candidate input changed: {path.name}")
    if tuple(freeze["candidate_systems"]) != ARM_NAMES:
        raise ValueError("candidate arms differ from frozen manifest")
    frozen_ids = freeze.get("evaluation_episode_ids")
    if frozen_ids is not None and frozen_ids != [
        row["episode_id"] for row in evaluation_panel()
    ]:
        raise ValueError("matched-arm panel differs from frozen manifest")
    return freeze


def _heldout_rows() -> list[dict[str, Any]]:
    rows = _read_jsonl(HELDOUT_PATH)
    if len(rows) != 70:
        raise ValueError(f"expected 70 heldout examples, found {len(rows)}")
    if any(row.get("split") != "heldout" for row in rows):
        raise ValueError("heldout evaluator received a non-heldout row")
    if any(row.get("protocol") != SOURCE_PROTOCOL for row in rows):
        raise ValueError("unexpected heldout source protocol")
    return rows


def _uses_tool(supervisor: Any, row: Mapping[str, Any]) -> bool:
    decision = supervisor.decide(
        row["input"]["observation"], row["input"]["proposed_action"],
    )
    return decision.decision == "USE_TOOL"


def _value_probability(supervisor: Any, row: Mapping[str, Any]) -> float:
    decision = supervisor.decide(
        row["input"]["observation"], row["input"]["proposed_action"],
    )
    return (
        float(decision.confidence)
        if decision.decision == "USE_TOOL"
        else 1.0 - float(decision.confidence)
    )


def _calibration(
    rows: Sequence[Mapping[str, Any]], probabilities: Sequence[float],
) -> dict[str, Any]:
    truth = [float(row["label"]["value"] == "HELPFUL") for row in rows]
    deltas = [float(row["outcome"]["reward_difference"]) for row in rows]
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, truth, strict=True)) / len(rows)
    ranked = sorted(
        zip(probabilities, truth, strict=True), key=lambda pair: pair[0], reverse=True,
    )
    positives = sum(truth)
    precision_sum = 0.0
    hits = 0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    mean_p = sum(probabilities) / len(probabilities)
    mean_delta = sum(deltas) / len(deltas)
    covariance = sum(
        (p - mean_p) * (delta - mean_delta)
        for p, delta in zip(probabilities, deltas, strict=True)
    )
    p_scale = math.sqrt(sum((p - mean_p) ** 2 for p in probabilities))
    d_scale = math.sqrt(sum((delta - mean_delta) ** 2 for delta in deltas))
    correlation = covariance / (p_scale * d_scale) if p_scale and d_scale else 0.0
    bins = []
    for low in (0.0, 0.2, 0.4, 0.6, 0.8):
        indices = [
            index for index, value in enumerate(probabilities)
            if low <= value < low + 0.2 or (low == 0.8 and value == 1.0)
        ]
        if indices:
            bins.append({
                "low": low,
                "high": low + 0.2,
                "count": len(indices),
                "mean_prediction": sum(probabilities[index] for index in indices)
                / len(indices),
                "observed_helpful_rate": sum(truth[index] for index in indices)
                / len(indices),
                "mean_reward_difference": sum(deltas[index] for index in indices)
                / len(indices),
            })
    return {
        "prevalence": positives / len(rows),
        "brier": brier,
        "average_precision": precision_sum / max(1, positives),
        "probability_reward_correlation": correlation,
        "bins": bins,
    }


def evaluate_counterfactuals() -> dict[str, Any]:
    freeze = verify_candidates()
    rows = _heldout_rows()
    artifact_before = sha256_file(ARTIFACT_PATH)
    supervisors = {
        "always_act": AlwaysActSupervisor(),
        "always_tool": AlwaysEvidenceOnceSupervisor(),
        "hard_coded_rules": HardCodedToolSupervisor(),
        "existing_imitation_classifier": LearnedToolSupervisor(
            EXISTING_ARTIFACT_PATH
        ),
        "transparent_value_rule": OutcomeValueRuleSupervisor(
            **freeze["transparent_value_rule"]
        ),
        "learned_outcome_value": PostEvidenceCompletionSupervisor(
            OutcomeValueSupervisor(ARTIFACT_PATH)
        ),
    }
    decisions = {
        name: [_uses_tool(supervisor, row) for row in rows]
        for name, supervisor in supervisors.items()
    }
    systems = {
        name: score_policy(rows, decisions[name]) for name in supervisors
    }
    probabilities = [
        _value_probability(supervisors["learned_outcome_value"], row)
        for row in rows
    ]
    systems["learned_outcome_value"]["calibration"] = _calibration(
        rows, probabilities,
    )
    paired_examples = sorted(
        (
            {
                "example_id": row["example_id"],
                "episode_id": row["episode_id"],
                "label": row["label"]["value"],
                "reward_difference": row["outcome"]["reward_difference"],
                "predicted_helpful_probability": probability,
                "learned_selected_tool": selected,
                "action_changed": row["outcome"]["action_changed"],
            }
            for row, probability, selected in zip(
                rows, probabilities, decisions["learned_outcome_value"], strict=True,
            )
        ),
        key=lambda row: float(row["reward_difference"]),
    )
    artifact_after = sha256_file(ARTIFACT_PATH)
    if artifact_after != artifact_before:
        raise AssertionError("heldout evaluation modified the learned artifact")
    result = {
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "split": "heldout",
        "examples": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "fit_or_tuning_calls": 0,
        "frozen_candidate_sha256": sha256_file(CANDIDATE_FREEZE_PATH),
        "artifact_sha256_before": artifact_before,
        "artifact_sha256_after": artifact_after,
        "systems": systems,
        "paired_examples": {
            "largest_regressions": paired_examples[:5],
            "largest_improvements": list(reversed(paired_examples[-5:])),
        },
        "development_threshold": freeze["selected_threshold"],
    }
    HELDOUT_RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _existing_arm_cost(records: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(row["cost_usd"]) for row in records)


def run_matched_arms(*, limit: int | None = None) -> dict[str, Any]:
    verify_candidates()
    supervisors = {
        "base_qwen": None,
        "existing_supervisor": LearnedToolSupervisor(EXISTING_ARTIFACT_PATH),
        "outcome_value_rule": OutcomeValueRuleSupervisor(
            **json.loads(CANDIDATE_FREEZE_PATH.read_text())["transparent_value_rule"]
        ),
        "learned_outcome_value": PostEvidenceCompletionSupervisor(
            OutcomeValueSupervisor(ARTIFACT_PATH)
        ),
    }
    existing = _read_jsonl(ARM_RECORDS)
    by_key = {(row["arm"], row["episode_id"]): row for row in existing}
    if len(by_key) != len(existing):
        raise ValueError("duplicate matched-arm record")
    todo = [
        (arm, row) for row in evaluation_panel() for arm in ARM_NAMES
        if (arm, row["episode_id"]) not in by_key
    ]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        todo = todo[:limit]
    for arm, panel_row in todo:
        prior = collected_cost(load_collected_episodes()) + _existing_arm_cost(existing)
        if prior >= HARD_CAP_USD:
            raise RuntimeError("frozen incremental budget cap already reached")
        sampler = BudgetedSampler(
            TinkerChatSampler(model=BASE_MODEL, experiment=f"{PROTOCOL}-{arm}"),
            prior_usd=prior,
        )
        spec = {
            key: panel_row[key]
            for key in (
                "season", "preset", "agent_slot", "seed", "mask_names",
                "enable_evidence",
            )
        }
        outcome = run_episode(spec, sampler=sampler, supervisor=supervisors[arm])
        usage = asdict(sampler.ledger)
        record = {
            "protocol": PROTOCOL,
            "created_at": utc_now(),
            "arm": arm,
            "episode_id": panel_row["episode_id"],
            "spec": spec,
            "outcome": outcome,
            "usage": usage,
            "cost_usd": usage_cost(usage),
        }
        _append_jsonl(ARM_RECORDS, record)
        existing.append(record)
        by_key[(arm, panel_row["episode_id"])] = record
    return materialize_arms()


def _bootstrap_delta(deltas: Sequence[float]) -> list[float]:
    rng = random.Random(20260906)
    values = []
    for _ in range(10_000):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        values.append(sum(sample) / len(sample))
    values.sort()
    return [values[249], values[9749]]


def _minimum_total_panel(deltas: Sequence[float]) -> int | None:
    if len(deltas) < 2:
        return None
    effect = abs(statistics.mean(deltas))
    if effect == 0:
        return None
    variance = statistics.variance(deltas)
    return math.ceil(((1.96 + 0.84) ** 2) * variance / (effect ** 2))


def materialize_arms() -> dict[str, Any]:
    records = _read_jsonl(ARM_RECORDS)
    episode_ids = [row["episode_id"] for row in evaluation_panel()]
    by_key = {(row["arm"], row["episode_id"]): row for row in records}
    arms = {}
    for arm in ARM_NAMES:
        rows = [by_key[(arm, episode_id)] for episode_id in episode_ids if (arm, episode_id) in by_key]
        outcomes = [row["outcome"] for row in rows]
        tool_calls = sum(int(row["tool_calls"]) for row in outcomes)
        successful_tool_calls = sum(int(row["tool_calls_ok"]) for row in outcomes)
        completed = sum(bool(row["task_completed"]) for row in outcomes)
        arms[arm] = {
            "episodes": len(rows),
            "complete": len(rows) == 30,
            "completed_episodes": completed,
            "mean_reward": (
                sum(float(row["reward"]) for row in outcomes) / len(rows)
                if rows else None
            ),
            "tool_calls": tool_calls,
            "successful_tool_calls": successful_tool_calls,
            "tool_accuracy": successful_tool_calls / max(1, tool_calls),
            "tools_per_completion": tool_calls / max(1, completed),
            "forced_tool_calls": sum(int(row["forced_tool_calls"]) for row in outcomes),
            "supervisor_interventions": sum(
                int(row["supervisor_interventions"]) for row in outcomes
            ),
            "blocked_actions": sum(int(row["blocked_actions"]) for row in outcomes),
            "fallback_picks": sum(int(row["fallback_picks"]) for row in outcomes),
            "model_requests": sum(int(row["model_requests"]) for row in outcomes),
            "incremental_usd": sum(float(row["cost_usd"]) for row in rows),
            "episode_rewards": {
                row["episode_id"]: float(row["outcome"]["reward"]) for row in rows
            },
        }
    base_rewards = arms["base_qwen"]["episode_rewards"]
    comparisons = {}
    for arm in ARM_NAMES[1:]:
        arm_rewards = arms[arm]["episode_rewards"]
        matched = sorted(set(base_rewards) & set(arm_rewards))
        deltas = [arm_rewards[key] - base_rewards[key] for key in matched]
        arm_cost_delta = float(arms[arm]["incremental_usd"]) - float(
            arms["base_qwen"]["incremental_usd"]
        )
        total_reward_delta = sum(deltas)
        without_largest_gain = sorted(deltas)[:-1] if len(deltas) > 1 else deltas
        estimated_total = _minimum_total_panel(deltas)
        comparisons[f"{arm}_vs_base_qwen"] = {
            "matched_episodes": len(matched),
            "mean_reward_delta": sum(deltas) / len(deltas) if deltas else None,
            "bootstrap_95_ci": _bootstrap_delta(deltas) if len(deltas) == 30 else None,
            "median_reward_delta": statistics.median(deltas) if deltas else None,
            "mean_without_largest_gain": (
                sum(without_largest_gain) / len(without_largest_gain)
                if without_largest_gain else None
            ),
            "wins": sum(value > 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
            "critical_improvements_at_least_25": sum(value >= 25 for value in deltas),
            "critical_regressions_at_least_25": sum(value <= -25 for value in deltas),
            "incremental_cost_vs_base_usd": arm_cost_delta,
            "reward_points_per_incremental_dollar": (
                total_reward_delta / arm_cost_delta if arm_cost_delta > 0 else None
            ),
            "estimated_total_panel_for_80pct_power": estimated_total,
            "estimated_additional_episodes_for_80pct_power": (
                max(0, estimated_total - len(deltas))
                if estimated_total is not None else None
            ),
        }
    result = {
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "arms": arms,
        "paired_comparisons": comparisons,
        "complete": all(row["complete"] for row in arms.values()),
        "incremental_usd": sum(row["incremental_usd"] for row in arms.values()),
    }
    ARM_RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_scorecard() -> dict[str, Any]:
    counterfactual = json.loads(HELDOUT_RESULT.read_text())
    arms = json.loads(ARM_RESULT.read_text())
    if not arms["complete"]:
        raise RuntimeError("matched-arm evaluation is incomplete")
    learned_cf = counterfactual["systems"]["learned_outcome_value"]
    learned_arm = arms["paired_comparisons"]["learned_outcome_value_vs_base_qwen"]
    calibration = learned_cf["calibration"]
    gates = {
        "positive_heldout_counterfactual_value": learned_cf["mean_reward_delta_vs_act"] > 0,
        "wins_more_than_losses": learned_arm["wins"] > learned_arm["losses"],
        "positive_full_episode_value": learned_arm["mean_reward_delta"] > 0,
        "not_driven_by_one_outlier": learned_arm["mean_without_largest_gain"] > 0,
        "critical_improvements_exceed_regressions": (
            learned_arm["critical_improvements_at_least_25"]
            > learned_arm["critical_regressions_at_least_25"]
        ),
        "selective_tool_use": (
            arms["arms"]["learned_outcome_value"]["forced_tool_calls"] < 30 * 15
        ),
        "predicted_value_tracks_observed_benefit": (
            calibration["probability_reward_correlation"] > 0
            and calibration["average_precision"] > calibration["prevalence"]
        ),
        "full_episode_bootstrap_ci_excludes_zero": learned_arm["bootstrap_95_ci"][0] > 0,
    }
    signal = all(gates.values())
    if signal:
        decision = "use_hybrid"
    elif (
        learned_cf["mean_reward_delta_vs_act"] > 0
        and learned_arm["mean_reward_delta"] > 0
        and (learned_arm["estimated_total_panel_for_80pct_power"] or 10_000) <= 100
    ):
        decision = "collect_more_data"
    elif any(
        arms["paired_comparisons"][f"{name}_vs_base_qwen"]["mean_reward_delta"] > 0
        for name in ("existing_supervisor", "outcome_value_rule")
    ):
        decision = "redesign_policy_or_tools"
    else:
        decision = "stop_supervision_as_primary_performance_lever"
    local_completed = (
        collected_cost(load_collected_episodes()) + float(arms["incremental_usd"])
    )
    if not SPEND_AUDIT.exists():
        raise RuntimeError("provider spend reconciliation is required before closeout")
    spend_audit = json.loads(SPEND_AUDIT.read_text())
    if not math.isclose(
        float(spend_audit["local_completed_usd"]), local_completed,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise RuntimeError("spend audit does not match completed request ledgers")
    total = float(spend_audit["exact_incremental_usd"])
    if total > HARD_CAP_USD:
        raise RuntimeError("experiment exceeded its frozen hard cap")
    scorecard = {
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "status": "complete",
        "decision": decision,
        "signal_gate": {
            **gates,
            "pass": signal,
        },
        "counterfactual_heldout": counterfactual,
        "matched_arms": arms,
        "spend": {
            "collection_usd": collected_cost(load_collected_episodes()),
            "matched_arms_usd": arms["incremental_usd"],
            "orphaned_interrupted_canary_usd": spend_audit[
                "orphaned_interrupted_canary_usd"
            ],
            "incremental_total_usd": total,
            "reconciliation": str(SPEND_AUDIT.relative_to(ROOT)),
            "target_usd": 5.0,
            "hard_cap_usd": HARD_CAP_USD,
        },
        "fit_or_tuning_on_heldout": False,
    }
    FINAL_SCORECARD.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n")
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("counterfactual", "run-arms", "materialize-arms", "scorecard"),
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "counterfactual":
        result = evaluate_counterfactuals()
    elif args.command == "run-arms":
        result = run_matched_arms(limit=args.limit)
    elif args.command == "materialize-arms":
        result = materialize_arms()
    else:
        result = write_scorecard()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
