"""Reproducible derived analysis for the completed tool-decision experiment."""
from __future__ import annotations

import json
import statistics
import time
from typing import Any, Mapping

from evals.tool_decision_scorecard import OUT_DIR, verify_candidates
from harness.tool_supervisor import HardCodedToolSupervisor, LearnedToolSupervisor
from training.tinker_backend import ROOT, sha256_file
from training.tool_decision_dataset import load_split
from training.tool_decision_supervisor import ARTIFACT_PATH

RESULT_PATH = OUT_DIR / "closeout-analysis.json"
PROTOCOL = "fantasy-alpha-tool-decision-closeout-v1"


def paired_summary(
    baseline: Mapping[str, float], contender: Mapping[str, float]
) -> dict[str, Any]:
    if set(baseline) != set(contender):
        raise ValueError("paired episode IDs differ")
    differences = {
        episode_id: float(contender[episode_id]) - float(baseline[episode_id])
        for episode_id in sorted(baseline)
    }
    values = list(differences.values())
    return {
        "episodes": len(values),
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
        "mean_paired_reward_change": sum(values) / len(values),
        "differences": differences,
    }


def _latency(supervisor: Any, rows: list[dict[str, Any]], repeats: int = 10) -> dict:
    # Warm the artifact and interpreter before recording local decision overhead.
    for row in rows[:10]:
        supervisor.decide(row["input"]["observation"], row["input"]["proposed_action"])
    samples_us: list[float] = []
    for _ in range(repeats):
        for row in rows:
            start = time.perf_counter_ns()
            supervisor.decide(
                row["input"]["observation"], row["input"]["proposed_action"]
            )
            samples_us.append((time.perf_counter_ns() - start) / 1_000)
    ordered = sorted(samples_us)
    return {
        "calls": len(samples_us),
        "mean_microseconds": statistics.fmean(samples_us),
        "median_microseconds": statistics.median(samples_us),
        "p95_microseconds": ordered[int(0.95 * (len(ordered) - 1))],
        "measurement": "warm local CPU decision function only",
    }


def build() -> dict[str, Any]:
    verify_candidates()
    scorecard_path = OUT_DIR / "scorecard.json"
    draftgym_path = OUT_DIR / "draftgym-scorecard.json"
    adversarial_path = OUT_DIR / "adversarial.json"
    scorecard = json.loads(scorecard_path.read_text())
    draftgym = json.loads(draftgym_path.read_text())
    adversarial = json.loads(adversarial_path.read_text())
    rows = load_split("heldout")
    base = draftgym["arms"]["base_qwen"]
    comparisons = {}
    for name in ("hard_coded_plus_qwen", "learned_supervisor_plus_qwen"):
        arm = draftgym["arms"][name]
        comparisons[name] = {
            **paired_summary(base["episode_rewards"], arm["episode_rewards"]),
            "tool_calls_per_completed_task": arm["tool_calls"] / arm["completed"],
            "incremental_cost_vs_base_usd": arm["cost_usd"] - base["cost_usd"],
        }
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "source_hashes": {
            str(scorecard_path.relative_to(ROOT)): sha256_file(scorecard_path),
            str(draftgym_path.relative_to(ROOT)): sha256_file(draftgym_path),
            str(adversarial_path.relative_to(ROOT)): sha256_file(adversarial_path),
        },
        "paired_draftgym": comparisons,
        "tool_calls_per_completed_task": {
            name: arm["tool_calls"] / arm["completed"]
            for name, arm in draftgym["arms"].items()
        },
        "local_supervisor_latency": {
            "hard_coded_supervisor_v1": _latency(HardCodedToolSupervisor(), rows),
            "learned_supervisor_v1": _latency(
                LearnedToolSupervisor(ARTIFACT_PATH), rows
            ),
        },
        "provider_end_to_end_latency": {
            "status": "not_retained",
            "reason": (
                "The frozen Tinker inference records retained tokens and cost but "
                "not per-request timestamps; no paid rerun was made solely to "
                "reconstruct an operational metric."
            ),
        },
        "adversarial_summary": {
            name: {
                "accuracy": system["metrics"]["accuracy"],
                "unsafe_direct_action_rate": system["metrics"][
                    "unsafe_direct_action_rate"
                ],
                "incorrect_examples": [
                    row["example_id"] for row in system["examples"]
                    if not row["correct"]
                ],
            }
            for name, system in adversarial["systems"].items()
        },
        "primary_decision": scorecard["decision"],
        "final_architecture_recommendation": (
            "redesign_outcome_labels_then_use_external_hybrid_supervisor"
        ),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
