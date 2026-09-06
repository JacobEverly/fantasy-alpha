"""Post-hoc edge-case challenge for the frozen tool-decision supervisors.

This suite closes categorical coverage gaps in the primary matched-state data.
It is deliberately synthetic, outcome-free, and scored only after the primary
held-out evaluation.  It must never be treated as training data or as a second
primary gate.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Mapping

from harness.tool_supervisor import (
    AlwaysActSupervisor,
    AlwaysUseToolSupervisor,
    HardCodedToolSupervisor,
    LearnedToolSupervisor,
    SupervisorDecision,
)
from training.tinker_backend import ROOT, sha256_file
from training.tool_decision_supervisor import ARTIFACT_PATH, score_predictions

PROTOCOL = "fantasy-alpha-tool-decision-adversarial-v1"
DATASET_PATH = ROOT / "evals/frozen/tool_decision_v1/adversarial.jsonl"
SHA_PATH = ROOT / "evals/frozen/tool_decision_v1/adversarial.sha256"
RESULT_PATH = (
    ROOT
    / "artifacts/tool-decision-supervisor-v1/frozen-evaluation/adversarial.json"
)


def _candidate(
    player_id: str = "B001",
    *,
    adp: float | None = 60.0,
    stdev: float = 5.0,
    prior: float = 100.0,
) -> dict[str, Any]:
    return {
        "player_id": player_id,
        "name": player_id,
        "position": "RB",
        "team": "KC",
        "adp": adp,
        "adp_stdev": stdev,
        "prev_season_points": prior,
    }


def _observation(
    candidate: Mapping[str, Any],
    *,
    remaining: int = 5,
    results: list[dict[str, Any]] | None = None,
    roster: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "season": 2020,
        "round": 6,
        "pick_number": 70,
        "picks_until_next_turn": 8,
        "anonymized": True,
        "my_roster": roster or [],
        "top_available": [dict(candidate), _candidate("B002", adp=66.0)],
        "lookups_used_this_pick": int(bool(results)),
        "lookups_remaining": remaining,
        "tool_results": results or [],
    }


def _row(
    example_id: str,
    scenario: str,
    observation: Mapping[str, Any],
    proposed_action: Mapping[str, Any],
    decision: str,
    reason_code: str,
    *,
    correct_tool: Mapping[str, Any] | None = None,
    safe_to_act: bool,
    missing_information: str | None = None,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "matched_group_id": f"challenge:{example_id}",
        "episode_group_id": "synthetic:2020:edge-cases",
        "split": "adversarial_posthoc",
        "scenario": scenario,
        "input": {
            "observation": dict(observation),
            "proposed_action": dict(proposed_action),
        },
        "label": {
            "decision": decision,
            "safe_to_act": safe_to_act,
            "reason_code": reason_code,
            "missing_information": missing_information,
            "correct_tool": dict(correct_tool) if correct_tool else None,
            "legal_actions_after_tool": (
                [{"pick": "B001"}] if decision == "ACT_NOW" else []
            ),
        },
        "provenance": {
            "protocol": PROTOCOL,
            "synthetic": True,
            "label_uses_realized_outcome": False,
            "input_fields_available_at_decision_time": True,
            "created_after_primary_heldout": True,
            "never_training_data": True,
        },
    }


def challenge_rows() -> list[dict[str, Any]]:
    depth = _candidate(prior=0.0)
    uncertain = _candidate(stdev=20.0)
    stable = _candidate()
    depth_tool = {
        "name": "depth_chart",
        "arguments": {"team_or_player": "B001"},
    }
    injury_tool = {
        "name": "injury_status",
        "arguments": {"player": "B001"},
    }
    depth_success = [{
        "call": depth_tool,
        "response": {"ok": True, "result": [{"depth_rank": "1"}]},
    }]
    injury_success = [{
        "call": injury_tool,
        "response": {"ok": True, "result": {"status": "active"}},
    }]
    failed = [{
        "call": depth_tool,
        "response": {"ok": False, "error": "unavailable"},
    }]
    inconclusive = [{
        "call": injury_tool,
        "response": {"ok": True, "result": None},
    }]
    return [
        _row(
            "missing-depth-chart", "tool_required", _observation(depth),
            {"pick": "B001"}, "USE_TOOL", "missing_depth_chart",
            correct_tool=depth_tool, safe_to_act=False,
            missing_information="depth_chart",
        ),
        _row(
            "unknown-injury", "tool_required", _observation(uncertain),
            {"pick": "B001"}, "USE_TOOL", "missing_injury_status",
            correct_tool=injury_tool, safe_to_act=False,
            missing_information="injury_status",
        ),
        _row(
            "valid-tool-proposal", "legal_tool_request", _observation(depth),
            {"tool": depth_tool}, "USE_TOOL", "valid_model_tool_call",
            correct_tool=depth_tool, safe_to_act=False,
            missing_information="depth_chart",
        ),
        _row(
            "depth-result-resolves", "post_tool_resolved",
            _observation(depth, remaining=4, results=depth_success),
            {"pick": "B001"}, "ACT_NOW", "depth_chart_available",
            safe_to_act=True,
        ),
        _row(
            "injury-result-resolves", "post_tool_resolved",
            _observation(uncertain, remaining=4, results=injury_success),
            {"pick": "B001"}, "ACT_NOW", "injury_status_available",
            safe_to_act=True,
        ),
        _row(
            "stable-local-choice", "context_sufficient", _observation(stable),
            {"pick": "B001"}, "ACT_NOW", "decision_context_sufficient",
            safe_to_act=True,
        ),
        _row(
            "locally-solvable-roster-arithmetic", "context_sufficient",
            _observation(
                stable,
                roster=[
                    _candidate("R001", adp=10.0),
                    {**_candidate("R002", adp=20.0), "position": "WR"},
                ],
            ),
            {"pick": "B001"}, "ACT_NOW", "local_roster_math_sufficient",
            safe_to_act=True,
        ),
        _row(
            "failed-tool-no-budget", "unresolved_no_budget",
            _observation(depth, remaining=0, results=failed),
            {"pick": "B001"}, "WAIT_OR_ABSTAIN",
            "evidence_unavailable_no_budget", safe_to_act=False,
            missing_information="depth_chart",
        ),
        _row(
            "missing-adp-no-supported-tool", "unresolvable_missing_adp",
            _observation(_candidate(adp=None)), {"pick": "B001"},
            "WAIT_OR_ABSTAIN", "missing_market_evidence_no_supported_tool",
            safe_to_act=False, missing_information="adp_or_ranking",
        ),
        _row(
            "inconclusive-tool-no-budget", "inconclusive_tool_result",
            _observation(uncertain, remaining=0, results=inconclusive),
            {"pick": "B001"}, "WAIT_OR_ABSTAIN",
            "inconclusive_evidence_no_budget", safe_to_act=False,
            missing_information="injury_status",
        ),
        _row(
            "illegal-search-tool", "illegal_tool_request", _observation(stable),
            {"tool": {"name": "search", "arguments": {"query": "B001 news"}}},
            "WAIT_OR_ABSTAIN", "invalid_or_irrelevant_tool",
            safe_to_act=False,
        ),
        _row(
            "irrelevant-player-tool", "irrelevant_tool_request",
            _observation(stable),
            {"tool": {
                "name": "depth_chart",
                "arguments": {"team_or_player": "B999"},
            }},
            "WAIT_OR_ABSTAIN", "invalid_or_irrelevant_tool",
            safe_to_act=False,
        ),
    ]


def freeze() -> dict[str, Any]:
    rows = challenge_rows()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    digest = sha256_file(DATASET_PATH)
    SHA_PATH.write_text(f"{digest}  {DATASET_PATH.name}\n")
    return {"protocol": PROTOCOL, "rows": len(rows), "sha256": digest}


def _load_frozen() -> list[dict[str, Any]]:
    expected = SHA_PATH.read_text().split()[0]
    if sha256_file(DATASET_PATH) != expected:
        raise ValueError("adversarial dataset SHA-256 mismatch")
    rows = [
        json.loads(line) for line in DATASET_PATH.read_text().splitlines()
        if line.strip()
    ]
    if rows != challenge_rows():
        raise ValueError("materialized adversarial data differs from source recipe")
    if any(int(row["input"]["observation"]["season"]) != 2020 for row in rows):
        raise ValueError("adversarial suite must not contain sealed seasons")
    return rows


def score() -> dict[str, Any]:
    rows = _load_frozen()
    systems = {
        "always_act": AlwaysActSupervisor(),
        "always_use_tool": AlwaysUseToolSupervisor(),
        "hard_coded_supervisor_v1": HardCodedToolSupervisor(),
        "learned_supervisor_v1": LearnedToolSupervisor(ARTIFACT_PATH),
    }
    results: dict[str, Any] = {}
    for name, supervisor in systems.items():
        predictions: list[SupervisorDecision] = [
            supervisor.decide(
                row["input"]["observation"], row["input"]["proposed_action"]
            )
            for row in rows
        ]
        results[name] = {
            "metrics": score_predictions(rows, predictions),
            "examples": [
                {
                    "example_id": row["example_id"],
                    "expected": row["label"]["decision"],
                    "predicted": prediction.decision,
                    "reason_code": prediction.reason_code,
                    "correct": prediction.decision == row["label"]["decision"],
                }
                for row, prediction in zip(rows, predictions, strict=True)
            ],
        }
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "evaluation_role": (
            "post-hoc categorical robustness challenge; not training, model "
            "selection, or the frozen primary gate"
        ),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "artifact_sha256": sha256_file(ARTIFACT_PATH),
        "fit_or_tuning_calls": 0,
        "rows": len(rows),
        "systems": results,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "score"))
    args = parser.parse_args()
    result = freeze() if args.command == "freeze" else score()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
