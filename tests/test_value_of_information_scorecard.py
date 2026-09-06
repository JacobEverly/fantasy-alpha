import json

from evals.value_of_information_scorecard import (
    ARM_NAMES,
    PostEvidenceCompletionSupervisor,
    _bootstrap_delta,
    evaluation_panel,
    materialize_arms,
)
from harness.tool_supervisor import SupervisorDecision


def test_arm_panel_is_unique_and_sealed_safe():
    panel = evaluation_panel()
    assert len(panel) == 30
    assert len({row["episode_id"] for row in panel}) == 30
    assert {row["season"] for row in panel} == {
        2015, 2016, 2017, 2019, 2020, 2021, 2022,
    }
    assert not {2013, 2018, 2023, 2024, 2025} & {
        row["season"] for row in panel
    }


def test_bootstrap_delta_is_deterministic_and_ordered():
    first = _bootstrap_delta([float(value) for value in range(30)])
    second = _bootstrap_delta([float(value) for value in range(30)])
    assert first == second
    assert first[0] < first[1]


def test_materialize_arms_builds_paired_comparisons(tmp_path, monkeypatch):
    records_path = tmp_path / "records.jsonl"
    result_path = tmp_path / "result.json"
    panel = [{"episode_id": f"episode-{index}"} for index in range(30)]
    monkeypatch.setattr(
        "evals.value_of_information_scorecard.evaluation_panel", lambda: panel,
    )
    monkeypatch.setattr(
        "evals.value_of_information_scorecard.ARM_RECORDS", records_path,
    )
    monkeypatch.setattr(
        "evals.value_of_information_scorecard.ARM_RESULT", result_path,
    )
    offsets = {
        "base_qwen": 0.0,
        "existing_supervisor": -2.0,
        "outcome_value_rule": 5.0,
        "learned_outcome_value": 10.0,
    }
    records = []
    for arm in ARM_NAMES:
        for row in panel:
            records.append({
                "arm": arm,
                "episode_id": row["episode_id"],
                "cost_usd": 0.01,
                "outcome": {
                    "reward": offsets[arm],
                    "task_completed": True,
                    "tool_calls": int(arm != "base_qwen"),
                    "tool_calls_ok": int(arm != "base_qwen"),
                    "forced_tool_calls": int(arm != "base_qwen"),
                    "supervisor_interventions": int(arm != "base_qwen"),
                    "blocked_actions": 0,
                    "fallback_picks": 0,
                    "model_requests": 15,
                },
            })
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    result = materialize_arms()
    assert result["complete"]
    learned = result["paired_comparisons"]["learned_outcome_value_vs_base_qwen"]
    assert learned["mean_reward_delta"] == 10.0
    assert learned["wins"] == 30


def test_post_evidence_wrapper_approves_reconsidered_pick_once():
    class Inner:
        def decide(self, observation, proposed_action):
            return SupervisorDecision(
                "USE_TOOL", 1.0, "would_repeat",
                ("injury_status",),
                {"name": "injury_status", "arguments": {"player": "B001"}},
            )

    observation = {
        "top_available": [{"player_id": "B001"}],
        "tool_results": [{
            "call": {
                "name": "injury_status", "arguments": {"player": "B001"},
            },
            "response": {"ok": True, "result": {"status": "healthy"}},
        }],
    }
    decision = PostEvidenceCompletionSupervisor(Inner()).decide(
        observation, {"pick": "B001"},
    )
    assert decision.decision == "ACT_NOW"
    assert decision.reason_code == "post_evidence_reconsideration_complete"
