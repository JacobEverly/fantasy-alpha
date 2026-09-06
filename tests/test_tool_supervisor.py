import json

import pytest

from harness.tool_supervisor import (
    AlwaysActSupervisor,
    AlwaysUseToolSupervisor,
    HardCodedToolSupervisor,
    SupervisorDecision,
    decision_features,
    decision_from_json,
    validate_proposed_tool,
)
from harness.value_supervisor import (
    AlwaysEvidenceOnceSupervisor,
    OutcomeValueRuleSupervisor,
)


def observation(*, prior=100.0, stdev=5.0, remaining=5, results=None):
    return {
        "season": 2020,
        "round": 6,
        "pick_number": 70,
        "picks_until_next_turn": 8,
        "my_roster": [],
        "top_available": [{
            "player_id": "B001", "name": "B001", "position": "RB",
            "team": "KC", "adp": 60.0, "adp_stdev": stdev,
            "prev_season_points": prior,
        }],
        "lookups_used_this_pick": 0,
        "lookups_remaining": remaining,
        "tool_results": results or [],
    }


def test_supervisor_decision_invariants_and_json_parser():
    with pytest.raises(ValueError):
        SupervisorDecision("NOPE", 1.0, "bad")
    with pytest.raises(ValueError):
        SupervisorDecision("USE_TOOL", 0.5, "missing")
    parsed = decision_from_json(json.dumps({
        "decision": "USE_TOOL",
        "confidence": 0.8,
        "reason_code": "need_evidence",
        "allowed_tools": ["depth_chart"],
        "required_tool": {
            "name": "depth_chart", "arguments": {"team_or_player": "B001"},
        },
    }))
    assert parsed.decision == "USE_TOOL"
    assert parsed.required_tool["name"] == "depth_chart"


def test_hard_coded_rules_cover_act_tool_and_wait():
    supervisor = HardCodedToolSupervisor()
    pick = {"pick": "B001"}
    assert supervisor.decide(observation(), pick).decision == "ACT_NOW"
    decision = supervisor.decide(observation(prior=0.0), pick)
    assert decision.decision == "USE_TOOL"
    assert decision.required_tool["name"] == "depth_chart"
    decision = supervisor.decide(observation(stdev=12.0), pick)
    assert decision.decision == "USE_TOOL"
    assert decision.required_tool["name"] == "injury_status"
    failed = [{
        "call": {"name": "depth_chart", "arguments": {"team_or_player": "B001"}},
        "response": {"ok": False, "error": "unavailable"},
    }]
    assert supervisor.decide(
        observation(prior=0.0, remaining=0, results=failed), pick
    ).decision == "WAIT_OR_ABSTAIN"
    successful = [{
        "call": {"name": "depth_chart", "arguments": {"team_or_player": "B001"}},
        "response": {"ok": True, "result": []},
    }]
    assert supervisor.decide(
        observation(prior=0.0, results=successful), pick
    ).decision == "ACT_NOW"


def test_tool_validation_and_trivial_baselines():
    obs = observation(prior=0.0)
    pick = {"pick": "B001"}
    assert AlwaysActSupervisor().decide(obs, pick).decision == "ACT_NOW"
    assert AlwaysUseToolSupervisor().decide(obs, pick).decision == "USE_TOOL"
    good = {"tool": {
        "name": "depth_chart", "arguments": {"team_or_player": "B001"},
    }}
    assert validate_proposed_tool(obs, good).decision == "USE_TOOL"
    bad = {"tool": {"name": "search", "arguments": {"query": "x"}}}
    assert validate_proposed_tool(obs, bad).decision == "WAIT_OR_ABSTAIN"
    invisible = {"tool": {
        "name": "depth_chart", "arguments": {"team_or_player": "B999"},
    }}
    assert validate_proposed_tool(obs, invisible).decision == "WAIT_OR_ABSTAIN"


def test_always_evidence_baseline_looks_up_once_then_acts():
    supervisor = AlwaysEvidenceOnceSupervisor()
    pick = {"pick": "B001"}
    assert supervisor.decide(observation(), pick).decision == "USE_TOOL"
    successful = [{
        "call": {"name": "injury_status", "arguments": {"player": "B001"}},
        "response": {"ok": True, "result": {"status": "healthy"}},
    }]
    assert supervisor.decide(
        observation(results=successful), pick
    ).decision == "ACT_NOW"


def test_outcome_value_rule_uses_only_its_frozen_conditions():
    supervisor = OutcomeValueRuleSupervisor(
        use_missing_prior=False, relative_adp_stdev_threshold=0.15,
    )
    pick = {"pick": "B001"}
    assert supervisor.decide(observation(prior=0.0), pick).decision == "ACT_NOW"
    assert supervisor.decide(observation(stdev=12.0), pick).decision == "USE_TOOL"


def test_feature_allowlist_uses_only_observation_and_proposed_pick():
    features = decision_features(observation(prior=0.0), {"pick": "B001"})
    assert features["candidate_missing_prior_points"] == 1.0
    assert features["result_status"] == "missing"
    assert "label" not in features and "reward" not in features
