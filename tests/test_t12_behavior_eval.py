import json

from training import t12_behavior_eval as behavior
from training.t12_dataset import load_t12_corpus


def test_behavior_suite_is_frozen_development_only_and_schema_complete():
    rows = behavior.build_suite(load_t12_corpus())
    assert len(rows) == 32
    assert len({row["meta"]["trace_id"] for row in rows}) == 32
    assert {row["meta"]["t12_schema"] for row in rows} == {
        "broad_bust", "broad_full_slate", "broad_season_threshold",
        "broad_weekly_h2h", "target_calibration", "target_draft_pick",
        "target_draft_recovery", "target_tool_call", "target_post_tool_pick",
    }


def test_response_validator_accepts_each_target_schema():
    for row in behavior.build_suite(load_t12_corpus()):
        result = behavior._validate_response(row, row["messages"][-1]["content"])
        assert result["parse_success"] is True
        assert result["legal"] is True
        assert result["target_match"] is True
        if row["meta"]["t12_schema"] == "target_tool_call":
            assert result["tool_valid"] is True


def test_response_validator_rejects_wrong_action_type():
    row = next(
        row for row in behavior.build_suite(load_t12_corpus())
        if row["meta"]["t12_schema"] == "target_tool_call"
    )
    result = behavior._validate_response(row, json.dumps({"pick": "B001"}))
    assert result["parse_success"] is True
    assert result["tool_valid"] is False
    assert result["legal"] is False
