import json

from training import t11_dataset as ds
from training.tinker_backend import grouped_development_split


def test_materialized_t11_dataset_is_valid_and_frozen():
    rows = ds.load_t11_corpus()
    result = ds.validate_rows(rows)
    manifest = json.loads(ds.MANIFEST_PATH.read_text())
    assert result["rows"] == 300
    assert result["subtypes"] == {
        "calibration_correction": 120,
        "direct_pick": 120,
        "post_tool_pick": 30,
        "tool_call": 30,
    }
    assert result["sealed_seasons_present"] == []
    assert result["realized_label_inputs"] == 0
    assert ds.sha256_file(ds.CORPUS_PATH) == manifest["corpus_sha256"]


def test_t11_build_is_deterministic():
    first = ds.build_forecast_rows()
    second = ds.build_forecast_rows()
    assert first == second
    assert len(first) == ds.TARGET_FORECAST_ROWS


def test_t11_tool_actions_are_paired_with_post_result_picks():
    rows = ds.load_t11_corpus()
    by_group = {}
    for row in rows:
        if row["meta"]["t11_subtype"] not in {"tool_call", "post_tool_pick"}:
            continue
        by_group.setdefault(row["meta"]["question_id"], []).append(row)
    assert len(by_group) == ds.TARGET_TOOL_GROUPS
    for group in by_group.values():
        assert {x["meta"]["t11_subtype"] for x in group} == {
            "tool_call", "post_tool_pick",
        }


def test_development_split_represents_every_target_behavior():
    _, development = grouped_development_split(ds.load_t11_corpus())
    subtypes = {row["meta"]["t11_subtype"] for row in development}
    families = {row["meta"]["family"] for row in development}
    assert subtypes == {
        "calibration_correction", "direct_pick", "tool_call", "post_tool_pick",
    }
    assert {
        "calibration_full_slate", "calibration_bust",
        "calibration_season_threshold", "calibration_weekly_h2h",
        "draft_direct", "draft_recovery", "draft_tool",
    } <= families
    tool_groups = {}
    for row in development:
        if row["meta"]["family"] == "draft_tool":
            tool_groups.setdefault(row["meta"]["question_id"], set()).add(
                row["meta"]["t11_subtype"]
            )
    assert tool_groups
    assert all(parts == {"tool_call", "post_tool_pick"} for parts in tool_groups.values())


def test_calibration_targets_are_outcome_blind():
    rows = ds.load_t11_corpus()
    forecasts = [r for r in rows if r["meta"]["t11_subtype"] == "calibration_correction"]
    assert forecasts
    for row in forecasts:
        meta = row["meta"]
        expected = round(max(
            0.08,
            min(0.85, meta["anchor_rate"] + max(
                -0.15, min(0.15, meta["source_teacher_p"] - meta["anchor_rate"])
            ) / 3.0),
        ), 2)
        assert meta["target_p"] == expected
        assert meta["label_uses_realized_outcome"] is False
