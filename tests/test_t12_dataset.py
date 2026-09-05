import json
import math

from training import t12_dataset as ds


def test_t12_build_is_deterministic_and_removes_conflicting_prompts():
    first = ds.build_rows()
    second = ds.build_rows()
    assert first == second
    result = ds.validate_rows(first)
    assert result["rows"] == 829
    assert result["source_rows"] == {"t1": 529, "t11": 300}
    assert result["superseded_conflicting_t1_prompts"] == 120
    assert result["superseded_conflicting_t1_rows"] == 156
    assert result["removed_duplicate_broad_teacher_variants"] == 126
    assert result["duplicate_prompts"] == 0
    assert result["conflicting_targets"] == 0


def test_t12_weighted_signal_is_exactly_seventy_thirty():
    result = ds.validate_rows(ds.build_rows())
    assert math.isclose(result["broad_fraction"], 0.70, abs_tol=1e-8)
    assert math.isclose(result["targeted_fraction"], 0.30, abs_tol=1e-8)
    assert result["complete_tool_transition_groups"] == 30


def test_materialized_t12_corpus_matches_manifest():
    rows = ds.load_t12_corpus()
    manifest = json.loads(ds.MANIFEST_PATH.read_text())
    assert ds.sha256_file(ds.CORPUS_PATH) == manifest["corpus_sha256"]
    assert ds.validate_rows(rows)["sealed_2025_untouched"] is True


def test_t12_canary_has_every_schema_and_twenty_complete_transitions():
    rows = ds.canary_subset(ds.build_rows())
    schemas = {row["meta"]["t12_schema"] for row in rows}
    assert schemas == set(ds.SCHEMA_MASS)
    groups = {}
    for row in rows:
        if row["meta"]["t12_schema"] in {
            "target_tool_call", "target_post_tool_pick",
        }:
            groups.setdefault(row["meta"]["question_id"], set()).add(
                row["meta"]["t12_schema"]
            )
    assert len(rows) == 230
    assert len(groups) == 20
    assert all(parts == {"target_tool_call", "target_post_tool_pick"}
               for parts in groups.values())
