import json

from training.override_dataset import load_override_corpus, validate_rows
from training.override_learnability import candidate_examples
from evals.override_holdout import _parse_action
from app.decision_card import example_payload, recommend


def test_materialized_override_corpus_is_masked_and_valid():
    rows = load_override_corpus()
    audit = validate_rows(rows)
    assert audit["rows"] == 1243
    assert audit["decisions"] == {"KEEP_ADP": 1187, "OVERRIDE": 56}
    assert audit["override_rate"] < 0.05


def test_override_targets_are_visible_and_candidate_matrix_is_complete():
    rows = load_override_corpus()
    examples = candidate_examples(rows)
    assert len(examples["state_labels"]) == len(rows)
    assert int(examples["state_labels"].sum()) == 56
    assert int(examples["y"].sum()) == 56
    for row in rows:
        target = json.loads(row["messages"][-1]["content"])
        if target["decision"] == "OVERRIDE":
            visible = json.loads(row["messages"][1]["content"].split("\n", 1)[1])
            assert target["candidate_id"] in {
                candidate["candidate_id"] for candidate in visible["candidates"]
            }


def test_holdout_action_parser_is_strict_and_allows_fenced_json():
    visible = {"B001", "B002"}
    assert _parse_action('{"decision":"KEEP_ADP"}', visible) == ("KEEP_ADP", None)
    assert _parse_action(
        '```json\n{"decision":"OVERRIDE","candidate_id":"B002"}\n```', visible
    ) == ("OVERRIDE", "B002")


def test_local_decision_card_can_reproduce_a_teacher_override():
    card = recommend(example_payload())
    assert card["decision"] == "OVERRIDE"
    assert card["selected_candidate_id"] != card["adp_candidate_id"]
    assert card["checks"]["expected_value_passed"] is True
    assert card["checks"]["downside_passed"] is True
