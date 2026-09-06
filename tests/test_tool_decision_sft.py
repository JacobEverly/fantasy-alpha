import json

from training import tool_decision_sft as sft
from training.tool_decision_dataset import load_split


def test_contrastive_rows_map_all_decisions_to_actions():
    by_decision = {}
    for example in load_split("train"):
        by_decision.setdefault(example["label"]["decision"], example)
    assert set(by_decision) == {"ACT_NOW", "USE_TOOL", "WAIT_OR_ABSTAIN"}
    assert "pick" in sft._target(by_decision["ACT_NOW"])
    assert "tool" in sft._target(by_decision["USE_TOOL"])
    assert "abstain" in sft._target(by_decision["WAIT_OR_ABSTAIN"])
    for example in by_decision.values():
        messages = sft._messages(example)
        assert [message["role"] for message in messages] == [
            "system", "user", "assistant",
        ]
        assert "Proposed action" in messages[1]["content"]


def test_materialized_corpus_is_balanced_and_canary_keeps_groups():
    if not sft.MANIFEST_PATH.exists():
        rows = sft.build_corpus()
    else:
        rows = sft.load_corpus()
    validation = sft.validate_corpus(rows)
    assert validation["rows"] == 564
    assert all(
        abs(value - 1 / 3) < 1e-9
        for value in validation["weighted_decision_fraction"].values()
    )
    canary = sft.canary_subset(rows)
    groups = {}
    for row in canary:
        groups.setdefault(row["meta"]["question_id"], set()).add(
            row["meta"]["decision"]
        )
    assert len(canary) == 160
    assert len(groups) == 40
    assert all(parts == {"ACT_NOW", "USE_TOOL", "WAIT_OR_ABSTAIN"}
               for parts in groups.values())


def test_model_action_parser_distinguishes_valid_and_invalid_actions():
    examples = load_split("development")
    by_decision = {
        row["label"]["decision"]: row for row in examples
    }
    for decision, example in by_decision.items():
        text = json.dumps(sft._target(example))
        parsed, structured, legal = sft._parse_model_decision(text, example)
        assert parsed.decision == decision
        assert structured is True
        assert legal is True
    parsed, structured, legal = sft._parse_model_decision("not json", examples[0])
    assert parsed.decision == "ACT_NOW"
    assert structured is False and legal is False
