from training import tool_decision_sft_v2 as v2


def test_revision_changes_only_weighting_metadata_and_keeps_messages():
    parent = v2.v1.load_corpus()
    revised = v2.build_corpus()
    validation = v2.validate_corpus(revised)
    assert validation["weighted_decision_fraction"] == {
        "ACT_NOW": 0.25,
        "USE_TOOL": 0.5,
        "WAIT_OR_ABSTAIN": 0.25,
    }
    assert [row["messages"] for row in parent] == [row["messages"] for row in revised]
    assert all(row["meta"]["revision"] for row in revised)


def test_revision_canary_has_eighty_complete_matched_groups():
    rows = v2.build_corpus()
    canary = v2.canary_subset(rows)
    groups = {}
    for row in canary:
        groups.setdefault(row["meta"]["question_id"], set()).add(
            row["meta"]["decision"]
        )
    assert len(canary) == 320
    assert len(groups) == 80
    assert all(parts == {"ACT_NOW", "USE_TOOL", "WAIT_OR_ABSTAIN"}
               for parts in groups.values())
