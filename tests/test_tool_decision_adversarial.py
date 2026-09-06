import json

from evals import tool_decision_adversarial as challenge


def test_challenge_covers_required_categorical_edges():
    rows = challenge.challenge_rows()
    assert len(rows) == 12
    assert {row["label"]["decision"] for row in rows} == {
        "ACT_NOW", "USE_TOOL", "WAIT_OR_ABSTAIN",
    }
    assert any("adp" in str(row["label"]["missing_information"]) for row in rows)
    assert any("arithmetic" in row["example_id"] for row in rows)
    assert any(row["scenario"] == "illegal_tool_request" for row in rows)
    assert any(row["scenario"] == "irrelevant_tool_request" for row in rows)
    assert all(row["provenance"]["never_training_data"] for row in rows)
    assert all(row["input"]["observation"]["season"] == 2020 for row in rows)


def test_materialized_challenge_is_frozen_and_matches_recipe():
    if not challenge.DATASET_PATH.exists():
        return
    expected = challenge.SHA_PATH.read_text().split()[0]
    assert challenge.sha256_file(challenge.DATASET_PATH) == expected
    rows = [
        json.loads(line)
        for line in challenge.DATASET_PATH.read_text().splitlines()
        if line.strip()
    ]
    assert rows == challenge.challenge_rows()
