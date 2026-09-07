import pytest

from training.outcome_decisions import FEATURE_ALLOWLIST
from training.outcome_ranker import encode


def features():
    row = {field: 0 for field in FEATURE_ALLOWLIST}
    row["candidate_position"] = "RB"
    return row


def test_encoder_is_fixed_and_one_hot():
    encoded = encode(features())
    assert encoded[:4] == [0.0, 1.0, 0.0, 0.0]
    assert len(encoded) == len(FEATURE_ALLOWLIST) + 3


def test_encoder_rejects_feature_drift():
    row = features()
    row["realized_points"] = 100
    with pytest.raises(ValueError, match="unexpected feature schema"):
        encode(row)
