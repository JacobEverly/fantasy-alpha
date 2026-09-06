import pytest

from evals.tool_decision_closeout import paired_summary


def test_paired_summary_reports_wins_ties_losses_and_mean():
    result = paired_summary(
        {"a": 1.0, "b": 2.0, "c": 3.0},
        {"a": 2.0, "b": 2.0, "c": 1.0},
    )
    assert result["wins"] == 1
    assert result["ties"] == 1
    assert result["losses"] == 1
    assert result["mean_paired_reward_change"] == pytest.approx(-1 / 3)


def test_paired_summary_rejects_unmatched_episode_ids():
    with pytest.raises(ValueError, match="episode IDs differ"):
        paired_summary({"a": 1.0}, {"b": 1.0})
