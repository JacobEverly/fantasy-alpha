import pytest

from evals.draftbench import DraftPlayer
from harness.league import LeagueConfig
from harness.scoring import PRESETS
from training.outcome_decisions import (
    FEATURE_ALLOWLIST,
    PrefixCounterfactualAgent,
    validate_rows,
)


def test_feature_allowlist_has_no_realized_or_identity_fields():
    fields = " ".join(FEATURE_ALLOWLIST)
    assert "points" in fields  # previous-season points are legal
    assert "realized" not in fields
    assert "completed" not in fields
    assert "name" not in fields
    assert "team" not in fields


def test_validation_rejects_label_leakage_and_single_action_state():
    row = {
        "state_id": "s",
        "episode_id": "e",
        "split": "development",
        "season": 2019,
        "features": {field: 0 for field in FEATURE_ALLOWLIST},
        "label": {"completed_roster_points": 1, "regret_to_best_candidate": 0, "is_best_candidate": True},
    }
    with pytest.raises(ValueError, match="multiple plausible"):
        validate_rows([row])
    bad = dict(row)
    bad["features"] = dict(row["features"], realized_points=100)
    with pytest.raises(ValueError, match="unexpected model feature"):
        validate_rows([bad, bad])


def test_prefix_agent_rejects_an_unreachable_matched_state():
    agent = PrefixCounterfactualAgent(["missing"], "branch")
    league = LeagueConfig(
        teams=2,
        roster={"QB": 1, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0, "BENCH": 0},
        scoring=PRESETS["ppr"],
    )
    board = [DraftPlayer("present", "P", "QB", "", 1.0, 1.0, 1, 0.0, None)]
    with pytest.raises(AssertionError, match="matched state"):
        agent.pick(board, [], league, 1)
