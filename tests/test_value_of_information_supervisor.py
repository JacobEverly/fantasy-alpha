import json

import pytest

from training.value_of_information_supervisor import (
    SOURCE_PROTOCOL,
    grouped_oof_probabilities,
    load_development,
    score_policy,
    select_threshold,
)


def _row(episode, index, label, delta, missing=0.0):
    player = {
        "player_id": f"P{episode}{index}",
        "position": "RB",
        "adp": 20.0 + index,
        "adp_stdev": 4.0,
        "prev_season_points": 0.0 if missing else 100.0,
    }
    return {
        "protocol": SOURCE_PROTOCOL,
        "example_id": f"e{episode}-{index}",
        "episode_id": f"episode-{episode}",
        "split": "development",
        "input": {
            "observation": {
                "round": index + 1,
                "pick_number": index + 1,
                "picks_until_next_turn": 5,
                "my_roster": [],
                "top_available": [player],
                "lookups_used_this_pick": 0,
                "lookups_remaining": 1,
                "tool_results": [],
            },
            "proposed_action": {"pick": player["player_id"]},
        },
        "intervention": {"computed_usd": 0.001},
        "branches": {
            "act": {"reward": 100.0},
            "intervene": {"reward": 100.0 + delta},
        },
        "label": {"value": label},
    }


def test_loader_rejects_heldout(tmp_path):
    row = _row(0, 0, "HELPFUL", 25.0)
    row["split"] = "heldout"
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="non-development"):
        load_development(path, exact=False)


def test_score_policy_uses_observed_counterfactual_rewards():
    rows = [_row(0, 0, "HELPFUL", 25.0), _row(0, 1, "HARMFUL", -25.0)]
    score = score_policy(rows, [True, False])
    assert score["mean_reward_delta_vs_act"] == 12.5
    assert score["tool_calls"] == 1
    assert score["harmful_selected"] == 0


def test_grouped_oof_and_threshold_selection_do_not_mix_episodes():
    rows = []
    for episode in range(6):
        rows.extend([
            _row(episode, 0, "HELPFUL", 25.0, missing=1.0),
            _row(episode, 1, "UNNECESSARY", 0.0),
        ])
    probabilities, folds = grouped_oof_probabilities(rows)
    threshold, curve = select_threshold(rows, probabilities)
    assert len(probabilities) == len(rows)
    assert 0.05 <= threshold <= 0.95
    assert len(curve) == 19
    for fold in folds:
        assert set(fold["test_episode_ids"])
        assert fold["train_episode_count"] < 6
