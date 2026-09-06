import json

import pytest

from training import tool_decision_dataset as ds


def test_episode_split_is_deterministic_stratified_and_disjoint():
    ids = [f"season{season}:slot{slot}:seed{seed}"
           for season in (2019, 2020)
           for slot in (1, 4, 7, 10, 12)
           for seed in (3, 11, 29)]
    first = ds._split_episodes(ids)
    second = ds._split_episodes(list(reversed(ids)))
    assert first == second
    for season in (2019, 2020):
        counts = {split: 0 for split in ds.SPLITS}
        for episode_id, split in first.items():
            if episode_id.startswith(f"season{season}:"):
                counts[split] += 1
        assert counts == {"train": 10, "development": 2, "heldout": 3}


def test_tool_choice_uses_only_visible_candidate_uncertainty():
    rookie = {"player_id": "B001", "prev_season_points": 0.0}
    volatile_veteran = {"player_id": "B002", "prev_season_points": 100.0}
    assert ds._tool_for_candidate(rookie) == {
        "name": "depth_chart", "arguments": {"team_or_player": "B001"},
    }
    assert ds._tool_for_candidate(volatile_veteran) == {
        "name": "injury_status", "arguments": {"player": "B002"},
    }


def test_materialized_dataset_matches_manifest_and_has_no_split_leakage():
    if not ds.MANIFEST_PATH.exists():
        pytest.skip("materialize the tool-decision dataset first")
    manifest = json.loads(ds.MANIFEST_PATH.read_text())
    rows = []
    episodes = {}
    groups = {}
    for split in ds.SPLITS:
        split_rows = ds.load_split(split)
        assert len(split_rows) == manifest["files"][split]["rows"]
        rows.extend(split_rows)
        for row in split_rows:
            episodes.setdefault(row["episode_group_id"], set()).add(split)
            groups.setdefault(row["matched_group_id"], set()).add(split)
    result = ds.validate_examples(rows)
    assert result["rows"] == 948
    assert result["matched_groups"] == 237
    assert result["tools"] == {"depth_chart": 177, "injury_status": 60}
    assert all(len(splits) == 1 for splits in episodes.values())
    assert all(len(splits) == 1 for splits in groups.values())
    assert result["sealed_seasons_present"] == []
