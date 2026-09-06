from dataclasses import dataclass

from evals.value_of_information import (
    BudgetedSampler,
    MATERIAL_REWARD_POINTS,
    PROTOCOL,
    collected_cost,
    episode_panel,
    label_intervention,
    materialize_split,
    observation_hash,
    replay_prefix,
    select_decision_points,
    usage_cost,
)
from test_draftgym import autopick_step, synthetic_gym


@dataclass
class Ledger:
    training_tokens: int = 0
    prefill_tokens: int = 0
    cached_prefill_tokens: int = 0
    sample_tokens: int = 0
    checkpoint_count: int = 0

    @property
    def usd(self):
        return usage_cost(self.__dict__)


def test_episode_panel_is_balanced_and_sealed_safe():
    panel = episode_panel()
    assert len(panel) == 70
    assert sum(row["split"] == "development" for row in panel) == 35
    assert sum(row["split"] == "heldout" for row in panel) == 35
    assert len({row["episode_id"] for row in panel}) == 70
    assert not {2013, 2018, 2023, 2024, 2025} & {row["season"] for row in panel}
    assert all(row["mask_names"] and row["enable_evidence"] for row in panel)


def test_action_prefix_replay_reproduces_exact_observation(monkeypatch):
    original = synthetic_gym(mask_names=True)
    observation = original.reset()
    actions = []
    for _ in range(2):
        action = autopick_step(original)
        actions.append(action)
        observation, _, done, _ = original.step(action)
        assert not done
    monkeypatch.setattr(
        "evals.value_of_information.DraftGym",
        lambda **kwargs: synthetic_gym(mask_names=True),
    )
    _, replayed = replay_prefix({}, actions)
    assert observation_hash(replayed) == observation_hash(observation)


def test_selection_uses_highest_and_lowest_visible_uncertainty():
    def point(index, prior, stdev):
        return {
            "decision_index": index,
            "parsed": True,
            "proposed_action": {"pick": "B001"},
            "observation": {
                "tool_results": [],
                "top_available": [{
                    "player_id": "B001", "adp": 50.0,
                    "adp_stdev": stdev, "prev_season_points": prior,
                }],
            },
        }
    selected = select_decision_points([
        point(0, 100.0, 5.0), point(1, 0.0, 5.0), point(2, 100.0, 20.0),
    ])
    assert [row["decision_index"] for row in selected] == [1, 0]


def test_outcome_labels_require_resolution_change_and_material_reward():
    base = {
        "intervention": {"resolved": True},
        "branches": {
            "act": {"action_legal": True, "actual_action": {"pick": "A"}, "reward": 0.0},
            "intervene": {"action_legal": True, "actual_action": {"pick": "B"}, "reward": MATERIAL_REWARD_POINTS},
        },
    }
    assert label_intervention(base)[0] == "HELPFUL"
    base["branches"]["intervene"]["reward"] = -MATERIAL_REWARD_POINTS
    assert label_intervention(base)[0] == "HARMFUL"
    base["branches"]["intervene"]["reward"] = 0.0
    assert label_intervention(base)[0] == "UNNECESSARY"
    base["intervention"]["resolved"] = False
    assert label_intervention(base)[0] == "UNRESOLVABLE"


def test_budgeted_sampler_stops_before_hard_cap():
    class Prompt:
        length = 20_000

    class Renderer:
        def build_generation_prompt(self, messages):
            return Prompt()

    class FakeSampler:
        ledger = Ledger()
        renderer = Renderer()

        def chat(self, *args, **kwargs):
            return {"content": "ok"}

    sampler = BudgetedSampler(FakeSampler(), prior_usd=9.99)
    try:
        sampler.chat([], max_tokens=1, temperature=0.0, seed=1)
    except RuntimeError as exc:
        assert "budget cap" in str(exc)
    else:
        raise AssertionError("expected budget guard to stop the call")


def test_materialize_split_is_rectangular_and_split_safe(tmp_path, monkeypatch):
    collection = tmp_path / "collection"
    dataset = tmp_path / "development.jsonl"
    monkeypatch.setattr("evals.value_of_information.ROOT", tmp_path)
    monkeypatch.setattr("evals.value_of_information.COLLECTION_DIR", collection)
    monkeypatch.setattr("evals.value_of_information.DATASET_DIR", dataset.parent)
    panel_row = episode_panel()[0]
    record = {
        "protocol": PROTOCOL,
        "episode_id": panel_row["episode_id"],
        "split": "development",
        "examples": [
            {
                "example_id": f"example-{index}",
                "split": "development",
                "label": {"value": "UNNECESSARY"},
            }
            for index in range(2)
        ],
        "cost": {"computed_usd": 0.01},
    }
    path = collection / "development" / "one.json"
    path.parent.mkdir(parents=True)
    path.write_text(__import__("json").dumps(record))
    manifest = materialize_split("development")
    assert manifest["episodes"] == 1
    assert manifest["examples"] == 2
    assert manifest["label_counts"] == {"HARMFUL": 0, "HELPFUL": 0,
                                         "UNNECESSARY": 2, "UNRESOLVABLE": 0}
    assert collected_cost([record]) == 0.01
