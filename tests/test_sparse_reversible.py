from dataclasses import dataclass

from test_draftgym import synthetic_gym

from envs.play_llm_reversible import run_reversible_episode
from harness.sparse_reversible import SparseReversibleController


def observation(*, result=None, round_number=2):
    tool_results = []
    if result is not None:
        tool_results = [
            {
                "call": {"name": "injury_status", "arguments": {"player": "1"}},
                "response": {"ok": True, "result": result},
            }
        ]
    return {
        "round": round_number,
        "lookups_remaining": 4,
        "my_roster": [{"player_id": "9", "position": "QB"}],
        "tool_results": tool_results,
        "top_available": [
            {
                "player_id": "1",
                "position": "RB",
                "adp": 20.0,
                "adp_stdev": 3.0,
                "prev_season_points": 200.0,
            },
            {
                "player_id": "2",
                "position": "WR",
                "adp": 25.0,
                "adp_stdev": 2.0,
                "prev_season_points": 190.0,
            },
            {
                "player_id": "3",
                "position": "QB",
                "adp": 26.0,
                "adp_stdev": 2.0,
                "prev_season_points": 250.0,
            },
        ],
    }


def test_controller_reverts_when_evidence_is_not_adverse():
    controller = SparseReversibleController()
    plan = controller.plan(observation(), {"pick": "1"})
    assert plan is not None
    result = controller.resolve(
        observation(
            result={
                "report_status": "Probable",
                "practice_status": "Full Participation",
            }
        ),
        {"pick": "2"},
    )
    assert result.final_action == {"pick": "1"}
    assert result.accepted_revision is False


def test_controller_accepts_safe_revision_after_adverse_evidence():
    controller = SparseReversibleController()
    assert controller.plan(observation(), {"pick": "1"}) is not None
    result = controller.resolve(
        observation(
            result={
                "report_status": "Doubtful",
                "practice_status": "Did Not Participate In Practice",
                "primary_injury": "Foot",
            }
        ),
        {"pick": "2"},
    )
    assert result.final_action == {"pick": "2"}
    assert result.accepted_revision is True


def test_controller_rejects_second_qb_and_never_plans_twice():
    controller = SparseReversibleController()
    assert controller.plan(observation(), {"pick": "1"}) is not None
    result = controller.resolve(
        observation(result={"report_status": "Out", "primary_injury": "Foot"}),
        {"pick": "3"},
    )
    assert result.final_action == {"pick": "1"}
    assert result.audit["roster_safe"] is False
    assert controller.plan(observation(), {"pick": "1"}) is None


def test_controller_only_triggers_in_frozen_round_window():
    controller = SparseReversibleController(first_round=1, last_round=4)
    assert controller.plan(observation(round_number=5), {"pick": "1"}) is None
    assert controller.plan(observation(round_number=1), {"pick": "1"}) is not None


@dataclass
class Ledger:
    training_tokens: int = 0
    prefill_tokens: int = 0
    cached_prefill_tokens: int = 0
    sample_tokens: int = 0
    checkpoint_count: int = 0


class FakeSampler:
    def __init__(self):
        self.ledger = Ledger()


def test_runner_limits_episode_to_one_tool_and_reverts(monkeypatch):
    gym = synthetic_gym(mask_names=False)
    original_observe = gym._observe

    def uncertain_observation():
        obs = original_observe()
        if not obs["done"] and obs["top_available"]:
            obs["top_available"][0]["adp_stdev"] = obs["top_available"][0]["adp"]
        return obs

    gym._observe = uncertain_observation
    monkeypatch.setattr("envs.play_llm_reversible.DraftGym", lambda **kwargs: gym)
    monkeypatch.setattr(
        "envs.play_llm_reversible._model_action",
        lambda sampler, obs, seed: (
            {"pick": obs["top_available"][0]["player_id"]},
            True,
        ),
    )
    result = run_reversible_episode(
        {},
        sampler=FakeSampler(),
        controller=SparseReversibleController(),
    )
    assert result["task_completed"] is True
    assert result["interventions"] == 1
    assert result["tool_calls"] == 1
    assert result["reverted_revisions"] == 1
    assert result["accepted_revisions"] == 0


def test_synthetic_replay_is_deterministic(monkeypatch):
    def one_run():
        gym = synthetic_gym(mask_names=False)
        original_observe = gym._observe

        def uncertain_observation():
            obs = original_observe()
            if not obs["done"] and obs["top_available"]:
                obs["top_available"][0]["adp_stdev"] = obs["top_available"][0]["adp"]
            return obs

        gym._observe = uncertain_observation
        monkeypatch.setattr("envs.play_llm_reversible.DraftGym", lambda **kwargs: gym)
        return run_reversible_episode(
            {},
            sampler=FakeSampler(),
            controller=SparseReversibleController(),
        )

    monkeypatch.setattr(
        "envs.play_llm_reversible._model_action",
        lambda sampler, obs, seed: (
            {"pick": obs["top_available"][0]["player_id"]},
            True,
        ),
    )
    first = one_run()
    second = one_run()
    assert first["reward"] == second["reward"]
    assert first["proposed_actions"] == second["proposed_actions"]
    assert first["decision_audit"] == second["decision_audit"]
