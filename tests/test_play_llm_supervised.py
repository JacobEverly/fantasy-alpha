from dataclasses import dataclass

from envs.play_llm_supervised import run_episode
from harness.tool_supervisor import HardCodedToolSupervisor
from test_draftgym import synthetic_gym


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

    def chat(self, messages, **kwargs):
        del messages, kwargs
        self.ledger.prefill_tokens += 10
        self.ledger.sample_tokens += 2
        return {"content": '{"pick":"1"}'}


def test_supervisor_runner_forces_tool_and_completes(monkeypatch):
    gym = synthetic_gym(mask_names=False)
    original = gym._observe

    def uncertain_observation():
        obs = original()
        if not obs["done"] and obs["top_available"]:
            obs["top_available"][0]["prev_season_points"] = 0.0
        return obs

    gym._observe = uncertain_observation
    monkeypatch.setattr("envs.play_llm_supervised.DraftGym", lambda **kwargs: gym)
    monkeypatch.setattr(
        "envs.play_llm_supervised._model_action",
        lambda sampler, obs, seed: ({"pick": obs["top_available"][0]["player_id"]}, True),
    )
    result = run_episode(
        {}, sampler=FakeSampler(), supervisor=HardCodedToolSupervisor(), request_cap=2
    )
    assert result["task_completed"] is True
    assert result["forced_tool_calls"] > 0
    assert result["tool_calls"] > 0
    assert result["fallback_picks"] > 0  # evidence is disabled in the synthetic gym


def test_unsupervised_runner_preserves_model_actions(monkeypatch):
    gym = synthetic_gym(mask_names=False)
    monkeypatch.setattr("envs.play_llm_supervised.DraftGym", lambda **kwargs: gym)
    monkeypatch.setattr(
        "envs.play_llm_supervised._model_action",
        lambda sampler, obs, seed: ({"pick": obs["top_available"][0]["player_id"]}, True),
    )
    result = run_episode({}, sampler=FakeSampler(), supervisor=None)
    assert result["task_completed"] is True
    assert result["supervisor_interventions"] == 0
