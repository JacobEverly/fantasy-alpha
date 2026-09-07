from dataclasses import dataclass

from envs.outcome_decision_protocol import (
    ALIGNED_SYSTEM_PROMPT,
    build_messages,
    decision_seed,
    model_action,
)


EPISODE = {
    "season": 2021,
    "preset": "ppr",
    "agent_slot": 7,
    "seed": 13,
    "mask_names": True,
}


def test_normal_seed_is_pick_stable_and_namespaces_are_disjoint():
    normal = decision_seed(EPISODE, 42, namespace="normal")
    assert normal == decision_seed(EPISODE, 42, namespace="normal")
    assert len(
        {
            normal,
            decision_seed(EPISODE, 42, namespace="repair", attempt=1),
            decision_seed(EPISODE, 42, namespace="tool"),
            decision_seed(EPISODE, 42, namespace="reconsideration"),
        }
    ) == 4
    assert normal != decision_seed(EPISODE, 43, namespace="normal")


def test_aligned_prompt_names_completed_team_objective():
    observation = {"pick_number": 1, "anonymized": True, "top_available": []}
    messages = build_messages(observation, prompt_variant="aligned")
    system = messages[0]["content"]
    assert system.startswith(ALIGNED_SYSTEM_PROMPT)
    assert "most total season points" in system
    assert "marginal starting-lineup contribution" in system
    assert "opportunity cost of waiting" in system
    assert "ANONYMIZED" in system


@dataclass
class Sampler:
    replies: list[str]

    def __post_init__(self):
        self.seeds = []

    def chat(self, messages, *, max_tokens, temperature, seed):
        self.seeds.append(seed)
        return {"content": self.replies.pop(0)}


def test_repair_never_advances_a_later_normal_seed():
    observation = {
        "pick_number": 18,
        "anonymized": False,
        "top_available": [{"player_id": "7"}],
    }
    sampler = Sampler(["not json", '{"pick":"7"}'])
    action, parsed, used = model_action(
        sampler, observation, EPISODE, prompt_variant="aligned"
    )
    assert parsed and action == {"pick": "7"}
    assert used == sampler.seeds
    later = decision_seed(EPISODE, 31, namespace="normal")
    assert later == decision_seed(EPISODE, 31, namespace="normal")
    assert later not in used
