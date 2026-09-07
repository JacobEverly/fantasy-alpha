"""Outcome-aligned prompt and call-order-independent sampling seeds.

The original runner is left untouched because completed experiments hash it.
New outcome-training experiments use this protocol instead.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from envs.play_llm import MASKED_NOTE, SYSTEM_PROMPT as ORIGINAL_SYSTEM_PROMPT, parse_action

ALIGNED_SYSTEM_PROMPT = (
    "You are making one consequential decision in a fantasy-football snake draft. "
    "Your objective is to build the legal roster expected to score the most total "
    "season points when a realistic manager sets the starting lineup each week. "
    "Do not optimize for a famous player, a plausible-looking pick, or this pick in "
    "isolation. Compare each candidate's marginal starting-lineup contribution, open "
    "roster requirements, positional scarcity, redundant bench value, picks until your "
    "next turn, probability of later availability, and the opportunity cost of waiting. "
    "Use evidence only when it could change the selection.\n"
    "Respond with STRICT JSON only—one object and no markdown.\n"
    'Pick now: {"pick":"<player_id>"}\n'
    'Or request evidence: {"tool":{"name":"<tool>","arguments":{...}}}\n'
    "A pick must be a player_id from top_available. Evidence calls are bounded by "
    "lookups_remaining."
)

SEED_NAMESPACES = frozenset({"normal", "repair", "tool", "reconsideration"})


def decision_seed(
    episode: Mapping[str, Any],
    pick_number: int,
    *,
    namespace: str = "normal",
    attempt: int = 0,
) -> int:
    """Stable seed independent of the number or order of other model calls."""
    if namespace not in SEED_NAMESPACES:
        raise ValueError(f"unknown decision seed namespace: {namespace}")
    identity = "|".join(
        [
            str(episode.get("season")),
            str(episode.get("preset", "ppr")),
            str(episode.get("agent_slot")),
            str(episode.get("seed")),
            str(bool(episode.get("mask_names", False))),
            str(int(pick_number)),
            namespace,
            str(int(attempt)),
        ]
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")


def build_messages(
    observation: Mapping[str, Any], *, prompt_variant: str = "aligned"
) -> list[dict[str, str]]:
    if prompt_variant == "aligned":
        system = ALIGNED_SYSTEM_PROMPT
    elif prompt_variant == "original":
        system = ORIGINAL_SYSTEM_PROMPT
    else:
        raise ValueError(f"unknown prompt variant: {prompt_variant}")
    if observation.get("anonymized"):
        system += MASKED_NOTE
    render = getattr(observation, "render_json", None)
    state = render() if callable(render) else __import__("json").dumps(observation)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Draft state (JSON):\n{state}\n\nRespond with ONE JSON action object now.",
        },
    ]


def model_action(
    sampler: Any,
    observation: Mapping[str, Any],
    episode: Mapping[str, Any],
    *,
    prompt_variant: str,
    namespace: str = "normal",
) -> tuple[dict, bool, list[int]]:
    """Sample and, at most once, repair with disjoint deterministic seeds."""
    pick_number = int(observation["pick_number"])
    messages = build_messages(observation, prompt_variant=prompt_variant)
    used: list[int] = []
    for attempt in range(2):
        seed_namespace = namespace if attempt == 0 else "repair"
        seed = decision_seed(
            episode,
            pick_number,
            namespace=seed_namespace,
            attempt=attempt,
        )
        used.append(seed)
        response = sampler.chat(messages, max_tokens=700, temperature=0.0, seed=seed)
        text = str(response["content"])
        try:
            return parse_action(text), True, used
        except ValueError as exc:
            if attempt == 0:
                messages += [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            f"Invalid ({type(exc).__name__}). Return only one JSON pick "
                            "or tool action."
                        ),
                    },
                ]
    return {}, False, used
