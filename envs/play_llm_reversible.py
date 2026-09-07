"""Run DraftGym with a bounded, reversible evidence intervention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from envs.draftgym import DraftGym
from envs.play_llm import MAX_REQUESTS_PER_PICK, autopick_action
from envs.play_llm_supervised import _model_action
from harness.sparse_reversible import SparseReversibleController


def _apply_pick(
    gym: DraftGym, action: Mapping[str, Any]
) -> tuple[Any, float, bool, Mapping[str, Any], bool, dict[str, Any]]:
    actual = dict(action)
    try:
        observation, reward, done, info = gym.step(actual)
        return observation, reward, done, info, True, actual
    except ValueError:
        fallback = autopick_action(gym)
        observation, reward, done, info = gym.step(fallback)
        return observation, reward, done, info, False, fallback


def run_reversible_episode(
    spec: Mapping[str, Any],
    *,
    sampler: Any,
    controller: SparseReversibleController | None = None,
    request_cap: int = MAX_REQUESTS_PER_PICK,
) -> dict[str, Any]:
    gym = DraftGym(**dict(spec))
    observation = gym.reset()
    if controller is not None:
        controller.reset_episode()
    start_ledger = asdict(sampler.ledger)
    stats: dict[str, Any] = {
        **dict(spec),
        "model_requests": 0,
        "parse_failures": 0,
        "fallback_picks": 0,
        "tool_calls": 0,
        "tool_calls_ok": 0,
        "interventions": 0,
        "accepted_revisions": 0,
        "reverted_revisions": 0,
        "decision_audit": [],
        "proposed_actions": [],
    }
    requests_this_pick = 0
    turn = 0
    reward = 0.0
    done = False
    info: Mapping[str, Any] = {}
    while not done:
        turn += 1
        if requests_this_pick >= request_cap:
            proposed = autopick_action(gym)
            parsed = False
            stats["fallback_picks"] += 1
        else:
            proposed, parsed = _model_action(
                sampler,
                observation,
                seed=int(spec.get("seed", 0)) * 10_000 + turn * 10,
            )
            stats["model_requests"] += 1 if parsed else 2
            requests_this_pick += 1 if parsed else 2
            if not parsed:
                stats["parse_failures"] += 1
                stats["fallback_picks"] += 1
                proposed = autopick_action(gym)
        stats["proposed_actions"].append(dict(proposed))

        if "tool" in proposed:
            # A model-authored lookup consumes the episode's sole evidence action.
            if controller is not None and controller.interventions_used >= 1:
                proposed = autopick_action(gym)
                stats["fallback_picks"] += 1
            else:
                if controller is not None:
                    controller.interventions_used += 1
                stats["tool_calls"] += 1
                observation, reward, done, info = gym.step(proposed)
                if observation.get("tool_results") and observation["tool_results"][-1][
                    "response"
                ].get("ok"):
                    stats["tool_calls_ok"] += 1
                continue

        plan = (
            controller.plan(observation, proposed) if controller is not None else None
        )
        if plan is not None:
            stats["interventions"] += 1
            stats["tool_calls"] += 1
            post_tool, reward, done, info = gym.step({"tool": plan.tool})
            if post_tool.get("tool_results") and post_tool["tool_results"][-1][
                "response"
            ].get("ok"):
                stats["tool_calls_ok"] += 1
            if done:
                raise AssertionError("evidence lookup unexpectedly ended episode")
            turn += 1
            reconsidered, reconsidered_parsed = _model_action(
                sampler,
                post_tool,
                seed=int(spec.get("seed", 0)) * 10_000 + turn * 10,
            )
            stats["model_requests"] += 1 if reconsidered_parsed else 2
            if not reconsidered_parsed or "pick" not in reconsidered:
                stats["parse_failures"] += int(not reconsidered_parsed)
                reconsidered = dict(plan.original_action)
            resolution = controller.resolve(post_tool, reconsidered)
            stats["decision_audit"].append(resolution.audit)
            if resolution.accepted_revision:
                stats["accepted_revisions"] += 1
            else:
                stats["reverted_revisions"] += 1
            observation, reward, done, info, legal, actual = _apply_pick(
                gym,
                resolution.final_action,
            )
            resolution.audit["executed_action"] = actual
            resolution.audit["executed_legally"] = legal
            if not legal:
                stats["fallback_picks"] += 1
            requests_this_pick = 0
            continue

        observation, reward, done, info, legal, _ = _apply_pick(gym, proposed)
        if not legal:
            stats["fallback_picks"] += 1
        requests_this_pick = 0

    end_ledger = asdict(sampler.ledger)
    stats.update(
        {
            "reward": float(reward),
            "task_completed": bool(done),
            "agent_points_realistic": info.get("agent_points_realistic"),
            "control_points_realistic": info.get("control_points_realistic"),
            "usage_delta": {
                key: end_ledger[key] - start_ledger[key] for key in start_ledger
            },
        }
    )
    if stats["interventions"] > 1 or stats["tool_calls"] > 1:
        raise AssertionError("sparse runner exceeded one evidence action")
    return stats
