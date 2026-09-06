"""Run a model through DraftGym with an optional decision-time supervisor."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from envs.draftgym import DraftGym
from envs.play_llm import MAX_REQUESTS_PER_PICK, autopick_action, build_messages, parse_action
from harness.tool_supervisor import ToolSupervisor


def _model_action(sampler: Any, obs: Mapping[str, Any], *, seed: int) -> tuple[dict, bool]:
    messages = build_messages(obs)
    for attempt in range(2):
        response = sampler.chat(
            messages, max_tokens=700, temperature=0.0, seed=seed + attempt,
        )
        text = str(response["content"])
        try:
            return parse_action(text), True
        except ValueError as exc:
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            f"Invalid ({type(exc).__name__}). Return only one JSON "
                            "pick or tool action."
                        ),
                    },
                ]
    return {}, False


def run_episode(
    spec: Mapping[str, Any], *, sampler: Any,
    supervisor: ToolSupervisor | None = None,
    request_cap: int = MAX_REQUESTS_PER_PICK,
) -> dict[str, Any]:
    """Execute a complete real DraftGym episode.

    The model proposes every action. A supervisor may approve it, replace a
    proposed pick with a required tool call, or block it and use the existing
    conservative autopick fallback. Tool results are injected by DraftGym and
    the model is called again on the updated observation.
    """
    gym = DraftGym(**dict(spec))
    obs = gym.reset()
    start_ledger = asdict(sampler.ledger)
    stats: dict[str, Any] = {
        **dict(spec),
        "model_requests": 0,
        "parse_failures": 0,
        "supervisor_interventions": 0,
        "forced_tool_calls": 0,
        "blocked_actions": 0,
        "approved_model_actions": 0,
        "tool_calls": 0,
        "tool_calls_ok": 0,
        "fallback_picks": 0,
        "decisions": [],
    }
    requests_this_pick = 0
    reward = 0.0
    done = False
    info: Mapping[str, Any] = {}
    turn = 0
    while not done:
        turn += 1
        if requests_this_pick >= request_cap:
            action = autopick_action(gym)
            stats["fallback_picks"] += 1
        else:
            action, parsed = _model_action(
                sampler, obs, seed=int(spec.get("seed", 0)) * 10_000 + turn * 10
            )
            stats["model_requests"] += 1 if parsed else 2
            requests_this_pick += 1 if parsed else 2
            if not parsed:
                stats["parse_failures"] += 1
                stats["fallback_picks"] += 1
                action = autopick_action(gym)

        if supervisor is not None:
            decision = supervisor.decide(obs, action)
            stats["decisions"].append(decision.to_dict())
            if decision.decision == "USE_TOOL" and "pick" in action:
                action = {"tool": decision.required_tool}
                stats["supervisor_interventions"] += 1
                stats["forced_tool_calls"] += 1
            elif decision.decision == "WAIT_OR_ABSTAIN":
                action = autopick_action(gym)
                stats["supervisor_interventions"] += 1
                stats["blocked_actions"] += 1
                stats["fallback_picks"] += 1
            else:
                stats["approved_model_actions"] += 1

        if "tool" in action:
            stats["tool_calls"] += 1
            obs, reward, done, info = gym.step(action)
            if obs["tool_results"] and obs["tool_results"][-1]["response"].get("ok"):
                stats["tool_calls_ok"] += 1
            continue

        try:
            obs, reward, done, info = gym.step(action)
        except ValueError:
            stats["fallback_picks"] += 1
            obs, reward, done, info = gym.step(autopick_action(gym))
        requests_this_pick = 0

    end_ledger = asdict(sampler.ledger)
    stats.update({
        "reward": reward,
        "task_completed": done,
        "agent_points_realistic": info.get("agent_points_realistic"),
        "control_points_realistic": info.get("control_points_realistic"),
        "usage_delta": {
            key: end_ledger[key] - start_ledger[key] for key in start_ledger
        },
    })
    return stats


def run_matched_panel(
    episodes: Sequence[Mapping[str, Any]], *, sampler_factory: Any,
    supervisors: Mapping[str, ToolSupervisor | None],
) -> dict[str, Any]:
    arms = {}
    for arm, supervisor in supervisors.items():
        sampler = sampler_factory()
        rows = [
            run_episode(spec, sampler=sampler, supervisor=supervisor)
            for spec in episodes
        ]
        arms[arm] = {
            "episodes": rows,
            "completed": sum(row["task_completed"] for row in rows),
            "mean_reward": sum(float(row["reward"]) for row in rows) / len(rows),
            "tool_calls": sum(int(row["tool_calls"]) for row in rows),
            "forced_tool_calls": sum(int(row["forced_tool_calls"]) for row in rows),
            "fallback_picks": sum(int(row["fallback_picks"]) for row in rows),
            "usage": asdict(sampler.ledger),
            "cost_usd": sampler.ledger.usd,
        }
    return {"arms": arms}
