#!/usr/bin/env python3
"""verifiers v1 adapter for DraftGym — Taskset/Harness/Env composition.

STATUS (2026-08-08): `pip install verifiers` resolved to **verifiers 0.1.14**
on the repo .venv (python 3.10) and imports cleanly. This adapter targets the
current v1 API (`verifiers.v1`: Taskset / Harness / Env — tasksets, harnesses,
and trace-capturing rollouts; v0 `MultiTurnEnv` is deprecated per
docs/training-risk-register.md). Verified against the installed package's
`verifiers/v1/README.md`:

- ``Taskset(source=..., rewards=[...])`` with a zero-arg lazy source loader;
- ``Harness(program=...)`` with an in-process async program
  ``async def program(task, state)`` that drives the model through
  ``state.get_client(api="chat")`` (the v1 interception endpoint, so
  trajectories/traces are captured by the framework);
- ``vf.Env(taskset=taskset, harness=harness)`` as the prime-rl-facing adapter;
- ``@vf.reward`` functions reading reference data from ``task`` and rollout
  results from ``state``.

The core env (envs/draftgym.py) stays zero-dep: this module import-guards
verifiers and degrades to a clear ImportError message at load_environment()
time when it is absent.

Season-split discipline (risk register #8, asserted here in code):
    train rollouts  = 2015-2022
    val rollouts    = 2023-2024   (checkpoint selection ONLY — point a
                                   [[orchestrator.eval.source]] at split="val")
    2025            = sealed holdout, refused by DraftGym itself.

Reward = DraftGym's terminal reward: realized season points of the agent
roster under the REALISTIC weekly-manager policy minus the same-seed
AutopickADP control under the same policy (risk register #7 — never
hindsight-optimal lineups).
"""
from __future__ import annotations

import json
from typing import Any, Iterator

try:
    import verifiers.v1 as vf

    HAVE_VERIFIERS = True
    _IMPORT_ERROR: str | None = None
except Exception as e:  # pragma: no cover - depends on install
    vf = None  # type: ignore[assignment]
    HAVE_VERIFIERS = False
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"

from envs.draftgym import HOLDOUT_SEASON

TRAIN_SEASONS = tuple(range(2015, 2023))  # 2015-2022
VAL_SEASONS = (2023, 2024)  # checkpoint-selection val ONLY
DEFAULT_PRESETS = ("ppr", "half_ppr", "standard")
DEFAULT_SLOTS = (1, 4, 7, 10, 12)
MAX_MODEL_REQUESTS_PER_PICK = 8


def _split_seasons(split: str) -> tuple[int, ...]:
    if split == "train":
        seasons = TRAIN_SEASONS
    elif split == "val":
        seasons = VAL_SEASONS
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    # Belt and braces: the sealed holdout can never enter a taskset.
    assert HOLDOUT_SEASON not in seasons
    assert not (split == "train" and any(s in VAL_SEASONS for s in seasons)), (
        "val seasons must never appear in the train split (risk register #8)"
    )
    return seasons


def episode_rows(
    split: str = "train",
    presets: tuple[str, ...] = DEFAULT_PRESETS,
    slots: tuple[int, ...] = DEFAULT_SLOTS,
    n_seeds: int = 4,
    teams: int = 12,
) -> Iterator[dict]:
    """One task row per DraftGym episode. Rows are plain JSON mappings; the
    program (not the row) instantiates the gym, so rows stay tiny."""
    for season in _split_seasons(split):
        for preset in presets:
            for slot in slots:
                for seed in range(n_seeds):
                    yield {
                        "task_id": f"draftgym-{season}-{preset}-t{teams}-s{slot}-r{seed}",
                        "season": season,
                        "preset": preset,
                        "agent_slot": slot,
                        "seed": seed,
                        "prompt": [
                            {
                                "role": "user",
                                "content": (
                                    f"Snake draft: {teams}-team {preset}, "
                                    f"slot {slot}, season {season}. The program "
                                    "drives the episode; act on each state."
                                ),
                            }
                        ],
                    }


async def draftgym_program(task: Any, state: Any) -> Any:
    """v1 Harness program: drives one full DraftGym episode.

    Model calls go through state.get_client() — the v1 interception endpoint —
    so every request/response lands in the framework-owned trajectory (traces
    for GRPO come from there, not from us).
    """
    from envs.draftgym import DraftGym
    from envs.play_llm import SYSTEM_PROMPT, autopick_action, build_messages, parse_action

    gym = DraftGym(
        season=task["season"],
        preset=task["preset"],
        agent_slot=task["agent_slot"],
        seed=task["seed"],
    )
    obs = gym.reset()
    client = state.get_client(api="chat")
    model = state.get_model()
    fallbacks = illegal = tool_calls = requests_this_pick = 0
    reward, done, info = 0.0, False, {}

    while not done:
        action = None
        if requests_this_pick < MAX_MODEL_REQUESTS_PER_PICK:
            messages = build_messages(obs)
            for attempt in range(2):  # initial + one repair retry
                response = await client.chat.completions.create(
                    model=model, messages=messages, max_tokens=700
                )
                requests_this_pick += 1
                text = response.choices[0].message.content or ""
                try:
                    action = parse_action(text)
                    break
                except ValueError as e:
                    if attempt == 0:
                        messages = messages + [
                            {"role": "assistant", "content": text},
                            {
                                "role": "user",
                                "content": f"Invalid ({e}). Reply with ONLY one "
                                "JSON action object.",
                            },
                        ]
        if action is None:
            action = autopick_action(gym)
            fallbacks += 1

        if "tool" in action:
            tool_calls += 1
            obs, reward, done, info = gym.step(action)
            continue
        try:
            obs, reward, done, info = gym.step(action)
        except ValueError:
            illegal += 1
            fallbacks += 1
            obs, reward, done, info = gym.step(autopick_action(gym))
        requests_this_pick = 0

    state["draftgym"] = {
        "reward": reward,
        "agent_points_realistic": info.get("agent_points_realistic"),
        "control_points_realistic": info.get("control_points_realistic"),
        "total_lookups": info.get("total_lookups", 0),
        "cap_hits": info.get("cap_hits", 0),
        "fallback_picks": fallbacks,
        "illegal_picks": illegal,
        "tool_calls": tool_calls,
        "picks": info.get("picks", []),
        "system_prompt": SYSTEM_PROMPT[:80] + "...",  # provenance breadcrumb
    }
    state["answer"] = json.dumps(info.get("agent_roster", []))
    return state


if HAVE_VERIFIERS:

    @vf.reward(weight=1.0)
    async def points_above_autopick(task, state) -> float:
        """Terminal DraftGym reward (realistic-manager points vs same-seed
        autopick control). Raw points scale (~±200); prime-rl's group-relative
        advantage normalizes within-group."""
        return float(state.get("draftgym", {}).get("reward", 0.0))

    @vf.metric
    async def fallback_picks(task, state) -> float:
        return float(state.get("draftgym", {}).get("fallback_picks", 0))

    @vf.metric
    async def tool_calls_used(task, state) -> float:
        return float(state.get("draftgym", {}).get("tool_calls", 0))

    def load_taskset(split: str = "train", **kwargs) -> "vf.Taskset":
        def source():
            yield from episode_rows(split=split, **kwargs)

        return vf.Taskset(
            source=source,
            rewards=[points_above_autopick],
            metrics=[fallback_picks, tool_calls_used],
        )

    def load_environment(split: str = "train", **kwargs) -> "vf.Env":
        """prime-rl entry point: vf.Env over the DraftGym taskset + a
        program-form harness (the program owns the multi-turn draft loop)."""
        return vf.Env(
            taskset=load_taskset(split=split, **kwargs),
            harness=vf.Harness(program=draftgym_program),
        )

else:  # pragma: no cover - depends on install

    def load_taskset(split: str = "train", **kwargs):
        raise ImportError(
            "verifiers is not installed in this interpreter "
            f"(import failed: {_IMPORT_ERROR}). `pip install verifiers` "
            "(v0.1.14 verified on py3.10, 2026-08-08) and retry."
        )

    load_environment = load_taskset
