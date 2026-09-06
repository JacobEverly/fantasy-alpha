"""Matched counterfactual collection for evidence-intervention value.

Each decision point is replayed from an identical action prefix.  One branch
accepts the model's proposed pick; the other performs one structured evidence
lookup and lets the same base model reconsider once.  Both branches then use
the same deterministic AutopickADP continuation, isolating the local treatment
instead of paying for—and confounding the result with—two long model rollouts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from envs.draftgym import DraftGym, Observation
from envs.play_llm import MAX_REQUESTS_PER_PICK, autopick_action, build_messages, parse_action
from harness.tool_supervisor import expected_tool, visible_candidate
from training.tinker_backend import (
    BASE_MODEL,
    PREFILL_USD_PER_MTOK,
    ROOT,
    SAMPLE_USD_PER_MTOK,
    CostLedger,
    TinkerChatSampler,
    require_api_key,
    sha256_file,
    utc_now,
)

PROTOCOL = "fantasy-alpha-value-of-information-v1"
SEASONS = (2015, 2016, 2017, 2019, 2020, 2021, 2022)
SLOTS = (1, 4, 7, 10, 12, 12, 10, 7, 4, 1)
SEEDS = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149)
SEALED_SEASONS = frozenset({2013, 2018, 2023, 2024, 2025})
MATERIAL_REWARD_POINTS = 25.0
TARGET_POINTS_PER_EPISODE = 2
TARGET_INCREMENTAL_USD = 5.0
HARD_CAP_USD = 10.0
PRIOR_TINKER_USD = 14.312830878

ARTIFACT_DIR = ROOT / "artifacts/value-of-information-v1"
COLLECTION_DIR = ARTIFACT_DIR / "collection"
DATASET_DIR = ROOT / "training/datasets/value_of_information_v1"
HELDOUT_PATH = ROOT / "evals/frozen/value_of_information_v1/heldout.jsonl"
CANDIDATE_FREEZE_PATH = ARTIFACT_DIR / "frozen-candidates.json"
SPEC_PATH = ROOT / "training/value-of-information-spec-v1.json"
SPEC_SHA_PATH = ROOT / "training/value-of-information-spec-v1.sha256"


def episode_panel() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season in SEASONS:
        for index, (slot, seed) in enumerate(zip(SLOTS, SEEDS, strict=True)):
            split = "development" if index < 5 else "heldout"
            rows.append({
                "episode_id": f"season{season}:slot{slot}:seed{seed}",
                "split": split,
                "season": season,
                "preset": "ppr",
                "agent_slot": slot,
                "seed": seed,
                "mask_names": True,
                "enable_evidence": True,
            })
    return rows


def verify_frozen_protocol() -> dict[str, Any]:
    expected = SPEC_SHA_PATH.read_text().split()[0]
    if sha256_file(SPEC_PATH) != expected:
        raise ValueError("value-of-information protocol SHA-256 mismatch")
    spec = json.loads(SPEC_PATH.read_text())
    if spec.get("protocol") != PROTOCOL:
        raise ValueError("unexpected value-of-information protocol")
    for relative, expected_hash in spec["implementation_hashes"].items():
        if sha256_file(ROOT / relative) != expected_hash:
            raise ValueError(f"frozen implementation changed: {relative}")
    if spec["episode_ids"] != [row["episode_id"] for row in episode_panel()]:
        raise ValueError("episode panel differs from frozen protocol")
    if spec["arm_evaluation_episode_ids"] != [
        row["episode_id"] for row in arm_evaluation_panel()
    ]:
        raise ValueError("arm panel differs from frozen protocol")
    return spec


def arm_evaluation_panel() -> list[dict[str, Any]]:
    """Thirty held-out episodes, balanced as closely as seven seasons permit."""
    heldout = [row for row in episode_panel() if row["split"] == "heldout"]
    by_season: dict[int, list[dict[str, Any]]] = {}
    for row in heldout:
        by_season.setdefault(int(row["season"]), []).append(row)
    selected = []
    for season in sorted(by_season):
        selected.extend(by_season[season][:4])
    selected.extend([by_season[min(by_season)][4], by_season[max(by_season)][4]])
    if len(selected) != 30 or len({row["episode_id"] for row in selected}) != 30:
        raise AssertionError("frozen arm panel must contain 30 unique episodes")
    return selected


def observation_hash(observation: Mapping[str, Any]) -> str:
    payload = json.dumps(observation, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def episode_spec(panel_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: panel_row[key]
        for key in (
            "season", "preset", "agent_slot", "seed", "mask_names",
            "enable_evidence",
        )
    }


def replay_prefix(
    spec: Mapping[str, Any], actions: Sequence[Mapping[str, Any]],
) -> tuple[DraftGym, Observation]:
    gym = DraftGym(**dict(spec))
    observation = gym.reset()
    for action in actions:
        observation, _, done, _ = gym.step(copy.deepcopy(dict(action)))
        if done:
            raise ValueError("action prefix reaches terminal state")
    return gym, observation


def _ledger_snapshot(sampler: Any) -> dict[str, int]:
    return {key: int(value) for key, value in asdict(sampler.ledger).items()}


def _ledger_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {key: int(after[key]) - int(before[key]) for key in before}


def usage_cost(usage: Mapping[str, int]) -> float:
    ledger = CostLedger(**{
        key: int(usage.get(key, 0)) for key in asdict(CostLedger())
    })
    return ledger.usd


class BudgetedSampler:
    """Hard-stop paid sampling before the experiment cap can be crossed."""

    def __init__(self, sampler: Any, *, prior_usd: float):
        self.sampler = sampler
        self.ledger = sampler.ledger
        self.prior_usd = float(prior_usd)

    def chat(
        self, messages: list[dict], *, max_tokens: int, temperature: float, seed: int,
    ) -> dict[str, Any]:
        prompt = self.sampler.renderer.build_generation_prompt(messages)
        worst_case_call_usd = (
            prompt.length * PREFILL_USD_PER_MTOK
            + max_tokens * SAMPLE_USD_PER_MTOK
        ) / 1_000_000
        projected = self.prior_usd + self.ledger.usd + worst_case_call_usd
        if projected > HARD_CAP_USD:
            raise RuntimeError(
                "next request could exceed the frozen incremental budget cap"
            )
        return self.sampler.chat(
            messages, max_tokens=max_tokens, temperature=temperature, seed=seed,
        )


def model_action(
    sampler: Any, observation: Observation, *, seed: int,
) -> dict[str, Any]:
    messages = build_messages(observation)
    before = _ledger_snapshot(sampler)
    started = time.perf_counter()
    responses: list[str] = []
    for attempt in range(2):
        response = sampler.chat(
            messages, max_tokens=700, temperature=0.0, seed=seed + attempt,
        )
        text = str(response["content"])
        responses.append(text)
        try:
            action = parse_action(text)
            parsed = True
            break
        except (ValueError, json.JSONDecodeError) as exc:
            parsed = False
            action = {}
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
    after = _ledger_snapshot(sampler)
    return {
        "action": action,
        "parsed": parsed,
        "responses": responses,
        "latency_seconds": time.perf_counter() - started,
        "usage": _ledger_delta(before, after),
        "seed": seed,
    }


def _apply_pick_or_fallback(
    gym: DraftGym, action: Mapping[str, Any],
) -> tuple[Observation, float, bool, Mapping[str, Any], dict[str, Any], bool]:
    actual = copy.deepcopy(dict(action))
    legal = "pick" in actual
    if legal:
        try:
            observation, reward, done, info = gym.step(actual)
            return observation, reward, done, info, actual, True
        except ValueError:
            legal = False
    actual = autopick_action(gym)
    observation, reward, done, info = gym.step(actual)
    return observation, reward, done, info, actual, legal


def autopick_to_terminal(
    gym: DraftGym, observation: Observation,
) -> dict[str, Any]:
    reward = 0.0
    done = bool(observation["done"])
    info: Mapping[str, Any] = {}
    actions: list[dict[str, Any]] = []
    while not done:
        action = autopick_action(gym)
        actions.append(dict(action))
        observation, reward, done, info = gym.step(action)
    return {
        "reward": float(reward),
        "info": dict(info),
        "continuation_actions": actions,
    }


def _uncertainty_score(point: Mapping[str, Any]) -> tuple[float, int]:
    observation = point["observation"]
    candidate = visible_candidate(observation, point["proposed_action"])
    if candidate is None:
        return (-1.0, int(point["decision_index"]))
    adp = max(float(candidate.get("adp") or 0.0), 1.0)
    relative_stdev = float(candidate.get("adp_stdev") or 0.0) / adp
    missing_prior = float(float(candidate.get("prev_season_points") or 0.0) <= 0)
    return (2.0 * missing_prior + relative_stdev, -int(point["decision_index"]))


def select_decision_points(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        dict(point) for point in points
        if point["parsed"]
        and set(point["proposed_action"]) == {"pick"}
        and not point["observation"].get("tool_results")
        and visible_candidate(point["observation"], point["proposed_action"])
        is not None
    ]
    if len(eligible) < TARGET_POINTS_PER_EPISODE:
        raise RuntimeError("episode has fewer than two eligible decision points")
    high = max(eligible, key=_uncertainty_score)
    remaining = [point for point in eligible if point["decision_index"] != high["decision_index"]]
    low = min(remaining, key=_uncertainty_score)
    return [high, low]


def run_baseline_episode(
    panel_row: Mapping[str, Any], sampler: Any,
) -> dict[str, Any]:
    spec = episode_spec(panel_row)
    gym = DraftGym(**spec)
    observation = gym.reset()
    prefix: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    requests_this_pick = 0
    request_index = 0
    model_requests = 0
    done = False
    reward = 0.0
    info: Mapping[str, Any] = {}
    start_usage = _ledger_snapshot(sampler)
    latency = 0.0
    while not done:
        request_index += 1
        if requests_this_pick >= MAX_REQUESTS_PER_PICK:
            result = {
                "action": autopick_action(gym), "parsed": False,
                "responses": [], "latency_seconds": 0.0,
                "usage": {key: 0 for key in start_usage},
                "seed": int(spec["seed"]) * 10000 + request_index * 10,
            }
        else:
            result = model_action(
                sampler, observation,
                seed=int(spec["seed"]) * 10000 + request_index * 10,
            )
            requests_this_pick += len(result["responses"])
            model_requests += len(result["responses"])
        latency += float(result["latency_seconds"])
        proposed = result["action"] or autopick_action(gym)
        if "tool" in proposed:
            prefix.append(copy.deepcopy(proposed))
            observation, reward, done, info = gym.step(proposed)
            continue
        points.append({
            "decision_index": len(points),
            "observation": copy.deepcopy(dict(observation)),
            "observation_sha256": observation_hash(observation),
            "prefix_actions": copy.deepcopy(prefix),
            "proposed_action": copy.deepcopy(proposed),
            "parsed": bool(result["parsed"]),
            "model_seed": int(result["seed"]),
            "model_usage": dict(result["usage"]),
            "model_latency_seconds": float(result["latency_seconds"]),
        })
        observation, reward, done, info, actual, _ = _apply_pick_or_fallback(
            gym, proposed
        )
        prefix.append(actual)
        requests_this_pick = 0
    end_usage = _ledger_snapshot(sampler)
    return {
        "episode_id": panel_row["episode_id"],
        "split": panel_row["split"],
        "spec": spec,
        "base_reward": float(reward),
        "base_info": dict(info),
        "base_actions": prefix,
        "model_requests": model_requests,
        "model_latency_seconds": latency,
        "usage": _ledger_delta(start_usage, end_usage),
        "selected_points": select_decision_points(points),
    }


def _tool_result_resolves(response: Mapping[str, Any]) -> bool:
    if not response.get("ok"):
        return False
    result = response.get("result")
    if result is None:
        return False
    if isinstance(result, (list, dict, str)) and not result:
        return False
    return True


def label_intervention(row: Mapping[str, Any]) -> tuple[str, str]:
    if not row["intervention"]["resolved"]:
        return "UNRESOLVABLE", "tool_failed_or_returned_no_evidence"
    immediate = row["branches"]["act"]
    intervention = row["branches"]["intervene"]
    if not immediate["action_legal"] and intervention["action_legal"]:
        return "HELPFUL", "prevented_illegal_action"
    changed = immediate["actual_action"] != intervention["actual_action"]
    delta = float(intervention["reward"] - immediate["reward"])
    if changed and delta >= MATERIAL_REWARD_POINTS:
        return "HELPFUL", "changed_action_positive_material_reward"
    if changed and delta <= -MATERIAL_REWARD_POINTS:
        return "HARMFUL", "changed_action_negative_material_reward"
    return "UNNECESSARY", (
        "evidence_did_not_change_action" if not changed
        else "changed_action_below_materiality_threshold"
    )


def evaluate_point(
    panel_row: Mapping[str, Any], point: Mapping[str, Any], sampler: Any,
) -> dict[str, Any]:
    spec = episode_spec(panel_row)
    act_gym, act_obs = replay_prefix(spec, point["prefix_actions"])
    if observation_hash(act_obs) != point["observation_sha256"]:
        raise AssertionError("ACT replay does not reproduce the frozen observation")
    act_obs, act_reward, act_done, act_info, act_action, act_legal = (
        _apply_pick_or_fallback(act_gym, point["proposed_action"])
    )
    act_terminal = (
        {
            "reward": float(act_reward),
            "info": dict(act_info),
            "continuation_actions": [],
        }
        if act_done else autopick_to_terminal(act_gym, act_obs)
    )

    tool_gym, tool_obs = replay_prefix(spec, point["prefix_actions"])
    if observation_hash(tool_obs) != point["observation_sha256"]:
        raise AssertionError("INTERVENE replay does not reproduce the frozen observation")
    candidate = visible_candidate(tool_obs, point["proposed_action"])
    if candidate is None:
        raise AssertionError("selected proposed pick is not visible")
    tool = expected_tool(candidate)
    post_tool_obs, _, _, _ = tool_gym.step({"tool": tool})
    response = copy.deepcopy(post_tool_obs["tool_results"][-1]["response"])
    resolved = _tool_result_resolves(response)
    if resolved:
        reconsidered = model_action(
            sampler, post_tool_obs, seed=int(point["model_seed"]),
        )
        post_action = reconsidered["action"]
        if set(post_action) != {"pick"}:
            post_action = autopick_action(tool_gym)
            post_parsed_pick = False
        else:
            post_parsed_pick = bool(reconsidered["parsed"])
    else:
        reconsidered = {
            "responses": [], "usage": {key: 0 for key in _ledger_snapshot(sampler)},
            "latency_seconds": 0.0, "seed": int(point["model_seed"]),
        }
        post_action = copy.deepcopy(point["proposed_action"])
        post_parsed_pick = False
    post_tool_obs, tool_reward, tool_done, tool_info, actual_post, post_legal = (
        _apply_pick_or_fallback(tool_gym, post_action)
    )
    tool_terminal = (
        {
            "reward": float(tool_reward),
            "info": dict(tool_info),
            "continuation_actions": [],
        }
        if tool_done else autopick_to_terminal(tool_gym, post_tool_obs)
    )

    row = {
        "protocol": PROTOCOL,
        "example_id": f"{panel_row['episode_id']}:decision{point['decision_index']}",
        "episode_id": panel_row["episode_id"],
        "split": panel_row["split"],
        "input": {
            "observation": copy.deepcopy(point["observation"]),
            "proposed_action": copy.deepcopy(point["proposed_action"]),
        },
        "intervention": {
            "tool": tool,
            "response": response,
            "resolved": resolved,
            "post_tool_model_action": copy.deepcopy(post_action),
            "post_tool_model_parsed_pick": post_parsed_pick,
            "usage": dict(reconsidered["usage"]),
            "computed_usd": usage_cost(reconsidered["usage"]),
            "latency_seconds": float(reconsidered["latency_seconds"]),
        },
        "branches": {
            "act": {
                "actual_action": act_action,
                "action_legal": act_legal,
                "reward": float(act_terminal["reward"]),
                "continuation_actions": act_terminal["continuation_actions"],
            },
            "intervene": {
                "actual_action": actual_post,
                "action_legal": post_legal,
                "reward": float(tool_terminal["reward"]),
                "continuation_actions": tool_terminal["continuation_actions"],
            },
        },
        "outcome": {
            "action_changed": act_action != actual_post,
            "reward_difference": float(tool_terminal["reward"] - act_terminal["reward"]),
        },
        "provenance": {
            "observation_sha256": point["observation_sha256"],
            "prefix_actions": copy.deepcopy(point["prefix_actions"]),
            "model": BASE_MODEL,
            "model_seed": int(point["model_seed"]),
            "masked": True,
            "fixed_continuation": "AutopickADP",
            "label_uses_realized_outcome": True,
            "outcome_fields_prohibited_as_features": True,
        },
    }
    label, reason = label_intervention(row)
    row["label"] = {
        "value": label,
        "reason_code": reason,
        "material_reward_points": MATERIAL_REWARD_POINTS,
    }
    return row


def _episode_path(panel_row: Mapping[str, Any]) -> Path:
    safe_id = str(panel_row["episode_id"]).replace(":", "__")
    return COLLECTION_DIR / str(panel_row["split"]) / f"{safe_id}.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def collected_episode_paths() -> list[Path]:
    return sorted(COLLECTION_DIR.glob("*/*.json")) if COLLECTION_DIR.exists() else []


def load_collected_episodes() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    valid_ids = {row["episode_id"]: row for row in episode_panel()}
    for path in collected_episode_paths():
        record = json.loads(path.read_text())
        episode_id = str(record.get("episode_id", ""))
        expected = valid_ids.get(episode_id)
        if expected is None or record.get("split") != expected["split"]:
            raise ValueError(f"invalid collected episode provenance: {path}")
        if record.get("protocol") != PROTOCOL:
            raise ValueError(f"unexpected protocol in {path}")
        records.append(record)
    if len({row["episode_id"] for row in records}) != len(records):
        raise ValueError("duplicate collected episode")
    return records


def collected_cost(records: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(row["cost"]["computed_usd"]) for row in records)


def materialize_split(split: str) -> dict[str, Any]:
    if split not in {"development", "heldout"}:
        raise ValueError("split must be development or heldout")
    episodes = [row for row in load_collected_episodes() if row["split"] == split]
    examples = [example for row in episodes for example in row["examples"]]
    expected_examples = len(episodes) * TARGET_POINTS_PER_EPISODE
    if len(examples) != expected_examples:
        raise ValueError("collected episode has an incomplete counterfactual pair")
    if any(example["split"] != split for example in examples):
        raise ValueError("example split does not match episode split")
    path = (
        DATASET_DIR / "development.jsonl" if split == "development" else HELDOUT_PATH
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in examples
    ))
    temporary.replace(path)
    manifest = {
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "split": split,
        "episodes": len(episodes),
        "examples": len(examples),
        "dataset_path": str(path.relative_to(ROOT)),
        "dataset_sha256": sha256_file(path),
        "episode_ids": sorted(row["episode_id"] for row in episodes),
        "label_counts": {
            label: sum(row["label"]["value"] == label for row in examples)
            for label in ("HELPFUL", "UNNECESSARY", "HARMFUL", "UNRESOLVABLE")
        },
        "incremental_usd": collected_cost(episodes),
        "complete": len(episodes) == 35 and len(examples) == 70,
    }
    _write_json(path.with_suffix(".manifest.json"), manifest)
    return manifest


def collect_split(split: str, *, limit: int | None = None) -> dict[str, Any]:
    verify_frozen_protocol()
    if split not in {"development", "heldout"}:
        raise ValueError("split must be development or heldout")
    if split == "heldout" and not CANDIDATE_FREEZE_PATH.exists():
        raise RuntimeError("freeze supervisor candidates before collecting heldout labels")
    require_api_key()
    panel = [row for row in episode_panel() if row["split"] == split]
    existing = {row["episode_id"]: row for row in load_collected_episodes()}
    todo = [row for row in panel if row["episode_id"] not in existing]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        todo = todo[:limit]
    prior_usd = collected_cost(list(existing.values()))
    if prior_usd >= HARD_CAP_USD:
        raise RuntimeError("frozen incremental budget cap already reached")
    sampler = BudgetedSampler(
        TinkerChatSampler(experiment=PROTOCOL), prior_usd=prior_usd,
    )
    for panel_row in todo:
        before = _ledger_snapshot(sampler)
        baseline = run_baseline_episode(panel_row, sampler)
        examples = [
            evaluate_point(panel_row, point, sampler)
            for point in baseline.pop("selected_points")
        ]
        after = _ledger_snapshot(sampler)
        usage = _ledger_delta(before, after)
        record = {
            "protocol": PROTOCOL,
            "created_at": utc_now(),
            "episode_id": panel_row["episode_id"],
            "split": split,
            "panel": dict(panel_row),
            "baseline": baseline,
            "examples": examples,
            "cost": {**usage, "computed_usd": usage_cost(usage)},
        }
        _write_json(_episode_path(panel_row), record)
    return materialize_split(split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("panel", "collect-development", "collect-heldout", "materialize"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("development", "heldout"))
    args = parser.parse_args()
    if args.command == "panel":
        print(json.dumps(episode_panel(), indent=2, sort_keys=True))
    elif args.command == "collect-development":
        print(json.dumps(
            collect_split("development", limit=args.limit), indent=2, sort_keys=True,
        ))
    elif args.command == "collect-heldout":
        print(json.dumps(
            collect_split("heldout", limit=args.limit), indent=2, sort_keys=True,
        ))
    elif args.command == "materialize":
        if not args.split:
            parser.error("materialize requires --split")
        print(json.dumps(materialize_split(args.split), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
