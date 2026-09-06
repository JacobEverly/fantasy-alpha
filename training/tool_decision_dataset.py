"""Build Fantasy Alpha's matched-state tool-decision dataset.

The unit of independence is a full DraftGym episode.  All examples derived
from the same season/slot/seed stay in one split, and every matched group
contains four decision surfaces:

* a proposed pick that requires one evidence lookup;
* the corresponding post-tool state where acting is now permitted;
* an unresolved lookup with no remaining lookup budget, which must abstain;
* a genuine nearby DraftGym state where the expert can act without a lookup.

Labels are produced without realized-season outcomes.  They encode a
conservative *decision policy*, not a claim that a lookup improves fantasy
points.  Downstream DraftGym evaluation is responsible for that second claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from envs.draftgym import DraftGym
from envs.play_llm import parse_action
from training.t11_dataset import (
    _collect_direct_candidates,
    _collect_tool_candidates,
    _extract_observation,
)
from training.tinker_backend import ROOT, SEALED_SEASONS, sha256_file

PROTOCOL = "fantasy-alpha-tool-decision-v1"
DATASET_SEED = 20260906
EXPECTED_GROUPS = 237
EXPECTED_ROWS_PER_GROUP = 4
SPLITS = ("train", "development", "heldout")
DECISIONS = ("ACT_NOW", "USE_TOOL", "WAIT_OR_ABSTAIN")

DATASET_DIR = ROOT / "training/datasets/tool_decision_v1"
TRAIN_PATH = DATASET_DIR / "train.jsonl"
DEVELOPMENT_PATH = DATASET_DIR / "development.jsonl"
HELDOUT_PATH = ROOT / "evals/frozen/tool_decision_v1/heldout.jsonl"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
CARD_PATH = ROOT / "docs/models/fantasy-alpha-tool-decision-v1-dataset-card.md"


def _stable_digest(value: object) -> str:
    return hashlib.sha256(f"{DATASET_SEED}:{value}".encode()).hexdigest()


def _episode_id(meta: Mapping[str, Any]) -> str:
    episode = meta["episode"]
    return (
        f"season{int(episode['season'])}:slot{int(episode['slot'])}:"
        f"seed{int(episode['seed'])}"
    )


def _split_episodes(episode_ids: Sequence[str]) -> dict[str, str]:
    """Deterministically assign whole episodes, stratified by season.

    Every historical season has fifteen source episodes.  Ten go to training,
    two to development, and three to the untouched internal held-out split.
    This is stronger than merely grouping the four variants of one state.
    """
    by_season: dict[int, list[str]] = defaultdict(list)
    for episode_id in set(episode_ids):
        season = int(episode_id.split(":", 1)[0].removeprefix("season"))
        by_season[season].append(episode_id)
    assigned: dict[str, str] = {}
    for season, ids in sorted(by_season.items()):
        ordered = sorted(ids, key=_stable_digest)
        if len(ordered) != 15:
            raise AssertionError(f"season {season}: expected 15 episodes, found {len(ordered)}")
        for index, episode_id in enumerate(ordered):
            assigned[episode_id] = (
                "train" if index < 10 else "development" if index < 12 else "heldout"
            )
    return assigned


def _candidate(obs: Mapping[str, Any], player_id: str) -> Mapping[str, Any]:
    return next(
        row for row in obs["top_available"]
        if str(row["player_id"]) == str(player_id)
    )


def _tool_for_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    player_id = str(candidate["player_id"])
    if float(candidate["prev_season_points"]) <= 0:
        return {
            "name": "depth_chart",
            "arguments": {"team_or_player": player_id},
        }
    return {
        "name": "injury_status",
        "arguments": {"player": player_id},
    }


def _execute_masked_tool(
    gyms: dict[int, DraftGym], *, season: int, tool: Mapping[str, Any],
) -> dict[str, Any]:
    gym = gyms.get(season)
    if gym is None:
        gym = DraftGym(
            season=season, preset="ppr", agent_slot=1, seed=0,
            mask_names=True, enable_evidence=True,
        )
        gym.reset()
        gyms[season] = gym
    store = gym._evidence_store()
    if store is None:
        raise RuntimeError(f"season {season}: evidence store is unavailable")
    response = gym._masked_dispatch(store, dict(tool))
    if not response.get("ok"):
        raise RuntimeError(f"season {season}: evidence lookup failed")
    return response


def _with_tool_result(
    obs: Mapping[str, Any], *, tool: Mapping[str, Any],
    response: Mapping[str, Any], exhausted: bool = False,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(obs))
    result["lookups_used_this_pick"] = int(result["lookups_used_this_pick"]) + 1
    result["lookups_remaining"] = (
        0 if exhausted else max(0, int(result["lookups_remaining"]) - 1)
    )
    result["tool_results"] = [{"call": dict(tool), "response": dict(response)}]
    return result


def _example(
    *, group_id: str, episode_id: str, split: str, scenario: str,
    observation: Mapping[str, Any], proposed_action: Mapping[str, Any],
    decision: str, reason_code: str, missing_information: str | None,
    correct_tool: Mapping[str, Any] | None, safe_to_act: bool,
    legal_actions_after_tool: Sequence[Mapping[str, Any]],
    source_trace_ids: Sequence[str],
) -> dict[str, Any]:
    example_id = f"{group_id}:{scenario}"
    return {
        "example_id": example_id,
        "matched_group_id": group_id,
        "episode_group_id": episode_id,
        "split": split,
        "scenario": scenario,
        "input": {
            "observation": copy.deepcopy(dict(observation)),
            "proposed_action": copy.deepcopy(dict(proposed_action)),
        },
        "label": {
            "decision": decision,
            "safe_to_act": safe_to_act,
            "reason_code": reason_code,
            "missing_information": missing_information,
            "correct_tool": copy.deepcopy(dict(correct_tool)) if correct_tool else None,
            "legal_actions_after_tool": [dict(action) for action in legal_actions_after_tool],
        },
        "provenance": {
            "protocol": PROTOCOL,
            "source_trace_ids": list(source_trace_ids),
            "label_uses_realized_outcome": False,
            "input_fields_available_at_decision_time": True,
        },
    }


def build_examples() -> list[dict[str, Any]]:
    tool_groups = _collect_tool_candidates()
    direct_rows = _collect_direct_candidates()
    if len(tool_groups) != EXPECTED_GROUPS:
        raise AssertionError(
            f"expected {EXPECTED_GROUPS} evidence groups, found {len(tool_groups)}"
        )

    direct_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in direct_rows:
        direct_by_episode[_episode_id(row["meta"])].append(row)
    for rows in direct_by_episode.values():
        rows.sort(key=lambda row: (
            int(row["meta"]["episode"]["round"]),
            _stable_digest(row["meta"]["trace_id"]),
        ))

    # Split against the complete 15-episode source grid for every season, not
    # only episodes that happen to contain a usable evidence opportunity.
    # Otherwise sparse-positive seasons would receive a different split ratio.
    episode_ids = (
        [_episode_id(call["meta"]) for call, _, _ in tool_groups]
        + [_episode_id(row["meta"]) for row in direct_rows]
    )
    split_by_episode = _split_episodes(episode_ids)
    used_direct: set[str] = set()
    gyms: dict[int, DraftGym] = {}
    examples: list[dict[str, Any]] = []

    for call_row, post_row, _ in sorted(
        tool_groups, key=lambda group: group[0]["meta"]["trace_id"]
    ):
        meta = call_row["meta"]
        episode_id = _episode_id(meta)
        split = split_by_episode[episode_id]
        group_id = str(meta["question_id"])
        before = _extract_observation(call_row["messages"][1]["content"])
        original_target = str(meta["tool_target"])
        proposed_pick = {"pick": original_target}
        candidate = _candidate(before, original_target)
        tool = _tool_for_candidate(candidate)
        tool_name = str(tool["name"])

        if tool_name == "depth_chart":
            actual_post = _extract_observation(post_row["messages"][1]["content"])
            response = actual_post["tool_results"][-1]["response"]
            post_obs = _with_tool_result(
                before, tool=tool, response=response,
            )
            post_action = parse_action(post_row["messages"][2]["content"])
        else:
            response = _execute_masked_tool(
                gyms, season=int(meta["season"]), tool=tool,
            )
            post_obs = _with_tool_result(before, tool=tool, response=response)
            post_action = proposed_pick

        failed_obs = _with_tool_result(
            before,
            tool=tool,
            response={"ok": False, "error": "evidence unavailable"},
            exhausted=True,
        )

        direct_candidates = [
            row for row in direct_by_episode[episode_id]
            if row["meta"]["trace_id"] not in used_direct
        ]
        if not direct_candidates:
            raise AssertionError(f"{episode_id}: no unused direct state for {group_id}")
        target_round = int(meta["episode"]["round"])
        direct_row = min(
            direct_candidates,
            key=lambda row: (
                abs(int(row["meta"]["episode"]["round"]) - target_round),
                _stable_digest(row["meta"]["trace_id"]),
            ),
        )
        used_direct.add(str(direct_row["meta"]["trace_id"]))
        direct_obs = _extract_observation(direct_row["messages"][1]["content"])
        direct_action = parse_action(direct_row["messages"][2]["content"])

        source_ids = [
            str(call_row["meta"]["trace_id"]),
            str(post_row["meta"]["trace_id"]),
            str(direct_row["meta"]["trace_id"]),
        ]
        common = {
            "group_id": group_id,
            "episode_id": episode_id,
            "split": split,
            "source_trace_ids": source_ids,
        }
        examples.extend([
            _example(
                **common,
                scenario="tool_required",
                observation=before,
                proposed_action=proposed_pick,
                decision="USE_TOOL",
                reason_code=f"missing_{tool_name}",
                missing_information=tool_name,
                correct_tool=tool,
                safe_to_act=False,
                legal_actions_after_tool=[post_action],
            ),
            _example(
                **common,
                scenario="post_tool_resolved",
                observation=post_obs,
                proposed_action=post_action,
                decision="ACT_NOW",
                reason_code=f"{tool_name}_available",
                missing_information=None,
                correct_tool=None,
                safe_to_act=True,
                legal_actions_after_tool=[post_action],
            ),
            _example(
                **common,
                scenario="unresolved_no_budget",
                observation=failed_obs,
                proposed_action=proposed_pick,
                decision="WAIT_OR_ABSTAIN",
                reason_code="evidence_unavailable_no_budget",
                missing_information=tool_name,
                correct_tool=None,
                safe_to_act=False,
                legal_actions_after_tool=[],
            ),
            _example(
                **common,
                scenario="context_sufficient",
                observation=direct_obs,
                proposed_action=direct_action,
                decision="ACT_NOW",
                reason_code="decision_context_sufficient",
                missing_information=None,
                correct_tool=None,
                safe_to_act=True,
                legal_actions_after_tool=[direct_action],
            ),
        ])
    return sorted(examples, key=lambda row: row["example_id"])


def _input_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(row["input"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_examples(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    expected_rows = EXPECTED_GROUPS * EXPECTED_ROWS_PER_GROUP
    if len(examples) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(examples)}")
    ids = [str(row.get("example_id", "")) for row in examples]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("example IDs must be nonempty and unique")

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    episodes: dict[str, set[str]] = defaultdict(set)
    scenarios: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    seasons: Counter[int] = Counter()
    input_hashes: Counter[str] = Counter()
    for row in examples:
        group = str(row.get("matched_group_id"))
        split = str(row.get("split"))
        episode = str(row.get("episode_group_id"))
        label = row.get("label") or {}
        decision = str(label.get("decision"))
        obs = (row.get("input") or {}).get("observation") or {}
        proposed = (row.get("input") or {}).get("proposed_action") or {}
        groups[group].append(row)
        episodes[episode].add(split)
        scenarios[str(row.get("scenario"))] += 1
        decisions[decision] += 1
        seasons[int(obs.get("season", -1))] += 1
        input_hashes[_input_hash(row)] += 1

        if split not in SPLITS:
            errors.append(f"{row.get('example_id')}: invalid split {split}")
        if decision not in DECISIONS:
            errors.append(f"{row.get('example_id')}: invalid decision {decision}")
        if int(obs.get("season", -1)) in SEALED_SEASONS:
            errors.append(f"{row.get('example_id')}: sealed season present")
        if set(proposed) not in ({"pick"}, {"tool"}):
            errors.append(f"{row.get('example_id')}: invalid proposed action")
        if row.get("provenance", {}).get("label_uses_realized_outcome") is not False:
            errors.append(f"{row.get('example_id')}: invalid label provenance")
        correct_tool = label.get("correct_tool")
        if decision == "USE_TOOL":
            if not isinstance(correct_tool, Mapping):
                errors.append(f"{row.get('example_id')}: missing correct tool")
            else:
                tools[str(correct_tool.get("name"))] += 1
        elif correct_tool is not None:
            errors.append(f"{row.get('example_id')}: non-tool label has a tool")

    expected_scenarios = {
        "context_sufficient": EXPECTED_GROUPS,
        "post_tool_resolved": EXPECTED_GROUPS,
        "tool_required": EXPECTED_GROUPS,
        "unresolved_no_budget": EXPECTED_GROUPS,
    }
    if dict(scenarios) != expected_scenarios:
        errors.append(f"scenario coverage mismatch: {dict(scenarios)}")
    if decisions != {
        "ACT_NOW": EXPECTED_GROUPS * 2,
        "USE_TOOL": EXPECTED_GROUPS,
        "WAIT_OR_ABSTAIN": EXPECTED_GROUPS,
    }:
        errors.append(f"decision coverage mismatch: {dict(decisions)}")
    if tools != {"depth_chart": 177, "injury_status": 60}:
        errors.append(f"tool coverage mismatch: {dict(tools)}")
    if len(groups) != EXPECTED_GROUPS or any(
        len(rows) != EXPECTED_ROWS_PER_GROUP
        or len({str(row["split"]) for row in rows}) != 1
        for rows in groups.values()
    ):
        errors.append("matched groups must have four rows in exactly one split")
    if any(len(splits) != 1 for splits in episodes.values()):
        errors.append("episode groups cross splits")
    duplicates = sum(count - 1 for count in input_hashes.values() if count > 1)
    if duplicates:
        errors.append(f"duplicate decision-time inputs: {duplicates}")
    serialized = json.dumps(examples, sort_keys=True).lower()
    if '"season": 2025' in serialized or "season 2025" in serialized:
        errors.append("2025 marker present")
    if errors:
        raise ValueError("tool-decision dataset validation failed:\n- " + "\n- ".join(errors))

    split_counts = Counter(str(row["split"]) for row in examples)
    split_groups = {
        split: len({str(row["matched_group_id"]) for row in examples if row["split"] == split})
        for split in SPLITS
    }
    split_episodes = {
        split: len({str(row["episode_group_id"]) for row in examples if row["split"] == split})
        for split in SPLITS
    }
    return {
        "valid": True,
        "rows": len(examples),
        "matched_groups": len(groups),
        "episode_groups": len(episodes),
        "split_rows": dict(sorted(split_counts.items())),
        "split_matched_groups": split_groups,
        "split_episode_groups": split_episodes,
        "scenarios": dict(sorted(scenarios.items())),
        "decisions": dict(sorted(decisions.items())),
        "tools": dict(sorted(tools.items())),
        "season_rows": {str(key): value for key, value in sorted(seasons.items())},
        "duplicate_decision_inputs": duplicates,
        "sealed_seasons_present": [],
        "labels_use_realized_outcomes": False,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def materialize() -> dict[str, Any]:
    examples = build_examples()
    validation = validate_examples(examples)
    paths = {
        "train": TRAIN_PATH,
        "development": DEVELOPMENT_PATH,
        "heldout": HELDOUT_PATH,
    }
    for split, path in paths.items():
        _write_jsonl(path, [row for row in examples if row["split"] == split])
    manifest = {
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_seed": DATASET_SEED,
        "split_policy": (
            "season-stratified whole-episode split: 10 train, 2 development, "
            "3 internal-heldout episodes per season"
        ),
        "evaluation_role": (
            "internal held-out development benchmark; not the sealed 2013, 2018, "
            "2023, 2024, or 2025 season gates"
        ),
        "files": {
            split: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "rows": sum(row["split"] == split for row in examples),
            }
            for split, path in paths.items()
        },
        "validation": validation,
        "limitations": [
            "Labels encode a conservative outcome-blind expert policy, not observed product lift.",
            "ACT_NOW has two surfaces (post-tool and context-sufficient), so class counts are 2:1:1 while scenario counts are balanced.",
            "The current tool surface covers structured depth-chart and injury lookups; named free-text search remains outside this masked benchmark.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_split(split: str, *, verify: bool = True) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    manifest = json.loads(MANIFEST_PATH.read_text())
    entry = manifest["files"][split]
    path = ROOT / entry["path"]
    if verify and sha256_file(path) != entry["sha256"]:
        raise ValueError(f"{split} dataset SHA-256 mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(row.get("split") != split for row in rows):
        raise ValueError(f"{split} file contains another split")
    return rows


def write_dataset_card(manifest: Mapping[str, Any]) -> None:
    validation = manifest["validation"]
    lines = [
        "# Fantasy Alpha tool-decision dataset v1",
        "",
        "A matched-state, outcome-blind dataset for deciding whether a proposed",
        "DraftGym pick may proceed, requires a structured evidence lookup, or must",
        "be blocked because evidence is unavailable.",
        "",
        "## Composition",
        "",
        f"- {validation['rows']} examples in {validation['matched_groups']} matched groups.",
        f"- {validation['episode_groups']} whole episodes; episodes never cross splits.",
        f"- Split rows: `{json.dumps(validation['split_rows'], sort_keys=True)}`.",
        f"- Scenario counts: `{json.dumps(validation['scenarios'], sort_keys=True)}`.",
        f"- Tool labels: `{json.dumps(validation['tools'], sort_keys=True)}`.",
        "- All states are anonymized and use only training seasons.",
        "- 2013, 2018, 2023, 2024, and 2025 are absent.",
        "",
        "## Matched design",
        "",
        "Each evidence group contributes a pre-tool decision, its successful",
        "post-tool state, a failed/no-budget state, and a genuine nearby state from",
        "the same episode where the expert policy acts without a lookup. Splitting",
        "by full episode prevents nearly adjacent draft states from leaking across",
        "training and evaluation.",
        "",
        "## Label provenance",
        "",
        "Labels come from the outcome-blind SurvivalSequencer workflow and strictly",
        "pre-as-of EvidenceStore responses. Realized season outcomes, verifier",
        "rewards, and future evidence never enter either inputs or labels.",
        "",
        "## Intended use",
        "",
        "Train and compare a narrow tool-decision supervisor. Held-out rows are for",
        "one frozen internal evaluation only and must never be passed to a trainer.",
        "This dataset does not establish that tool use improves season points; that",
        "requires a later paired DraftGym run.",
        "",
        "## Known limitations",
        "",
        *[f"- {item}" for item in manifest["limitations"]],
        "",
    ]
    CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    CARD_PATH.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-card", action="store_true")
    args = parser.parse_args()
    manifest = materialize()
    if args.write_card:
        write_dataset_card(manifest)
    print(json.dumps(manifest["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
