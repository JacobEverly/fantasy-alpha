"""Frozen held-out and DraftGym evaluation for tool-decision supervisors."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from envs.play_llm_supervised import run_episode
from harness.tool_supervisor import (
    AlwaysActSupervisor,
    AlwaysUseToolSupervisor,
    HardCodedToolSupervisor,
    LearnedToolSupervisor,
    SupervisorDecision,
)
from training import tool_decision_sft as sft
from training.tinker_backend import (
    BASE_MODEL,
    CACHED_PREFILL_USD_PER_MTOK,
    PREFILL_USD_PER_MTOK,
    ROOT,
    SAMPLE_USD_PER_MTOK,
    TinkerChatSampler,
    sha256_file,
)
from training.tool_decision_dataset import load_split
from training.tool_decision_supervisor import (
    ARTIFACT_PATH,
    evaluate_supervisor,
    score_predictions,
    verify_frozen_spec,
)

PROTOCOL = "fantasy-alpha-tool-decision-frozen-evaluation-v1"
CANDIDATE_MANIFEST = ROOT / "training/tool-decision-candidates-v1.json"
CANDIDATE_MANIFEST_SHA = ROOT / "training/tool-decision-candidates-v1.sha256"
OUT_DIR = ROOT / "artifacts/tool-decision-supervisor-v1/frozen-evaluation"
MODEL_CHECKPOINT = OUT_DIR / "base-heldout-records.jsonl"
MODEL_RESULT = OUT_DIR / "base-heldout.json"
LOCAL_RESULT = OUT_DIR / "local-heldout.json"
DRAFTGYM_CHECKPOINT = OUT_DIR / "draftgym-records.jsonl"
DRAFTGYM_RESULT = OUT_DIR / "draftgym-scorecard.json"
FINAL_SCORECARD = OUT_DIR / "scorecard.json"

FROZEN_EPISODES = (
    {"season": 2015, "preset": "ppr", "agent_slot": 1, "seed": 11,
     "mask_names": True, "enable_evidence": True},
    {"season": 2016, "preset": "ppr", "agent_slot": 12, "seed": 3,
     "mask_names": True, "enable_evidence": True},
    {"season": 2017, "preset": "ppr", "agent_slot": 4, "seed": 29,
     "mask_names": True, "enable_evidence": True},
    {"season": 2019, "preset": "ppr", "agent_slot": 1, "seed": 11,
     "mask_names": True, "enable_evidence": True},
    {"season": 2020, "preset": "ppr", "agent_slot": 7, "seed": 11,
     "mask_names": True, "enable_evidence": True},
    {"season": 2021, "preset": "ppr", "agent_slot": 12, "seed": 3,
     "mask_names": True, "enable_evidence": True},
    {"season": 2022, "preset": "ppr", "agent_slot": 4, "seed": 11,
     "mask_names": True, "enable_evidence": True},
)


def verify_candidates() -> dict[str, Any]:
    verify_frozen_spec()
    expected = CANDIDATE_MANIFEST_SHA.read_text().split()[0]
    if sha256_file(CANDIDATE_MANIFEST) != expected:
        raise ValueError("candidate manifest SHA-256 mismatch")
    manifest = json.loads(CANDIDATE_MANIFEST.read_text())
    for relative, expected_hash in manifest["hashes"].items():
        if sha256_file(ROOT / relative) != expected_hash:
            raise ValueError(f"candidate input changed: {relative}")
    if tuple(manifest["frozen_episode_ids"]) != tuple(
        f"season{row['season']}:slot{row['agent_slot']}:seed{row['seed']}"
        for row in FROZEN_EPISODES
    ):
        raise ValueError("DraftGym panel differs from candidate manifest")
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _usage_cost(usage: Mapping[str, int]) -> float:
    return (
        int(usage.get("prefill_tokens", 0)) * PREFILL_USD_PER_MTOK
        + int(usage.get("cached_prefill_tokens", 0)) * CACHED_PREFILL_USD_PER_MTOK
        + int(usage.get("sample_tokens", 0)) * SAMPLE_USD_PER_MTOK
    ) / 1_000_000


def run_local_heldout() -> dict[str, Any]:
    verify_candidates()
    if LOCAL_RESULT.exists():
        return json.loads(LOCAL_RESULT.read_text())
    rows = load_split("heldout")
    learned = LearnedToolSupervisor(ARTIFACT_PATH)
    artifact_hash_before = sha256_file(ARTIFACT_PATH)
    systems = {
        "always_act": evaluate_supervisor(rows, AlwaysActSupervisor()),
        "always_use_tool": evaluate_supervisor(rows, AlwaysUseToolSupervisor()),
        "hard_coded_supervisor_v1": evaluate_supervisor(
            rows, HardCodedToolSupervisor()
        ),
        "learned_supervisor_v1": evaluate_supervisor(rows, learned),
    }
    artifact_hash_after = sha256_file(ARTIFACT_PATH)
    if artifact_hash_after != artifact_hash_before:
        raise AssertionError("held-out evaluator modified learned artifact")
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "split": "heldout",
        "rows": len(rows),
        "systems": systems,
        "fit_or_tuning_calls": 0,
        "artifact_sha256_before": artifact_hash_before,
        "artifact_sha256_after": artifact_hash_after,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_base_heldout() -> dict[str, Any]:
    verify_candidates()
    if MODEL_RESULT.exists():
        return json.loads(MODEL_RESULT.read_text())
    examples = load_split("heldout")
    existing = _read_jsonl(MODEL_CHECKPOINT)
    by_id = {str(row["example_id"]): row for row in existing}
    if len(by_id) != len(existing):
        raise ValueError("duplicate model held-out checkpoint rows")
    sampler = TinkerChatSampler(
        model=BASE_MODEL, experiment="tool-decision-frozen-heldout-v1"
    )
    for index, example in enumerate(examples):
        example_id = str(example["example_id"])
        if example_id in by_id:
            continue
        before = asdict(sampler.ledger)
        response = sampler.chat(
            sft._messages(example)[:-1], max_tokens=220,
            temperature=0.0, seed=sft.SEED + index,
        )
        after = asdict(sampler.ledger)
        prediction, structured, legal = sft._parse_model_decision(
            response["content"], example
        )
        row = {
            "example_id": example_id,
            "prediction": prediction.to_dict(),
            "structured": structured,
            "legal": legal,
            "response": response["content"],
            "usage": {key: after[key] - before[key] for key in before},
        }
        _append_jsonl(MODEL_CHECKPOINT, row)
        by_id[example_id] = row
    if len(by_id) != len(examples):
        raise RuntimeError("base held-out evaluation is incomplete")
    ordered = [by_id[str(example["example_id"])] for example in examples]
    predictions = [
        SupervisorDecision(
            decision=row["prediction"]["decision"],
            confidence=float(row["prediction"]["confidence"]),
            reason_code=row["prediction"]["reason_code"],
            allowed_tools=tuple(row["prediction"]["allowed_tools"]),
            required_tool=row["prediction"]["required_tool"],
        )
        for row in ordered
    ]
    metrics = score_predictions(examples, predictions)
    metrics["structured_action_rate"] = sum(row["structured"] for row in ordered) / len(ordered)
    metrics["legal_action_rate"] = sum(row["legal"] for row in ordered) / len(ordered)
    metrics["post_tool_completion_rate"] = sum(
        prediction.decision == "ACT_NOW" and row["legal"]
        for example, prediction, row in zip(examples, predictions, ordered, strict=True)
        if example["scenario"] == "post_tool_resolved"
    ) / sum(example["scenario"] == "post_tool_resolved" for example in examples)
    gate = metrics["primary_gate"]
    gate["structured_action_rate"] = metrics["structured_action_rate"] >= 0.95
    gate["post_tool_completion_rate"] = metrics["post_tool_completion_rate"] >= 0.95
    gate["pass"] = all(value for key, value in gate.items() if key != "pass")
    usage = {
        key: sum(int(row["usage"][key]) for row in ordered)
        for key in ordered[0]["usage"]
    }
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "model": BASE_MODEL,
        "split": "heldout",
        "rows": len(examples),
        "metrics": metrics,
        "usage": usage,
        "cost_usd": _usage_cost(usage),
        "fit_or_tuning_calls": 0,
    }
    MODEL_RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_draftgym_panel() -> dict[str, Any]:
    verify_candidates()
    if DRAFTGYM_RESULT.exists():
        return json.loads(DRAFTGYM_RESULT.read_text())
    learned = LearnedToolSupervisor(ARTIFACT_PATH)
    arms = {
        "base_qwen": None,
        "hard_coded_plus_qwen": HardCodedToolSupervisor(),
        "learned_supervisor_plus_qwen": learned,
    }
    existing = _read_jsonl(DRAFTGYM_CHECKPOINT)
    by_key = {
        (str(row["arm"]), str(row["episode_id"])): row for row in existing
    }
    if len(by_key) != len(existing):
        raise ValueError("duplicate DraftGym checkpoint rows")
    for arm, supervisor in arms.items():
        for spec in FROZEN_EPISODES:
            episode_id = (
                f"season{spec['season']}:slot{spec['agent_slot']}:seed{spec['seed']}"
            )
            if (arm, episode_id) in by_key:
                continue
            sampler = TinkerChatSampler(
                model=BASE_MODEL,
                experiment=f"tool-decision-draftgym-{arm}",
            )
            outcome = run_episode(spec, sampler=sampler, supervisor=supervisor)
            row = {"arm": arm, "episode_id": episode_id, "outcome": outcome}
            _append_jsonl(DRAFTGYM_CHECKPOINT, row)
            by_key[(arm, episode_id)] = row
    ordered = [
        by_key[(arm, f"season{spec['season']}:slot{spec['agent_slot']}:seed{spec['seed']}")]
        for arm in arms for spec in FROZEN_EPISODES
    ]
    summaries = {}
    for arm in arms:
        rows = [row["outcome"] for row in ordered if row["arm"] == arm]
        usage = {
            key: sum(int(row["usage_delta"][key]) for row in rows)
            for key in rows[0]["usage_delta"]
        }
        summaries[arm] = {
            "episodes": len(rows),
            "completed": sum(row["task_completed"] for row in rows),
            "mean_reward": sum(float(row["reward"]) for row in rows) / len(rows),
            "median_reward": sorted(float(row["reward"]) for row in rows)[len(rows) // 2],
            "tool_calls": sum(int(row["tool_calls"]) for row in rows),
            "successful_tool_calls": sum(int(row["tool_calls_ok"]) for row in rows),
            "forced_tool_calls": sum(int(row["forced_tool_calls"]) for row in rows),
            "blocked_actions": sum(int(row["blocked_actions"]) for row in rows),
            "fallback_picks": sum(int(row["fallback_picks"]) for row in rows),
            "model_requests": sum(int(row["model_requests"]) for row in rows),
            "usage": usage,
            "cost_usd": _usage_cost(usage),
            "episode_rewards": {
                row_id: next(
                    float(row["reward"]) for row in rows
                    if f"season{row['season']}:slot{row['agent_slot']}:seed{row['seed']}" == row_id
                )
                for row_id in [
                    f"season{spec['season']}:slot{spec['agent_slot']}:seed{spec['seed']}"
                    for spec in FROZEN_EPISODES
                ]
            },
        }
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "episodes": list(FROZEN_EPISODES),
        "arms": summaries,
        "incremental_cost_usd": sum(item["cost_usd"] for item in summaries.values()),
        "scientific_limit": (
            "Seven historical internal-heldout episodes measure harness behavior; "
            "they are too few for a stable fantasy-point quality claim."
        ),
    }
    DRAFTGYM_RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_scorecard() -> dict[str, Any]:
    verify_candidates()
    local = json.loads(LOCAL_RESULT.read_text())
    base = json.loads(MODEL_RESULT.read_text())
    draftgym = json.loads(DRAFTGYM_RESULT.read_text())
    development = json.loads(
        (ROOT / "artifacts/tool-decision-supervisor-v1/development-report.json").read_text()
    )
    canary_v1 = json.loads(
        (ROOT / "artifacts/tool-decision-supervisor-v1/tinker-canary-v1/development-behavior.json").read_text()
    )
    canary_v2 = json.loads(
        (ROOT / "artifacts/tool-decision-supervisor-v1/tinker-canary-v2/development-behavior.json").read_text()
    )
    experiment_spend = (
        0.229296615
        + 0.684672391
        + 0.225781425
        + 1.33366358
        + 0.225304047
        + float(base["cost_usd"])
        + float(draftgym["incremental_cost_usd"])
    )
    learned_heldout = local["systems"]["learned_supervisor_v1"]
    decision = (
        "use_external_supervisor"
        if learned_heldout["primary_gate"]["pass"]
        else "redesign_data_and_policy"
    )
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "decision": decision,
        "development": development["development_metrics"],
        "heldout": {
            **local["systems"],
            "untouched_base_qwen35_9b": base["metrics"],
            "contrastive_canary_v1": {
                "evaluation_role": "development_rejected",
                "metrics": canary_v1["metrics"],
            },
            "contrastive_canary_v2": {
                "evaluation_role": "development_rejected",
                "metrics": canary_v2["metrics"],
            },
        },
        "draftgym": draftgym,
        "spend": {
            "incremental_experiment_usd": experiment_spend,
            "cumulative_tinker_usd": 10.855193358 + experiment_spend,
            "target_under_five_usd": experiment_spend < 5.0,
            "hard_cap_under_ten_usd": experiment_spend < 10.0,
        },
        "training_decision": (
            "stop_contrastive_sft_after_two_failed_development_canaries"
        ),
        "rl_started": False,
        "sealed_seasons_opened": [],
    }
    FINAL_SCORECARD.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("local-heldout", "base-heldout", "draftgym", "scorecard"),
    )
    args = parser.parse_args()
    if args.command == "local-heldout":
        result = run_local_heldout()
    elif args.command == "base-heldout":
        result = run_base_heldout()
    elif args.command == "draftgym":
        result = run_draftgym_panel()
    else:
        result = write_scorecard()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
