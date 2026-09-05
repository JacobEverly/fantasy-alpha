"""Development-only format gates and checkpoint selection for T1.2."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from envs.play_llm import parse_action
from training.filter import parse_final_p
from training.t11_dataset import _extract_observation
from training.t12_dataset import load_t12_corpus
from training.tinker_backend import (
    ROOT,
    TinkerChatSampler,
    grouped_development_split,
    safe_text,
    sha256_file,
    utc_now,
)

ROOT_DIR = ROOT / "artifacts/tinker-sft-t12"
TRAIN_SUMMARY = ROOT_DIR / "t12-full/run-summary.json"
CANARY_SUMMARY = ROOT_DIR / "canary-v2/run-summary.json"
SELECTION_PATH = ROOT_DIR / "t12-full/checkpoint-selection.json"
BEHAVIOR_DIR = ROOT_DIR / "development-behavior"
EVAL_SEED = 20260808


def build_suite(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Freeze a compact, schema-complete suite from the grouped development split."""
    _, development = grouped_development_split(rows, fraction=0.10, seed=EVAL_SEED)
    by_schema: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in development:
        by_schema[row["meta"]["t12_schema"]].append(row)
    selected: list[dict[str, Any]] = []
    counts = {
        "broad_bust": 3,
        "broad_full_slate": 3,
        "broad_season_threshold": 3,
        "broad_weekly_h2h": 3,
        "target_calibration": 6,
        "target_draft_pick": 6,
        "target_draft_recovery": 2,
    }
    for schema, count in counts.items():
        selected.extend(sorted(
            by_schema[schema], key=lambda row: row["meta"]["trace_id"]
        )[:count])

    tool_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for schema in ("target_tool_call", "target_post_tool_pick"):
        for row in by_schema[schema]:
            tool_groups[row["meta"]["question_id"]].append(row)
    for group in sorted(tool_groups)[:3]:
        pair = tool_groups[group]
        if {row["meta"]["t12_schema"] for row in pair} != {
            "target_tool_call", "target_post_tool_pick",
        }:
            raise AssertionError("development tool transition is incomplete")
        selected.extend(pair)
    selected.sort(key=lambda row: row["meta"]["trace_id"])
    if len(selected) != 32 or {row["meta"]["t12_schema"] for row in selected} != {
        "broad_bust", "broad_full_slate", "broad_season_threshold",
        "broad_weekly_h2h", "target_calibration", "target_draft_pick",
        "target_draft_recovery", "target_tool_call", "target_post_tool_pick",
    }:
        raise AssertionError("T1.2 development behavior suite is not schema complete")
    return selected


def suite_trace_ids() -> list[str]:
    return [row["meta"]["trace_id"] for row in build_suite(load_t12_corpus())]


def _validate_response(row: Mapping[str, Any], content: str) -> dict[str, Any]:
    schema = str(row["meta"]["t12_schema"])
    target = str(row["messages"][-1]["content"])
    result = {
        "schema": schema,
        "parse_success": False,
        "legal": True,
        "tool_valid": None,
        "target_match": False,
    }
    if schema.startswith("broad_") or schema == "target_calibration":
        p = parse_final_p(content)
        target_p = parse_final_p(target)
        result.update({
            "parse_success": p is not None and 0 <= p <= 1,
            "target_match": p is not None and target_p is not None and abs(p - target_p) <= 0.10,
            "parsed_probability": p,
            "target_probability": target_p,
        })
        return result

    obs = _extract_observation(str(row["messages"][1]["content"]))
    visible = {str(candidate["player_id"]) for candidate in obs["top_available"]}
    try:
        action = parse_action(content)
    except ValueError:
        return result
    result["parse_success"] = True
    expected = json.loads(target)
    result["target_match"] = action == expected
    if schema == "target_tool_call":
        tool = action.get("tool")
        valid = bool(
            isinstance(tool, Mapping)
            and tool.get("name") == "depth_chart"
            and str((tool.get("arguments") or {}).get("team_or_player", "")) in visible
        )
        result["tool_valid"] = valid
        result["legal"] = valid
    else:
        valid = str(action.get("pick", "")) in visible
        result["legal"] = valid
        if "tool" in action:
            result["tool_valid"] = False
    return result


def evaluate_checkpoint(
    model_path: str, *, label: str, out_path: Path,
) -> dict[str, Any]:
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if existing.get("status") == "complete" and existing.get("model_path") == model_path:
            return existing
        raise RuntimeError(f"refusing to overwrite behavior evaluation: {out_path}")
    rows = build_suite(load_t12_corpus())
    sampler = TinkerChatSampler(model_path=model_path, experiment="tinker-sft-t12")
    records = []
    for index, row in enumerate(rows):
        before = asdict(sampler.ledger)
        response = sampler.chat(
            list(row["messages"][:-1]), max_tokens=700,
            temperature=0.0, seed=EVAL_SEED + index,
        )
        after = asdict(sampler.ledger)
        checked = _validate_response(row, response["content"])
        records.append({
            "trace_id": row["meta"]["trace_id"],
            **checked,
            "usage_delta": {key: after[key] - before[key] for key in before},
        })
    structure = sum(record["parse_success"] for record in records) / len(records)
    tool_records = [record for record in records if record["schema"] == "target_tool_call"]
    payload = {
        "status": "complete",
        "completed_at": utc_now(),
        "label": label,
        "model_path": model_path,
        "suite_trace_ids": [record["trace_id"] for record in records],
        "suite_rows": len(records),
        "structured_coverage": structure,
        "irreparable_responses": sum(not record["parse_success"] for record in records),
        "illegal_actions": sum(not record["legal"] for record in records),
        "tool_valid_rate": (
            sum(record["tool_valid"] is True for record in tool_records) / len(tool_records)
        ),
        "target_match_rate": sum(record["target_match"] for record in records) / len(records),
        "schema_coverage": {
            schema: all(record["parse_success"] for record in records if record["schema"] == schema)
            for schema in sorted({record["schema"] for record in records})
        },
        "records": records,
        "cost": {**asdict(sampler.ledger), "computed_usd": sampler.ledger.usd},
    }
    payload["gate"] = {
        "structured_coverage_at_least_98pct": structure >= 0.98,
        "no_irreparable_responses": payload["irreparable_responses"] == 0,
        "no_illegal_actions": payload["illegal_actions"] == 0,
        "valid_tool_calls": payload["tool_valid_rate"] == 1.0,
        "every_schema_valid": all(payload["schema_coverage"].values()),
    }
    payload["gate"]["pass"] = all(payload["gate"].values())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def evaluate_canary() -> dict[str, Any]:
    summary = json.loads(CANARY_SUMMARY.read_text())
    if summary.get("status") != "complete":
        raise ValueError("T1.2 canary training is incomplete")
    return evaluate_checkpoint(
        str(summary["final_sampler_path"]), label="canary-final",
        out_path=ROOT_DIR / "canary-v2/behavior-gate.json",
    )


def select_checkpoint() -> dict[str, Any]:
    if SELECTION_PATH.exists():
        return json.loads(SELECTION_PATH.read_text())
    summary = json.loads(TRAIN_SUMMARY.read_text())
    candidates = [
        {
            "label": f"step-{entry['step']}",
            "step": int(entry["step"]),
            "development_nll": float(entry["development_nll"]),
            "sampler_path": str(entry["sampler_path"]),
        }
        for entry in summary["checkpoints"]
        if "fraction" in entry
    ]
    candidates.append({
        "label": "final",
        "step": int(summary["steps"]),
        "development_nll": float(summary["final_development_nll"]),
        "sampler_path": str(summary["final_sampler_path"]),
    })
    if len(candidates) != 2:
        raise ValueError("T1.2 selection requires exactly halfway and final checkpoints")
    evaluated = []
    for candidate in candidates:
        behavior = evaluate_checkpoint(
            candidate["sampler_path"], label=candidate["label"],
            out_path=BEHAVIOR_DIR / f"{candidate['label']}.json",
        )
        evaluated.append({**candidate, "behavior": {
            key: behavior[key] for key in (
                "structured_coverage", "irreparable_responses", "illegal_actions",
                "tool_valid_rate", "target_match_rate", "cost", "gate",
            )
        }})
    selected = max(evaluated, key=lambda candidate: (
        candidate["behavior"]["structured_coverage"],
        -candidate["behavior"]["illegal_actions"],
        candidate["behavior"]["tool_valid_rate"],
        candidate["behavior"]["target_match_rate"],
        -candidate["development_nll"],
        -candidate["step"],
    ))
    payload = {
        "selected_at": utc_now(),
        "selection_uses_heldout_labels": False,
        "selection_suite": "frozen grouped development prompts only",
        "priority": [
            "schema validity", "legal draft behavior", "tool-policy validity",
            "development target quality", "development NLL", "earliest tie-break",
        ],
        "training_summary_sha256": sha256_file(TRAIN_SUMMARY),
        "candidates": evaluated,
        "selected": selected,
    }
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("canary", "select"))
    args = parser.parse_args(argv)
    result = evaluate_canary() if args.command == "canary" else select_checkpoint()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI boundary redacts provider errors
        raise SystemExit(f"T1.2 behavior evaluation failed: {safe_text(exc)}") from None
