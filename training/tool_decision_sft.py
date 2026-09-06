"""Contrastive Tinker SFT for Fantasy Alpha's tool-decision boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from envs.play_llm import SYSTEM_PROMPT, _first_json_object, parse_action
from harness.tool_supervisor import SupervisorDecision
from training.tinker_backend import (
    BASE_MODEL,
    ROOT,
    RunConfig,
    TinkerChatSampler,
    corpus_manifest,
    estimate_run_cost,
    grouped_development_split,
    run_sft,
    safe_text,
    sha256_file,
)
from training.tool_decision_dataset import load_split
from training.tool_decision_supervisor import score_predictions, verify_frozen_spec

PROTOCOL = "fantasy-alpha-tool-decision-contrastive-sft-v1"
CORPUS_PATH = ROOT / "training/datasets/tool_decision_v1/contrastive_sft_train.jsonl"
MANIFEST_PATH = ROOT / "training/datasets/tool_decision_v1/contrastive_sft_manifest.json"
CANARY_DIR = ROOT / "artifacts/tool-decision-supervisor-v1/tinker-canary-v1"
BASE_EVAL_PATH = ROOT / "artifacts/tool-decision-supervisor-v1/base-development.json"
CANARY_EVAL_PATH = CANARY_DIR / "development-behavior.json"
SEED = 20260906
EXPECTED_ROWS = 564
CANARY_GROUPS = 40

REVIEW_ADDENDUM = (
    "\nYou are now reviewing one proposed pick before it is executed. "
    "If the visible state supports it, repeat the pick JSON exactly. "
    "If material evidence is missing, return one permitted tool call. "
    "If evidence already failed and no lookup budget remains, return "
    '{"abstain":{"reason_code":"evidence_unavailable_no_budget"}}. '
    "Return exactly one JSON object."
)


def _target(example: Mapping[str, Any]) -> dict[str, Any]:
    decision = example["label"]["decision"]
    if decision == "ACT_NOW":
        return dict(example["input"]["proposed_action"])
    if decision == "USE_TOOL":
        return {"tool": dict(example["label"]["correct_tool"])}
    if decision == "WAIT_OR_ABSTAIN":
        return {"abstain": {"reason_code": str(example["label"]["reason_code"])}}
    raise ValueError(f"unknown decision: {decision}")


def _messages(example: Mapping[str, Any]) -> list[dict[str, str]]:
    obs = example["input"]["observation"]
    proposed = example["input"]["proposed_action"]
    system = SYSTEM_PROMPT + (
        "\nThis draft is ANONYMIZED: use only depth_chart and injury_status "
        "with board ids from top_available."
    ) + REVIEW_ADDENDUM
    user = (
        "Draft state (JSON):\n"
        + json.dumps(obs, sort_keys=True)
        + "\n\nProposed action (JSON):\n"
        + json.dumps(proposed, sort_keys=True)
        + "\n\nReview and return ONE JSON action object now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": json.dumps(_target(example), sort_keys=True, separators=(",", ":")),
        },
    ]


def build_corpus() -> list[dict[str, Any]]:
    examples = load_split("train")
    counts = Counter(row["label"]["decision"] for row in examples)
    rows = []
    for example in examples:
        decision = str(example["label"]["decision"])
        # Equal optimizer-level mass for all three decisions. Tinker's datum
        # renderer normalizes each assistant response before this row scale.
        loss_weight = len(examples) / (len(counts) * counts[decision])
        obs = example["input"]["observation"]
        rows.append({
            "messages": _messages(example),
            "meta": {
                "track": "anonymized",
                "trace_id": f"tool-decision-sft:{example['example_id']}",
                "bench": "draftgym_tool_decision",
                "family": str(example["scenario"]),
                "season": int(obs["season"]),
                "question_id": str(example["matched_group_id"]),
                "episode_group_id": str(example["episode_group_id"]),
                "source_example_id": str(example["example_id"]),
                "decision": decision,
                "loss_weight": loss_weight,
                "label_uses_realized_outcome": False,
            },
        })
    return sorted(rows, key=lambda row: row["meta"]["trace_id"])


def validate_corpus(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = []
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    ids = [str(row.get("meta", {}).get("trace_id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("trace ids must be unique and nonempty")
    decisions = Counter()
    groups: dict[str, set[str]] = {}
    weighted = Counter()
    for row in rows:
        meta = row.get("meta") or {}
        decision = str(meta.get("decision"))
        decisions[decision] += 1
        weighted[decision] += float(meta.get("loss_weight") or 0.0)
        groups.setdefault(str(meta.get("question_id")), set()).add(
            str(meta.get("episode_group_id"))
        )
        if int(meta.get("season", -1)) in {2013, 2018, 2023, 2024, 2025}:
            errors.append(f"{meta.get('trace_id')}: sealed season")
        if meta.get("label_uses_realized_outcome") is not False:
            errors.append(f"{meta.get('trace_id')}: invalid provenance")
        if [message.get("role") for message in row.get("messages", [])] != [
            "system", "user", "assistant",
        ]:
            errors.append(f"{meta.get('trace_id')}: invalid messages")
    if decisions != {"ACT_NOW": 282, "USE_TOOL": 141, "WAIT_OR_ABSTAIN": 141}:
        errors.append(f"decision counts mismatch: {dict(decisions)}")
    if len(groups) != 141 or any(len(episodes) != 1 for episodes in groups.values()):
        errors.append("SFT matched groups are incomplete or cross episodes")
    if any(abs(value / sum(weighted.values()) - 1 / 3) > 1e-9 for value in weighted.values()):
        errors.append(f"decision loss mass is not balanced: {dict(weighted)}")
    if errors:
        raise ValueError("contrastive SFT validation failed:\n- " + "\n- ".join(errors))
    return {
        "valid": True,
        "rows": len(rows),
        "matched_groups": len(groups),
        "decisions": dict(sorted(decisions.items())),
        "weighted_decision_fraction": {
            key: value / sum(weighted.values()) for key, value in sorted(weighted.items())
        },
        "sealed_seasons_present": [],
        "label_uses_realized_outcome": False,
    }


def materialize() -> dict[str, Any]:
    verify_frozen_spec()
    rows = build_corpus()
    validation = validate_corpus(rows)
    CORPUS_PATH.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    config = canary_config()
    rendered = corpus_manifest(rows, config, source_path=CORPUS_PATH)
    manifest = {
        "protocol": PROTOCOL,
        "corpus_path": str(CORPUS_PATH.relative_to(ROOT)),
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "source_train_sha256": json.loads(
            (ROOT / "training/datasets/tool_decision_v1/manifest.json").read_text()
        )["files"]["train"]["sha256"],
        "validation": validation,
        "rendered": rendered,
        "format": {
            "base_model": BASE_MODEL,
            "renderer": config.renderer,
            "output_actions": ["pick", "tool", "abstain"],
            "same_draftgym_tool_envelope": True,
            "candidate_review_is_second_pass": True,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_corpus() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if sha256_file(CORPUS_PATH) != manifest["corpus_sha256"]:
        raise ValueError("contrastive SFT corpus SHA-256 mismatch")
    rows = [json.loads(line) for line in CORPUS_PATH.read_text().splitlines() if line.strip()]
    validate_corpus(rows)
    return rows


def canary_subset(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["meta"]["question_id"]), []).append(row)
    chosen = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest(),
    )[:CANARY_GROUPS]
    selected = [row for group in chosen for row in groups[group]]
    if len(selected) != CANARY_GROUPS * 4:
        raise AssertionError("canary must preserve forty complete matched groups")
    return sorted(selected, key=lambda row: row["meta"]["trace_id"])


def canary_config() -> RunConfig:
    return RunConfig(
        run_name="fantasy-alpha-tool-decision-canary-v1",
        experiment="tool-decision-supervisor-v1",
        lora_rank=32,
        learning_rate=1e-4,
        batch_size=32,
        epochs=1,
        max_length=8192,
        development_fraction=0.10,
        seed=SEED,
        checkpoint_every_epochs=1,
    )


def preflight() -> dict[str, Any]:
    spec = verify_frozen_spec()
    rows = load_corpus()
    subset = canary_subset(rows)
    train_rows, dev_rows = grouped_development_split(
        subset, fraction=canary_config().development_fraction, seed=SEED
    )
    estimate = estimate_run_cost(train_rows, dev_rows, canary_config())
    # Development inference: untouched base plus one adapter, 164 prompts each.
    # Use actual rendered corpus mean length as a conservative prompt proxy.
    rendered = json.loads(MANIFEST_PATH.read_text())["rendered"]
    mean_prompt = rendered["total_rendered_tokens"] / rendered["rows"]
    dev_rows_count = spec["dataset"]["development"]["rows"]
    inference_prefill = int(mean_prompt * dev_rows_count * 2)
    inference_output = dev_rows_count * 2 * spec["model_settings"]["max_tokens"]
    from training.tinker_backend import PREFILL_USD_PER_MTOK, SAMPLE_USD_PER_MTOK

    inference_usd = (
        inference_prefill * PREFILL_USD_PER_MTOK
        + inference_output * SAMPLE_USD_PER_MTOK
    ) / 1_000_000
    total = float(estimate["estimated_training_usd"]) + inference_usd
    if total >= float(spec["budget"]["hard_incremental_cap_usd"]):
        raise RuntimeError("projected canary workload exceeds hard incremental cap")
    return {
        "protocol": PROTOCOL,
        "rows": len(subset),
        "matched_groups": CANARY_GROUPS,
        "config": asdict(canary_config()),
        "training_estimate": estimate,
        "conservative_two_arm_development_inference_usd": inference_usd,
        "projected_canary_total_usd": total,
        "target_under_five_usd": total < 5.0,
        "hard_cap_under_ten_usd": total < 10.0,
        "heldout_rows_seen": 0,
    }


def run_canary() -> dict[str, Any]:
    info = preflight()
    CANARY_DIR.mkdir(parents=True, exist_ok=True)
    preflight_path = CANARY_DIR / "preflight.json"
    if not preflight_path.exists():
        preflight_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return run_sft(
        canary_subset(load_corpus()),
        canary_config(),
        CANARY_DIR,
        hard_cap_usd=10.0,
        source_path=CORPUS_PATH,
    )


def _parse_model_decision(
    text: str, example: Mapping[str, Any]
) -> tuple[SupervisorDecision, bool, bool]:
    blob = _first_json_object(text)
    if blob is None:
        return SupervisorDecision(
            "ACT_NOW", 0.0, "invalid_response_fallback"
        ), False, False
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return SupervisorDecision(
            "ACT_NOW", 0.0, "invalid_response_fallback"
        ), False, False
    if not isinstance(payload, Mapping):
        return SupervisorDecision(
            "ACT_NOW", 0.0, "invalid_response_fallback"
        ), False, False
    if "abstain" in payload:
        legal = set(payload) == {"abstain"}
        return SupervisorDecision(
            "WAIT_OR_ABSTAIN", 1.0, "model_abstained"
        ), True, legal
    try:
        action = parse_action(blob)
    except ValueError:
        return SupervisorDecision(
            "ACT_NOW", 0.0, "invalid_response_fallback"
        ), False, False
    if "tool" in action:
        tool = action["tool"]
        return SupervisorDecision(
            "USE_TOOL", 1.0, "model_requested_tool",
            (str(tool["name"]),), tool,
        ), True, tool == example["label"].get("correct_tool")
    expected_pick = example["input"]["proposed_action"].get("pick")
    legal = str(action.get("pick")) == str(expected_pick)
    return SupervisorDecision(
        "ACT_NOW", 1.0, "model_approved_pick"
    ), True, legal


def evaluate_model(model_path: str | None, *, out_path: Path, label: str) -> dict[str, Any]:
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        expected_path = model_path or BASE_MODEL
        if existing.get("status") == "complete" and existing.get("model_path") == expected_path:
            return existing
        raise RuntimeError(f"refusing to overwrite model evaluation: {out_path}")
    examples = load_split("development")
    sampler = TinkerChatSampler(
        model_path=model_path,
        model=BASE_MODEL,
        experiment="tool-decision-supervisor-v1",
    )
    predictions = []
    structured = []
    legal = []
    records = []
    for index, example in enumerate(examples):
        response = sampler.chat(
            _messages(example)[:-1],
            max_tokens=220,
            temperature=0.0,
            seed=SEED + index,
        )
        prediction, is_structured, is_legal = _parse_model_decision(
            response["content"], example
        )
        predictions.append(prediction)
        structured.append(is_structured)
        legal.append(is_legal)
        records.append({
            "example_id": example["example_id"],
            "truth": example["label"]["decision"],
            "prediction": prediction.to_dict(),
            "structured": is_structured,
            "legal": is_legal,
            "response": safe_text(response["content"]),
            "usage": {
                "prompt_tokens": response["prompt_tokens"],
                "output_tokens": response["output_tokens"],
            },
        })
    metrics = score_predictions(examples, predictions)
    metrics["structured_action_rate"] = sum(structured) / len(structured)
    metrics["legal_action_rate"] = sum(legal) / len(legal)
    metrics["post_tool_completion_rate"] = sum(
        prediction.decision == "ACT_NOW" and ok
        for example, prediction, ok in zip(examples, predictions, legal, strict=True)
        if example["scenario"] == "post_tool_resolved"
    ) / sum(example["scenario"] == "post_tool_resolved" for example in examples)
    gate = metrics["primary_gate"]
    gate["structured_action_rate"] = metrics["structured_action_rate"] >= 0.95
    gate["post_tool_completion_rate"] = metrics["post_tool_completion_rate"] >= 0.95
    gate["pass"] = all(value for key, value in gate.items() if key != "pass")
    payload = {
        "protocol": PROTOCOL,
        "status": "complete",
        "label": label,
        "model_path": model_path or BASE_MODEL,
        "split": "development",
        "heldout_rows_seen": 0,
        "metrics": metrics,
        "cost": {**asdict(sampler.ledger), "computed_usd": sampler.ledger.usd},
        "records": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def evaluate_base() -> dict[str, Any]:
    return evaluate_model(None, out_path=BASE_EVAL_PATH, label="untouched-base")


def evaluate_canary() -> dict[str, Any]:
    summary = json.loads((CANARY_DIR / "run-summary.json").read_text())
    if summary.get("status") != "complete":
        raise RuntimeError("Tinker canary is incomplete")
    return evaluate_model(
        str(summary["final_sampler_path"]),
        out_path=CANARY_EVAL_PATH,
        label="contrastive-canary-v1",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materialize")
    sub.add_parser("preflight")
    sub.add_parser("run-canary")
    sub.add_parser("eval-base")
    sub.add_parser("eval-canary")
    args = parser.parse_args()
    if args.command == "materialize":
        result = materialize()
    elif args.command == "preflight":
        result = preflight()
    elif args.command == "run-canary":
        result = run_canary()
    elif args.command == "eval-base":
        result = evaluate_base()
    else:
        result = evaluate_canary()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
