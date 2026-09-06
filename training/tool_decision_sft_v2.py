"""One preregistered revision of the tool-decision contrastive canary."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from training import tool_decision_sft as v1
from training.tinker_backend import (
    ROOT,
    RunConfig,
    corpus_manifest,
    estimate_run_cost,
    grouped_development_split,
    run_sft,
    sha256_file,
)

PROTOCOL = "fantasy-alpha-tool-decision-contrastive-sft-v2"
CORPUS_PATH = ROOT / "training/datasets/tool_decision_v1/contrastive_sft_train_v2.jsonl"
MANIFEST_PATH = ROOT / "training/datasets/tool_decision_v1/contrastive_sft_manifest_v2.json"
CANARY_DIR = ROOT / "artifacts/tool-decision-supervisor-v1/tinker-canary-v2"
CANARY_EVAL_PATH = CANARY_DIR / "development-behavior.json"
CANARY_GROUPS = 80
DECISION_MASS = {
    "ACT_NOW": 0.25,
    "USE_TOOL": 0.50,
    "WAIT_OR_ABSTAIN": 0.25,
}


def build_corpus() -> list[dict[str, Any]]:
    rows = copy.deepcopy(v1.load_corpus())
    counts = Counter(str(row["meta"]["decision"]) for row in rows)
    for row in rows:
        decision = str(row["meta"]["decision"])
        row["meta"]["loss_weight"] = (
            len(rows) * DECISION_MASS[decision] / counts[decision]
        )
        row["meta"]["trace_id"] = row["meta"]["trace_id"].replace(
            "tool-decision-sft:", "tool-decision-sft-v2:"
        )
        row["meta"]["revision"] = "tool_mass_50pct_and_80_groups"
    return rows


def validate_corpus(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != v1.EXPECTED_ROWS:
        raise ValueError("v2 row count changed")
    weighted = Counter()
    for row in rows:
        decision = str(row["meta"]["decision"])
        weighted[decision] += float(row["meta"]["loss_weight"])
        if row["meta"].get("revision") != "tool_mass_50pct_and_80_groups":
            raise ValueError("v2 revision provenance is missing")
    total = sum(weighted.values())
    achieved = {key: value / total for key, value in sorted(weighted.items())}
    if any(abs(achieved[key] - value) > 1e-9 for key, value in DECISION_MASS.items()):
        raise ValueError(f"v2 decision mass mismatch: {achieved}")
    return {
        "valid": True,
        "rows": len(rows),
        "weighted_decision_fraction": achieved,
        "changed_fields": ["meta.loss_weight", "meta.trace_id", "meta.revision"],
        "messages_unchanged_from_v1": True,
        "heldout_rows_seen": 0,
    }


def config() -> RunConfig:
    base = v1.canary_config()
    return RunConfig(
        **{
            **asdict(base),
            "run_name": "fantasy-alpha-tool-decision-canary-v2",
        }
    )


def materialize() -> dict[str, Any]:
    v1.verify_frozen_spec()
    rows = build_corpus()
    validation = validate_corpus(rows)
    CORPUS_PATH.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    rendered = corpus_manifest(rows, config(), source_path=CORPUS_PATH)
    manifest = {
        "protocol": PROTOCOL,
        "parent_corpus_sha256": sha256_file(v1.CORPUS_PATH),
        "corpus_path": str(CORPUS_PATH.relative_to(ROOT)),
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "revision_basis": {
            "v1_canary_required_tool_recall": 0.0,
            "v1_canary_immediate_action_retention": 1.0,
            "change": "increase representative groups 40 to 80 and USE_TOOL loss mass from one-third to one-half",
        },
        "validation": validation,
        "rendered": rendered,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_corpus() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if sha256_file(CORPUS_PATH) != manifest["corpus_sha256"]:
        raise ValueError("v2 contrastive corpus SHA-256 mismatch")
    rows = [json.loads(line) for line in CORPUS_PATH.read_text().splitlines() if line.strip()]
    validate_corpus(rows)
    return rows


def canary_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["meta"]["question_id"]), []).append(row)
    chosen = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{v1.SEED}:v2:{value}".encode()).hexdigest(),
    )[:CANARY_GROUPS]
    selected = [row for group in chosen for row in groups[group]]
    if len(selected) != CANARY_GROUPS * 4:
        raise AssertionError("v2 canary must preserve eighty complete groups")
    return sorted(selected, key=lambda row: row["meta"]["trace_id"])


def preflight() -> dict[str, Any]:
    rows = canary_subset(load_corpus())
    train_rows, dev_rows = grouped_development_split(
        rows, fraction=config().development_fraction, seed=v1.SEED
    )
    estimate = estimate_run_cost(train_rows, dev_rows, config())
    inference_estimate = json.loads(
        (ROOT / "artifacts/tool-decision-supervisor-v1/tinker-preflight.json").read_text()
    )["conservative_two_arm_development_inference_usd"] / 2
    projected = float(estimate["estimated_training_usd"]) + inference_estimate
    if 1.139750431 + projected >= 10.0:
        raise RuntimeError("cumulative experiment spend would exceed the hard cap")
    return {
        "protocol": PROTOCOL,
        "rows": len(rows),
        "matched_groups": CANARY_GROUPS,
        "training_estimate": estimate,
        "adapter_development_inference_estimate_usd": inference_estimate,
        "spent_before_revision_usd": 1.139750431,
        "projected_revision_usd": projected,
        "projected_experiment_total_usd": 1.139750431 + projected,
        "target_under_five_usd": 1.139750431 + projected < 5.0,
        "hard_cap_under_ten_usd": 1.139750431 + projected < 10.0,
        "heldout_rows_seen": 0,
    }


def run_canary() -> dict[str, Any]:
    info = preflight()
    CANARY_DIR.mkdir(parents=True, exist_ok=True)
    (CANARY_DIR / "preflight.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n"
    )
    return run_sft(
        canary_subset(load_corpus()), config(), CANARY_DIR,
        hard_cap_usd=10.0, source_path=CORPUS_PATH,
    )


def evaluate_canary() -> dict[str, Any]:
    summary = json.loads((CANARY_DIR / "run-summary.json").read_text())
    if summary.get("status") != "complete":
        raise RuntimeError("revised Tinker canary is incomplete")
    return v1.evaluate_model(
        str(summary["final_sampler_path"]),
        out_path=CANARY_EVAL_PATH,
        label="contrastive-canary-v2",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materialize")
    sub.add_parser("preflight")
    sub.add_parser("run-canary")
    sub.add_parser("eval-canary")
    args = parser.parse_args()
    if args.command == "materialize":
        result = materialize()
    elif args.command == "preflight":
        result = preflight()
    elif args.command == "run-canary":
        result = run_canary()
    else:
        result = evaluate_canary()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
