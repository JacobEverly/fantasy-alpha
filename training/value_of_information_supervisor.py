"""Train a development-only supervisor on measured intervention outcomes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from harness.tool_supervisor import (
    HardCodedToolSupervisor,
    LearnedToolSupervisor,
    decision_features,
)
from harness.value_supervisor import OutcomeValueRuleSupervisor
from evals.value_of_information import arm_evaluation_panel, verify_frozen_protocol
from training.tinker_backend import ROOT, sha256_file, utc_now

PROTOCOL = "fantasy-alpha-outcome-value-supervisor-v1"
SOURCE_PROTOCOL = "fantasy-alpha-value-of-information-v1"
DATASET_PATH = ROOT / "training/datasets/value_of_information_v1/development.jsonl"
ARTIFACT_DIR = ROOT / "artifacts/value-of-information-v1"
ARTIFACT_PATH = ARTIFACT_DIR / "outcome-value-supervisor.joblib"
REPORT_PATH = ARTIFACT_DIR / "development-report.json"
CANDIDATE_FREEZE_PATH = ARTIFACT_DIR / "frozen-candidates.json"
RANDOM_SEED = 20260906
THRESHOLDS = tuple(value / 20 for value in range(1, 20))
RULE_STDEV_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 999.0)
EXISTING_ARTIFACT_PATH = (
    ROOT / "artifacts/tool-decision-supervisor-v1/learned-supervisor.joblib"
)


def _deps() -> dict[str, Any]:
    try:
        import joblib
        import sklearn.feature_extraction
        import sklearn.linear_model
        import sklearn.model_selection
        import sklearn.pipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install fantasy-alpha[baselines]") from exc
    return {
        "joblib": joblib,
        "DictVectorizer": sklearn.feature_extraction.DictVectorizer,
        "LogisticRegression": sklearn.linear_model.LogisticRegression,
        "GroupKFold": sklearn.model_selection.GroupKFold,
        "Pipeline": sklearn.pipeline.Pipeline,
    }


def load_development(path: Path = DATASET_PATH, *, exact: bool = True) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if exact and len(rows) != 70:
        raise ValueError(f"expected 70 development examples, found {len(rows)}")
    if any(row.get("split") != "development" for row in rows):
        raise ValueError("trainer refuses non-development rows")
    if any(row.get("protocol") != SOURCE_PROTOCOL for row in rows):
        raise ValueError("unexpected source protocol")
    groups = [str(row.get("episode_id", "")) for row in rows]
    if any(not value for value in groups) or len(set(groups)) * 2 != len(rows):
        raise ValueError("expected exactly two examples per development episode")
    if len({str(row["example_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate example_id")
    return rows


def _extract(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, float | str]], list[int], list[str]]:
    features = [
        decision_features(row["input"]["observation"], row["input"]["proposed_action"])
        for row in rows
    ]
    labels = [int(row["label"]["value"] == "HELPFUL") for row in rows]
    groups = [str(row["episode_id"]) for row in rows]
    return features, labels, groups


def _fit(features: Sequence[Mapping[str, Any]], labels: Sequence[int]) -> Any | None:
    if len(set(labels)) < 2:
        return None
    d = _deps()
    model = d["Pipeline"]([
        ("features", d["DictVectorizer"](sparse=True)),
        ("classifier", d["LogisticRegression"](
            class_weight="balanced", C=1.0, max_iter=2000,
            random_state=RANDOM_SEED,
        )),
    ])
    model.fit(list(features), list(labels))
    return model


def _probabilities(
    model: Any | None, features: Sequence[Mapping[str, Any]], labels: Sequence[int],
) -> list[float]:
    if model is None:
        prevalence = sum(labels) / max(1, len(labels))
        return [prevalence] * len(features)
    classes = [int(value) for value in model.classes_]
    return [
        float(dict(zip(classes, row, strict=True)).get(1, 0.0))
        for row in model.predict_proba(list(features))
    ]


def _uses_tool_from_rules(row: Mapping[str, Any]) -> bool:
    decision = HardCodedToolSupervisor().decide(
        row["input"]["observation"], row["input"]["proposed_action"],
    )
    return decision.decision == "USE_TOOL"


def _supervisor_uses_tool(supervisor: Any, row: Mapping[str, Any]) -> bool:
    return supervisor.decide(
        row["input"]["observation"], row["input"]["proposed_action"],
    ).decision == "USE_TOOL"


def select_rule(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    curve = []
    for use_missing in (False, True):
        for threshold in RULE_STDEV_THRESHOLDS:
            supervisor = OutcomeValueRuleSupervisor(
                use_missing_prior=use_missing,
                relative_adp_stdev_threshold=threshold,
            )
            decisions = [_supervisor_uses_tool(supervisor, row) for row in rows]
            curve.append({
                "use_missing_prior": use_missing,
                "relative_adp_stdev_threshold": threshold,
                **score_policy(rows, decisions),
            })
    selected = max(
        curve,
        key=lambda row: (
            row["mean_reward"], -row["harmful_selected"],
            -row["incremental_sampling_usd"], -row["tool_calls"],
            row["relative_adp_stdev_threshold"], not row["use_missing_prior"],
        ),
    )
    config = {
        "use_missing_prior": bool(selected["use_missing_prior"]),
        "relative_adp_stdev_threshold": float(
            selected["relative_adp_stdev_threshold"]
        ),
    }
    return config, curve


def grouped_oof_rule_decisions(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[bool], list[dict[str, Any]]]:
    _, _, groups = _extract(rows)
    splitter = _deps()["GroupKFold"](n_splits=min(5, len(set(groups))))
    decisions = [False] * len(rows)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(rows, groups=groups), start=1,
    ):
        train_rows = [rows[index] for index in train_idx]
        config, _ = select_rule(train_rows)
        supervisor = OutcomeValueRuleSupervisor(**config)
        for index in test_idx:
            decisions[index] = _supervisor_uses_tool(supervisor, rows[index])
        folds.append({
            "fold": fold,
            "config": config,
            "test_episode_ids": sorted({groups[index] for index in test_idx}),
        })
    return decisions, folds


def score_policy(
    rows: Sequence[Mapping[str, Any]], uses_tool: Sequence[bool],
) -> dict[str, Any]:
    if len(rows) != len(uses_tool):
        raise ValueError("row/decision length mismatch")
    rewards = [
        float(row["branches"]["intervene" if tool else "act"]["reward"])
        for row, tool in zip(rows, uses_tool, strict=True)
    ]
    act_rewards = [float(row["branches"]["act"]["reward"]) for row in rows]
    costs = [
        float(row["intervention"].get("computed_usd", 0.0)) if tool else 0.0
        for row, tool in zip(rows, uses_tool, strict=True)
    ]
    helpful = [row["label"]["value"] == "HELPFUL" for row in rows]
    harmful = [row["label"]["value"] == "HARMFUL" for row in rows]
    selected = [index for index, value in enumerate(uses_tool) if value]
    return {
        "examples": len(rows),
        "mean_reward": sum(rewards) / len(rewards),
        "mean_reward_delta_vs_act": (
            sum(reward - base for reward, base in zip(rewards, act_rewards, strict=True))
            / len(rows)
        ),
        "tool_calls": sum(uses_tool),
        "tool_rate": sum(uses_tool) / len(rows),
        "incremental_sampling_usd": sum(costs),
        "helpful_recall": sum(uses_tool[index] for index, value in enumerate(helpful) if value)
        / max(1, sum(helpful)),
        "harmful_selected": sum(harmful[index] for index in selected),
        "selected_precision": sum(helpful[index] for index in selected) / max(1, len(selected)),
    }


def select_threshold(
    rows: Sequence[Mapping[str, Any]], probabilities: Sequence[float],
) -> tuple[float, list[dict[str, Any]]]:
    curve = []
    for threshold in THRESHOLDS:
        metrics = score_policy(rows, [value >= threshold for value in probabilities])
        curve.append({"threshold": threshold, **metrics})
    selected = max(
        curve,
        key=lambda row: (
            row["mean_reward"], -row["harmful_selected"],
            -row["incremental_sampling_usd"], -row["tool_calls"], row["threshold"],
        ),
    )
    return float(selected["threshold"]), curve


def grouped_oof_probabilities(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[dict[str, Any]]]:
    features, labels, groups = _extract(rows)
    unique_groups = sorted(set(groups))
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise ValueError("need at least two episode groups")
    splitter = _deps()["GroupKFold"](n_splits=n_splits)
    probabilities = [0.0] * len(rows)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(features, labels, groups), start=1,
    ):
        train_features = [features[index] for index in train_idx]
        train_labels = [labels[index] for index in train_idx]
        model = _fit(train_features, train_labels)
        test_features = [features[index] for index in test_idx]
        predicted = _probabilities(model, test_features, train_labels)
        for index, probability in zip(test_idx, predicted, strict=True):
            probabilities[index] = probability
        folds.append({
            "fold": fold,
            "train_episode_count": len({groups[index] for index in train_idx}),
            "test_episode_ids": sorted({groups[index] for index in test_idx}),
            "train_positive_count": sum(train_labels),
        })
    return probabilities, folds


def train(path: Path = DATASET_PATH) -> dict[str, Any]:
    verify_frozen_protocol()
    rows = load_development(path)
    features, labels, groups = _extract(rows)
    oof_probabilities, folds = grouped_oof_probabilities(rows)
    threshold, curve = select_threshold(rows, oof_probabilities)
    rule_oof_decisions, rule_folds = grouped_oof_rule_decisions(rows)
    rule_config, rule_curve = select_rule(rows)
    model = _fit(features, labels)
    constant = sum(labels) / len(labels)
    d = _deps()
    payload = {
        "protocol": PROTOCOL,
        "model": model,
        "constant_probability": constant,
        "threshold": threshold,
        "metadata": {
            "created_at": utc_now(),
            "source_dataset": str(path.relative_to(ROOT)),
            "source_sha256": sha256_file(path),
            "rows": len(rows),
            "episode_groups": len(set(groups)),
            "feature_source": "harness.tool_supervisor.decision_features",
            "outcome_fields_used_as_features": False,
            "random_seed": RANDOM_SEED,
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    d["joblib"].dump(payload, ARTIFACT_PATH)
    existing_supervisor = LearnedToolSupervisor(EXISTING_ARTIFACT_PATH)
    systems = {
        "always_act": score_policy(rows, [False] * len(rows)),
        "always_tool": score_policy(rows, [True] * len(rows)),
        "hard_coded_rules": score_policy(rows, [_uses_tool_from_rules(row) for row in rows]),
        "existing_imitation_classifier": score_policy(rows, [
            _supervisor_uses_tool(existing_supervisor, row) for row in rows
        ]),
        "transparent_value_rule_grouped_oof": score_policy(
            rows, rule_oof_decisions,
        ),
        "learned_grouped_oof": score_policy(
            rows, [value >= threshold for value in oof_probabilities],
        ),
    }
    report = {
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "split": "development",
        "fit_uses_heldout": False,
        "rows": len(rows),
        "episodes": len(set(groups)),
        "label_counts": dict(sorted(Counter(row["label"]["value"] for row in rows).items())),
        "selected_threshold": threshold,
        "selection_rule": "maximize grouped-OOF mean reward; break ties by harm, cost, calls",
        "systems": systems,
        "threshold_curve": curve,
        "folds": folds,
        "transparent_value_rule": rule_config,
        "transparent_value_rule_curve": rule_curve,
        "transparent_value_rule_folds": rule_folds,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "artifact_sha256": sha256_file(ARTIFACT_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    freeze = {
        "protocol": PROTOCOL,
        "frozen_at": utc_now(),
        "heldout_labels_seen": False,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "artifact_sha256": sha256_file(ARTIFACT_PATH),
        "development_dataset": str(path.relative_to(ROOT)),
        "development_dataset_sha256": sha256_file(path),
        "selected_threshold": threshold,
        "candidate_systems": [
            "base_qwen", "existing_supervisor", "outcome_value_rule",
            "learned_outcome_value",
        ],
        "existing_supervisor_artifact": str(EXISTING_ARTIFACT_PATH.relative_to(ROOT)),
        "existing_supervisor_artifact_sha256": sha256_file(EXISTING_ARTIFACT_PATH),
        "transparent_value_rule": rule_config,
        "evaluation_episode_ids": [
            row["episode_id"] for row in arm_evaluation_panel()
        ],
    }
    CANDIDATE_FREEZE_PATH.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train",))
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
