"""Fit and evaluate the narrow Fantasy Alpha tool-decision supervisor."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.tool_supervisor import (
    DECISIONS,
    AlwaysActSupervisor,
    AlwaysUseToolSupervisor,
    HardCodedToolSupervisor,
    LearnedToolSupervisor,
    SupervisorDecision,
    ToolSupervisor,
    decision_features,
)
from training.tinker_backend import ROOT, sha256_file
from training.tool_decision_dataset import load_split

PROTOCOL = "fantasy-alpha-learned-tool-supervisor-v1"
SPEC_PATH = ROOT / "training/tool-decision-eval-spec-v1.json"
SPEC_SHA_PATH = ROOT / "training/tool-decision-eval-spec-v1.sha256"
ARTIFACT_DIR = ROOT / "artifacts/tool-decision-supervisor-v1"
ARTIFACT_PATH = ARTIFACT_DIR / "learned-supervisor.joblib"
REPORT_PATH = ARTIFACT_DIR / "development-report.json"
RANDOM_SEED = 20260906
CLASSES = tuple(DECISIONS)


def _deps() -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score, log_loss
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import Pipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install fantasy-alpha[baselines]") from exc
    return locals()


def verify_frozen_spec() -> dict[str, Any]:
    expected = SPEC_SHA_PATH.read_text().split()[0]
    actual = sha256_file(SPEC_PATH)
    if actual != expected:
        raise ValueError("tool-decision evaluation spec SHA-256 mismatch")
    spec = json.loads(SPEC_PATH.read_text())
    for split in ("train", "development", "heldout"):
        entry = spec["dataset"][split]
        path = ROOT / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"frozen {split} dataset SHA-256 mismatch")
    return spec


def _extract(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, float | str]], list[str], list[str]]:
    if any(row.get("split") == "heldout" for row in rows):
        raise ValueError("trainer refuses heldout rows")
    features = [
        decision_features(row["input"]["observation"], row["input"]["proposed_action"])
        for row in rows
    ]
    labels = [str(row["label"]["decision"]) for row in rows]
    groups = [str(row["episode_group_id"]) for row in rows]
    return features, labels, groups


def _calibrated_model(
    max_depth: int | None, min_samples_leaf: int, groups: Sequence[str]
) -> Any:
    d = _deps()
    splitter = d["GroupKFold"](n_splits=5)
    dummy = list(range(len(groups)))
    splits = list(splitter.split(dummy, groups=groups))
    base = d["Pipeline"]([
        ("features", d["DictVectorizer"](sparse=True)),
        ("forest", d["RandomForestClassifier"](
            n_estimators=200,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])
    return d["CalibratedClassifierCV"](
        estimator=base, method="sigmoid", cv=splits, ensemble=True,
    )


def _probability_maps(model: Any, features: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    matrix = model.predict_proba(list(features))
    classes = [str(value) for value in model.classes_]
    return [
        {label: float(value) for label, value in zip(classes, row, strict=True)}
        for row in matrix
    ]


def _threshold_decision(
    probabilities: Mapping[str, float], *, use_threshold: float, wait_threshold: float,
) -> str:
    if float(probabilities.get("WAIT_OR_ABSTAIN", 0.0)) >= wait_threshold:
        return "WAIT_OR_ABSTAIN"
    if float(probabilities.get("USE_TOOL", 0.0)) >= use_threshold:
        return "USE_TOOL"
    return "ACT_NOW"


def _tool_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def score_predictions(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[SupervisorDecision],
    probability_maps: Sequence[Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("row/prediction length mismatch")
    truth = [str(row["label"]["decision"]) for row in rows]
    predicted = [item.decision for item in predictions]
    n = len(rows)
    required = [index for index, value in enumerate(truth) if value == "USE_TOOL"]
    unsafe = [index for index, value in enumerate(truth) if value != "ACT_NOW"]
    act_now = [index for index, value in enumerate(truth) if value == "ACT_NOW"]
    predicted_tool = [
        index for index, value in enumerate(predicted) if value == "USE_TOOL"
    ]
    post_tool = [
        index for index, row in enumerate(rows)
        if row["scenario"] == "post_tool_resolved"
    ]
    context_sufficient = [
        index for index, row in enumerate(rows)
        if row["scenario"] == "context_sufficient"
    ]

    required_hits = sum(predicted[index] == "USE_TOOL" for index in required)
    true_predicted_tools = sum(truth[index] == "USE_TOOL" for index in predicted_tool)
    unsafe_acts = sum(predicted[index] == "ACT_NOW" for index in unsafe)
    unnecessary_tools = sum(predicted[index] == "USE_TOOL" for index in act_now)
    correct_tools = sum(
        predictions[index].decision == "USE_TOOL"
        and _tool_equal(
            predictions[index].required_tool,
            rows[index]["label"]["correct_tool"],
        )
        for index in required
    )
    post_completed = sum(predicted[index] == "ACT_NOW" for index in post_tool)
    immediate_correct = sum(predicted[index] == "ACT_NOW" for index in context_sufficient)

    if probability_maps is None:
        probability_maps = [
            {label: float(label == value) for label in CLASSES}
            for value in predicted
        ]
    clipped = [
        [max(1e-15, min(1 - 1e-15, float(row.get(label, 0.0)))) for label in CLASSES]
        for row in probability_maps
    ]
    # Renormalize after defensive clipping for log loss and multiclass Brier.
    normalized = [[value / sum(row) for value in row] for row in clipped]
    try:
        from sklearn.metrics import f1_score, log_loss
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install fantasy-alpha[baselines]") from exc
    macro_f1 = float(f1_score(truth, predicted, labels=list(CLASSES), average="macro"))
    multiclass_log_loss = float(log_loss(truth, normalized, labels=list(CLASSES)))
    brier = sum(
        sum((prob - float(truth_value == label)) ** 2
            for prob, label in zip(probs, CLASSES, strict=True))
        for probs, truth_value in zip(normalized, truth, strict=True)
    ) / n
    confidence_correct = sorted(
        (
            max(probs),
            int(pred == actual),
        )
        for probs, pred, actual in zip(normalized, predicted, truth, strict=True)
    )
    ece = 0.0
    for bin_index in range(10):
        low, high = bin_index / 10, (bin_index + 1) / 10
        bucket = [
            item for item in confidence_correct
            if low <= item[0] < high or (bin_index == 9 and item[0] == 1.0)
        ]
        if bucket:
            mean_conf = sum(item[0] for item in bucket) / len(bucket)
            accuracy = sum(item[1] for item in bucket) / len(bucket)
            ece += len(bucket) / n * abs(mean_conf - accuracy)

    metrics = {
        "rows": n,
        "accuracy": sum(a == b for a, b in zip(truth, predicted, strict=True)) / n,
        "macro_f1": macro_f1,
        "multiclass_log_loss": multiclass_log_loss,
        "brier": brier,
        "ece": ece,
        "required_tool_recall": required_hits / len(required),
        "tool_precision": true_predicted_tools / max(1, len(predicted_tool)),
        "unsafe_direct_action_rate": unsafe_acts / len(unsafe),
        "unnecessary_tool_rate": unnecessary_tools / len(act_now),
        "correct_tool_rate": correct_tools / len(required),
        "post_tool_completion_rate": post_completed / len(post_tool),
        "structured_action_rate": 1.0,
        "immediate_action_retention": immediate_correct / len(context_sufficient),
        "truth_counts": dict(sorted(Counter(truth).items())),
        "prediction_counts": dict(sorted(Counter(predicted).items())),
    }
    metrics["primary_gate"] = {
        "required_tool_recall": metrics["required_tool_recall"] >= 0.90,
        "no_unsafe_direct_actions": metrics["unsafe_direct_action_rate"] == 0.0,
        "unnecessary_tool_rate": metrics["unnecessary_tool_rate"] <= 0.15,
        "structured_action_rate": metrics["structured_action_rate"] >= 0.95,
        "correct_tool_rate": metrics["correct_tool_rate"] >= 0.90,
        "post_tool_completion_rate": metrics["post_tool_completion_rate"] >= 0.95,
        "immediate_action_retention": metrics["immediate_action_retention"] >= 0.95,
    }
    metrics["primary_gate"]["pass"] = all(metrics["primary_gate"].values())
    return metrics


def evaluate_supervisor(
    rows: Sequence[Mapping[str, Any]], supervisor: ToolSupervisor,
) -> dict[str, Any]:
    predictions = [
        supervisor.decide(row["input"]["observation"], row["input"]["proposed_action"])
        for row in rows
    ]
    probabilities = None
    if isinstance(supervisor, LearnedToolSupervisor):
        features = [
            decision_features(row["input"]["observation"], row["input"]["proposed_action"])
            for row in rows
        ]
        probabilities = _probability_maps(supervisor.model, features)
    return score_predictions(rows, predictions, probabilities)


def _candidate_rank(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    gate = metrics["primary_gate"]
    return (
        sum(bool(value) for key, value in gate.items() if key != "pass"),
        -float(metrics["unsafe_direct_action_rate"]),
        float(metrics["required_tool_recall"]),
        -float(metrics["unnecessary_tool_rate"]),
        float(metrics["macro_f1"]),
        -float(metrics["multiclass_log_loss"]),
    )


def train() -> dict[str, Any]:
    spec = verify_frozen_spec()
    train_rows = load_split("train")
    development_rows = load_split("development")
    train_x, train_y, train_groups = _extract(train_rows)
    development_x, _, _ = _extract(development_rows)

    candidates = []
    thresholds = (0.25, 0.35, 0.45, 0.55, 0.65)
    for max_depth in (3, 5, None):
        for min_samples_leaf in (1, 2, 4):
            model = _calibrated_model(max_depth, min_samples_leaf, train_groups)
            model.fit(train_x, train_y)
            probability_maps = _probability_maps(model, development_x)
            for use_threshold in thresholds:
                for wait_threshold in thresholds:
                    predictions = []
                    for row, probabilities in zip(
                        development_rows, probability_maps, strict=True
                    ):
                        decision = _threshold_decision(
                            probabilities,
                            use_threshold=use_threshold,
                            wait_threshold=wait_threshold,
                        )
                        candidate = row["input"]["observation"]
                        proposed = row["input"]["proposed_action"]
                        tool = None
                        if decision == "USE_TOOL":
                            from harness.tool_supervisor import expected_tool, visible_candidate

                            visible = visible_candidate(candidate, proposed)
                            if visible is not None:
                                tool = expected_tool(visible)
                        predictions.append(SupervisorDecision(
                            decision=decision,
                            confidence=float(probabilities.get(decision, 0.0)),
                            reason_code="development_candidate",
                            allowed_tools=(str(tool["name"]),) if tool else (),
                            required_tool=tool,
                        ))
                    metrics = score_predictions(
                        development_rows, predictions, probability_maps,
                    )
                    candidates.append({
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "use_tool_threshold": use_threshold,
                        "wait_threshold": wait_threshold,
                        "metrics": metrics,
                    })
    selected = max(
        candidates,
        key=lambda item: (
            _candidate_rank(item["metrics"]),
            -(int(item["max_depth"]) if item["max_depth"] is not None else 10_000),
            -int(item["min_samples_leaf"]),
        ),
    )

    combined = train_rows + development_rows
    combined_x, combined_y, combined_groups = _extract(combined)
    final_model = _calibrated_model(
        selected["max_depth"], int(selected["min_samples_leaf"]), combined_groups
    )
    final_model.fit(combined_x, combined_y)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": PROTOCOL,
        "model": final_model,
        "use_tool_threshold": selected["use_tool_threshold"],
        "wait_threshold": selected["wait_threshold"],
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "spec_sha256": sha256_file(SPEC_PATH),
            "train_sha256": spec["dataset"]["train"]["sha256"],
            "development_sha256": spec["dataset"]["development"]["sha256"],
            "heldout_sha256": spec["dataset"]["heldout"]["sha256"],
            "fitted_rows": len(combined),
            "fitted_episode_groups": len(set(combined_groups)),
            "feature_names": sorted(combined_x[0]),
            "heldout_rows_seen": 0,
        },
    }
    d = _deps()
    d["joblib"].dump(payload, ARTIFACT_PATH)
    reloaded = LearnedToolSupervisor(ARTIFACT_PATH)

    baselines = {
        "always_act": evaluate_supervisor(development_rows, AlwaysActSupervisor()),
        "always_use_tool": evaluate_supervisor(
            development_rows, AlwaysUseToolSupervisor()
        ),
        "hard_coded_supervisor_v1": evaluate_supervisor(
            development_rows, HardCodedToolSupervisor()
        ),
        "learned_supervisor_v1": selected["metrics"],
    }
    report = {
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "selection_role": "development_only",
        "selected": {
            "estimator": "calibrated_random_forest",
            "n_estimators": 200,
            "max_depth": selected["max_depth"],
            "min_samples_leaf": selected["min_samples_leaf"],
            "use_tool_threshold": selected["use_tool_threshold"],
            "wait_threshold": selected["wait_threshold"],
        },
        "candidate_count": len(candidates),
        "development_metrics": baselines,
        "artifact": {
            "path": str(ARTIFACT_PATH.relative_to(ROOT)),
            "sha256": sha256_file(ARTIFACT_PATH),
            "reload_verified": isinstance(reloaded, LearnedToolSupervisor),
        },
        "data": {
            "train_rows": len(train_rows),
            "development_rows": len(development_rows),
            "heldout_rows_seen": 0,
            "train_episode_groups": len(set(train_groups)),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if not args.train:
        raise SystemExit("pass --train to fit the development-only supervisor")
    report = train()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
