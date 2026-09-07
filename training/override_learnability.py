"""Grouped local learnability gate for the high-confidence override corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from training.override_dataset import CORPUS_PATH, ROOT, load_override_corpus

REPORT_PATH = ROOT / "artifacts/high-confidence-override-v1/local-learnability.json"
POSITIONS = ("QB", "RB", "WR", "TE")
NUMERIC_CANDIDATE_FIELDS = (
    "candidate_board_rank", "candidate_adp", "candidate_adp_stdev",
    "candidate_prev_season_points", "candidate_survival_to_next_turn",
)
CONTEXT_FIELDS = (
    "round", "overall_pick", "draft_slot", "picks_until_next_turn",
    "roster_qb", "roster_rb", "roster_wr", "roster_te",
)


def _payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(str(row["messages"][1]["content"]).split("\n", 1)[1])


def _candidate_vector(
    context: Mapping[str, Any], adp: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[float]:
    vector = [float(context[name]) for name in CONTEXT_FIELDS]
    vector.extend(float(candidate["candidate_position"] == pos) for pos in POSITIONS)
    vector.extend(float(adp["candidate_position"] == pos) for pos in POSITIONS)
    for name in NUMERIC_CANDIDATE_FIELDS:
        candidate_value = float(candidate[name])
        adp_value = float(adp[name])
        vector.extend((candidate_value, adp_value, candidate_value - adp_value))
    return vector


def candidate_examples(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    x: list[list[float]] = []
    y: list[int] = []
    state_indices: list[int] = []
    candidate_ids: list[str] = []
    state_labels: list[int] = []
    target_ids: list[str | None] = []
    episodes: list[str] = []
    for state_index, row in enumerate(rows):
        payload = _payload(row)
        candidates = sorted(payload["candidates"], key=lambda item: item["candidate_board_rank"])
        adp = candidates[0]
        target = json.loads(str(row["messages"][-1]["content"]))
        override = target["decision"] == "OVERRIDE"
        target_id = str(target["candidate_id"]) if override else None
        state_labels.append(int(override))
        target_ids.append(target_id)
        episodes.append(str(row["meta"]["episode_id"]))
        for candidate in candidates[1:]:
            cid = str(candidate["candidate_id"])
            x.append(_candidate_vector(payload["context"], adp, candidate))
            y.append(int(override and cid == target_id))
            state_indices.append(state_index)
            candidate_ids.append(cid)
    return {
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=int),
        "state_indices": np.asarray(state_indices, dtype=int),
        "candidate_ids": candidate_ids,
        "state_labels": np.asarray(state_labels, dtype=int),
        "target_ids": target_ids,
        "episodes": np.asarray(episodes),
    }


def _state_predictions(data: Mapping[str, Any], probabilities: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n_states = len(data["state_labels"])
    scores = np.zeros(n_states, dtype=float)
    candidates = [""] * n_states
    for state in range(n_states):
        indices = np.flatnonzero(data["state_indices"] == state)
        best = int(indices[int(np.argmax(probabilities[indices]))])
        scores[state] = float(probabilities[best])
        candidates[state] = str(data["candidate_ids"][best])
    return scores, candidates


def _metrics(
    labels: np.ndarray, target_ids: Sequence[str | None], predicted_ids: Sequence[str],
    scores: np.ndarray, threshold: float,
) -> dict[str, Any]:
    actions = scores >= threshold
    tp = int(np.sum(actions & (labels == 1)))
    fp = int(np.sum(actions & (labels == 0)))
    fn = int(np.sum(~actions & (labels == 1)))
    exact = sum(
        bool(actions[index]) and labels[index] == 1 and predicted_ids[index] == target_ids[index]
        for index in range(len(labels))
    )
    return {
        "threshold": threshold,
        "true_override_states": int(labels.sum()),
        "predicted_override_states": int(actions.sum()),
        "true_positives": tp,
        "false_overrides": fp,
        "false_negatives": fn,
        "override_precision": tp / (tp + fp) if tp + fp else 1.0,
        "override_recall": tp / (tp + fn) if tp + fn else 0.0,
        "exact_action_precision": exact / int(actions.sum()) if actions.sum() else 1.0,
        "exact_action_recall": exact / int(labels.sum()) if labels.sum() else 0.0,
        "false_override_rate_on_keep_states": fp / int((labels == 0).sum()),
        "intervention_rate": float(actions.mean()),
    }


def run(
    rows: Sequence[dict[str, Any]], *, group_mode: str = "episode"
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold

    data = candidate_examples(rows)
    x, y = data["x"], data["y"]
    if group_mode == "episode":
        state_groups = data["episodes"]
        split_description = "five-fold stratified episode-grouped out-of-fold predictions"
    elif group_mode == "prompt":
        state_groups = np.asarray([
            hashlib.sha256(
                " ".join(str(row["messages"][1]["content"]).split()).encode()
            ).hexdigest()
            for row in rows
        ])
        split_description = (
            "post-hoc five-fold split grouped by exact visible prompt identity; "
            "duplicate seed states cannot cross folds"
        )
    else:
        raise ValueError(f"unknown group mode: {group_mode}")
    candidate_groups = state_groups[data["state_indices"]]
    candidate_state_labels = data["state_labels"][data["state_indices"]]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260907)
    probabilities = np.zeros(len(y), dtype=float)
    fold_rows = []
    for fold, (train, test) in enumerate(
        splitter.split(x, candidate_state_labels, groups=candidate_groups), 1
    ):
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=1.0, random_state=20260907 + fold,
        )
        weights = np.where(y[train] == 1, 16.0, 1.0)
        model.fit(x[train], y[train], sample_weight=weights)
        probabilities[test] = model.predict_proba(x[test])[:, 1]
        fold_rows.append({
            "fold": fold,
            "training_candidate_rows": len(train),
            "test_candidate_rows": len(test),
            "training_positive_candidates": int(y[train].sum()),
            "test_positive_candidates": int(y[test].sum()),
            "test_episode_groups": len(set(candidate_groups[test])),
        })
    scores, predicted_ids = _state_predictions(data, probabilities)
    labels = data["state_labels"]
    # This is a diagnostic gate, not the frozen model evaluation. Select the
    # most useful operating point under a strict 2% false-override constraint.
    candidates = []
    for threshold in np.unique(np.concatenate(([0.0, 1.0], scores))):
        metrics = _metrics(labels, data["target_ids"], predicted_ids, scores, float(threshold))
        if metrics["false_override_rate_on_keep_states"] <= 0.02:
            candidates.append(metrics)
    selected = max(
        candidates,
        key=lambda item: (
            item["exact_action_recall"], item["override_precision"],
            -item["intervention_rate"], item["threshold"],
        ),
    )
    prevalence = float(labels.mean())
    state_ap = float(average_precision_score(labels, scores))
    gates = {
        "state_average_precision_at_least_4x_prevalence": bool(state_ap >= 4 * prevalence),
        "override_precision_at_least_50pct": bool(selected["override_precision"] >= 0.50),
        "exact_action_recall_at_least_25pct": bool(selected["exact_action_recall"] >= 0.25),
        "false_override_rate_at_most_2pct": bool(selected["false_override_rate_on_keep_states"] <= 0.02),
    }
    gates["pass"] = all(gates.values())
    return {
        "protocol": "fantasy-alpha-override-local-learnability-v1",
        "purpose": "paid-training gate only; not a product or held-out performance claim",
        "split": split_description,
        "group_mode": group_mode,
        "unique_state_groups": len(set(state_groups)),
        "states": len(rows),
        "candidate_rows": len(y),
        "override_prevalence": prevalence,
        "state_average_precision": state_ap,
        "selected_operating_point": selected,
        "gates": gates,
        "folds": fold_rows,
        "paid_spend_usd": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument("--group-by-prompt", action="store_true")
    args = parser.parse_args()
    report = run(
        load_override_corpus(CORPUS_PATH),
        group_mode="prompt" if args.group_by_prompt else "episode",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
