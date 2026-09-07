"""Small local decision card backed by the frozen risk-aware ranker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from training.outcome_decisions import FEATURE_ALLOWLIST
from training.outcome_ranker import encode
from training.override_dataset import load_override_corpus
from training.risk_aware_ranker import load_artifact

CONTEXT_FIELDS = {
    "round", "overall_pick", "draft_slot", "picks_until_next_turn",
    "roster_qb", "roster_rb", "roster_wr", "roster_te",
}
CANDIDATE_FIELDS = set(FEATURE_ALLOWLIST) - CONTEXT_FIELDS


def recommend(payload: Mapping[str, Any]) -> dict[str, Any]:
    context = payload.get("context")
    candidates = payload.get("candidates")
    if not isinstance(context, Mapping) or not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("input needs a context object and at least two candidates")
    if set(context) != CONTEXT_FIELDS:
        raise ValueError("context fields do not match the decision-card schema")
    ordered = sorted(candidates, key=lambda item: int(item["candidate_board_rank"]))[:5]
    if int(ordered[0]["candidate_board_rank"]) != 1:
        raise ValueError("the ADP candidate must have board rank 1")
    features = []
    for candidate in ordered:
        if set(candidate) != CANDIDATE_FIELDS | {"candidate_id"}:
            raise ValueError("candidate fields do not match the decision-card schema")
        features.append({**context, **{key: candidate[key] for key in CANDIDATE_FIELDS}})
    artifact = load_artifact()
    x = np.asarray([encode(row) for row in features], dtype=float)
    means = artifact["models"]["mean"].predict(x)
    downsides = artifact["models"]["downside"].predict(x)
    alternative = max(
        range(1, len(ordered)),
        key=lambda index: (
            float(means[index]), float(downsides[index]),
            -float(ordered[index]["candidate_adp"]),
        ),
    )
    mean_edge = float(means[alternative] - means[0])
    downside_edge = float(downsides[alternative] - downsides[0])
    mean_margin = float(artifact["policy"]["mean_margin"])
    downside_margin = float(artifact["policy"]["downside_margin"])
    override = mean_edge >= mean_margin and downside_edge >= downside_margin
    selected = ordered[alternative] if override else ordered[0]
    return {
        "decision": "OVERRIDE" if override else "KEEP_ADP",
        "selected_candidate_id": str(selected["candidate_id"]),
        "adp_candidate_id": str(ordered[0]["candidate_id"]),
        "alternative_considered": str(ordered[alternative]["candidate_id"]),
        "checks": {
            "expected_value_edge": round(mean_edge, 2),
            "expected_value_required": mean_margin,
            "expected_value_passed": mean_edge >= mean_margin,
            "downside_edge": round(downside_edge, 2),
            "downside_required": downside_margin,
            "downside_passed": downside_edge >= downside_margin,
        },
        "explanation": (
            "Override: the alternative cleared both the expected-value and downside safeguards."
            if override else
            "Keep ADP: the best visible alternative did not clear both safeguards."
        ),
        "scope": "Historical decision support; not a forecast guarantee.",
    }


def example_payload() -> dict[str, Any]:
    row = next(
        row for row in load_override_corpus()
        if json.loads(row["messages"][-1]["content"])["decision"] == "OVERRIDE"
    )
    return json.loads(row["messages"][1]["content"].split("\n", 1)[1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Get a conservative ADP decision card")
    parser.add_argument("--input", type=Path, help="JSON file containing context and candidates")
    parser.add_argument("--example", action="store_true", help="run a bundled masked example")
    args = parser.parse_args()
    if bool(args.input) == bool(args.example):
        parser.error("choose exactly one of --input or --example")
    payload = example_payload() if args.example else json.loads(args.input.read_text())
    print(json.dumps(recommend(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
