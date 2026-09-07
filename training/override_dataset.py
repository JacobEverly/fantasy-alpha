"""Build the narrow SFT corpus for high-confidence ADP overrides.

The frozen risk-aware ranker is the teacher.  Inputs contain only information
available at the draft turn; teacher scores and realized outcomes stay in
metadata and never enter the model prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from training.outcome_decisions import FEATURE_ALLOWLIST, ROWS_PATH
from training.outcome_ranker import encode, load_rows
from training.risk_aware_ranker import MODEL_PATH, load_artifact

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "training/datasets/override_sft_v1.jsonl"
MANIFEST_PATH = ROOT / "training/datasets/override_sft_v1.manifest.json"
SYSTEM_PROMPT = (
    "You are a conservative fantasy-football draft decision policy. Choose "
    "KEEP_ADP unless the visible evidence strongly supports one listed alternative. "
    "Return only compact JSON matching one of these schemas: "
    '{"decision":"KEEP_ADP"} or '
    '{"decision":"OVERRIDE","candidate_id":"B000"}.'
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _groups(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    return [
        sorted(group, key=lambda row: int(row["features"]["candidate_board_rank"]))[:5]
        for _, group in sorted(grouped.items())
    ]


def _visible_prompt(rows: Sequence[Mapping[str, Any]]) -> str:
    first = rows[0]
    shared = first["features"]
    context_names = (
        "round", "overall_pick", "draft_slot", "picks_until_next_turn",
        "roster_qb", "roster_rb", "roster_wr", "roster_te",
    )
    context = {name: shared[name] for name in context_names}
    candidates = []
    candidate_names = (
        "candidate_board_rank", "candidate_position", "candidate_adp",
        "candidate_adp_stdev", "candidate_prev_season_points",
        "candidate_survival_to_next_turn",
    )
    for row in rows:
        candidates.append({
            "candidate_id": row["candidate_id"],
            **{name: row["features"][name] for name in candidate_names},
        })
    return (
        "Choose the draft action from this masked state. Candidate board rank 1 is "
        "the ADP default.\n"
        + json.dumps({"context": context, "candidates": candidates}, separators=(",", ":"))
    )


def build_rows() -> list[dict[str, Any]]:
    source = load_rows()
    artifact = load_artifact()
    mean_margin = float(artifact["policy"]["mean_margin"])
    downside_margin = float(artifact["policy"]["downside_margin"])
    result = []
    for group in _groups(source):
        x = np.asarray([encode(row["features"]) for row in group], dtype=float)
        means = artifact["models"]["mean"].predict(x)
        downsides = artifact["models"]["downside"].predict(x)
        alternative = max(
            range(1, len(group)),
            key=lambda index: (
                float(means[index]), float(downsides[index]),
                -float(group[index]["features"]["candidate_adp"]),
            ),
        )
        mean_edge = float(means[alternative] - means[0])
        downside_edge = float(downsides[alternative] - downsides[0])
        override = mean_edge >= mean_margin and downside_edge >= downside_margin
        one_margin_only = (mean_edge >= mean_margin) != (downside_edge >= downside_margin)
        near_boundary = not override and max(mean_edge, downside_edge) >= 50.0
        if override:
            target = {
                "decision": "OVERRIDE",
                "candidate_id": str(group[alternative]["candidate_id"]),
            }
            loss_weight = 8.0
            difficulty = "positive_override"
        else:
            target = {"decision": "KEEP_ADP"}
            loss_weight = 2.0 if (one_margin_only or near_boundary) else 1.0
            difficulty = "hard_keep" if (one_margin_only or near_boundary) else "easy_keep"
        first = group[0]
        result.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _visible_prompt(group)},
                {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
            ],
            "meta": {
                "trace_id": f"override:{first['state_id']}",
                "bench": "draftgym",
                "family": "high_confidence_override",
                "season": int(first["season"]),
                "question_id": str(first["episode_id"]),
                "episode_id": str(first["episode_id"]),
                "state_id": str(first["state_id"]),
                "decision": target["decision"],
                "target_candidate_id": target.get("candidate_id"),
                "difficulty": difficulty,
                "loss_weight": loss_weight,
                "teacher_mean_edge": mean_edge,
                "teacher_downside_edge": downside_edge,
            },
        })
    return sorted(result, key=lambda row: row["meta"]["trace_id"])


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty override corpus")
    trace_ids = [str(row["meta"]["trace_id"]) for row in rows]
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("duplicate override trace ID")
    decisions: Counter[str] = Counter()
    difficulty: Counter[str] = Counter()
    seasons: Counter[int] = Counter()
    for row in rows:
        if [message.get("role") for message in row["messages"]] != [
            "system", "user", "assistant"
        ]:
            raise ValueError("invalid message roles")
        prompt = "\n".join(message["content"] for message in row["messages"][:-1]).lower()
        if any(marker in prompt for marker in (
            "teacher_mean_edge", "teacher_downside_edge", "completed_roster_points",
            "regret_to_best", "is_best_candidate",
        )):
            raise ValueError(f"label leaked into prompt for {row['meta']['trace_id']}")
        target = json.loads(row["messages"][-1]["content"])
        decision = str(target.get("decision"))
        if decision not in {"KEEP_ADP", "OVERRIDE"}:
            raise ValueError("invalid target decision")
        visible = json.loads(row["messages"][1]["content"].split("\n", 1)[1])
        visible_ids = {str(candidate["candidate_id"]) for candidate in visible["candidates"]}
        if decision == "OVERRIDE" and str(target.get("candidate_id")) not in visible_ids:
            raise ValueError("override target is not visible")
        if decision == "KEEP_ADP" and "candidate_id" in target:
            raise ValueError("KEEP_ADP target must not name a candidate")
        weight = float(row["meta"]["loss_weight"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("invalid loss weight")
        decisions[decision] += 1
        difficulty[str(row["meta"]["difficulty"])] += 1
        seasons[int(row["meta"]["season"])] += 1
    return {
        "rows": len(rows),
        "decisions": dict(sorted(decisions.items())),
        "override_rate": decisions["OVERRIDE"] / len(rows),
        "difficulty": dict(sorted(difficulty.items())),
        "seasons": {str(key): value for key, value in sorted(seasons.items())},
        "weighted_rows": sum(float(row["meta"]["loss_weight"]) for row in rows),
    }


def write(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    audit = validate_rows(rows)
    manifest = {
        "protocol": "fantasy-alpha-high-confidence-override-sft-v1",
        "objective": "imitate the frozen 80/80 risk-aware teacher while defaulting to ADP",
        "source_rows_path": str(ROWS_PATH.relative_to(ROOT)),
        "source_rows_sha256": _sha256(ROWS_PATH),
        "teacher_artifact_path": str(MODEL_PATH.relative_to(ROOT)),
        "teacher_artifact_sha256": _sha256(MODEL_PATH),
        "input_feature_allowlist": list(FEATURE_ALLOWLIST),
        "teacher_scores_visible_to_model": False,
        "realized_outcomes_visible_to_model": False,
        "positive_loss_weight": 8.0,
        "hard_negative_loss_weight": 2.0,
        "audit": audit,
        "corpus_sha256": _sha256(CORPUS_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_override_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate_rows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = build_rows()
    manifest = write(rows) if args.write else {"audit": validate_rows(rows)}
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
