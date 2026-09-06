"""Build and validate the T1.2 broad-plus-targeted SFT mixture.

T1.2 retains the broad T1 forecast corpus while adding T1.1's short JSON
actions.  Because the two sources differ by roughly thirty-fold in assistant
target length, rows carry explicit loss weights.  The weights allocate 70% of
the supervised token mass to broad retention and 30% to targeted behavior.

The 120 T1.1 forecast corrections reuse prompts from T1 with different targets.
Those corrected targets supersede 156 older T1 sample rows so a prompt never
has two conflicting labels.  No examples are copied or synthesized, and 2025
remains sealed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from training.t11_dataset import (
    CORPUS_PATH as T11_CORPUS_PATH,
)
from training.t11_dataset import (
    load_t11_corpus,
)
from training.t11_dataset import (
    validate_rows as validate_t11_rows,
)
from training.tinker_backend import (
    DEFAULT_CORPUS as T1_CORPUS_PATH,
)
from training.tinker_backend import (
    ROOT,
    RunConfig,
    grouped_development_split,
    load_corpus,
    render_rows,
    sha256_file,
)

CORPUS_PATH = ROOT / "training/datasets/t12_mixture_v2.jsonl"
MANIFEST_PATH = ROOT / "training/datasets/t12_mixture_v2.manifest.json"
DATASET_CARD_PATH = ROOT / "docs/models/fantasy-alpha-t12-dataset-card.md"
DATASET_SEED = 20260905
EXPECTED_ROWS = 829

# Fractions of total weighted assistant-token mass.  Broad families are kept
# equal so the smaller bust family is not washed out.  The targeted allocation
# gives complete evidence transitions and recovery actions meaningful weight
# despite their nine- or twenty-token answers.
SCHEMA_MASS = {
    "broad_bust": 0.175,
    "broad_full_slate": 0.175,
    "broad_season_threshold": 0.175,
    "broad_weekly_h2h": 0.175,
    "target_calibration": 0.05,
    "target_draft_pick": 0.07,
    "target_draft_recovery": 0.03,
    "target_tool_call": 0.10,
    "target_post_tool_pick": 0.05,
}


def _prompt_text(row: Mapping[str, Any]) -> str:
    return "\n".join(str(message["content"]) for message in row["messages"][:-1])


def _prompt_hash(row: Mapping[str, Any]) -> str:
    normalized = " ".join(_prompt_text(row).split()).lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _schema(row: Mapping[str, Any], source: str) -> str:
    meta = row["meta"]
    if source == "t1":
        return f"broad_{meta['family']}"
    subtype = meta["t11_subtype"]
    if subtype == "calibration_correction":
        return "target_calibration"
    if subtype == "tool_call":
        return "target_tool_call"
    if subtype == "post_tool_pick":
        return "target_post_tool_pick"
    if "recovery" in meta.get("categories", []):
        return "target_draft_recovery"
    return "target_draft_pick"


def _decorate(row: Mapping[str, Any], *, source: str, schema: str) -> dict[str, Any]:
    result = copy.deepcopy(row)
    meta = result["meta"]
    original_trace = str(meta["trace_id"])
    meta.update({
        "trace_id": f"t12:{source}:{original_trace}",
        "t12_source": source,
        "t12_source_trace_id": original_trace,
        "t12_schema": schema,
        "t12_dataset_seed": DATASET_SEED,
    })
    return result


def _target_counts(rows: Sequence[dict[str, Any]]) -> list[int]:
    config = RunConfig(run_name="t12-token-audit", epochs=1)
    datums, _, _ = render_rows(rows, config)
    return [
        sum(float(weight) > 0 for weight in datum.loss_fn_inputs["weights"].data)
        for datum in datums
    ]


def build_rows() -> list[dict[str, Any]]:
    broad = load_corpus()
    targeted = load_t11_corpus()
    validate_t11_rows(targeted)

    targeted_prompt_hashes = {_prompt_hash(row) for row in targeted}
    retained_variants = [row for row in broad if _prompt_hash(row) not in targeted_prompt_hashes]
    superseded = len(broad) - len(retained_variants)
    # The 120 corrected prompt identities map to 156 T1 rows because T1 kept
    # multiple teacher samples for some questions.  Every old variant must be
    # removed to avoid same-prompt/different-target supervision.
    if superseded != 156:
        raise AssertionError(f"expected 156 superseded T1 rows, found {superseded}")
    # T1 also contains multiple stochastic teacher answers for 126 remaining
    # prompt identities. Pick the lowest trace ID without consulting outcomes,
    # so every prompt has one deterministic target and selection cannot leak.
    retained_by_prompt: dict[str, dict[str, Any]] = {}
    for row in sorted(retained_variants, key=lambda item: item["meta"]["trace_id"]):
        retained_by_prompt.setdefault(_prompt_hash(row), row)
    retained_broad = list(retained_by_prompt.values())
    if len(retained_variants) - len(retained_broad) != 126:
        raise AssertionError("expected 126 duplicate broad teacher variants")

    rows = [
        _decorate(row, source="t1", schema=_schema(row, "t1"))
        for row in retained_broad
    ] + [
        _decorate(row, source="t11", schema=_schema(row, "t11"))
        for row in targeted
    ]
    rows.sort(key=lambda row: row["meta"]["trace_id"])

    token_counts = _target_counts(rows)
    by_schema: dict[str, int] = defaultdict(int)
    for row, count in zip(rows, token_counts, strict=True):
        by_schema[row["meta"]["t12_schema"]] += count
    if set(by_schema) != set(SCHEMA_MASS):
        raise AssertionError(f"unexpected T1.2 schemas: {sorted(by_schema)}")
    if not math.isclose(sum(SCHEMA_MASS.values()), 1.0):
        raise AssertionError("T1.2 schema mass must sum to one")

    for row, count in zip(rows, token_counts, strict=True):
        schema = row["meta"]["t12_schema"]
        # The cookbook normalizes each datum's assistant weights to sum to one.
        # Multiplying by the row's target length first restores token-level
        # weighting; the schema scale then makes actual optimizer loss mass
        # equal the frozen allocation while keeping average row mass at one.
        row["meta"]["loss_weight"] = (
            count * len(rows) * SCHEMA_MASS[schema] / by_schema[schema]
        )
    return rows


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    ids = [str(row.get("meta", {}).get("trace_id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("trace IDs must be nonempty and unique")

    prompt_targets: dict[str, set[str]] = defaultdict(set)
    prompt_rows: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    season_counts: Counter[int] = Counter()
    tool_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        meta = row.get("meta") or {}
        tid = str(meta.get("trace_id"))
        roles = [message.get("role") for message in row.get("messages", [])]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"{tid}: invalid message roles")
            continue
        season = int(meta.get("season", -1))
        if season == 2025 or season in {2013, 2018, 2023, 2024}:
            errors.append(f"{tid}: sealed season {season}")
        source = str(meta.get("t12_source"))
        schema = str(meta.get("t12_schema"))
        source_counts[source] += 1
        schema_counts[schema] += 1
        season_counts[season] += 1
        if schema not in SCHEMA_MASS:
            errors.append(f"{tid}: unknown schema {schema}")
        weight = meta.get("loss_weight")
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            errors.append(f"{tid}: invalid loss weight")
        fingerprint = _prompt_hash(row)
        prompt_rows[fingerprint] += 1
        prompt_targets[fingerprint].add(str(row["messages"][-1]["content"]))
        if schema in {"target_tool_call", "target_post_tool_pick"}:
            tool_groups[str(meta.get("question_id"))].add(schema)
        serialized = json.dumps(row, sort_keys=True).lower()
        if any(marker in serialized for marker in ('"season": 2025', 'season 2025')):
            errors.append(f"{tid}: 2025 marker present")

    duplicate_prompts = sum(count - 1 for count in prompt_rows.values() if count > 1)
    conflicting = sum(len(targets) > 1 for targets in prompt_targets.values())
    if duplicate_prompts or conflicting:
        errors.append(
            f"prompt uniqueness failed: duplicates={duplicate_prompts}, conflicts={conflicting}"
        )
    if source_counts != {"t1": 529, "t11": 300}:
        errors.append(f"source coverage mismatch: {dict(source_counts)}")
    expected_schema_counts = {
        "broad_bust": 59,
        "broad_full_slate": 155,
        "broad_season_threshold": 160,
        "broad_weekly_h2h": 155,
        "target_calibration": 120,
        "target_draft_pick": 96,
        "target_draft_recovery": 24,
        "target_tool_call": 30,
        "target_post_tool_pick": 30,
    }
    if dict(schema_counts) != expected_schema_counts:
        errors.append(f"schema coverage mismatch: {dict(schema_counts)}")
    if len(tool_groups) != 30 or any(
        schemas != {"target_tool_call", "target_post_tool_pick"}
        for schemas in tool_groups.values()
    ):
        errors.append("tool transitions must contain paired call and post-result rows")

    if errors:
        raise ValueError("T1.2 dataset validation failed:\n- " + "\n- ".join(errors))

    concrete = [dict(row) for row in rows]
    token_counts = _target_counts(concrete)
    datums, _, _ = render_rows(
        concrete, RunConfig(run_name="t12-validation", epochs=1)
    )
    weighted: Counter[str] = Counter()
    raw: Counter[str] = Counter()
    for row, count, datum in zip(rows, token_counts, datums, strict=True):
        schema = str(row["meta"]["t12_schema"])
        raw[schema] += count
        weighted[schema] += sum(datum.loss_fn_inputs["weights"].data)
    weighted_total = sum(weighted.values())
    achieved = {schema: weighted[schema] / weighted_total for schema in sorted(weighted)}
    for schema, target in SCHEMA_MASS.items():
        if not math.isclose(achieved[schema], target, abs_tol=1e-8):
            raise ValueError(f"{schema}: weighted mass is not the frozen target")
    return {
        "valid": True,
        "rows": len(rows),
        "source_rows": dict(sorted(source_counts.items())),
        "schema_rows": dict(sorted(schema_counts.items())),
        "season_rows": dict(sorted(season_counts.items())),
        "raw_assistant_tokens": dict(sorted(raw.items())),
        "weighted_assistant_mass": dict(sorted(weighted.items())),
        "weighted_schema_fraction": achieved,
        "broad_fraction": sum(
            value for schema, value in achieved.items() if schema.startswith("broad_")
        ),
        "targeted_fraction": sum(
            value for schema, value in achieved.items() if schema.startswith("target_")
        ),
        "superseded_conflicting_t1_prompts": 120,
        "superseded_conflicting_t1_rows": 156,
        "removed_duplicate_broad_teacher_variants": 126,
        "duplicate_prompts": duplicate_prompts,
        "conflicting_targets": conflicting,
        "complete_tool_transition_groups": len(tool_groups),
        "sealed_2025_untouched": 2025 not in season_counts,
    }


def materialize() -> dict[str, Any]:
    rows = build_rows()
    validation = validate_rows(rows)
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = {
        "protocol": "fantasy-alpha-t12-broad-targeted-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_seed": DATASET_SEED,
        "corpus_path": str(CORPUS_PATH.relative_to(ROOT)),
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "source_sha256": {
            str(T1_CORPUS_PATH.relative_to(ROOT)): sha256_file(T1_CORPUS_PATH),
            str(T11_CORPUS_PATH.relative_to(ROOT)): sha256_file(T11_CORPUS_PATH),
        },
        "weighting": {
            "method": "per-example assistant-token loss scaling",
            "schema_mass": SCHEMA_MASS,
            "rationale": (
                "short JSON actions would contribute only about 3% of unweighted "
                "assistant-token loss; explicit weights freeze a 70/30 retention/correction mix"
            ),
        },
        "validation": validation,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_t12_corpus(*, exact: bool = True) -> list[dict[str, Any]]:
    if not CORPUS_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("materialize the T1.2 corpus first")
    manifest = json.loads(MANIFEST_PATH.read_text())
    if exact and sha256_file(CORPUS_PATH) != manifest.get("corpus_sha256"):
        raise ValueError("frozen T1.2 corpus SHA-256 mismatch")
    rows = [json.loads(line) for line in CORPUS_PATH.read_text().splitlines() if line.strip()]
    validate_rows(rows)
    return rows


def canary_subset(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Representative 230-row subset with twenty complete tool transitions."""
    train_rows, _ = grouped_development_split(
        rows, fraction=0.10, seed=RunConfig(run_name="t12-canary-split").seed
    )
    by_schema: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_schema[row["meta"]["t12_schema"]].append(row)
    selected: list[dict[str, Any]] = []
    for schema in (
        "broad_bust", "broad_full_slate", "broad_season_threshold",
        "broad_weekly_h2h", "target_calibration", "target_draft_pick",
        "target_draft_recovery",
    ):
        ordered = sorted(by_schema[schema], key=lambda row: row["meta"]["trace_id"])
        count = {
            "broad_bust": 25,
            "broad_full_slate": 25,
            "broad_season_threshold": 25,
            "broad_weekly_h2h": 25,
            "target_calibration": 30,
            "target_draft_pick": 40,
            "target_draft_recovery": 20,
        }[schema]
        selected.extend(ordered[:count])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for schema in ("target_tool_call", "target_post_tool_pick"):
        for row in by_schema[schema]:
            groups[row["meta"]["question_id"]].append(row)
    for key in sorted(groups)[:20]:
        selected.extend(groups[key])
    if len(selected) != 230:
        raise AssertionError("T1.2 canary must contain 230 rows")
    return sorted(selected, key=lambda row: row["meta"]["trace_id"])


def write_dataset_card(manifest: Mapping[str, Any]) -> None:
    validation = manifest["validation"]
    lines = [
        "# Fantasy Alpha T1.2 broad-plus-targeted dataset v2",
        "",
        "This development-only mixture is designed to retain broad forecast behavior while",
        "teaching short draft and tool actions. It contains 529 retained T1 traces and all",
        "300 T1.1 traces. The 120 prompt identities corrected by T1.1 correspond to 156",
        "old T1 sample rows; every old answer is superseded rather than duplicated. Another",
        "126 stochastic teacher variants are deterministically deduplicated without outcomes.",
        "",
        f"- Rows: {validation['rows']}",
        f"- SHA-256: `{manifest['corpus_sha256']}`",
        "- Supervision: last assistant response only",
        "- Weighted learning signal: 70% broad retention / 30% targeted behavior",
        "- Revision: tool-call share increased from 5% to 10% after the first canary",
        "- Tool transitions: 30 complete call → result-conditioned pick groups",
        "- Realized outcomes are not target inputs; 2025 remains sealed",
        "",
        "## Schema allocation",
        "",
        "| schema | rows | weighted assistant-token share |",
        "|---|---:|---:|",
    ]
    for schema, share in validation["weighted_schema_fraction"].items():
        lines.append(f"| {schema} | {validation['schema_rows'][schema]} | {share:.1%} |")
    lines += [
        "",
        "## Important limitation",
        "",
        "The draft labels imitate an outcome-blind rule policy. They can teach valid tool",
        "use, legality, and roster discipline, but they do not prove superior player selection.",
        "The frozen development evaluation must determine whether those behaviors transfer.",
    ]
    DATASET_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_CARD_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("materialize", "validate"))
    args = parser.parse_args()
    manifest = materialize() if args.command == "materialize" else json.loads(
        MANIFEST_PATH.read_text()
    )
    rows = load_t12_corpus()
    validation = validate_rows(rows)
    if args.command == "materialize":
        write_dataset_card(manifest)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
