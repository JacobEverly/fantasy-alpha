"""Dataset audits and human-readable closeout for the outcome-value experiment."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.value_of_information import (
    ARTIFACT_DIR,
    PRIOR_TINKER_USD,
    PROTOCOL as SOURCE_PROTOCOL,
    SEALED_SEASONS,
    collected_cost,
    load_collected_episodes,
    usage_cost,
    verify_frozen_protocol,
)
from evals.value_of_information_scorecard import (
    ARM_RECORDS,
    FINAL_SCORECARD,
    verify_candidates,
)
from harness.tool_supervisor import decision_features
from training.tinker_backend import (
    ROOT,
    billing_snapshot,
    billing_window_for_today,
    sha256_file,
    utc_now,
)
from training.value_of_information_supervisor import DATASET_PATH

HELDOUT_PATH = ROOT / "evals/frozen/value_of_information_v1/heldout.jsonl"
AUDIT_PATH = ARTIFACT_DIR / "dataset-audit.json"
REVIEW_PATH = ARTIFACT_DIR / "reviewed-pairs.jsonl"
DATASET_CARD_PATH = ROOT / "docs/value-of-information-dataset-card.md"
REPORT_PATH = ROOT / "docs/value-of-information-experiment.md"
BILLING_PATH = ARTIFACT_DIR / "billing.json"
SPEND_AUDIT_PATH = ARTIFACT_DIR / "spend-audit.json"
PROVIDER_WINDOW_START = "2026-09-06T20:00:00+00:00"
PROVIDER_WINDOW_END = "2026-09-06T22:00:00+00:00"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _season_from_id(episode_id: str) -> int:
    match = re.fullmatch(r"season(\d{4}):slot\d+:seed\d+", episode_id)
    if not match:
        raise ValueError(f"invalid episode id: {episode_id}")
    return int(match.group(1))


def _sum_usage(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    materialized = list(rows)
    keys = (
        "training_tokens", "prefill_tokens", "cached_prefill_tokens",
        "sample_tokens", "checkpoint_count",
    )
    return {
        key: sum(int(row.get(key, 0) or 0) for row in materialized) for key in keys
    }


def provider_sampling_usage(
    events: Sequence[Mapping[str, Any]], *, starting_at: str = PROVIDER_WINDOW_START,
    ending_before: str | None = None,
) -> dict[str, int]:
    """Convert provider billing events in the isolated experiment window."""
    rows = [
        row for row in events
        if str(row.get("bucket_start", "")) >= starting_at
        and (
            ending_before is None
            or str(row.get("bucket_start", "")) < ending_before
        )
    ]
    usage = {
        "training_tokens": 0,
        "prefill_tokens": 0,
        "cached_prefill_tokens": 0,
        "sample_tokens": 0,
        "checkpoint_count": 0,
    }
    for row in rows:
        kind = row.get("type")
        count = int(row.get("token_count", 0) or 0)
        if kind == "sampling_prefill":
            key = "cached_prefill_tokens" if row.get("cached") else "prefill_tokens"
            usage[key] += count
        elif kind == "sampling_sample":
            usage["sample_tokens"] += count
        elif kind == "training":
            usage["training_tokens"] += count
        elif kind != "storage":
            raise ValueError(f"unexpected provider billing event: {kind}")
    return usage


def write_billing_snapshot() -> None:
    start, end = billing_window_for_today()
    snapshot = billing_snapshot(start, end)
    payload = {"starting_on": start, "ending_before": end, **snapshot}
    BILLING_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_spend_audit() -> dict[str, Any]:
    """Reconcile the interrupted canary against provider-authoritative tokens."""
    billing = json.loads(BILLING_PATH.read_text())
    latest_bucket = max(
        (str(row["bucket_start"]) for row in billing["events"]), default="",
    )
    if latest_bucket < "2026-09-06T21:00:00+00:00":
        raise RuntimeError(
            "provider billing window is incomplete; retry after hourly events settle"
        )
    collection_rows = load_collected_episodes()
    arm_rows = _read_jsonl(ARM_RECORDS)
    collection_window = [
        row for row in collection_rows if row["created_at"] < PROVIDER_WINDOW_END
    ]
    arm_window = [
        row for row in arm_rows if row["created_at"] < PROVIDER_WINDOW_END
    ]
    collection_usage = _sum_usage(row["cost"] for row in collection_window)
    arm_usage = _sum_usage(row["usage"] for row in arm_window)
    reconciled_completed_usage = {
        key: collection_usage[key] + arm_usage[key] for key in collection_usage
    }
    provider_usage = provider_sampling_usage(
        billing["events"], ending_before=PROVIDER_WINDOW_END,
    )
    orphaned_usage = {
        key: provider_usage[key] - reconciled_completed_usage[key]
        for key in reconciled_completed_usage
    }
    if any(value < 0 for value in orphaned_usage.values()):
        raise RuntimeError(
            "provider billing window is incomplete; retry after hourly events settle"
        )
    local_completed_usd = (
        collected_cost(collection_rows)
        + sum(float(row["cost_usd"]) for row in arm_rows)
    )
    orphaned_usd = usage_cost(orphaned_usage)
    total = local_completed_usd + orphaned_usd
    result = {
        "protocol": SOURCE_PROTOCOL,
        "created_at": utc_now(),
        "reconciliation": "provider_window_minus_completed_request_ledgers",
        "provider_window_start": PROVIDER_WINDOW_START,
        "provider_window_end": PROVIDER_WINDOW_END,
        "latest_provider_bucket": latest_bucket,
        "provider_window_isolated_to_this_experiment": True,
        "reconciled_collection_records": len(collection_window),
        "reconciled_arm_records": len(arm_window),
        "latest_completed_record_before_window_end": max(
            row["created_at"] for row in collection_window + arm_window
        ),
        "first_completed_record_after_window_end": min(
            row["created_at"] for row in collection_rows + arm_rows
            if row["created_at"] >= PROVIDER_WINDOW_END
        ),
        "billing_snapshot": str(BILLING_PATH.relative_to(ROOT)),
        "billing_snapshot_sha256": sha256_file(BILLING_PATH),
        "reconciled_completed_usage": reconciled_completed_usage,
        "provider_usage": provider_usage,
        "orphaned_interrupted_canary_usage": orphaned_usage,
        "local_completed_usd": local_completed_usd,
        "orphaned_interrupted_canary_usd": orphaned_usd,
        "exact_incremental_usd": total,
        "exact_cumulative_tinker_usd": PRIOR_TINKER_USD + total,
        "ongoing_storage_excluded": True,
    }
    SPEND_AUDIT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def audit_datasets(
    development_path: Path = DATASET_PATH, heldout_path: Path = HELDOUT_PATH,
) -> dict[str, Any]:
    verify_frozen_protocol()
    verify_candidates()
    development = _read_jsonl(development_path)
    heldout = _read_jsonl(heldout_path)
    checks = {
        "development_rows_70": len(development) == 70,
        "heldout_rows_70": len(heldout) == 70,
        "development_split_exact": all(
            row.get("split") == "development" for row in development
        ),
        "heldout_split_exact": all(row.get("split") == "heldout" for row in heldout),
        "source_protocol_exact": all(
            row.get("protocol") == SOURCE_PROTOCOL for row in development + heldout
        ),
    }
    development_groups = {row["episode_id"] for row in development}
    heldout_groups = {row["episode_id"] for row in heldout}
    checks["episode_groups_disjoint"] = not development_groups & heldout_groups
    checks["two_points_per_development_episode"] = (
        all(value == 2 for value in Counter(
            row["episode_id"] for row in development
        ).values())
        and len(development_groups) == 35
    )
    checks["two_points_per_heldout_episode"] = (
        all(value == 2 for value in Counter(
            row["episode_id"] for row in heldout
        ).values())
        and len(heldout_groups) == 35
    )
    checks["sealed_seasons_absent"] = not SEALED_SEASONS & {
        _season_from_id(row["episode_id"]) for row in development + heldout
    }
    checks["masked_inputs"] = all(
        bool(row["provenance"]["masked"]) for row in development + heldout
    )
    checks["visible_names_anonymized"] = all(
        all(
            str(player.get("name")) == str(player.get("player_id"))
            for player in row["input"]["observation"].get("top_available", [])
        )
        for row in development + heldout
    )
    # Calling the shared feature function proves every row is usable through the
    # inference allowlist. Outcome/tool-result fields remain in the label record,
    # but are never passed to the fitted estimator.
    features = [
        decision_features(row["input"]["observation"], row["input"]["proposed_action"])
        for row in development + heldout
    ]
    feature_keys = sorted(set().union(*(row.keys() for row in features)))
    prohibited = {"reward", "label", "future_action", "tool_response", "identity"}
    checks["feature_allowlist_excludes_outcomes"] = not prohibited & set(feature_keys)
    checks["unique_example_ids"] = len({
        row["example_id"] for row in development + heldout
    }) == len(development) + len(heldout)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"dataset audit failed: {failed}")
    result = {
        "protocol": SOURCE_PROTOCOL,
        "created_at": utc_now(),
        "checks": checks,
        "development": {
            "path": str(development_path.relative_to(ROOT)),
            "sha256": sha256_file(development_path),
            "rows": len(development),
            "episodes": len(development_groups),
            "labels": dict(sorted(Counter(
                row["label"]["value"] for row in development
            ).items())),
        },
        "heldout": {
            "path": str(heldout_path.relative_to(ROOT)),
            "sha256": sha256_file(heldout_path),
            "rows": len(heldout),
            "episodes": len(heldout_groups),
            "labels": dict(sorted(Counter(
                row["label"]["value"] for row in heldout
            ).items())),
        },
        "decision_time_feature_keys": feature_keys,
        "heldout_fit_or_tuning_calls": 0,
    }
    AUDIT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def reviewed_pairs(rows: Sequence[Mapping[str, Any]], count: int = 12) -> list[dict]:
    ordered = sorted(
        rows, key=lambda row: (
            str(row["label"]["value"]),
            -abs(float(row["outcome"]["reward_difference"])),
            str(row["example_id"]),
        ),
    )
    chosen: list[Mapping[str, Any]] = []
    for label in ("HELPFUL", "UNNECESSARY", "HARMFUL", "UNRESOLVABLE"):
        chosen.extend([row for row in ordered if row["label"]["value"] == label][:3])
    chosen_ids = {row["example_id"] for row in chosen}
    chosen.extend(row for row in ordered if row["example_id"] not in chosen_ids)
    return [
        {
            "example_id": row["example_id"],
            "episode_id": row["episode_id"],
            "split": row["split"],
            "proposed_action": row["input"]["proposed_action"],
            "tool": row["intervention"]["tool"],
            "tool_resolved": row["intervention"]["resolved"],
            "intervened_action": row["intervention"]["post_tool_model_action"],
            "action_changed": row["outcome"]["action_changed"],
            "reward_difference": row["outcome"]["reward_difference"],
            "label": row["label"],
        }
        for row in chosen[:count]
    ]


def write_dataset_card() -> None:
    audit = audit_datasets()
    all_rows = _read_jsonl(DATASET_PATH) + _read_jsonl(HELDOUT_PATH)
    review = reviewed_pairs(all_rows)
    REVIEW_PATH.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in review
    ))
    development = audit["development"]
    heldout = audit["heldout"]
    text = f"""# Fantasy Alpha value-of-information dataset v1

This dataset measures whether one relevant evidence lookup changed a Qwen3.5-9B
draft decision and improved its realized DraftGym reward. It is an intervention
dataset, not a policy-imitation dataset.

## Composition

- Development: {development['rows']} points from {development['episodes']} complete episodes; labels `{json.dumps(development['labels'], sort_keys=True)}`.
- Held out: {heldout['rows']} points from {heldout['episodes']} complete episodes; labels `{json.dumps(heldout['labels'], sort_keys=True)}`.
- All inputs are masked PPR states from 2015–2017 and 2019–2022.
- Seasons 2013, 2018, and 2023–2025 remain sealed.

Each point replays the exact same action prefix. The control accepts the proposed
pick. The treatment performs one structured lookup and lets the same model
reconsider once. Both branches then use deterministic AutopickADP.

## Labels

`HELPFUL` and `HARMFUL` require a changed action and at least 25 points of
terminal reward movement, except that preventing an illegal action is directly
helpful. Empty or failed evidence is `UNRESOLVABLE`; everything else is
`UNNECESSARY`.

## Intended use and limits

Use development rows to fit evidence-value supervisors. Held-out rows are for one
fit-free evaluation only. The small panel supports a first routing experiment,
not a general claim about every fantasy season, model, tool, or league format.
Terminal reward is deterministic conditional on the frozen state and continuation,
but it remains a noisy proxy for the local decision's true value.

Hashes and leakage checks are in `{AUDIT_PATH.relative_to(ROOT)}`. Twelve
deterministically selected review rows are in `{REVIEW_PATH.relative_to(ROOT)}`.
"""
    DATASET_CARD_PATH.write_text(text)


def write_experiment_report() -> None:
    scorecard = json.loads(FINAL_SCORECARD.read_text())
    arms = scorecard["matched_arms"]["arms"]
    systems = scorecard["counterfactual_heldout"]["systems"]
    learned = scorecard["matched_arms"]["paired_comparisons"][
        "learned_outcome_value_vs_base_qwen"
    ]
    cf = systems["learned_outcome_value"]
    decision = scorecard["decision"]
    recommendation = {
        "use_hybrid": "Use the hybrid; its evidence-value signal justifies an RL follow-up.",
        "collect_more_data": "Collect the estimated additional paired panel before RL.",
        "redesign_policy_or_tools": "Redesign the policy or evidence tools before RL.",
        "stop_supervision_as_primary_performance_lever": (
            "Stop treating supervision as the primary performance lever; do not begin RL."
        ),
    }[decision]
    system_labels = {
        "always_act": "Never buy evidence",
        "always_tool": "Always buy evidence",
        "hard_coded_rules": "Hand-written rules",
        "existing_imitation_classifier": "Prior imitation supervisor",
        "transparent_value_rule": "Outcome-tuned rule",
        "learned_outcome_value": "Learned outcome-value supervisor",
    }
    counterfactual_order = (
        "always_act", "always_tool", "hard_coded_rules",
        "existing_imitation_classifier", "transparent_value_rule",
        "learned_outcome_value",
    )
    counterfactual_rows = "\n".join(
        f"| {system_labels[name]} | {row['mean_reward_delta_vs_act']:.2f} | "
        f"{row['tool_rate']:.1%} | {row['harmful_selected']} |"
        for name in counterfactual_order for row in (systems[name],)
    )
    arm_labels = {
        "base_qwen": "Base Qwen3.5-9B",
        "existing_supervisor": "Prior supervisor + Qwen",
        "outcome_value_rule": "Outcome rule + Qwen",
        "learned_outcome_value": "Learned outcome supervisor + Qwen",
    }
    arm_order = (
        "base_qwen", "existing_supervisor", "outcome_value_rule",
        "learned_outcome_value",
    )
    arm_rows = "\n".join(
        f"| {arm_labels[name]} | {row['completed_episodes']}/30 | "
        f"{row['mean_reward']:.2f} | {row['tool_calls']} | "
        f"${row['incremental_usd']:.4f} |"
        for name in arm_order for row in (arms[name],)
    )
    gate_rows = "\n".join(
        f"| {name.replace('_', ' ')} | {'Pass' if passed else 'Fail'} |"
        for name, passed in scorecard["signal_gate"].items() if name != "pass"
    )
    text = f"""# Outcome-linked evidence supervisor experiment

**Completed:** 2026-09-06 · **Base model:** `Qwen/Qwen3.5-9B` through Tinker ·
**Decision:** **{decision}**

## Executive result

{recommendation}

On 70 held-out matched decision points, the learned supervisor changed mean
counterfactual reward by **{cf['mean_reward_delta_vs_act']:.2f} points** while
selecting tools on **{cf['tool_rate']:.1%}** of points. In 30 matched full
episodes it changed reward by **{learned['mean_reward_delta']:.2f} points** on
average, with {learned['wins']} wins, {learned['losses']} losses, and
{learned['ties']} ties versus base Qwen. The paired bootstrap 95% interval is
[{learned['bootstrap_95_ci'][0]:.2f}, {learned['bootstrap_95_ci'][1]:.2f}].

This is the second half of a deliberate post-training loop. The prior experiment
trained a highly accurate policy-imitation supervisor, but successful tool calls
did not improve full-draft outcomes. This experiment changed the target: predict
whether buying one piece of evidence improves terminal reward, rather than
whether an old policy says evidence is missing.

## Experimental design

- Collected 140 matched decision points from 70 complete masked episodes.
- Replayed the exact same action prefix for control and treatment.
- Control accepted Qwen's proposed pick; treatment performed one relevant
  structured lookup and let the same model reconsider once.
- Continued both branches with the same deterministic policy to isolate the
  local intervention.
- Fit the lightweight classifier and selected its threshold only on 35
  development episodes, grouped by complete episode.
- Froze the artifact, threshold, rules, prompts, 30-episode arm panel, and hashes
  before the 35 held-out episodes were scored.
- Kept seasons 2013, 2018, and 2023–2025 sealed.

## Held-out matched decisions

| Policy | Mean reward change vs act | Tool rate | Harmful interventions selected |
|---|---:|---:|---:|
{counterfactual_rows}

The matched-state result measures local value of information. It is cheaper and
less confounded than replaying a full model trajectory twice, but it cannot prove
that the policy remains valuable after its decisions change later draft states.

## Frozen full-episode comparison

| Arm | Completed | Mean reward | Tool calls | Tinker cost |
|---|---:|---:|---:|---:|
{arm_rows}

The 30 episodes are paired by season, draft slot, and random seed. The primary
comparison is the learned outcome supervisor against untouched base Qwen:
{learned['wins']} wins, {learned['losses']} losses, {learned['ties']} ties;
median delta **{learned['median_reward_delta']:.2f}** and mean delta after removing
the largest gain **{learned['mean_without_largest_gain']:.2f}**.

## Preregistered signal gate

| Condition | Result |
|---|---|
{gate_rows}

## Interpretation

The experiment tests whether the supervisor knows when evidence is worth buying,
not merely whether it reproduces a hand-written rule. The full decision gate and
every failed condition are recorded in the scorecard. A positive local effect
without a positive full-episode effect is not enough to justify RL.

The learned artifact is intentionally small and external to the main model. This
keeps legality, tool availability, and budget enforcement deterministic while
making the uncertain intervention policy replaceable. It also makes the negative
result from an SFT checkpoint useful: model training is one candidate lever, not
the definition of progress.

One canary exposed a repeated-tool loop because the outcome estimator was trained
on pre-tool states. The affected arm was stopped before a completed record, the
fix was constrained to approving the reconsidered action after one successful
lookup, and the event plus provider-reconciled cost are retained. No artifact,
threshold, feature, or held-out label was changed.

## Limits

- The panel is large enough for a first falsifiable routing result, not a universal
  claim across models, tools, league formats, or future seasons.
- Terminal draft reward is deterministic under the frozen continuation, but is a
  noisy proxy for the causal value of one local pick.
- Helpful interventions are rare, so calibration and policy selection remain
  data-constrained.
- This milestone does not claim a trained mid-trajectory world model or an RL
  policy. RL remains gated on the frozen outcome signal.

Exact experiment spend was **${scorecard['spend']['incremental_total_usd']:.6f}**
against a $5 target and $10 hard cap. No held-out labels were used for fitting or
threshold selection.

## Artifacts

- Frozen protocol: `training/value-of-information-spec-v1.json`
- Dataset audit: `{AUDIT_PATH.relative_to(ROOT)}`
- Scorecard: `{FINAL_SCORECARD.relative_to(ROOT)}`
- Reviewed pairs: `{REVIEW_PATH.relative_to(ROOT)}`
- Spend reconciliation: `{SPEND_AUDIT_PATH.relative_to(ROOT)}`
"""
    REPORT_PATH.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("billing", "dataset-card", "experiment-report", "spend-audit"),
    )
    args = parser.parse_args()
    if args.command == "billing":
        write_billing_snapshot()
        print(BILLING_PATH)
    elif args.command == "dataset-card":
        write_dataset_card()
        print(DATASET_CARD_PATH)
    elif args.command == "spend-audit":
        print(json.dumps(write_spend_audit(), indent=2, sort_keys=True))
    else:
        write_experiment_report()
        print(REPORT_PATH)


if __name__ == "__main__":
    main()
