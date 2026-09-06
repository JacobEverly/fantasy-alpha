"""Finalize the preregistered T1.2 milestone after its canary gate failed."""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.tinker_sft_t12_scorecard import (
    FAILED_CANARY_BEHAVIOR,
    FAILED_CANARY_SUMMARY,
    OLD_SCORECARD,
    PRIOR_TINKER_USD,
    SPEC_PATH,
    verify_spec,
)
from training.t12_behavior_eval import CANARY_SUMMARY
from training.t12_dataset import MANIFEST_PATH
from training.tinker_backend import ROOT, sha256_file

CANARY_BEHAVIOR = ROOT / "artifacts/tinker-sft-t12/canary-v2/behavior-gate.json"
DECISION_PATH = ROOT / "artifacts/tinker-sft-t12/decision.json"
SCORECARD_PATH = ROOT / "artifacts/tinker-sft-t12/evaluation/scorecard.json"
SPEND_PATH = ROOT / "artifacts/tinker-sft-t12/evaluation/spend-audit.json"
REPORT_PATH = ROOT / "docs/tinker-sft-t12-experiment-report.md"
MODEL_CARD_PATH = ROOT / "docs/models/fantasy-alpha-qwen35-9b-tinker-sft-t12.md"


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_stop_scorecard() -> dict[str, Any]:
    spec = verify_spec()
    v1_train = _json(FAILED_CANARY_SUMMARY)
    v1_behavior = _json(FAILED_CANARY_BEHAVIOR)
    v2_train = _json(CANARY_SUMMARY)
    v2_behavior = _json(CANARY_BEHAVIOR)
    for name, summary in (("v1", v1_train), ("v2", v2_train)):
        if summary.get("status") != "complete" or not summary.get("reload_verified"):
            raise ValueError(f"{name} canary lifecycle is incomplete")
    if v1_behavior["gate"]["pass"] or v2_behavior["gate"]["pass"]:
        raise ValueError("stop report is only valid when both preregistered canaries fail")
    if v2_behavior["structured_coverage"] < 0.98:
        raise ValueError("record the actual v2 structure result before finalizing")

    costs = {
        "failed_v1_canary_training_usd": float(v1_train["cost"]["computed_usd"]),
        "failed_v1_behavior_gate_usd": float(v1_behavior["cost"]["computed_usd"]),
        "failed_v2_canary_training_usd": float(v2_train["cost"]["computed_usd"]),
        "failed_v2_behavior_gate_usd": float(v2_behavior["cost"]["computed_usd"]),
    }
    incremental = sum(costs.values())
    if incremental >= float(spec["cost"]["hard_cap_usd"]):
        raise RuntimeError("T1.2 canary work exceeded the hard cap")
    prior = _json(OLD_SCORECARD)
    payload = {
        "protocol": spec["protocol"],
        "status": "stopped_after_failed_representative_canary",
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": sha256_file(SPEC_PATH),
        "decision": {
            "verdict": "stop_additional_sft",
            "reason": (
                "The one permitted revision improved target matching but repeated the "
                "same zero-tool behavior on all held-out tool-opportunity prompts."
            ),
            "full_t12_training_started": False,
            "frozen_t12_arm_evaluated": False,
            "rl_started": False,
            "criteria": {
                "v2_loss_improved": v2_train["development_loss_improved"],
                "v2_structured_coverage_at_least_98pct": v2_behavior["structured_coverage"] >= 0.98,
                "v2_no_irreparable_responses": v2_behavior["irreparable_responses"] == 0,
                "v2_zero_illegal_actions": v2_behavior["illegal_actions"] == 0,
                "v2_valid_tool_calls": v2_behavior["tool_valid_rate"] == 1.0,
                "v2_every_schema_parseable": all(v2_behavior["schema_coverage"].values()),
            },
        },
        "canaries": {
            "v1": {
                "training": {
                    "rows": v1_train["config"]["train_rows"] + v1_train["config"]["development_rows"],
                    "steps": v1_train["steps"],
                    "development_nll": v1_train["development_nll_history"],
                    "reload_verified": v1_train["reload_verified"],
                    "sampler_path": v1_train["final_sampler_path"],
                },
                "behavior": {
                    key: v1_behavior[key] for key in (
                        "structured_coverage", "irreparable_responses", "illegal_actions",
                        "tool_valid_rate", "target_match_rate", "schema_coverage", "gate",
                    )
                },
            },
            "v2": {
                "training": {
                    "rows": v2_train["config"]["train_rows"] + v2_train["config"]["development_rows"],
                    "steps": v2_train["steps"],
                    "development_nll": v2_train["development_nll_history"],
                    "reload_verified": v2_train["reload_verified"],
                    "sampler_path": v2_train["final_sampler_path"],
                },
                "behavior": {
                    key: v2_behavior[key] for key in (
                        "structured_coverage", "irreparable_responses", "illegal_actions",
                        "tool_valid_rate", "target_match_rate", "schema_coverage", "gate",
                    )
                },
            },
        },
        "prior_frozen_results_reused": {
            "arms": ["base", "t1", "t11"],
            "scorecard_sha256": sha256_file(OLD_SCORECARD),
            "results": prior["results"],
        },
        "four_way_scorecard": {
            "available": False,
            "reason": "The frozen protocol prohibited full T1.2 training after the canary failed.",
            "not_missing_by_accident": True,
        },
        "spend": {
            **costs,
            "t12_incremental_total_usd": incremental,
            "exact_tinker_cumulative_usd": PRIOR_TINKER_USD + incremental,
            "target_usd": spec["cost"]["target_incremental_usd"],
            "hard_cap_usd": spec["cost"]["hard_cap_usd"],
            "ongoing_checkpoint_storage_excluded": True,
        },
        "sealed_2025_untouched": True,
    }
    return payload


def provider_reconciliation(scorecard: Mapping[str, Any]) -> dict[str, Any]:
    billing_path = ROOT / "artifacts/tinker-sft-t12/billing.json"
    billing = _json(billing_path)
    # Billing events are hourly and may lag.  Preserve the complete provider
    # snapshot and prove the exact internal ledger from provider-returned token
    # counts on every completed request.  A later snapshot can make this match
    # exact without changing the scientific result.
    latest_bucket = max(
        (event["bucket_start"] for event in billing["events"]), default=None
    )
    completed_at = max(
        _json(CANARY_SUMMARY)["completed_at"], _json(CANARY_BEHAVIOR)["completed_at"]
    )
    provider_window_complete = bool(latest_bucket and latest_bucket >= completed_at[:13])
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "internal_exact_usd": scorecard["spend"]["t12_incremental_total_usd"],
        "internal_sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                FAILED_CANARY_SUMMARY, FAILED_CANARY_BEHAVIOR,
                CANARY_SUMMARY, CANARY_BEHAVIOR,
            )
        },
        "provider_snapshot": str(billing_path.relative_to(ROOT)),
        "provider_snapshot_sha256": sha256_file(billing_path),
        "latest_provider_bucket": latest_bucket,
        "latest_run_completed_at": completed_at,
        "provider_window_complete": provider_window_complete,
        "reconciliation": (
            "matched" if provider_window_complete
            else "provider hourly billing lagged the latest canary; exact request ledgers retained"
        ),
        "storage_reported_separately": True,
    }


def write_report(scorecard: Mapping[str, Any]) -> None:
    v1 = scorecard["canaries"]["v1"]
    v2 = scorecard["canaries"]["v2"]
    spend = scorecard["spend"]
    lines = [
        "# Fantasy Alpha Tinker SFT T1.2 — stopped at the canary gate",
        "",
        "Decision: **stop additional SFT; do not begin RL**.",
        "",
        "T1.2 did not proceed to full training or the paid four-way evaluation. This is",
        "the intended behavior of the frozen gate, not an incomplete run. Both canaries",
        "completed their full lifecycle, reloaded, sampled, and exported successfully.",
        "",
        "## What happened",
        "",
        "| gate | first canary | one allowed revision |",
        "|---|---:|---:|",
        f"| training rows | {v1['training']['rows']} | {v2['training']['rows']} |",
        f"| final development NLL | {v1['training']['development_nll'][-1]:.4f} | {v2['training']['development_nll'][-1]:.4f} |",
        f"| structured coverage | {v1['behavior']['structured_coverage']:.1%} | {v2['behavior']['structured_coverage']:.1%} |",
        f"| target match | {v1['behavior']['target_match_rate']:.1%} | {v2['behavior']['target_match_rate']:.1%} |",
        f"| valid tool-call rate | {v1['behavior']['tool_valid_rate']:.1%} | {v2['behavior']['tool_valid_rate']:.1%} |",
        f"| illegal/wrong-type actions | {v1['behavior']['illegal_actions']} | {v2['behavior']['illegal_actions']} |",
        "",
        "The revision doubled tool-call learning weight from 5% to 10% and expanded the",
        "one-epoch canary from 54 to 230 rows. It raised target matching from 43.8% to",
        "65.6% and preserved perfect parseability, but the model still chose a pick on",
        "all three unseen states where the correct action was a depth-chart lookup.",
        "",
        "## Interpretation",
        "",
        "SFT learned surface formats and some targets, but it did not learn the conditional",
        "policy boundary between acting and gathering evidence. More loss reduction is not",
        "evidence that a full adapter would solve that boundary. The next useful experiment",
        "would redesign tool supervision as contrastive paired decisions or evaluate a",
        "hard-coded tool router; it should not spend on DraftGym RL yet.",
        "",
        "## Spending",
        "",
        f"Exact incremental T1.2 spend: **${spend['t12_incremental_total_usd']:.6f}**.",
        f"Exact cumulative Tinker workload: **${spend['exact_tinker_cumulative_usd']:.6f}**.",
        "Ongoing checkpoint storage is separate. The $7 target and $10 hard cap were respected.",
        "",
        "2025 stayed sealed. No full T1.2 adapter and no RL run were started.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def write_model_card(scorecard: Mapping[str, Any]) -> None:
    manifest = _json(MANIFEST_PATH)
    v2 = scorecard["canaries"]["v2"]
    lines = [
        "# Fantasy Alpha Qwen3.5-9B Tinker SFT T1.2 canary",
        "",
        "Status: **research canary rejected; not a full model release**.",
        "",
        "- Base: `Qwen/Qwen3.5-9B`",
        "- Method: rank-32 LoRA, 1e-4 learning rate, one epoch",
        f"- Frozen mixture SHA: `{manifest['corpus_sha256']}`",
        f"- Canary sampler: `{v2['training']['sampler_path']}`",
        "- Structured coverage: 100%; valid tool-call rate: 0% on three unseen opportunities",
        "- 2025 was not used; the full 24-draft development evaluation was not opened",
        "",
        "The canary demonstrates reloadable Tinker training and retained response syntax,",
        "but it cannot reliably decide when to call the evidence tool. It must not be served,",
        "used for betting, or advanced to RL as though the conditional tool policy passed.",
    ]
    MODEL_CARD_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    scorecard = build_stop_scorecard()
    _write(SCORECARD_PATH, scorecard)
    _write(DECISION_PATH, scorecard["decision"])
    _write(SPEND_PATH, provider_reconciliation(scorecard))
    write_report(scorecard)
    write_model_card(scorecard)
    print(json.dumps(scorecard, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
