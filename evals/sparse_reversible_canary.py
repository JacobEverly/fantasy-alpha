"""Frozen paired full-trajectory canary for one reversible intervention."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from envs.play_llm_reversible import run_reversible_episode
from harness.sparse_reversible import SparseReversibleController
from training.tinker_backend import (
    BASE_MODEL,
    PREFILL_USD_PER_MTOK,
    ROOT,
    SAMPLE_USD_PER_MTOK,
    TinkerChatSampler,
    require_api_key,
    sha256_file,
    utc_now,
)

PROTOCOL = "fantasy-alpha-sparse-reversible-canary-v1"
SEALED_SEASONS = frozenset({2013, 2018, 2023, 2024, 2025})
DEVELOPMENT_SEASONS = (2015, 2016, 2017, 2019, 2020, 2021, 2022)
SPEC_PATH = ROOT / "training/sparse-reversible-canary-v1.json"
SPEC_SHA_PATH = ROOT / "training/sparse-reversible-canary-v1.sha256"
ARTIFACT_DIR = ROOT / "artifacts/sparse-reversible-canary-v1"
RECORDS_PATH = ARTIFACT_DIR / "matched-records.jsonl"
SCORECARD_PATH = ARTIFACT_DIR / "scorecard.json"
TARGET_USD = 2.0
HARD_CAP_USD = 5.0
CONTROLLER_CONFIG = {
    "relative_adp_stdev_threshold": 0.10,
    "first_round": 1,
    "last_round": 4,
    "max_adp_reach": 18.0,
}


def evaluation_panel() -> list[dict[str, Any]]:
    slots = (1, 4, 7, 10)
    seeds = (211, 223, 227, 229)
    rows = []
    for season in DEVELOPMENT_SEASONS:
        for slot, seed in zip(slots, seeds, strict=True):
            rows.append(
                {
                    "episode_id": f"season{season}:slot{slot}:seed{seed}",
                    "season": season,
                    "preset": "ppr",
                    "agent_slot": slot,
                    "seed": seed,
                    "mask_names": True,
                    "enable_evidence": True,
                }
            )
    rows.extend(
        [
            {
                "episode_id": "season2015:slot12:seed233",
                "season": 2015,
                "preset": "ppr",
                "agent_slot": 12,
                "seed": 233,
                "mask_names": True,
                "enable_evidence": True,
            },
            {
                "episode_id": "season2022:slot12:seed239",
                "season": 2022,
                "preset": "ppr",
                "agent_slot": 12,
                "seed": 239,
                "mask_names": True,
                "enable_evidence": True,
            },
        ]
    )
    if len(rows) != 30 or len({row["episode_id"] for row in rows}) != 30:
        raise AssertionError("canary panel must contain 30 unique episodes")
    if any(int(row["season"]) in SEALED_SEASONS for row in rows):
        raise AssertionError("canary panel includes a sealed season")
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def freeze_protocol() -> dict[str, Any]:
    if SPEC_PATH.exists() or SPEC_SHA_PATH.exists():
        raise RuntimeError("refusing to overwrite an existing frozen protocol")
    source_files = (
        "envs/play_llm_reversible.py",
        "harness/sparse_reversible.py",
        "envs/draftgym.py",
        "envs/play_llm.py",
        "training/tinker_backend.py",
    )
    payload = {
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "frozen_before_paid_outcomes": True,
        "research_question": "Can one conservative, reversible evidence intervention improve complete DraftGym reward over untouched base Qwen?",
        "model": {
            "name": BASE_MODEL,
            "temperature": 0.0,
            "max_tokens": 700,
            "renderer": "qwen3_5_disable_thinking",
        },
        "controller": CONTROLLER_CONFIG,
        "controller_rationale": "development-only transparent rule: relative ADP uncertainty >=10% had positive local value with zero harmful selections; restrict to rounds 1-4, one lookup, and revert unless evidence is adverse",
        "panel": evaluation_panel(),
        "staging": {
            "stage_one_pairs": 8,
            "early_kill": "stop only for an invariant failure, or after 8 pairs when there are at least 3 critical regressions, zero critical improvements, and paired mean delta <= -25",
        },
        "success_gates": {
            "paired_mean_positive": True,
            "ten_percent_trimmed_mean_positive": True,
            "wins_exceed_losses": True,
            "positive_without_largest_gain": True,
            "critical_improvements_at_least_regressions": True,
            "zero_illegal_accepted_actions": True,
            "max_one_intervention": True,
        },
        "claims": "a positive result is a development canary for the inference-time controller, not proof that Qwen weights improved or that the result generalizes",
        "budget": {"target_usd": TARGET_USD, "hard_cap_usd": HARD_CAP_USD},
        "sealed_seasons": sorted(SEALED_SEASONS),
        "implementation_hashes": {
            path: sha256_file(ROOT / path) for path in source_files
        },
    }
    SPEC_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = sha256_file(SPEC_PATH)
    SPEC_SHA_PATH.write_text(f"{digest}  {SPEC_PATH.name}\n")
    return payload


def verify_protocol() -> dict[str, Any]:
    expected = SPEC_SHA_PATH.read_text().split()[0]
    if sha256_file(SPEC_PATH) != expected:
        raise ValueError("sparse reversible protocol SHA-256 mismatch")
    payload = json.loads(SPEC_PATH.read_text())
    if payload.get("protocol") != PROTOCOL or not payload.get(
        "frozen_before_paid_outcomes"
    ):
        raise ValueError("unexpected or unfrozen sparse reversible protocol")
    if payload.get("panel") != evaluation_panel():
        raise ValueError("canary panel differs from frozen protocol")
    if payload.get("controller") != CONTROLLER_CONFIG:
        raise ValueError("controller config differs from frozen protocol")
    for relative, expected_hash in payload["implementation_hashes"].items():
        if sha256_file(ROOT / relative) != expected_hash:
            raise ValueError(f"frozen implementation changed: {relative}")
    return payload


def usage_cost(usage: Mapping[str, int]) -> float:
    return (
        int(usage.get("prefill_tokens", 0)) * PREFILL_USD_PER_MTOK
        + int(usage.get("cached_prefill_tokens", 0)) * 0.132
        + int(usage.get("sample_tokens", 0)) * SAMPLE_USD_PER_MTOK
    ) / 1_000_000


class BudgetedSampler:
    def __init__(self, sampler: Any, *, prior_usd: float):
        self.sampler = sampler
        self.renderer = sampler.renderer
        self.ledger = sampler.ledger
        self.prior_usd = float(prior_usd)

    def chat(self, messages, *, max_tokens, temperature, seed):
        prompt = self.renderer.build_generation_prompt(messages)
        reserve = (
            prompt.length * PREFILL_USD_PER_MTOK + max_tokens * SAMPLE_USD_PER_MTOK
        ) / 1_000_000
        if self.prior_usd + self.ledger.usd + reserve > HARD_CAP_USD:
            raise RuntimeError("next request could exceed the frozen $5 hard cap")
        return self.sampler.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )


def _completed_cost(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(row["cost_usd"]) for row in rows)


def _early_kill(rows: Sequence[Mapping[str, Any]]) -> bool:
    by_key = {(row["arm"], row["episode_id"]): row for row in rows}
    ids = [row["episode_id"] for row in evaluation_panel()[:8]]
    if not all(
        (arm, episode_id) in by_key
        for episode_id in ids
        for arm in ("base", "treatment")
    ):
        return False
    deltas = [
        float(by_key[("treatment", key)]["outcome"]["reward"])
        - float(by_key[("base", key)]["outcome"]["reward"])
        for key in ids
    ]
    return (
        sum(delta <= -25 for delta in deltas) >= 3
        and sum(delta >= 25 for delta in deltas) == 0
        and statistics.fmean(deltas) <= -25
    )


def run_panel(*, limit_pairs: int | None = None) -> dict[str, Any]:
    verify_protocol()
    require_api_key()
    existing = _read_jsonl(RECORDS_PATH)
    keys = {(row["arm"], row["episode_id"]) for row in existing}
    if len(keys) != len(existing):
        raise ValueError("duplicate canary record")
    panel = (
        evaluation_panel()[:limit_pairs]
        if limit_pairs is not None
        else evaluation_panel()
    )
    for panel_row in panel:
        if _early_kill(existing):
            break
        spec = {
            key: panel_row[key]
            for key in (
                "season",
                "preset",
                "agent_slot",
                "seed",
                "mask_names",
                "enable_evidence",
            )
        }
        for arm in ("base", "treatment"):
            key = (arm, panel_row["episode_id"])
            if key in keys:
                continue
            prior = _completed_cost(existing)
            sampler = BudgetedSampler(
                TinkerChatSampler(model=BASE_MODEL, experiment=PROTOCOL),
                prior_usd=prior,
            )
            controller = (
                SparseReversibleController(**CONTROLLER_CONFIG)
                if arm == "treatment"
                else None
            )
            outcome = run_reversible_episode(
                spec, sampler=sampler, controller=controller
            )
            usage = dict(outcome["usage_delta"])
            record = {
                "protocol": PROTOCOL,
                "created_at": utc_now(),
                "arm": arm,
                "episode_id": panel_row["episode_id"],
                "spec": spec,
                "outcome": outcome,
                "usage": usage,
                "cost_usd": usage_cost(usage),
            }
            _append_jsonl(RECORDS_PATH, record)
            existing.append(record)
            keys.add(key)
    return materialize_scorecard(require_complete=False)


def _bootstrap(deltas: Sequence[float]) -> list[float] | None:
    if len(deltas) < 2:
        return None
    rng = random.Random(20260906)
    samples = sorted(
        statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(10_000)
    )
    return [samples[249], samples[9749]]


def _trimmed_mean(deltas: Sequence[float]) -> float | None:
    if not deltas:
        return None
    trim = int(len(deltas) * 0.10)
    ordered = sorted(deltas)
    kept = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.fmean(kept)


def materialize_scorecard(*, require_complete: bool = True) -> dict[str, Any]:
    verify_protocol()
    rows = _read_jsonl(RECORDS_PATH)
    panel_by_id = {row["episode_id"]: row for row in evaluation_panel()}
    for row in rows:
        if row.get("protocol") != PROTOCOL or row.get("arm") not in {
            "base",
            "treatment",
        }:
            raise ValueError("unexpected canary record provenance")
        panel_row = panel_by_id.get(str(row.get("episode_id")))
        if panel_row is None:
            raise ValueError("canary record is outside the frozen panel")
        expected_spec = {
            key: panel_row[key]
            for key in (
                "season",
                "preset",
                "agent_slot",
                "seed",
                "mask_names",
                "enable_evidence",
            )
        }
        if row.get("spec") != expected_spec:
            raise ValueError("canary record spec differs from frozen panel")
        if int(expected_spec["season"]) in SEALED_SEASONS:
            raise ValueError("sealed season found in canary records")
    by_key = {(row["arm"], row["episode_id"]): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("duplicate canary record")
    matched_ids = [
        row["episode_id"]
        for row in evaluation_panel()
        if ("base", row["episode_id"]) in by_key
        and ("treatment", row["episode_id"]) in by_key
    ]
    if require_complete and len(matched_ids) != 30 and not _early_kill(rows):
        raise RuntimeError(
            "canary is incomplete and did not hit its frozen early kill gate"
        )
    deltas = [
        float(by_key[("treatment", key)]["outcome"]["reward"])
        - float(by_key[("base", key)]["outcome"]["reward"])
        for key in matched_ids
    ]
    treatment = [by_key[("treatment", key)]["outcome"] for key in matched_ids]
    matched_prefixes = True
    for key in matched_ids:
        base_outcome = by_key[("base", key)]["outcome"]
        treatment_outcome = by_key[("treatment", key)]["outcome"]
        audits = treatment_outcome["decision_audit"]
        if audits:
            prefix_length = int(audits[0]["trigger"]["round"])
            matched_prefixes = matched_prefixes and (
                base_outcome["proposed_actions"][:prefix_length]
                == treatment_outcome["proposed_actions"][:prefix_length]
            )
    without_largest = sorted(deltas)[:-1] if len(deltas) > 1 else deltas
    metrics = {
        "matched_pairs": len(matched_ids),
        "mean_delta": statistics.fmean(deltas) if deltas else None,
        "ten_percent_trimmed_mean_delta": _trimmed_mean(deltas),
        "median_delta": statistics.median(deltas) if deltas else None,
        "mean_without_largest_gain": statistics.fmean(without_largest)
        if without_largest
        else None,
        "wins": sum(delta > 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "critical_improvements": sum(delta >= 25 for delta in deltas),
        "critical_regressions": sum(delta <= -25 for delta in deltas),
        "bootstrap_95_ci": _bootstrap(deltas),
    }
    invariants = {
        "max_one_intervention": all(
            int(row["interventions"]) <= 1 for row in treatment
        ),
        "max_one_tool_call": all(int(row["tool_calls"]) <= 1 for row in treatment),
        "zero_illegal_accepted_actions": all(
            all(
                not event["accepted_revision"] or event["executed_legally"]
                for event in row["decision_audit"]
            )
            for row in treatment
        ),
        "all_completed": all(bool(row["task_completed"]) for row in treatment),
        "matched_action_prefixes": matched_prefixes,
        "sealed_seasons_absent": all(
            int(row["spec"]["season"]) not in SEALED_SEASONS for row in rows
        ),
    }
    gates = {
        "paired_mean_positive": bool(deltas and metrics["mean_delta"] > 0),
        "ten_percent_trimmed_mean_positive": bool(
            deltas and metrics["ten_percent_trimmed_mean_delta"] > 0
        ),
        "wins_exceed_losses": metrics["wins"] > metrics["losses"],
        "positive_without_largest_gain": bool(
            deltas and metrics["mean_without_largest_gain"] > 0
        ),
        "critical_improvements_at_least_regressions": metrics["critical_improvements"]
        >= metrics["critical_regressions"],
        **invariants,
    }
    complete = len(matched_ids) == 30 or _early_kill(rows)
    passed = complete and all(gates.values()) and len(matched_ids) == 30
    result = {
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "complete": complete,
        "early_kill": _early_kill(rows),
        "decision": (
            "replicate_and_generate_trajectory_preferences"
            if passed
            else "revise_or_stop_sparse_intervention"
            if complete
            else "continue_canary"
        ),
        "signal_gate": {**gates, "pass": passed},
        "paired": metrics,
        "arms": {
            arm: {
                "episodes": sum(row["arm"] == arm for row in rows),
                "mean_reward": statistics.fmean(
                    float(row["outcome"]["reward"]) for row in rows if row["arm"] == arm
                )
                if any(row["arm"] == arm for row in rows)
                else None,
                "cost_usd": sum(
                    float(row["cost_usd"]) for row in rows if row["arm"] == arm
                ),
            }
            for arm in ("base", "treatment")
        },
        "intervention_summary": {
            "interventions": sum(int(row["interventions"]) for row in treatment),
            "accepted_revisions": sum(
                int(row["accepted_revisions"]) for row in treatment
            ),
            "reverted_revisions": sum(
                int(row["reverted_revisions"]) for row in treatment
            ),
            "tool_calls": sum(int(row["tool_calls"]) for row in treatment),
        },
        "incremental_spend_usd": _completed_cost(rows),
        "spend_status": "exact request-ledger cost; provider reconciliation pending",
        "fit_or_tuning_on_canary": False,
        "episode_deltas": dict(zip(matched_ids, deltas, strict=True)),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SCORECARD_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "scorecard"))
    parser.add_argument("--limit-pairs", type=int)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_protocol()
    elif args.command == "run":
        result = run_panel(limit_pairs=args.limit_pairs)
    else:
        result = materialize_scorecard()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
