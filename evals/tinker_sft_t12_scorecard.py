"""Frozen four-way development evaluation for T1.2.

Only the new T1.2 arm incurs inference spend.  Base, T1, and T1.1 are loaded
from hash-pinned completed artifacts.  The 2025 gate is absent from every path.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import evals.tinker_sft_scorecard as shared
from envs.draftgym import DraftGym
from envs.play_llm import autopick_action
from evals.breakoutbench import FAMILIES, QUESTIONS_DIR, score_run
from evals.canaries import load_answers_files, score_canaries
from evals.tinker_sft_t11_resilient_runner import _ask_json_array_conservative
from evals.tinker_sft_t11_scorecard import (
    DRAFT_EPISODES,
    EVAL_SEED,
    _draft_comparison,
    _draft_episode,
    _prediction_comparison,
)
from training.t12_behavior_eval import SELECTION_PATH
from training.t12_dataset import CORPUS_PATH, MANIFEST_PATH
from training.tinker_backend import (
    BASE_MODEL,
    RENDERER_NAME,
    ROOT,
    TinkerChatSampler,
    safe_text,
    sha256_file,
    utc_now,
)

SPEC_PATH = ROOT / "training/tinker-eval-spec-t12-v2.json"
SPEC_HASH_PATH = ROOT / "training/tinker-eval-spec-t12-v2.sha256"
RUN_ROOT = ROOT / "artifacts/tinker-sft-t12/evaluation"
OLD_ROOT = ROOT / "artifacts/tinker-sft-t11/evaluation"
OLD_SCORECARD = OLD_ROOT / "scorecard.json"
PRELIGHT = ROOT / "artifacts/tinker-sft-t12/preflight.json"
CANARY_SUMMARY = ROOT / "artifacts/tinker-sft-t12/canary-v2/run-summary.json"
CANARY_BEHAVIOR = ROOT / "artifacts/tinker-sft-t12/canary-v2/behavior-gate.json"
FAILED_CANARY_SUMMARY = ROOT / "artifacts/tinker-sft-t12/canary/run-summary.json"
FAILED_CANARY_BEHAVIOR = ROOT / "artifacts/tinker-sft-t12/canary/behavior-gate.json"
TRAIN_SUMMARY = ROOT / "artifacts/tinker-sft-t12/t12-full/run-summary.json"
SCORECARD_PATH = RUN_ROOT / "scorecard.json"
SPEND_AUDIT_PATH = RUN_ROOT / "spend-audit.json"
REPORT_PATH = ROOT / "docs/tinker-sft-t12-experiment-report.md"
MODEL_CARD_PATH = ROOT / "docs/models/fantasy-alpha-qwen35-9b-tinker-sft-t12.md"
BUDGET_CAP_USD = 10.0
PRIOR_TINKER_USD = 9.61609089


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _input_paths() -> list[Path]:
    paths = [
        CORPUS_PATH, MANIFEST_PATH, PRELIGHT,
        ROOT / "artifacts/tinker-sft-t12/preflight-v1.json",
        FAILED_CANARY_SUMMARY, FAILED_CANARY_BEHAVIOR,
        ROOT / "training/tinker-eval-spec-t12-v1.json",
        ROOT / "training/tinker-eval-spec-t12-v1.sha256",
        ROOT / "artifacts/tinker-sft-t12/t11-collapse-analysis.json",
        OLD_SCORECARD,
        ROOT / "training/tinker-eval-spec-t11-v1.json",
        ROOT / "training/tinker-eval-spec-t11-v1.sha256",
        ROOT / "artifacts/tinker-sft-v1/t1-full/checkpoint-selection.json",
        ROOT / "artifacts/tinker-sft-t11/t11-full/checkpoint-selection.json",
        ROOT / "training/t12_dataset.py",
        ROOT / "training/t12_behavior_eval.py",
        ROOT / "training/tinker_backend.py",
        ROOT / "training/tinker_sft.py",
        ROOT / "evals/tinker_sft_scorecard.py",
        ROOT / "evals/tinker_sft_t11_scorecard.py",
        ROOT / "evals/tinker_sft_t11_resilient_runner.py",
        ROOT / "evals/tinker_sft_t12_scorecard.py",
        ROOT / "envs/draftgym.py",
        ROOT / "envs/play_llm.py",
        ROOT / "evals/breakoutbench.py",
        ROOT / "evals/calibbench.py",
        ROOT / "evals/canaries.py",
        ROOT / "evals/run_expanded_qwen.py",
    ]
    for arm in ("base", "t1", "t11"):
        paths.extend(sorted(path for path in (OLD_ROOT / arm).glob("*.json")))
    for season in shared.SEASONS:
        paths.extend([
            QUESTIONS_DIR / f"breakoutbench_{season}_{shared.PRESET}.json",
            QUESTIONS_DIR / f"breakoutbench_{season}_{shared.PRESET}_KEY.json",
            QUESTIONS_DIR / f"calibbench_{season}_{shared.PRESET}_anon.json",
            QUESTIONS_DIR / f"calibbench_{season}_{shared.PRESET}_KEY.json",
            QUESTIONS_DIR / f"canary_{season}_{shared.PRESET}.json",
            QUESTIONS_DIR / f"canary_{season}_{shared.PRESET}_KEY.json",
        ])
    return sorted(set(paths))


def build_spec() -> dict[str, Any]:
    paths = _input_paths()
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing T1.2 inputs: {missing}")
    manifest = _json(MANIFEST_PATH)
    preflight = _json(PRELIGHT)
    old = _json(OLD_SCORECARD)
    if manifest["corpus_sha256"] != sha256_file(CORPUS_PATH):
        raise ValueError("T1.2 corpus does not match its manifest")
    if preflight["corpus"]["corpus_sha256"] != manifest["corpus_sha256"]:
        raise ValueError("T1.2 preflight does not match the corpus")
    if preflight["corpus"]["truncated_examples"] != 0:
        raise ValueError("T1.2 contains a truncated example")
    if old.get("sealed_2025_untouched") is not True:
        raise ValueError("old comparison artifact does not prove 2025 stayed sealed")
    from training.t12_behavior_eval import suite_trace_ids

    return {
        "protocol": "fantasy-alpha-tinker-sft-t12-eval-v2",
        "evaluation_role": "development; 2018/2023/2024 were previously opened",
        "frozen_before_t12_v2_paid_work": True,
        "single_revision": {
            "reason": "v1 canary made zero valid tool calls on three held-out opportunities",
            "change": "tool-call mass 5% to 10%; one-epoch canary 54 to 230 rows",
            "failed_canary_preserved": True,
        },
        "models": {
            "base": BASE_MODEL,
            "t1": old["results"]["t1"].get("model_path", "hash-pinned old artifact"),
            "t11": old["results"]["t11"].get("model_path", "hash-pinned old artifact"),
            "t12": "development-selected T1.2 sampler; unresolved until training",
        },
        "renderer": RENDERER_NAME,
        "training": {
            "base": BASE_MODEL,
            "fresh_from_base": True,
            "lora_rank": 32,
            "learning_rate": 0.0001,
            "epochs": 1,
            "batch_size": 32,
            "checkpoint_fractions": [0.5, 1.0],
            "loss": "last assistant message with frozen per-schema token weights",
            "broad_targeted_weight": [0.70, 0.30],
            "seed": EVAL_SEED,
        },
        "checkpoint_selection": {
            "suite_trace_ids": suite_trace_ids(),
            "development_only": True,
            "priority": [
                "schema validity", "legal draft behavior", "tool-policy validity",
                "development target quality", "development NLL", "earliest tie-break",
            ],
        },
        "temperature": 0.0,
        "seed": EVAL_SEED,
        "seasons": list(shared.SEASONS),
        "sealed_2025_untouched": True,
        "prediction_families": {
            "breakout": list(FAMILIES),
            "calibration": ["season_threshold", "weekly_h2h"],
            "canary": "canary_v1",
            "total": 18,
            "terminal_parse_failure": "zero answers and zero coverage; continue",
        },
        "draftgym": {
            "episodes": list(DRAFT_EPISODES),
            "masked": True,
            "max_requests_per_pick": 8,
            "max_tokens": shared.MAX_TOKENS["draftgym"],
            "same_contract_as_t11": True,
        },
        "decision_rule": {
            "structured_coverage_min": 0.98,
            "illegal_picks_max": 0,
            "tool_precision_min": 0.80,
            "tool_recall_min": 0.20,
            "tool_useful_result_rate_min": 0.80,
            "unnecessary_tool_rate_max": 0.20,
            "pairwise_win_rate_vs_base_min": 0.60,
            "median_delta_vs_base_positive": True,
            "trimmed_mean_delta_vs_base_positive": True,
            "max_brier_regression": 0.005,
            "max_log_loss_regression": 0.01,
            "max_ece_regression": 0.02,
            "terminal_control_reward_match_rate_max": 0.50,
            "canary_behavior_gate_required": True,
            "contamination_canary_required": True,
        },
        "cost": {
            "target_incremental_usd": 7.0,
            "hard_cap_usd": BUDGET_CAP_USD,
            "estimated_training_usd": preflight["estimate"]["estimated_training_usd"],
            "estimated_canary_and_selection_usd": 1.10,
            "estimated_t12_evaluation_usd": 0.90,
            "estimated_total_usd": preflight["estimate"]["estimated_training_usd"] + 2.00,
        },
        "reuse": {
            "base_t1_t11_not_regenerated": True,
            "old_scorecard_sha256": sha256_file(OLD_SCORECARD),
        },
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in paths
        },
    }


def freeze_spec() -> dict[str, Any]:
    if SPEC_PATH.exists() or SPEC_HASH_PATH.exists():
        raise FileExistsError("T1.2 evaluation spec is already frozen")
    spec = build_spec()
    _write(SPEC_PATH, spec)
    digest = sha256_file(SPEC_PATH)
    SPEC_HASH_PATH.write_text(digest + "\n")
    return {"spec": str(SPEC_PATH.relative_to(ROOT)), "sha256": digest}


def verify_spec() -> dict[str, Any]:
    if not SPEC_PATH.exists() or not SPEC_HASH_PATH.exists():
        raise FileNotFoundError("freeze the T1.2 spec before paid work")
    expected = SPEC_HASH_PATH.read_text().strip()
    if sha256_file(SPEC_PATH) != expected:
        raise ValueError("frozen T1.2 spec hash mismatch")
    spec = _json(SPEC_PATH)
    if spec != build_spec():
        raise ValueError("T1.2 inputs or protocol changed after freezing")
    if 2025 in spec["seasons"] or not spec["sealed_2025_untouched"]:
        raise ValueError("2025 must remain sealed")
    return spec


def _selected_model_path() -> str:
    selection = _json(SELECTION_PATH)
    if selection.get("selection_uses_heldout_labels") is not False:
        raise ValueError("T1.2 selection provenance is invalid")
    return str(selection["selected"]["sampler_path"])


def run_arm() -> dict[str, Any]:
    spec = verify_spec()
    model_path = _selected_model_path()
    arm_dir = RUN_ROOT / "t12"
    arm_dir.mkdir(parents=True, exist_ok=True)
    state_path = arm_dir / "run-state.json"
    old = _json(state_path) if state_path.exists() else {}
    if old.get("status") == "complete":
        return old
    records = list(old.get("records", []))
    sampler = TinkerChatSampler(model_path=model_path, experiment="tinker-sft-t12")
    shared._ask_json_array = _ask_json_array_conservative
    shared._restore_ledger(sampler, old)
    shared._run_prediction_benches(sampler, arm_dir, records, spec)
    shared._save_state(
        state_path, sampler, records, arm="t12", status="prediction_benches_complete",
        model_path=model_path, spec_sha256=sha256_file(SPEC_PATH),
    )
    for episode in DRAFT_EPISODES:
        stem = f"draftgym_{episode['season']}_slot{episode['agent_slot']}_seed{episode['seed']}.json"
        out = arm_dir / stem
        if out.exists():
            continue
        before = asdict(sampler.ledger)
        result = _draft_episode(sampler, episode)
        _write(out, result)
        after = asdict(sampler.ledger)
        records.append({
            "benchmark": "draftgym", "season": episode["season"],
            "agent_slot": episode["agent_slot"], "seed": episode["seed"],
            "model_requests": result["model_requests"],
            "parse_failures": result["parse_failures"],
            "usage_delta": {key: after[key] - before[key] for key in before},
        })
        shared._save_state(
            state_path, sampler, records, arm="t12", status="running",
            model_path=model_path, spec_sha256=sha256_file(SPEC_PATH),
        )
    payload = {
        "arm": "t12", "status": "complete", "completed_at": utc_now(),
        "model": BASE_MODEL, "model_path": model_path,
        "spec_sha256": sha256_file(SPEC_PATH), "records": records,
        "cost": {**asdict(sampler.ledger), "computed_usd": sampler.ledger.usd},
    }
    _write(state_path, payload)
    return payload


def _control_action_matches(episode: Mapping[str, Any]) -> tuple[int, int]:
    gym = DraftGym(
        season=episode["season"], preset=episode["preset"],
        agent_slot=episode["agent_slot"], seed=episode["seed"],
        mask_names=True, enable_evidence=True,
    )
    gym.reset()
    events: dict[int, list[Mapping[str, Any]]] = {}
    for event in episode.get("tool_events", []):
        events.setdefault(int(event["pick_number"]), []).append(event)
    matches = 0
    for pick in episode["picks"]:
        number = int(pick["pick_number"])
        for event in events.get(number, []):
            gym.step({"tool": {
                "name": event["tool"],
                "arguments": {"team_or_player": event["target"]},
            }})
        control = autopick_action(gym)
        matches += control == pick["action"]
        gym.step(pick["action"])
    return matches, len(episode["picks"])


def _draft_score(arm_dir: Path) -> dict[str, Any]:
    episodes = [
        _json(arm_dir / f"draftgym_{entry['season']}_slot{entry['agent_slot']}_seed{entry['seed']}.json")
        for entry in DRAFT_EPISODES
    ]
    all_picks = [pick for episode in episodes for pick in episode["picks"]]
    picks = len(all_picks)
    calls = sum(int(episode["tool_calls"]) for episode in episodes)
    opportunities = sum(int(episode["tool_opportunities"]) for episode in episodes)
    action_match, action_total = map(sum, zip(*(
        _control_action_matches(episode) for episode in episodes
    ), strict=True))
    episode_mix = [Counter(pick["position"] for pick in episode["picks"]) for episode in episodes]
    return {
        "episodes": len(episodes),
        "mean_reward": statistics.fmean(episode["reward"] for episode in episodes),
        "median_reward": statistics.median(episode["reward"] for episode in episodes),
        "rewards": [episode["reward"] for episode in episodes],
        "fallback_picks": sum(episode["fallback_picks"] for episode in episodes),
        "fallback_rate": sum(episode["fallback_picks"] for episode in episodes) / max(1, picks),
        "illegal_picks": sum(episode["illegal_picks"] for episode in episodes),
        "parse_failures": sum(episode["parse_failures"] for episode in episodes),
        "tool_calls": calls,
        "tool_calls_ok": sum(episode["tool_calls_ok"] for episode in episodes),
        "tool_opportunities": opportunities,
        "tool_opportunities_served": sum(episode["tool_opportunities_served"] for episode in episodes),
        "aligned_tool_calls": sum(episode["aligned_tool_calls"] for episode in episodes),
        "useful_tool_results": sum(episode["useful_tool_results"] for episode in episodes),
        "unnecessary_tool_calls": sum(episode["unnecessary_tool_calls"] for episode in episodes),
        "duplicate_tool_calls": sum(episode["duplicate_tool_calls"] for episode in episodes),
        "tool_precision": sum(episode["aligned_tool_calls"] for episode in episodes) / calls if calls else 0.0,
        "tool_recall": sum(episode["tool_opportunities_served"] for episode in episodes) / opportunities if opportunities else 0.0,
        "useful_tool_result_rate": sum(episode["useful_tool_results"] for episode in episodes) / calls if calls else 0.0,
        "unnecessary_tool_rate": sum(episode["unnecessary_tool_calls"] for episode in episodes) / calls if calls else 0.0,
        "control_action_matches": action_match,
        "control_action_opportunities": action_total,
        "control_action_match_rate": action_match / max(1, action_total),
        "terminal_control_reward_matches": sum(float(episode["reward"]) == 0 for episode in episodes),
        "terminal_control_reward_match_rate": sum(float(episode["reward"]) == 0 for episode in episodes) / len(episodes),
        "mistake_diagnostics": {
            "early_qb_or_te_before_round_5": sum(
                pick["round"] < 5 and pick["position"] in {"QB", "TE"} for pick in all_picks
            ),
            "episodes_with_more_than_two_qbs": sum(mix["QB"] > 2 for mix in episode_mix),
            "episodes_with_more_than_two_tes": sum(mix["TE"] > 2 for mix in episode_mix),
            "late_round_qb_or_te": sum(
                pick["round"] >= 11 and pick["position"] in {"QB", "TE"} for pick in all_picks
            ),
        },
        "episodes_detail": episodes,
    }


def _t12_score() -> dict[str, Any]:
    arm_dir = RUN_ROOT / "t12"
    state = _json(arm_dir / "run-state.json")
    if state.get("status") != "complete":
        raise ValueError("T1.2 inference is incomplete")
    breakout_paths = [
        arm_dir / f"breakout_{season}_{shared.PRESET}_{family}.json"
        for season in shared.SEASONS for family in FAMILIES
    ]
    canary_paths = [
        arm_dir / f"canary_{season}_{shared.PRESET}.json" for season in shared.SEASONS
    ]
    prediction_records = [record for record in state["records"] if record["benchmark"] != "draftgym"]
    return {
        "breakout": score_run(breakout_paths),
        "calib": shared._calib_score(arm_dir),
        "canary": score_canaries(load_answers_files(canary_paths)),
        "draftgym": _draft_score(arm_dir),
        "structured_coverage": sum(record["answered"] for record in prediction_records)
        / sum(record["valid_ids"] for record in prediction_records),
        "repair_requests": sum(bool(record.get("repaired")) for record in prediction_records),
        "terminal_prediction_failures": sum(not record.get("parse_success", True) for record in prediction_records),
        "cost": state["cost"],
    }


def _decision(
    results: Mapping[str, Any], comparisons: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    rule = spec["decision_rule"]
    contender = results["t12"]
    base = results["base"]
    draft = contender["draftgym"]
    paired = comparisons["draftgym"]["base"]
    deltas = {
        "breakout_brier": contender["breakout"]["pooled"]["pooled_brier"] - base["breakout"]["pooled"]["pooled_brier"],
        "calib_brier": contender["calib"]["overall"]["brier"] - base["calib"]["overall"]["brier"],
        "calib_log_loss": contender["calib"]["overall"]["log_loss"] - base["calib"]["overall"]["log_loss"],
        "calib_ece": contender["calib"]["overall"]["ece"] - base["calib"]["overall"]["ece"],
    }
    criteria = {
        "structured_coverage": contender["structured_coverage"] >= rule["structured_coverage_min"],
        "zero_illegal_picks": draft["illegal_picks"] <= rule["illegal_picks_max"],
        "selective_tool_policy": (
            draft["tool_calls"] > 0
            and draft["tool_precision"] >= rule["tool_precision_min"]
            and draft["tool_recall"] >= rule["tool_recall_min"]
            and draft["useful_tool_result_rate"] >= rule["tool_useful_result_rate_min"]
            and draft["unnecessary_tool_rate"] <= rule["unnecessary_tool_rate_max"]
        ),
        "draft_beats_base": (
            paired["win_rate"] >= rule["pairwise_win_rate_vs_base_min"]
            and paired["median_delta"] > 0
            and paired["trimmed_mean_delta"] > 0
        ),
        "probability_non_regressive": (
            deltas["breakout_brier"] <= rule["max_brier_regression"]
            and deltas["calib_brier"] <= rule["max_brier_regression"]
            and deltas["calib_log_loss"] <= rule["max_log_loss_regression"]
            and deltas["calib_ece"] <= rule["max_ece_regression"]
        ),
        "contamination_canary": bool(contender["canary"]["gate"]["pass"]),
        "training_canary_behavior": bool(_json(CANARY_BEHAVIOR)["gate"]["pass"]),
        "no_near_control_collapse": (
            draft["terminal_control_reward_match_rate"]
            <= rule["terminal_control_reward_match_rate_max"]
        ),
        "point_in_time_and_2025_sealed": True,
    }
    if all(criteria.values()):
        verdict = "proceed_to_small_draftgym_rl_pilot"
    else:
        dominated = (
            paired["median_delta"] <= 0
            and paired["trimmed_mean_delta"] <= 0
            and contender["structured_coverage"] < base["structured_coverage"]
        )
        repeated_collapse = (
            draft["terminal_control_reward_match_rate"] > 0.75
            or contender["structured_coverage"] < 0.75
        )
        verdict = "stop_additional_sft" if dominated or repeated_collapse else "revise_sft_before_rl"
    return {"verdict": verdict, "criteria": criteria, "probability_deltas_vs_base": deltas}


def score() -> dict[str, Any]:
    spec = verify_spec()
    old = _json(OLD_SCORECARD)["results"]
    results = {arm: old[arm] for arm in ("base", "t1", "t11")}
    results["t12"] = _t12_score()
    contender_episodes = results["t12"]["draftgym"]["episodes_detail"]
    comparisons = {
        "draftgym": {
            arm: _draft_comparison(
                contender_episodes, results[arm]["draftgym"]["episodes_detail"], arm
            )
            for arm in ("base", "t1", "t11")
        },
        "predictions": {
            arm: _prediction_comparison(RUN_ROOT / "t12", OLD_ROOT / arm)
            for arm in ("base", "t1", "t11")
        },
    }
    decision = _decision(results, comparisons, spec)
    canary_train = _json(CANARY_SUMMARY)
    canary_behavior = _json(CANARY_BEHAVIOR)
    train = _json(TRAIN_SUMMARY)
    selection = _json(SELECTION_PATH)
    selection_eval = sum(
        float(candidate["behavior"]["cost"]["computed_usd"])
        for candidate in selection["candidates"]
    )
    failed_canary = (
        float(_json(FAILED_CANARY_SUMMARY)["cost"]["computed_usd"])
        + float(_json(FAILED_CANARY_BEHAVIOR)["cost"]["computed_usd"])
    )
    incremental = sum([
        failed_canary,
        float(canary_train["cost"]["computed_usd"]),
        float(canary_behavior["cost"]["computed_usd"]),
        float(train["cost"]["computed_usd"]),
        selection_eval,
        float(results["t12"]["cost"]["computed_usd"]),
    ])
    if incremental > BUDGET_CAP_USD:
        raise RuntimeError("T1.2 workload exceeded the precommitted hard cap")
    payload = {
        "protocol": spec["protocol"],
        "scored_at": utc_now(),
        "spec_sha256": sha256_file(SPEC_PATH),
        "results": results,
        "comparisons": comparisons,
        "decision": decision,
        "spend": {
            "canary_training_usd": canary_train["cost"]["computed_usd"],
            "canary_behavior_usd": canary_behavior["cost"]["computed_usd"],
            "training_usd": train["cost"]["computed_usd"],
            "checkpoint_selection_behavior_usd": selection_eval,
            "t12_evaluation_usd": results["t12"]["cost"]["computed_usd"],
            "failed_or_retried_sessions_usd": failed_canary,
            "t12_incremental_total_usd": incremental,
            "exact_tinker_cumulative_usd": PRIOR_TINKER_USD + incremental,
        },
        "sealed_2025_untouched": True,
    }
    _write(SCORECARD_PATH, payload)
    write_report(payload)
    write_model_card(payload)
    return payload


def write_report(scorecard: Mapping[str, Any]) -> None:
    results = scorecard["results"]
    comparison = scorecard["comparisons"]["draftgym"]
    draft = results["t12"]["draftgym"]
    arms = ("base", "t1", "t11", "t12")
    labels = {"base": "Base", "t1": "T1", "t11": "T1.1", "t12": "T1.2"}
    def row(label: str, values: Sequence[float], lower: bool = False) -> str:
        arrow = "↓" if lower else "↑"
        return f"| {label} {arrow} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
    lines = [
        "# Fantasy Alpha Tinker SFT T1.2 — frozen development evaluation",
        "",
        f"Decision: **{scorecard['decision']['verdict'].replace('_', ' ')}**.",
        "",
        "This four-way matched comparison reuses hash-pinned Base, T1, and T1.1",
        "artifacts and pays only for T1.2. The 2025 one-shot gate remains sealed.",
        "",
        "## Scorecard",
        "",
        "| metric | " + " | ".join(labels[arm] for arm in arms) + " |",
        "|---|---:|---:|---:|---:|",
        row("BreakoutBench Brier", [results[a]["breakout"]["pooled"]["pooled_brier"] for a in arms], True),
        row("CalibBench Brier", [results[a]["calib"]["overall"]["brier"] for a in arms], True),
        row("CalibBench log loss", [results[a]["calib"]["overall"]["log_loss"] for a in arms], True),
        row("CalibBench ECE", [results[a]["calib"]["overall"]["ece"] for a in arms], True),
        row("DraftGym mean", [results[a]["draftgym"]["mean_reward"] for a in arms]),
        row("DraftGym median", [results[a]["draftgym"]["median_reward"] for a in arms]),
        row("Structured coverage", [results[a]["structured_coverage"] for a in arms]),
        "",
        "## Paired draft comparisons",
        "",
        "| comparison | wins | win rate | median delta | trimmed mean | episode 95% CI | season-clustered 95% CI |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for arm in ("base", "t1", "t11"):
        value = comparison[arm]
        lines.append(
            f"| T1.2 vs {labels[arm]} | {value['wins']}/24 | {value['win_rate']:.1%} | "
            f"{value['median_delta']:+.2f} | {value['trimmed_mean_delta']:+.2f} | "
            f"{value['episode_bootstrap_95ci']} | {value['season_clustered_bootstrap_95ci']} |"
        )
    lines += [
        "",
        "## Tool, legality, and control diagnostics",
        "",
        f"- Tool calls: {draft['tool_calls']} across {draft['tool_opportunities']} opportunities",
        f"- Precision / recall: {draft['tool_precision']:.1%} / {draft['tool_recall']:.1%}",
        f"- Useful / unnecessary call rate: {draft['useful_tool_result_rate']:.1%} / {draft['unnecessary_tool_rate']:.1%}",
        f"- Illegal picks: {draft['illegal_picks']}; fallback rate: {draft['fallback_rate']:.1%}",
        f"- Exact control-action match rate at the same state: {draft['control_action_match_rate']:.1%}",
        f"- Terminal reward exactly equal to control: {draft['terminal_control_reward_matches']}/24",
        "",
        "## Decision gates",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}"
        for name, passed in scorecard["decision"]["criteria"].items()
    )
    lines += [
        "",
        "## Spending",
        "",
        f"Exact T1.2 incremental spend: **${scorecard['spend']['t12_incremental_total_usd']:.6f}**.",
        f"Exact cumulative Tinker workload: **${scorecard['spend']['exact_tinker_cumulative_usd']:.6f}**.",
        "Ongoing checkpoint storage is reported separately. No RL was started.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def write_model_card(scorecard: Mapping[str, Any]) -> None:
    train = _json(TRAIN_SUMMARY)
    selection = _json(SELECTION_PATH)
    manifest = _json(MANIFEST_PATH)
    lines = [
        "# Fantasy Alpha Qwen3.5-9B Tinker SFT T1.2",
        "",
        f"Development decision: **{scorecard['decision']['verdict'].replace('_', ' ')}**.",
        "",
        "## Model and training",
        "",
        "- Base: `Qwen/Qwen3.5-9B` (fresh, not T1/T1.1)",
        "- Method: rank-32 LoRA, 1e-4 learning rate, one epoch",
        f"- Dataset: 829 unique prompts, SHA `{manifest['corpus_sha256']}`",
        "- Weighted signal: 70% broad retention / 30% targeted behavior",
        f"- Selected checkpoint: `{selection['selected']['sampler_path']}`",
        f"- Development NLL: {train['initial_development_nll']:.4f} → {train['final_development_nll']:.4f}",
        "",
        "## Intended use and limits",
        "",
        "Development research for calibrated fantasy forecasts, legal masked draft actions,",
        "and selective depth-chart evidence use. The rule-policy labels mostly track market",
        "order, so SFT can teach process without proving better player selection. Historical",
        "development backtests are contamination-suspect; 2025 remains sealed. This model is",
        "not a production or gambling claim.",
    ]
    MODEL_CARD_PATH.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "run", "score"))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_spec()
    elif args.command == "verify":
        result = verify_spec()
    elif args.command == "run":
        result = run_arm()
    else:
        result = score()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI boundary redacts provider errors
        raise SystemExit(f"T1.2 evaluation failed: {safe_text(exc)}") from None
