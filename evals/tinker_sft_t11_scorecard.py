#!/usr/bin/env python3
"""Frozen three-way development evaluation for targeted Tinker SFT T1.1.

Inference is resumable and never scores an arm.  Only the final ``score``
command opens answer keys, after base, T1, and T1.1 outputs are complete.
The 2025 gate is deliberately absent from every code path.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.breakoutbench import FAMILIES, QUESTIONS_DIR, score_run
from evals.canaries import load_answers_files, score_canaries
from evals.tinker_sft_scorecard import (
    MAX_TOKENS,
    PRESET,
    SEASONS,
    _calib_score,
    _json,
    _prediction_rows,
    _restore_ledger,
    _run_prediction_benches,
    _save_state,
    _write,
)
from envs.draftgym import DraftGym
from envs.play_llm import autopick_action, build_messages, parse_action
from training.t11_dataset import (
    CORPUS_PATH,
    MANIFEST_PATH,
    _tool_opportunity,
)
from training.tinker_backend import (
    BASE_MODEL,
    RENDERER_NAME,
    ROOT,
    TinkerChatSampler,
    safe_text,
    sha256_file,
    utc_now,
)

SPEC_PATH = ROOT / "training/tinker-eval-spec-t11-v1.json"
SPEC_HASH_PATH = ROOT / "training/tinker-eval-spec-t11-v1.sha256"
RUN_ROOT = ROOT / "artifacts/tinker-sft-t11/evaluation"
TRAIN_SUMMARY = ROOT / "artifacts/tinker-sft-t11/t11-full/run-summary.json"
CANARY_SUMMARY = ROOT / "artifacts/tinker-sft-t11/canary/run-summary.json"
T11_SELECTION = ROOT / "artifacts/tinker-sft-t11/t11-full/checkpoint-selection.json"
T1_SELECTION = ROOT / "artifacts/tinker-sft-v1/t1-full/checkpoint-selection.json"
T1_SCORECARD = ROOT / "artifacts/tinker-sft-v1/evaluation/scorecard.json"
SCORECARD_PATH = RUN_ROOT / "scorecard.json"
REPORT_PATH = ROOT / "docs/tinker-sft-t11-experiment-report.md"
MODEL_CARD_PATH = ROOT / "docs/models/fantasy-alpha-qwen35-9b-tinker-sft-t11.md"
BUDGET_CAP_USD = 10.0
# Keep the original T1 sampling seed for an apples-to-apples model comparison.
EVAL_SEED = 20260808

DRAFT_EPISODES = tuple(
    {
        "season": season, "preset": PRESET, "agent_slot": slot, "seed": seed,
        "mask_names": True,
    }
    for season in SEASONS
    for slot in (1, 4, 7, 10)
    for seed in (11, 29)
)


def _input_paths() -> list[Path]:
    paths = [
        CORPUS_PATH,
        MANIFEST_PATH,
        ROOT / "artifacts/tinker-sft-t11/preflight.json",
        T1_SELECTION,
        T1_SCORECARD,
        ROOT / "evals/tinker_sft_scorecard.py",
        ROOT / "evals/tinker_sft_t11_scorecard.py",
        ROOT / "training/t11_dataset.py",
        ROOT / "training/tinker_backend.py",
        ROOT / "training/tinker_sft.py",
        ROOT / "envs/draftgym.py",
        ROOT / "envs/play_llm.py",
        ROOT / "evals/breakoutbench.py",
        ROOT / "evals/calibbench.py",
        ROOT / "evals/canaries.py",
        ROOT / "evals/run_expanded_qwen.py",
    ]
    for season in SEASONS:
        paths.extend([
            QUESTIONS_DIR / f"breakoutbench_{season}_{PRESET}.json",
            QUESTIONS_DIR / f"breakoutbench_{season}_{PRESET}_KEY.json",
            QUESTIONS_DIR / f"calibbench_{season}_{PRESET}_anon.json",
            QUESTIONS_DIR / f"calibbench_{season}_{PRESET}_KEY.json",
            QUESTIONS_DIR / f"canary_{season}_{PRESET}.json",
            QUESTIONS_DIR / f"canary_{season}_{PRESET}_KEY.json",
        ])
    return sorted(paths)


def build_spec() -> dict[str, Any]:
    inputs = _input_paths()
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing T1.1 evaluation inputs: {missing}")
    manifest = _json(MANIFEST_PATH)
    if manifest["corpus_sha256"] != sha256_file(CORPUS_PATH):
        raise ValueError("T1.1 corpus does not match its manifest")
    preflight = _json(ROOT / "artifacts/tinker-sft-t11/preflight.json")
    if (
        preflight["corpus"]["corpus_sha256"] != manifest["corpus_sha256"]
        or preflight["corpus"]["truncated_examples"] != 0
        or preflight["corpus"]["rows"] != 300
    ):
        raise ValueError("T1.1 preflight does not match the validated corpus")
    return {
        "protocol": "fantasy-alpha-tinker-sft-t11-eval-v1",
        "evaluation_role": "development; 2018/2023/2024 were opened in T1",
        "frozen_before_t11_training": True,
        "models": {
            "base": BASE_MODEL,
            "t1": _json(T1_SELECTION)["selected"]["sampler_path"],
            "t11": "development-selected T1.1 sampler; unresolved until training",
        },
        "renderer": RENDERER_NAME,
        "training": {
            "backend": "Tinker 0.27.1 / tinker-cookbook 0.5.7",
            "lora_rank": 32, "batch_size": 32, "epochs": 2,
            "learning_rate": 0.0003, "final_learning_rate_fraction": 0.1,
            "optimizer": "Adam(beta1=.9,beta2=.95,eps=1e-8,grad_clip=1.0)",
            "loss": "last assistant message only", "seed": 20260808,
            "checkpoint_every_epochs": 1,
        },
        "corpus_sha256": manifest["corpus_sha256"],
        "temperature": 0.0,
        "seed": EVAL_SEED,
        "preset": PRESET,
        "seasons": list(SEASONS),
        "sealed_2025_untouched": True,
        "breakoutbench": {
            "variant": "anonymized", "grounded": True,
            "families": list(FAMILIES), "max_tokens": MAX_TOKENS["breakout"],
        },
        "calibbench": {
            "variant": "anonymized", "families": ["season_threshold", "weekly_h2h"],
            "max_tokens": MAX_TOKENS["calib"],
        },
        "canary": {"protocol": "canary_v1", "max_tokens": MAX_TOKENS["canary"]},
        "draftgym": {
            "masked": True, "episodes": list(DRAFT_EPISODES),
            "max_tokens": MAX_TOKENS["draftgym"], "max_requests_per_pick": 8,
            "tool_opportunity": (
                "round>=5 and selected candidate has zero prior points or "
                "adp_stdev/adp>=0.12; aligned calls use depth_chart on a visible candidate"
            ),
            "mistake_diagnostics": [
                "QB/TE before round 5", "episodes with more than two QBs",
                "episodes with more than two TEs", "late-round QB/TE picks",
            ],
        },
        "repair_policy": "one identical strict-JSON repair after parse failure",
        "checkpoint_selection": (
            "lowest T1.1 development NLL among epochs 1 and 2; earliest wins ties"
        ),
        "paired_analysis": {
            "episode_bootstrap_repetitions": 10000,
            "season_clustered_bootstrap_repetitions": 10000,
            "seed": EVAL_SEED,
        },
        "decision_rule": {
            "pairwise_win_rate_min": 0.60,
            "median_pairwise_delta_must_be_positive": True,
            "trimmed_pairwise_mean_must_be_positive": True,
            "max_brier_regression": 0.005,
            "max_log_loss_regression": 0.01,
            "max_ece_regression": 0.02,
            "tool_precision_min": 0.80,
            "tool_recall_min": 0.20,
            "tool_useful_result_rate_min": 0.80,
            "unnecessary_tool_rate_max": 0.20,
            "structured_coverage_min": 0.98,
            "illegal_picks_max": 0,
            "fallback_rate_regression_max": 0.02,
            "proceed": "all precommitted criteria pass against both base and T1",
            "revise": "some directional value exists but at least one criterion fails",
            "stop": "T1.1 is clearly dominated or introduces a major regression",
        },
        "cost": {
            "incremental_hard_cap_usd": BUDGET_CAP_USD,
            "estimated_training_usd": preflight["estimate"]["estimated_training_usd"],
            "estimated_canary_usd": 0.20,
            "estimated_three_arm_evaluation_usd": 2.60,
            "estimated_incremental_total_usd": 4.51,
        },
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in inputs
        },
    }


def freeze_spec() -> dict[str, Any]:
    if SPEC_PATH.exists() or SPEC_HASH_PATH.exists():
        raise FileExistsError("T1.1 evaluation spec is already frozen")
    spec = build_spec()
    _write(SPEC_PATH, spec)
    digest = sha256_file(SPEC_PATH)
    SPEC_HASH_PATH.write_text(digest + "\n")
    return {"spec": str(SPEC_PATH.relative_to(ROOT)), "sha256": digest}


def verify_spec() -> dict[str, Any]:
    if not SPEC_PATH.exists() or not SPEC_HASH_PATH.exists():
        raise FileNotFoundError("freeze the T1.1 evaluation spec before paid work")
    expected = SPEC_HASH_PATH.read_text().strip()
    if sha256_file(SPEC_PATH) != expected:
        raise ValueError("frozen T1.1 evaluation spec hash mismatch")
    spec = _json(SPEC_PATH)
    if spec != build_spec():
        raise ValueError("T1.1 inputs or protocol changed after freezing")
    if 2025 in spec["seasons"] or not spec["sealed_2025_untouched"]:
        raise ValueError("2025 must remain sealed")
    return spec


def select_checkpoint() -> dict[str, Any]:
    spec = verify_spec()
    if (RUN_ROOT / "t11").exists():
        raise ValueError("refusing checkpoint selection after T1.1 inference began")
    summary = _json(TRAIN_SUMMARY)
    if summary.get("status") != "complete" or summary["config"]["epochs"] != 2:
        raise ValueError("expected one complete two-epoch T1.1 run")
    candidates = [
        {
            "epoch": int(row["epoch"]),
            "development_nll": float(row["development_nll"]),
            "sampler_path": row["sampler_path"],
        }
        for row in summary["checkpoints"]
    ]
    candidates.append({
        "epoch": 2,
        "development_nll": float(summary["final_development_nll"]),
        "sampler_path": summary["final_sampler_path"],
    })
    selected = min(candidates, key=lambda x: (x["development_nll"], x["epoch"]))
    payload = {
        "selected_at": utc_now(), "selection_uses_evaluation_labels": False,
        "rule": spec["checkpoint_selection"], "candidates": candidates,
        "selected": selected, "training_summary_sha256": sha256_file(TRAIN_SUMMARY),
    }
    _write(T11_SELECTION, payload)
    return payload


def _opportunity_ids(obs: Mapping[str, Any]) -> set[str]:
    if obs.get("tool_results"):
        return set()
    return {
        str(candidate["player_id"])
        for candidate in obs.get("top_available", [])
        if _tool_opportunity(obs, candidate)
    }


def _nonempty_tool_result(obs: Mapping[str, Any]) -> bool:
    if not obs.get("tool_results"):
        return False
    response = obs["tool_results"][-1].get("response", {})
    result = response.get("result") if response.get("ok") else None
    return result is not None and result != [] and result != {}


def _draft_episode(
    sampler: TinkerChatSampler, episode_spec: Mapping[str, Any]
) -> dict[str, Any]:
    gym = DraftGym(**episode_spec)
    obs = gym.reset()
    stats: dict[str, Any] = {
        **episode_spec, "model_requests": 0, "fallback_picks": 0,
        "parse_failures": 0, "illegal_picks": 0, "tool_calls": 0,
        "tool_calls_ok": 0, "tool_opportunities": 0,
        "tool_opportunities_served": 0, "aligned_tool_calls": 0,
        "useful_tool_results": 0, "unnecessary_tool_calls": 0,
        "duplicate_tool_calls": 0, "picks": [], "tool_events": [],
    }
    reward, done, info = 0.0, False, {}
    requests_this_pick = 0
    counted_opportunities: set[int] = set()
    served_opportunities: set[int] = set()
    calls_this_pick: set[tuple[str, str]] = set()
    while not done:
        opportunity_ids = _opportunity_ids(obs)
        pick_number = int(obs["pick_number"])
        if opportunity_ids and pick_number not in counted_opportunities:
            counted_opportunities.add(pick_number)
            stats["tool_opportunities"] += 1
        if requests_this_pick >= 8:
            action = autopick_action(gym)
            stats["fallback_picks"] += 1
        else:
            action = None
            messages = build_messages(obs)
            for attempt in range(2):
                result = sampler.chat(
                    messages, max_tokens=MAX_TOKENS["draftgym"],
                    temperature=0.0, seed=EVAL_SEED,
                )
                stats["model_requests"] += 1
                requests_this_pick += 1
                try:
                    action = parse_action(result["content"])
                    break
                except ValueError as exc:
                    stats["parse_failures"] += 1
                    if attempt == 0:
                        messages += [
                            {"role": "assistant", "content": result["content"]},
                            {"role": "user", "content":
                             f"Invalid ({safe_text(exc)}). Reply with ONLY one JSON action object."},
                        ]
            if action is None:
                action = autopick_action(gym)
                stats["fallback_picks"] += 1
        if "tool" in action:
            stats["tool_calls"] += 1
            tool = action["tool"]
            name = str(tool.get("name", ""))
            args = tool.get("arguments", {})
            target = str(args.get("team_or_player", args.get("player", "")))
            signature = (name, target)
            aligned = name == "depth_chart" and target in opportunity_ids
            duplicate = signature in calls_this_pick
            calls_this_pick.add(signature)
            if aligned and not duplicate:
                stats["aligned_tool_calls"] += 1
                served_opportunities.add(pick_number)
            else:
                stats["unnecessary_tool_calls"] += 1
            if duplicate:
                stats["duplicate_tool_calls"] += 1
            obs, reward, done, info = gym.step(action)
            ok = bool(obs["tool_results"][-1]["response"].get("ok"))
            useful = _nonempty_tool_result(obs)
            stats["tool_calls_ok"] += int(ok)
            stats["useful_tool_results"] += int(useful)
            stats["tool_events"].append({
                "pick_number": pick_number, "tool": name, "target": target,
                "aligned": aligned and not duplicate, "duplicate": duplicate,
                "ok": ok, "useful_result": useful,
            })
            continue
        selected = next(
            (candidate for candidate in obs["top_available"]
             if candidate["player_id"] == str(action.get("pick"))),
            None,
        )
        roster_before = Counter(x["position"] for x in obs["my_roster"])
        round_number = int(obs["round"])
        try:
            obs, reward, done, info = gym.step(action)
        except ValueError:
            stats["illegal_picks"] += 1
            stats["fallback_picks"] += 1
            action = autopick_action(gym)
            obs, reward, done, info = gym.step(action)
        stats["picks"].append({
            "pick_number": pick_number, "round": round_number, "action": action,
            "position": selected["position"] if selected else None,
            "adp": selected["adp"] if selected else None,
            "prev_season_points": selected["prev_season_points"] if selected else None,
            "roster_counts_before": dict(sorted(roster_before.items())),
        })
        requests_this_pick = 0
        calls_this_pick = set()
    stats["tool_opportunities_served"] = len(served_opportunities)
    stats.update({
        "reward": reward,
        "agent_points_realistic": info.get("agent_points_realistic"),
        "control_points_realistic": info.get("control_points_realistic"),
        "total_lookups": info.get("total_lookups", 0),
        "cap_hits": info.get("cap_hits", 0),
    })
    return stats


def _model_path(arm: str) -> str | None:
    if arm == "base":
        return None
    if arm == "t1":
        selection = _json(T1_SELECTION)
        if selection.get("selection_uses_heldout_labels") is not False:
            raise ValueError("T1 checkpoint provenance is invalid")
        return str(selection["selected"]["sampler_path"])
    selection = _json(T11_SELECTION)
    if selection.get("selection_uses_evaluation_labels") is not False:
        raise ValueError("T1.1 checkpoint provenance is invalid")
    return str(selection["selected"]["sampler_path"])


def run_arm(arm: str) -> dict[str, Any]:
    spec = verify_spec()
    if arm not in {"base", "t1", "t11"}:
        raise ValueError("arm must be base, t1, or t11")
    model_path = _model_path(arm)
    arm_dir = RUN_ROOT / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    state_path = arm_dir / "run-state.json"
    old = _json(state_path) if state_path.exists() else {}
    if old.get("status") == "complete":
        return old
    records = list(old.get("records", []))
    sampler = TinkerChatSampler(model_path=model_path, experiment="tinker-sft-t11")
    _restore_ledger(sampler, old)
    _run_prediction_benches(sampler, arm_dir, records, spec)
    _save_state(
        state_path, sampler, records, arm=arm, status="prediction_benches_complete",
        model_path=model_path, spec_sha256=sha256_file(SPEC_PATH),
    )
    for episode in DRAFT_EPISODES:
        stem = (
            f"draftgym_{episode['season']}_slot{episode['agent_slot']}_seed{episode['seed']}.json"
        )
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
        _save_state(
            state_path, sampler, records, arm=arm, status="running",
            model_path=model_path, spec_sha256=sha256_file(SPEC_PATH),
        )
    payload = {
        "arm": arm, "status": "complete", "completed_at": utc_now(),
        "model": BASE_MODEL, "model_path": model_path,
        "spec_sha256": sha256_file(SPEC_PATH), "records": records,
        "cost": {**asdict(sampler.ledger), "computed_usd": sampler.ledger.usd},
    }
    _write(state_path, payload)
    return payload


def _draft_score(arm_dir: Path) -> dict[str, Any]:
    episodes = [
        _json(arm_dir / (
            f"draftgym_{entry['season']}_slot{entry['agent_slot']}_seed{entry['seed']}.json"
        ))
        for entry in DRAFT_EPISODES
    ]
    picks = sum(len(entry["picks"]) for entry in episodes)
    calls = sum(int(entry["tool_calls"]) for entry in episodes)
    opportunities = sum(int(entry["tool_opportunities"]) for entry in episodes)
    all_picks = [pick for entry in episodes for pick in entry["picks"]]
    position_mix = Counter(pick["position"] for pick in all_picks if pick["position"])
    episode_mix = [Counter(pick["position"] for pick in entry["picks"]) for entry in episodes]
    return {
        "episodes": len(episodes),
        "mean_reward": statistics.fmean(entry["reward"] for entry in episodes),
        "median_reward": statistics.median(entry["reward"] for entry in episodes),
        "rewards": [entry["reward"] for entry in episodes],
        "fallback_picks": sum(entry["fallback_picks"] for entry in episodes),
        "fallback_rate": sum(entry["fallback_picks"] for entry in episodes) / max(1, picks),
        "illegal_picks": sum(entry["illegal_picks"] for entry in episodes),
        "parse_failures": sum(entry["parse_failures"] for entry in episodes),
        "tool_calls": calls,
        "tool_calls_ok": sum(entry["tool_calls_ok"] for entry in episodes),
        "tool_opportunities": opportunities,
        "tool_opportunities_served": sum(
            entry["tool_opportunities_served"] for entry in episodes
        ),
        "aligned_tool_calls": sum(entry["aligned_tool_calls"] for entry in episodes),
        "useful_tool_results": sum(entry["useful_tool_results"] for entry in episodes),
        "unnecessary_tool_calls": sum(entry["unnecessary_tool_calls"] for entry in episodes),
        "duplicate_tool_calls": sum(entry["duplicate_tool_calls"] for entry in episodes),
        "tool_precision": (
            sum(entry["aligned_tool_calls"] for entry in episodes) / calls if calls else 0.0
        ),
        "tool_recall": (
            sum(entry["tool_opportunities_served"] for entry in episodes) / opportunities
            if opportunities else 0.0
        ),
        "useful_tool_result_rate": (
            sum(entry["useful_tool_results"] for entry in episodes) / calls if calls else 0.0
        ),
        "unnecessary_tool_rate": (
            sum(entry["unnecessary_tool_calls"] for entry in episodes) / calls
            if calls else 0.0
        ),
        "mistake_diagnostics": {
            "early_qb_or_te_before_round_5": sum(
                pick["round"] < 5 and pick["position"] in {"QB", "TE"}
                for pick in all_picks
            ),
            "episodes_with_more_than_two_qbs": sum(mix["QB"] > 2 for mix in episode_mix),
            "episodes_with_more_than_two_tes": sum(mix["TE"] > 2 for mix in episode_mix),
            "late_round_qb_or_te": sum(
                pick["round"] >= 11 and pick["position"] in {"QB", "TE"}
                for pick in all_picks
            ),
            "position_mix": dict(sorted(position_mix.items())),
        },
        "episodes_detail": episodes,
    }


def _arm_score(arm: str) -> dict[str, Any]:
    arm_dir = RUN_ROOT / arm
    state = _json(arm_dir / "run-state.json")
    if state.get("status") != "complete":
        raise ValueError(f"{arm} inference is incomplete")
    breakout_paths = [
        arm_dir / f"breakout_{season}_{PRESET}_{family}.json"
        for season in SEASONS for family in FAMILIES
    ]
    canary_paths = [arm_dir / f"canary_{season}_{PRESET}.json" for season in SEASONS]
    prediction_records = [r for r in state["records"] if r["benchmark"] != "draftgym"]
    coverage = sum(r["answered"] for r in prediction_records) / sum(
        r["valid_ids"] for r in prediction_records
    )
    return {
        "breakout": score_run(breakout_paths),
        "calib": _calib_score(arm_dir),
        "canary": score_canaries(load_answers_files(canary_paths)),
        "draftgym": _draft_score(arm_dir),
        "structured_coverage": coverage,
        "repair_requests": sum(bool(r.get("repaired")) for r in prediction_records),
        "cost": state["cost"],
    }


def _bootstrap(values: Sequence[float], *, clustered: Sequence[int] | None = None) -> list[float]:
    rng = random.Random(EVAL_SEED)
    if clustered is None:
        draws = sorted(
            statistics.fmean(values[rng.randrange(len(values))] for _ in values)
            for _ in range(10000)
        )
    else:
        groups: dict[int, list[float]] = defaultdict(list)
        for value, group in zip(values, clustered, strict=True):
            groups[int(group)].append(value)
        group_means = [statistics.fmean(groups[key]) for key in sorted(groups)]
        draws = sorted(
            statistics.fmean(group_means[rng.randrange(len(group_means))] for _ in group_means)
            for _ in range(10000)
        )
    return [draws[249], draws[9749]]


def _trimmed_mean(values: Sequence[float]) -> float:
    ordered = sorted(values)
    trim = max(1, int(len(ordered) * 0.10))
    return statistics.fmean(ordered[trim:-trim])


def _draft_comparison(
    contender: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]],
    reference_name: str,
) -> dict[str, Any]:
    rows, deltas, seasons = [], [], []
    for candidate, baseline in zip(contender, reference, strict=True):
        key_c = (candidate["season"], candidate["agent_slot"], candidate["seed"])
        key_b = (baseline["season"], baseline["agent_slot"], baseline["seed"])
        if key_c != key_b:
            raise ValueError("DraftGym pairing mismatch")
        delta = float(candidate["reward"]) - float(baseline["reward"])
        deltas.append(delta)
        seasons.append(int(candidate["season"]))
        rows.append({
            "season": candidate["season"], "agent_slot": candidate["agent_slot"],
            "seed": candidate["seed"], "reference_reward": baseline["reward"],
            "t11_reward": candidate["reward"], "delta": delta,
        })
    return {
        "reference": reference_name, "episodes": rows,
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "trimmed_mean_delta": _trimmed_mean(deltas),
        "wins": sum(value > 0 for value in deltas),
        "losses": sum(value < 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "win_rate": sum(value > 0 for value in deltas) / len(deltas),
        "episode_bootstrap_95ci": _bootstrap(deltas),
        "season_clustered_bootstrap_95ci": _bootstrap(deltas, clustered=seasons),
        "largest_positive_delta": max(deltas),
        "largest_negative_delta": min(deltas),
    }


def _prediction_comparison(contender_dir: Path, reference_dir: Path) -> dict[str, Any]:
    contender, reference = _prediction_rows(contender_dir), _prediction_rows(reference_dir)
    common = sorted(set(contender) & set(reference))
    rows, deltas, seasons = [], [], []
    for key in common:
        candidate, baseline = contender[key], reference[key]
        y = int(candidate["y"])
        improvement = (float(baseline["p"]) - y) ** 2 - (float(candidate["p"]) - y) ** 2
        deltas.append(improvement)
        seasons.append(int(candidate["season"]))
        rows.append({
            "benchmark": candidate["benchmark"], "family": candidate["family"],
            "season": candidate["season"], "question_id": candidate["question_id"],
            "y": y, "reference_p": baseline["p"], "t11_p": candidate["p"],
            "brier_improvement": improvement,
        })
    return {
        "matched_examples": len(rows),
        "mean_brier_improvement": statistics.fmean(deltas),
        "episode_bootstrap_95ci": _bootstrap(deltas),
        "season_clustered_bootstrap_95ci": _bootstrap(deltas, clustered=seasons),
        "largest_improvements": sorted(rows, key=lambda x: -x["brier_improvement"])[:20],
        "largest_regressions": sorted(rows, key=lambda x: x["brier_improvement"])[:20],
    }


def _decision(results: Mapping[str, Any], comparisons: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    rule = spec["decision_rule"]
    t11 = results["t11"]
    pairwise = comparisons["draftgym"]
    pairwise_pass = {
        name: (
            values["win_rate"] >= rule["pairwise_win_rate_min"]
            and values["median_delta"] > 0
            and values["trimmed_mean_delta"] > 0
        )
        for name, values in pairwise.items()
    }
    probability_pass: dict[str, bool] = {}
    probability_deltas: dict[str, dict[str, float]] = {}
    for name in ("base", "t1"):
        ref = results[name]
        deltas = {
            "breakout_brier": t11["breakout"]["pooled"]["pooled_brier"]
            - ref["breakout"]["pooled"]["pooled_brier"],
            "calib_brier": t11["calib"]["overall"]["brier"]
            - ref["calib"]["overall"]["brier"],
            "calib_log_loss": t11["calib"]["overall"]["log_loss"]
            - ref["calib"]["overall"]["log_loss"],
            "calib_ece": t11["calib"]["overall"]["ece"]
            - ref["calib"]["overall"]["ece"],
        }
        probability_deltas[name] = deltas
        probability_pass[name] = (
            deltas["breakout_brier"] <= rule["max_brier_regression"]
            and deltas["calib_brier"] <= rule["max_brier_regression"]
            and deltas["calib_log_loss"] <= rule["max_log_loss_regression"]
            and deltas["calib_ece"] <= rule["max_ece_regression"]
        )
    draft = t11["draftgym"]
    tool_pass = (
        draft["tool_precision"] >= rule["tool_precision_min"]
        and draft["tool_recall"] >= rule["tool_recall_min"]
        and draft["useful_tool_result_rate"] >= rule["tool_useful_result_rate_min"]
        and draft["unnecessary_tool_rate"] <= rule["unnecessary_tool_rate_max"]
    )
    structure_pass = (
        t11["structured_coverage"] >= rule["structured_coverage_min"]
        and draft["illegal_picks"] <= rule["illegal_picks_max"]
        and draft["fallback_rate"]
        <= results["base"]["draftgym"]["fallback_rate"] + rule["fallback_rate_regression_max"]
    )
    canary_pass = bool(t11["canary"]["gate"]["pass"])
    criteria = {
        "draft_beats_base": pairwise_pass["base"],
        "draft_beats_t1": pairwise_pass["t1"],
        "prediction_non_regressive_vs_base": probability_pass["base"],
        "prediction_non_regressive_vs_t1": probability_pass["t1"],
        "tool_policy": tool_pass,
        "structured_output_and_legality": structure_pass,
        "canary": canary_pass,
        "contamination_and_point_in_time": True,
    }
    if all(criteria.values()):
        verdict = "proceed_to_small_draftgym_rl_pilot"
    else:
        draft_dominated = all(
            pairwise[name]["median_delta"] <= 0 and pairwise[name]["trimmed_mean_delta"] <= 0
            for name in pairwise
        )
        major_probability = any(
            values["calib_brier"] > 2 * rule["max_brier_regression"]
            or values["calib_log_loss"] > 2 * rule["max_log_loss_regression"]
            for values in probability_deltas.values()
        )
        verdict = (
            "stop_sft_training" if draft_dominated or major_probability
            else "revise_sft_before_rl"
        )
    return {
        "verdict": verdict, "criteria": criteria,
        "probability_deltas": probability_deltas,
    }


def score() -> dict[str, Any]:
    spec = verify_spec()
    results = {arm: _arm_score(arm) for arm in ("base", "t1", "t11")}
    t11_episodes = results["t11"]["draftgym"]["episodes_detail"]
    comparisons = {
        "draftgym": {
            arm: _draft_comparison(
                t11_episodes, results[arm]["draftgym"]["episodes_detail"], arm
            )
            for arm in ("base", "t1")
        },
        "predictions": {
            arm: _prediction_comparison(RUN_ROOT / "t11", RUN_ROOT / arm)
            for arm in ("base", "t1")
        },
    }
    decision = _decision(results, comparisons, spec)
    canary = _json(CANARY_SUMMARY)
    train = _json(TRAIN_SUMMARY)
    incremental = (
        float(canary["cost"]["computed_usd"])
        + float(train["cost"]["computed_usd"])
        + sum(float(results[arm]["cost"]["computed_usd"]) for arm in results)
    )
    if incremental > BUDGET_CAP_USD:
        raise RuntimeError("completed T1.1 workload exceeded the precommitted budget cap")
    prior_tinker = float(_json(T1_SCORECARD)["spend"]["incremental_total_usd"])
    payload = {
        "protocol": spec["protocol"], "scored_at": utc_now(),
        "spec_sha256": sha256_file(SPEC_PATH), "results": results,
        "comparisons": comparisons, "decision": decision,
        "spend": {
            "canary_usd": canary["cost"]["computed_usd"],
            "training_usd": train["cost"]["computed_usd"],
            "base_evaluation_usd": results["base"]["cost"]["computed_usd"],
            "t1_evaluation_usd": results["t1"]["cost"]["computed_usd"],
            "t11_evaluation_usd": results["t11"]["cost"]["computed_usd"],
            "t11_incremental_total_usd": incremental,
            "exact_tinker_cumulative_usd": prior_tinker + incremental,
        },
        "sealed_2025_untouched": True,
    }
    _write(SCORECARD_PATH, payload)
    write_report(payload)
    write_model_card(payload)
    return payload


def _metric_row(label: str, values: Sequence[float], *, lower: bool = False) -> str:
    rendered = " | ".join(f"{value:.4f}" for value in values)
    return f"| {label} {'↓' if lower else '↑'} | {rendered} |"


def write_report(scorecard: Mapping[str, Any]) -> None:
    results = scorecard["results"]
    comparison = scorecard["comparisons"]["draftgym"]
    tool = results["t11"]["draftgym"]
    spend = scorecard["spend"]
    lines = [
        "# Fantasy Alpha targeted Tinker SFT T1.1 — frozen development evaluation",
        "",
        f"Decision: **{scorecard['decision']['verdict'].replace('_', ' ')}**.",
        "",
        "This is a three-way matched comparison of the untouched Qwen3.5-9B base,",
        "the original T1 adapter, and the targeted T1.1 adapter. The 2018/2023/2024",
        "suite is development evidence because it was opened during T1. The 2025",
        "one-shot gate remains sealed.",
        "",
        "## Scorecard",
        "",
        "| metric | base | T1 | T1.1 |",
        "|---|---:|---:|---:|",
        _metric_row("BreakoutBench Brier", [
            results[x]["breakout"]["pooled"]["pooled_brier"] for x in ("base", "t1", "t11")
        ], lower=True),
        _metric_row("CalibBench Brier", [
            results[x]["calib"]["overall"]["brier"] for x in ("base", "t1", "t11")
        ], lower=True),
        _metric_row("CalibBench log loss", [
            results[x]["calib"]["overall"]["log_loss"] for x in ("base", "t1", "t11")
        ], lower=True),
        _metric_row("CalibBench ECE", [
            results[x]["calib"]["overall"]["ece"] for x in ("base", "t1", "t11")
        ], lower=True),
        _metric_row("BreakoutBench top-10 lift", [
            results[x]["breakout"]["pooled"]["lift"] for x in ("base", "t1", "t11")
        ]),
        _metric_row("Masked DraftGym mean reward", [
            results[x]["draftgym"]["mean_reward"] for x in ("base", "t1", "t11")
        ]),
        _metric_row("Masked DraftGym median reward", [
            results[x]["draftgym"]["median_reward"] for x in ("base", "t1", "t11")
        ]),
        _metric_row("Structured answer coverage", [
            results[x]["structured_coverage"] for x in ("base", "t1", "t11")
        ]),
        "",
        "## Paired draft result",
        "",
        "| comparison | wins | win rate | median delta | trimmed mean | episode 95% CI | season-clustered 95% CI |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name in ("base", "t1"):
        row = comparison[name]
        lines.append(
            f"| T1.1 vs {name} | {row['wins']}/{len(row['episodes'])} | {row['win_rate']:.1%} | "
            f"{row['median_delta']:+.2f} | {row['trimmed_mean_delta']:+.2f} | "
            f"{row['episode_bootstrap_95ci']} | {row['season_clustered_bootstrap_95ci']} |"
        )
    lines += [
        "",
        "Every paired episode:",
        "",
        "| season/slot/seed | base | T1 | T1.1 | T1.1−base | T1.1−T1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_base = {
        (row["season"], row["agent_slot"], row["seed"]): row
        for row in comparison["base"]["episodes"]
    }
    by_t1 = {
        (row["season"], row["agent_slot"], row["seed"]): row
        for row in comparison["t1"]["episodes"]
    }
    for key in sorted(by_base):
        base_row, t1_row = by_base[key], by_t1[key]
        lines.append(
            f"| {key[0]}/{key[1]}/{key[2]} | {base_row['reference_reward']:+.2f} | "
            f"{t1_row['reference_reward']:+.2f} | {base_row['t11_reward']:+.2f} | "
            f"{base_row['delta']:+.2f} | {t1_row['delta']:+.2f} |"
        )
    lines += [
        "",
        "## Mistake-prevention diagnostics",
        "",
        "| diagnostic | base | T1 | T1.1 |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "early_qb_or_te_before_round_5", "episodes_with_more_than_two_qbs",
        "episodes_with_more_than_two_tes", "late_round_qb_or_te",
    ):
        lines.append(
            f"| {key.replace('_', ' ')} | "
            + " | ".join(str(results[arm]["draftgym"]["mistake_diagnostics"][key])
                         for arm in ("base", "t1", "t11"))
            + " |"
        )
    lines += [
        "",
        "These are predeclared process diagnostics. They identify roster/timing mistakes",
        "prevented or introduced without using post-season outcomes as labels.",
        "",
        "## Evidence-tool behavior",
        "",
        f"T1.1 made {tool['tool_calls']} calls across {tool['tool_opportunities']} predeclared",
        f"opportunities: precision {tool['tool_precision']:.1%}, recall {tool['tool_recall']:.1%},",
        f"useful-result rate {tool['useful_tool_result_rate']:.1%}, and unnecessary-call rate",
        f"{tool['unnecessary_tool_rate']:.1%}. These are policy-alignment metrics under the",
        "frozen opportunity definition, not a claim that every lookup causally improved reward.",
        "",
        "## Spending",
        "",
        f"Exact T1.1 incremental spend: **${spend['t11_incremental_total_usd']:.6f}**.",
        f"Exact cumulative Tinker workload spend: **${spend['exact_tinker_cumulative_usd']:.6f}**.",
        "Ongoing checkpoint storage is separate.",
        "",
        "## Precommitted decision",
        "",
    ]
    for criterion, passed in scorecard["decision"]["criteria"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {criterion.replace('_', ' ')}")
    lines += [
        "",
        "The JSON scorecard contains every episode, paired delta, prediction regression,",
        "tool event, bootstrap interval, and exact price-based cost. No RL was started.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def write_model_card(scorecard: Mapping[str, Any]) -> None:
    decision = scorecard["decision"]["verdict"].replace("_", " ")
    train = _json(TRAIN_SUMMARY)
    manifest = _json(MANIFEST_PATH)
    lines = [
        "# Fantasy Alpha Qwen3.5-9B Tinker SFT T1.1",
        "",
        f"Development decision: **{decision}**.",
        "",
        "## Model",
        "",
        "- Base: `Qwen/Qwen3.5-9B`",
        "- Method: Tinker LoRA SFT, rank 32, no-thinking Qwen renderer",
        f"- Dataset: 300 targeted traces, SHA `{manifest['corpus_sha256']}`",
        f"- Selected checkpoint: `{_json(T11_SELECTION)['selected']['sampler_path']}`",
        f"- Training steps: {train['steps']}; development NLL "
        f"{train['initial_development_nll']:.4f} → {train['final_development_nll']:.4f}",
        "",
        "## Intended use",
        "",
        "Development research for masked fantasy draft actions, conservative probability",
        "forecasts, and selective structured depth-chart lookups. It is not a production",
        "claim, gambling model, or evidence of general football expertise.",
        "",
        "## Limits",
        "",
        "The evaluation seasons were previously opened and are development-only. Historical",
        "backtests remain contamination-suspect. The 2025 gate is still sealed, and no RL",
        "should begin unless the precommitted T1.1 criteria passed. The rule-policy teacher",
        "has only a small, unresolved historical edge over ADP, so imitation can improve",
        "discipline without guaranteeing superior player selection.",
    ]
    MODEL_CARD_PATH.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    run = sub.add_parser("run")
    run.add_argument("--arm", required=True, choices=("base", "t1", "t11"))
    sub.add_parser("select-checkpoint")
    sub.add_parser("score")
    args = ap.parse_args(argv)
    if args.command == "freeze":
        result = freeze_spec()
    elif args.command == "run":
        result = run_arm(args.arm)
    elif args.command == "select-checkpoint":
        result = select_checkpoint()
    else:
        result = score()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(f"T1.1 evaluation failed: {safe_text(exc)}") from None
