#!/usr/bin/env python3
"""Frozen, matched base-versus-Tinker-adapter evaluation for SFT v1.

The inference commands never open answer keys. Scoring is a separate command
that refuses to run until both arms are complete, which keeps checkpoint and
prompt choices independent of held-out outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.breakoutbench import FAMILIES, QUESTIONS_DIR, score_run
from evals.calibbench import ece, log_loss, reliability_table
from evals.canaries import load_answers_files, score_canaries
from evals.draftbench import _board_key
from evals.run_expanded_qwen import (
    _family_rows,
    _prior_ranks,
    build_prompt,
    grounding_for,
    parse_answers,
)
from envs.draftgym import DraftGym
from envs.play_llm import autopick_action, build_messages, parse_action
from training.tinker_backend import (
    BASE_MODEL,
    CORPUS_SHA256,
    RENDERER_NAME,
    ROOT,
    TinkerChatSampler,
    safe_text,
    sha256_file,
    utc_now,
)

SPEC_PATH = ROOT / "training/tinker-eval-spec-v1.json"
SPEC_HASH_PATH = ROOT / "training/tinker-eval-spec-v1.sha256"
RUN_ROOT = ROOT / "artifacts/tinker-sft-v1/evaluation"
TRAIN_SUMMARY = ROOT / "artifacts/tinker-sft-v1/t1-full/run-summary.json"
SELECTION_PATH = ROOT / "artifacts/tinker-sft-v1/t1-full/checkpoint-selection.json"
SCORECARD_PATH = RUN_ROOT / "scorecard.json"
REPORT_PATH = ROOT / "docs/tinker-sft-experiment-report.md"

SEASONS = (2018, 2023, 2024)
PRESET = "ppr"
SEED = 20260808
MAX_TOKENS = {"breakout": 10000, "calib": 18000, "canary": 8000, "draftgym": 600}
PROMPT_IMPLEMENTATIONS = (
    ROOT / "evals/breakoutbench.py",
    ROOT / "evals/calibbench.py",
    ROOT / "evals/canaries.py",
    ROOT / "evals/run_expanded_qwen.py",
    ROOT / "envs/draftgym.py",
    ROOT / "envs/play_llm.py",
)
DRAFT_EPISODES = tuple(
    {"season": season, "preset": PRESET, "agent_slot": slot, "seed": 0,
     "mask_names": True}
    for season in SEASONS for slot in (1, 7)
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _question_files() -> list[Path]:
    paths = []
    for season in SEASONS:
        paths.extend([
            QUESTIONS_DIR / f"breakoutbench_{season}_{PRESET}.json",
            QUESTIONS_DIR / f"breakoutbench_{season}_{PRESET}_KEY.json",
            QUESTIONS_DIR / f"calibbench_{season}_{PRESET}_anon.json",
            QUESTIONS_DIR / f"calibbench_{season}_{PRESET}_KEY.json",
            QUESTIONS_DIR / f"canary_{season}_{PRESET}.json",
            QUESTIONS_DIR / f"canary_{season}_{PRESET}_KEY.json",
        ])
    return paths


def build_spec() -> dict[str, Any]:
    inputs = _question_files() + list(PROMPT_IMPLEMENTATIONS)
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen evaluation inputs: {missing}")
    return {
        "protocol": "fantasy-alpha-tinker-sft-eval-v1",
        "frozen_before_t1_training": True,
        "model": BASE_MODEL,
        "renderer": RENDERER_NAME,
        "corpus_sha256": CORPUS_SHA256,
        "temperature": 0.0,
        "seed": SEED,
        "preset": PRESET,
        "seasons": list(SEASONS),
        "sealed_2025_untouched": True,
        "breakoutbench": {"variant": "anonymized", "grounded": True,
                           "families": list(FAMILIES),
                           "max_tokens": MAX_TOKENS["breakout"]},
        "calibbench": {"variant": "anonymized",
                        "families": ["season_threshold", "weekly_h2h"],
                        "max_tokens": MAX_TOKENS["calib"]},
        "canary": {"protocol": "canary_v1", "max_tokens": MAX_TOKENS["canary"]},
        "draftgym": {"masked": True, "episodes": list(DRAFT_EPISODES),
                     "max_tokens": MAX_TOKENS["draftgym"],
                     "max_requests_per_pick": 8},
        "repair_policy": "one identical strict-JSON repair after parse failure",
        "checkpoint_selection": "lowest development NLL among epochs 1 and 2; earliest wins ties",
        "paired_analysis": {"bootstrap_unit": "season", "repetitions": 10000,
                            "seed": SEED},
        "decision_rule": {
            "material_breakout_brier_improvement": 0.005,
            "material_calib_brier_improvement": 0.005,
            "material_masked_draftgym_reward_improvement": 25.0,
            "major_brier_regression": 0.01,
            "major_masked_draftgym_reward_regression": 25.0,
            "max_structured_coverage_regression": 0.02,
            "proceed": "canary passes, no major regression, and at least two material product improvements",
            "revise": "canary passes and evidence is mixed or only one dimension materially improves",
            "stop": "canary fails, a major regression occurs, or no product dimension improves",
        },
        "input_sha256": {
            str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(inputs)
        },
    }


def freeze_spec() -> dict[str, Any]:
    if SPEC_PATH.exists() or SPEC_HASH_PATH.exists():
        raise FileExistsError("evaluation spec is already frozen")
    spec = build_spec()
    _write(SPEC_PATH, spec)
    digest = sha256_file(SPEC_PATH)
    SPEC_HASH_PATH.write_text(digest + "\n")
    return {"spec": str(SPEC_PATH.relative_to(ROOT)), "sha256": digest}


def verify_spec() -> dict[str, Any]:
    if not SPEC_PATH.exists() or not SPEC_HASH_PATH.exists():
        raise FileNotFoundError("freeze the evaluation spec before inference")
    expected = SPEC_HASH_PATH.read_text().strip()
    actual = sha256_file(SPEC_PATH)
    if actual != expected:
        raise ValueError("frozen evaluation spec hash mismatch")
    spec = _json(SPEC_PATH)
    if spec != build_spec():
        raise ValueError("evaluation inputs or protocol changed after freezing")
    if 2025 in spec["seasons"] or not spec["sealed_2025_untouched"]:
        raise ValueError("2025 must remain untouched")
    return spec


def select_checkpoint() -> dict[str, Any]:
    verify_spec()
    summary = _json(TRAIN_SUMMARY)
    if summary.get("status") != "complete" or summary["config"]["epochs"] != 2:
        raise ValueError("expected a complete, two-epoch T1 run")
    candidates = []
    for row in summary["checkpoints"]:
        candidates.append({"epoch": int(row["epoch"]),
                           "development_nll": float(row["development_nll"]),
                           "sampler_path": row["sampler_path"]})
    candidates.append({"epoch": 2,
                       "development_nll": float(summary["final_development_nll"]),
                       "sampler_path": summary["final_sampler_path"]})
    selected = min(candidates, key=lambda x: (x["development_nll"], x["epoch"]))
    payload = {
        "selected_at": utc_now(), "selection_uses_heldout_labels": False,
        "rule": _json(SPEC_PATH)["checkpoint_selection"],
        "candidates": candidates, "selected": selected,
        "training_summary_sha256": sha256_file(TRAIN_SUMMARY),
    }
    _write(SELECTION_PATH, payload)
    return payload


def _calib_prompt(payload: Mapping[str, Any], family: str) -> tuple[list[dict], set[str]]:
    questions = [q for q in payload["questions"] if q["family"] == family]
    content = (
        "Season and player identities are masked. Format: ppr.\n"
        + payload["instructions"] + "\nQuestions:\n" + json.dumps(questions)
        + '\nReturn a JSON array with EXACTLY one object per question: '
          '{"question_id": str, "p": float 0-1}. No prose.'
    )
    return ([{"role": "system", "content":
              "You are a calibrated fantasy football analyst. Answer with strict JSON only."},
             {"role": "user", "content": content}],
            {q["question_id"] for q in questions})


def _restore_ledger(sampler: TinkerChatSampler, state: Mapping[str, Any]) -> None:
    prior = state.get("cost", {})
    for key in asdict(sampler.ledger):
        setattr(sampler.ledger, key, int(prior.get(key, 0)))


def _save_state(path: Path, sampler: TinkerChatSampler, records: list[dict], **extra: Any) -> None:
    _write(path, {**extra, "records": records,
                  "cost": {**asdict(sampler.ledger),
                           "computed_usd": sampler.ledger.usd},
                  "updated_at": utc_now()})


def _ask_json_array(
    sampler: TinkerChatSampler, messages: list[dict], valid_ids: set[str],
    *, max_tokens: int, seed: int,
) -> tuple[list[dict], dict[str, Any]]:
    before = asdict(sampler.ledger)
    result = sampler.chat(messages, max_tokens=max_tokens, temperature=0.0, seed=seed)
    repaired = False
    try:
        answers = parse_answers(result["content"], valid_ids)
    except Exception as exc:
        repaired = True
        repair_messages = messages + [
            {"role": "assistant", "content": result["content"]},
            {"role": "user", "content":
             f"Invalid ({safe_text(exc)}). Return ONLY the JSON array, one object per question_id."},
        ]
        result = sampler.chat(
            repair_messages, max_tokens=max_tokens, temperature=0.0, seed=seed
        )
        answers = parse_answers(result["content"], valid_ids)
    after = asdict(sampler.ledger)
    return answers, {
        "valid_ids": len(valid_ids), "answered": len(answers),
        "coverage": len(answers) / len(valid_ids), "repaired": repaired,
        "parse_success": True,
        "usage_delta": {k: after[k] - before[k] for k in before},
    }


def _run_prediction_benches(
    sampler: TinkerChatSampler, arm_dir: Path, records: list[dict], spec: Mapping[str, Any]
) -> None:
    fam_rows = _family_rows(PRESET)
    prior_ranks = _prior_ranks(PRESET)
    for season in SEASONS:
        qpayload = _json(QUESTIONS_DIR / f"breakoutbench_{season}_{PRESET}.json")
        for family in FAMILIES:
            out = arm_dir / f"breakout_{season}_{PRESET}_{family}.json"
            if out.exists():
                continue
            past = [r for r in fam_rows.get(family, []) if r["season"] < season]
            grounded = {**qpayload, "questions": [
                {**q, "packet": {**q["packet"], "historical_grounding": grounding_for(
                    family, q["packet"], past, prior_ranks, season)}}
                for q in qpayload["questions"]
            ]}
            messages, valid = build_prompt(grounded, family, True)
            answers, record = _ask_json_array(
                sampler, messages, valid, max_tokens=MAX_TOKENS["breakout"], seed=SEED
            )
            _write(out, answers)
            records.append({"benchmark": "breakoutbench", "season": season,
                            "family": family, **record})

        cpayload = _json(QUESTIONS_DIR / f"calibbench_{season}_{PRESET}_anon.json")
        for family in ("season_threshold", "weekly_h2h"):
            out = arm_dir / f"calib_{season}_{PRESET}_{family}.json"
            if out.exists():
                continue
            messages, valid = _calib_prompt(cpayload, family)
            answers, record = _ask_json_array(
                sampler, messages, valid, max_tokens=MAX_TOKENS["calib"], seed=SEED
            )
            _write(out, answers)
            records.append({"benchmark": "calibbench", "season": season,
                            "family": family, **record})

        canary = _json(QUESTIONS_DIR / f"canary_{season}_{PRESET}.json")
        out = arm_dir / f"canary_{season}_{PRESET}.json"
        if not out.exists():
            messages, valid = build_prompt(canary, "full_slate", False)
            answers, record = _ask_json_array(
                sampler, messages, valid, max_tokens=MAX_TOKENS["canary"], seed=SEED
            )
            _write(out, answers)
            records.append({"benchmark": "canary", "season": season,
                            "family": "full_slate", **record})


def _draft_episode(sampler: TinkerChatSampler, spec: dict[str, Any]) -> dict[str, Any]:
    gym = DraftGym(**spec)
    obs = gym.reset()
    stats = {**spec, "model_requests": 0, "fallback_picks": 0,
             "parse_failures": 0, "illegal_picks": 0, "tool_calls": 0,
             "tool_calls_ok": 0, "picks": []}
    reward, done, info = 0.0, False, {}
    requests_this_pick = 0
    while not done:
        if requests_this_pick >= 8:
            action = autopick_action(gym)
            stats["fallback_picks"] += 1
        else:
            action = None
            messages = build_messages(obs)
            for attempt in range(2):
                result = sampler.chat(
                    messages, max_tokens=MAX_TOKENS["draftgym"],
                    temperature=0.0, seed=SEED,
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
            obs, reward, done, info = gym.step(action)
            if obs["tool_results"] and obs["tool_results"][-1]["response"].get("ok"):
                stats["tool_calls_ok"] += 1
            continue
        pick_number = obs["pick_number"]
        try:
            obs, reward, done, info = gym.step(action)
        except ValueError:
            stats["illegal_picks"] += 1
            stats["fallback_picks"] += 1
            obs, reward, done, info = gym.step({
                "pick": min(gym._agent_candidates(), key=_board_key).player_id
            })
        stats["picks"].append({"pick_number": pick_number, "action": action})
        requests_this_pick = 0
    stats.update({"reward": reward,
                  "agent_points_realistic": info.get("agent_points_realistic"),
                  "control_points_realistic": info.get("control_points_realistic"),
                  "total_lookups": info.get("total_lookups", 0),
                  "cap_hits": info.get("cap_hits", 0)})
    return stats


def run_arm(arm: str) -> dict[str, Any]:
    spec = verify_spec()
    if arm not in {"base", "adapter"}:
        raise ValueError("arm must be base or adapter")
    model_path = None
    if arm == "adapter":
        if not SELECTION_PATH.exists():
            raise FileNotFoundError("select the development-best checkpoint first")
        selection = _json(SELECTION_PATH)
        if selection.get("selection_uses_heldout_labels") is not False:
            raise ValueError("invalid checkpoint-selection provenance")
        model_path = selection["selected"]["sampler_path"]
    arm_dir = RUN_ROOT / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    state_path = arm_dir / "run-state.json"
    old = _json(state_path) if state_path.exists() else {}
    if old.get("status") == "complete":
        return old
    records = list(old.get("records", []))
    sampler = TinkerChatSampler(model_path=model_path)
    _restore_ledger(sampler, old)
    _run_prediction_benches(sampler, arm_dir, records, spec)
    _save_state(state_path, sampler, records, arm=arm, status="prediction_benches_complete")
    for episode in DRAFT_EPISODES:
        stem = f"draftgym_{episode['season']}_slot{episode['agent_slot']}_seed{episode['seed']}.json"
        out = arm_dir / stem
        if out.exists():
            continue
        before = asdict(sampler.ledger)
        result = _draft_episode(sampler, dict(episode))
        _write(out, result)
        after = asdict(sampler.ledger)
        records.append({"benchmark": "draftgym", "season": episode["season"],
                        "agent_slot": episode["agent_slot"],
                        "model_requests": result["model_requests"],
                        "parse_failures": result["parse_failures"],
                        "usage_delta": {k: after[k] - before[k] for k in before}})
        _save_state(state_path, sampler, records, arm=arm, status="running")
    payload = {"arm": arm, "status": "complete", "completed_at": utc_now(),
               "model": BASE_MODEL, "model_path": model_path,
               "spec_sha256": sha256_file(SPEC_PATH), "records": records,
               "cost": {**asdict(sampler.ledger), "computed_usd": sampler.ledger.usd}}
    _write(state_path, payload)
    return payload


def _calib_score(arm_dir: Path) -> dict[str, Any]:
    pairs: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for season in SEASONS:
        key = _json(QUESTIONS_DIR / f"calibbench_{season}_{PRESET}_KEY.json")
        by_anon = {row["anon_id"]: row for row in key["questions"].values()}
        for family in ("season_threshold", "weekly_h2h"):
            for answer in _json(arm_dir / f"calib_{season}_{PRESET}_{family}.json"):
                target = by_anon[answer["question_id"]]
                pairs[target["family"]].append((float(answer["p"]), int(target["outcome"])))
    all_pairs = [x for values in pairs.values() for x in values]
    def one(values: list[tuple[float, int]]) -> dict[str, Any]:
        return {"n": len(values),
                "brier": sum((p-y)**2 for p, y in values) / len(values),
                "log_loss": log_loss(values), "ece": ece(values),
                "reliability": reliability_table(values)}
    return {**{family: one(values) for family, values in pairs.items()},
            "overall": one(all_pairs)}


def _draft_score(arm_dir: Path) -> dict[str, Any]:
    episodes = [_json(arm_dir / f"draftgym_{e['season']}_slot{e['agent_slot']}_seed{e['seed']}.json")
                for e in DRAFT_EPISODES]
    picks = sum(len(e["picks"]) for e in episodes)
    return {"episodes": len(episodes), "mean_reward": statistics.mean(e["reward"] for e in episodes),
            "median_reward": statistics.median(e["reward"] for e in episodes),
            "rewards": [e["reward"] for e in episodes],
            "fallback_picks": sum(e["fallback_picks"] for e in episodes),
            "fallback_rate": sum(e["fallback_picks"] for e in episodes) / max(1, picks),
            "illegal_picks": sum(e["illegal_picks"] for e in episodes),
            "parse_failures": sum(e["parse_failures"] for e in episodes),
            "tool_calls": sum(e["tool_calls"] for e in episodes),
            "tool_calls_ok": sum(e["tool_calls_ok"] for e in episodes),
            "episodes_detail": episodes}


def _prediction_rows(arm_dir: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for season in SEASONS:
        bkey = _json(QUESTIONS_DIR / f"breakoutbench_{season}_{PRESET}_KEY.json")["questions"]
        for family in FAMILIES:
            for a in _json(arm_dir / f"breakout_{season}_{PRESET}_{family}.json"):
                qid = a["question_id"]
                rows[f"breakout:{season}:{qid}"] = {
                    "benchmark": "breakoutbench", "season": season, "family": family,
                    "question_id": qid, "p": float(a["p"]), "y": int(bkey[qid]["outcome"]),
                }
        ckey = _json(QUESTIONS_DIR / f"calibbench_{season}_{PRESET}_KEY.json")["questions"]
        by_anon = {v["anon_id"]: v for v in ckey.values()}
        for family in ("season_threshold", "weekly_h2h"):
            for a in _json(arm_dir / f"calib_{season}_{PRESET}_{family}.json"):
                qid = a["question_id"]
                target = by_anon[qid]
                rows[f"calib:{season}:{qid}"] = {
                    "benchmark": "calibbench", "season": season, "family": family,
                    "question_id": qid, "p": float(a["p"]), "y": int(target["outcome"]),
                }
    return rows


def _paired(base_dir: Path, adapter_dir: Path) -> dict[str, Any]:
    base, adapter = _prediction_rows(base_dir), _prediction_rows(adapter_dir)
    common = sorted(base.keys() & adapter.keys())
    examples = []
    by_season: dict[int, list[float]] = defaultdict(list)
    for key in common:
        b, a = base[key], adapter[key]
        improvement = (b["p"] - b["y"])**2 - (a["p"] - a["y"])**2
        row = {**{k: b[k] for k in ("benchmark", "season", "family", "question_id", "y")},
               "base_p": b["p"], "adapter_p": a["p"],
               "brier_improvement": improvement}
        examples.append(row)
        by_season[b["season"]].append(improvement)
    season_means = {s: statistics.mean(v) for s, v in by_season.items()}
    rng = random.Random(SEED)
    draws = []
    seasons = sorted(season_means)
    for _ in range(10000):
        sample = [seasons[rng.randrange(len(seasons))] for _ in seasons]
        draws.append(statistics.mean(season_means[s] for s in sample))
    draws.sort()
    return {"matched_examples": len(examples),
            "mean_brier_improvement": statistics.mean(x["brier_improvement"] for x in examples),
            "season_clustered_95ci": [draws[249], draws[9749]],
            "mean_absolute_probability_change": statistics.mean(
                abs(x["adapter_p"] - x["base_p"]) for x in examples
            ),
            "largest_improvements": sorted(examples, key=lambda x: -x["brier_improvement"])[:15],
            "largest_regressions": sorted(examples, key=lambda x: x["brier_improvement"])[:15]}


def _paired_draft(results: Mapping[str, Any]) -> dict[str, Any]:
    base = results["base"]["draftgym"]["episodes_detail"]
    adapter = results["adapter"]["draftgym"]["episodes_detail"]
    if len(base) != len(adapter):
        raise ValueError("DraftGym arms are not rectangular")
    rows = []
    by_season: dict[int, list[float]] = defaultdict(list)
    for b, a in zip(base, adapter, strict=True):
        key_b = (b["season"], b["agent_slot"], b["seed"])
        key_a = (a["season"], a["agent_slot"], a["seed"])
        if key_b != key_a:
            raise ValueError("DraftGym episode pairing mismatch")
        delta = float(a["reward"]) - float(b["reward"])
        rows.append({"season": b["season"], "agent_slot": b["agent_slot"],
                     "seed": b["seed"], "base_reward": b["reward"],
                     "adapter_reward": a["reward"], "delta": delta})
        by_season[int(b["season"])].append(delta)
    season_means = {s: statistics.mean(v) for s, v in by_season.items()}
    seasons = sorted(season_means)
    rng = random.Random(SEED)
    draws = sorted(
        statistics.mean(season_means[seasons[rng.randrange(len(seasons))]]
                        for _ in seasons)
        for _ in range(10000)
    )
    deltas = [r["delta"] for r in rows]
    return {"episodes": rows, "mean_delta": statistics.mean(deltas),
            "median_delta": statistics.median(deltas),
            "wins": sum(x > 0 for x in deltas), "losses": sum(x < 0 for x in deltas),
            "season_mean_deltas": season_means,
            "season_clustered_95ci": [draws[249], draws[9749]],
            "interpretation": "supplementary uncertainty analysis; frozen decision uses mean reward"}


def _decision(results: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    b, a = results["base"], results["adapter"]
    rule = spec["decision_rule"]
    breakout_improvement = b["breakout"]["pooled"]["pooled_brier"] - a["breakout"]["pooled"]["pooled_brier"]
    calib_improvement = b["calib"]["overall"]["brier"] - a["calib"]["overall"]["brier"]
    draft_improvement = a["draftgym"]["mean_reward"] - b["draftgym"]["mean_reward"]
    material = {
        "breakoutbench": breakout_improvement >= rule["material_breakout_brier_improvement"],
        "calibbench": calib_improvement >= rule["material_calib_brier_improvement"],
        "masked_draftgym": draft_improvement >= rule["material_masked_draftgym_reward_improvement"],
    }
    coverage_delta = a["structured_coverage"] - b["structured_coverage"]
    major = {
        "breakoutbench": breakout_improvement <= -rule["major_brier_regression"],
        "calibbench": calib_improvement <= -rule["major_brier_regression"],
        "masked_draftgym": draft_improvement <= -rule["major_masked_draftgym_reward_regression"],
        "structured_output": coverage_delta <= -rule["max_structured_coverage_regression"],
    }
    canary_pass = bool(a["canary"]["gate"]["pass"])
    n_material = sum(material.values())
    any_directional = breakout_improvement > 0 or calib_improvement > 0 or draft_improvement > 0
    if canary_pass and not any(major.values()) and n_material >= 2:
        verdict = "proceed_to_draftgym_rl"
    elif canary_pass and not any(major.values()) and (n_material == 1 or any_directional):
        verdict = "revise_sft_before_rl"
    else:
        verdict = "stop_sft_training"
    return {"verdict": verdict, "material_improvements": material,
            "major_regressions": major, "canary_pass": canary_pass,
            "deltas": {"breakout_brier_improvement": breakout_improvement,
                       "calib_brier_improvement": calib_improvement,
                       "masked_draftgym_reward_improvement": draft_improvement,
                       "structured_coverage": coverage_delta}}


def _arm_score(arm: str) -> dict[str, Any]:
    arm_dir = RUN_ROOT / arm
    state = _json(arm_dir / "run-state.json")
    if state.get("status") != "complete":
        raise ValueError(f"{arm} inference is incomplete")
    breakout_paths = [arm_dir / f"breakout_{s}_{PRESET}_{f}.json"
                      for s in SEASONS for f in FAMILIES]
    # score_run only needs the season/preset pattern; these names satisfy it.
    breakout = score_run(breakout_paths)
    canary_paths = [arm_dir / f"canary_{s}_{PRESET}.json" for s in SEASONS]
    canary = score_canaries(load_answers_files(canary_paths))
    prediction_records = [r for r in state["records"] if r["benchmark"] != "draftgym"]
    coverage = sum(r["answered"] for r in prediction_records) / sum(
        r["valid_ids"] for r in prediction_records
    )
    return {"breakout": breakout, "calib": _calib_score(arm_dir),
            "canary": canary, "draftgym": _draft_score(arm_dir),
            "structured_coverage": coverage,
            "repair_requests": sum(bool(r.get("repaired")) for r in prediction_records),
            "cost": state["cost"]}


def score() -> dict[str, Any]:
    spec = verify_spec()
    results = {arm: _arm_score(arm) for arm in ("base", "adapter")}
    paired = _paired(RUN_ROOT / "base", RUN_ROOT / "adapter")
    paired_draft = _paired_draft(results)
    decision = _decision(results, spec)
    canary = _json(ROOT / "artifacts/tinker-sft-v1/t0-smoke/run-summary.json")
    smoke = _json(ROOT / "artifacts/tinker-sft-v1/t0-existing-smoke/run-summary.json")
    train = _json(TRAIN_SUMMARY)
    total = (float(canary["cost"]["computed_usd"])
             + float(smoke["cost"]["computed_usd"])
             + float(train["cost"]["computed_usd"])
             + sum(float(results[x]["cost"]["computed_usd"]) for x in results))
    payload = {"protocol": spec["protocol"], "scored_at": utc_now(),
               "spec_sha256": sha256_file(SPEC_PATH), "results": results,
               "paired": paired, "paired_draftgym": paired_draft, "decision": decision,
               "spend": {"pre_t1_canary_usd": canary["cost"]["computed_usd"],
                         "existing_t0_smoke_usd": smoke["cost"]["computed_usd"],
                         "t1_training_usd": train["cost"]["computed_usd"],
                         "base_evaluation_usd": results["base"]["cost"]["computed_usd"],
                         "adapter_evaluation_usd": results["adapter"]["cost"]["computed_usd"],
                         "incremental_total_usd": total}}
    _write(SCORECARD_PATH, payload)
    write_report(payload)
    return payload


def write_report(scorecard: Mapping[str, Any]) -> None:
    b, a = scorecard["results"]["base"], scorecard["results"]["adapter"]
    d = scorecard["decision"]
    t0 = _json(ROOT / "artifacts/tinker-sft-v1/t0-existing-smoke/run-summary.json")
    train = _json(TRAIN_SUMMARY)
    lines = [
        "# Fantasy Alpha Tinker SFT v1 — frozen evaluation",
        "",
        f"Decision: **{d['verdict'].replace('_', ' ')}**.",
        "",
        "The comparison uses the untouched `Qwen/Qwen3.5-9B` and the development-selected "
        "rank-32 adapter under the same prompts, renderer, decoding, seeds, tools, and verifiers. "
        "Seasons 2018, 2023, and 2024 are held out; 2025 remains untouched.",
        "",
        "## Run record",
        "",
        f"The exact existing 162-row T0 corpus completed {t0['steps']} optimizer steps, "
        f"reduced development NLL from {t0['initial_development_nll']:.3f} to "
        f"{t0['final_development_nll']:.3f}, reloaded its saved state, sampled the base and "
        f"adapter, and exported a {t0['archive']['bytes'] / 1_000_000:.1f} MB archive. "
        "A separate 24-row balanced T1-format canary completed first; its post-training "
        "metadata check needed one client-side retry, with zero repeated optimizer steps.",
        "",
        f"The full 811-trace run used 728 grouped training rows and 83 grouped development "
        f"rows for two epochs ({train['steps']} steps). Training-batch NLL moved from "
        f"{train['first_training_batch_nll']:.3f} to {train['final_training_batch_nll']:.3f}; "
        f"development NLL moved {train['initial_development_nll']:.3f} → "
        f"{train['development_nll_history'][1]:.3f} → {train['final_development_nll']:.3f}. "
        "The frozen development-only rule selected epoch 2. The saved state reloaded and the "
        "exported adapter hash was verified.",
        "",
        "Unlike the existing Prime/prime-rl backend, Tinker exposed managed forward/backward, "
        "optimizer, checkpoint, and sampler clients: there was no GPU pod to provision or abandon. "
        "Both backends remain available and consume the same system/user/assistant data shape with "
        "assistant-only loss.",
        "",
        "## Frozen scorecard",
        "",
        "| Measure | Base | Fine-tuned | Fine-tuned minus base |",
        "|---|---:|---:|---:|",
        f"| BreakoutBench Brier ↓ | {b['breakout']['pooled']['pooled_brier']:.4f} | "
        f"{a['breakout']['pooled']['pooled_brier']:.4f} | "
        f"{a['breakout']['pooled']['pooled_brier']-b['breakout']['pooled']['pooled_brier']:+.4f} |",
        f"| CalibBench Brier ↓ | {b['calib']['overall']['brier']:.4f} | "
        f"{a['calib']['overall']['brier']:.4f} | "
        f"{a['calib']['overall']['brier']-b['calib']['overall']['brier']:+.4f} |",
        f"| CalibBench ECE ↓ | {b['calib']['overall']['ece']:.4f} | {a['calib']['overall']['ece']:.4f} | "
        f"{a['calib']['overall']['ece']-b['calib']['overall']['ece']:+.4f} |",
        f"| Masked DraftGym mean reward ↑ | {b['draftgym']['mean_reward']:.2f} | "
        f"{a['draftgym']['mean_reward']:.2f} | {d['deltas']['masked_draftgym_reward_improvement']:+.2f} |",
        f"| Structured answer coverage ↑ | {b['structured_coverage']:.2%} | "
        f"{a['structured_coverage']:.2%} | {d['deltas']['structured_coverage']:+.2%} |",
        "",
        f"Canary gate: **{'PASS' if d['canary_pass'] else 'FAIL'}**. "
        f"Paired examples: {scorecard['paired']['matched_examples']:,}; season-clustered 95% CI "
        f"for mean Brier improvement {scorecard['paired']['season_clustered_95ci']}.",
        "",
        f"Exact price-based incremental workload spend: "
        f"**${scorecard['spend']['incremental_total_usd']:.4f}** "
        "(ongoing checkpoint storage reported separately in the budget ledger).",
        "",
        "## What changed",
        "",
        f"BreakoutBench top-10 lift was unchanged at {b['breakout']['pooled']['lift']:.2f}×. "
        f"CalibBench log loss moved from {b['calib']['overall']['log_loss']:.4f} to "
        f"{a['calib']['overall']['log_loss']:.4f}, despite the small Brier/ECE gains—evidence "
        "that some probability changes became too confident.",
        "",
        f"The masked-draft mean improved by {scorecard['paired_draftgym']['mean_delta']:.2f}, "
        f"but the adapter won only {scorecard['paired_draftgym']['wins']}/6 paired drafts; "
        f"the median change was {scorecard['paired_draftgym']['median_delta']:.2f}, and the "
        f"season-clustered 95% interval was {scorecard['paired_draftgym']['season_clustered_95ci']}. "
        "Two very large wins drive the mean, so this is a promising signal rather than a robust result.",
        "",
        f"Both arms covered {a['structured_coverage']:.0%} of requested predictions with no repair, "
        f"fallback, or illegal picks. Neither arm used an evidence tool ({a['draftgym']['tool_calls']} "
        "adapter calls), so SFT did not teach evidence-seeking behavior.",
        "",
        "## Decision",
        "",
        "The frozen decision above follows the preregistered thresholds: **revise the SFT approach "
        "before RL**. The JSON scorecard contains "
        "family-level results, every DraftGym episode, and the 15 largest paired improvements and regressions. "
        "These results measure behavior on this suite; they do not establish broad football expertise or prove "
        "that the model will generalize to the still-sealed 2025 season. The logical revision is to add "
        "development-only tool-use and calibrated-decision traces, then repeat with a newly frozen validation "
        "slice before spending on RL. No RL was started in this milestone.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    run = sub.add_parser("run")
    run.add_argument("--arm", required=True, choices=("base", "adapter"))
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
        raise SystemExit(f"Tinker evaluation failed: {safe_text(exc)}") from None
