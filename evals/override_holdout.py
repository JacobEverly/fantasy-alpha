"""One-shot 2025 evaluation for the high-confidence override adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from envs.draftgym import realistic_roster_points
from evals.draftbench import (
    DEFAULT_ROSTER, DEFAULT_TEAMS, AutopickADP, DraftAgent, DraftSimulator,
    _board_key, feasible_players, load_board, team_on_the_clock,
)
from evals.outcome_headroom import robust_metrics
from harness.league import LeagueConfig
from harness.scoring import PRESETS
from training.outcome_decisions import decision_features
from training.outcome_ranker import encode
from training.override_dataset import SYSTEM_PROMPT, _visible_prompt
from training.risk_aware_ranker import RiskAwareRankerAgent, load_artifact
from training.tinker_backend import ROOT, TinkerChatSampler, sha256_file, utc_now

SPEC_PATH = ROOT / "training/override-eval-spec-v1.json"
TRAIN_SUMMARY = ROOT / "artifacts/high-confidence-override-v1/tinker-run/run-summary.json"
RESULT_PATH = ROOT / "artifacts/high-confidence-override-v1/holdout-2025.json"


def _parse_action(content: str, visible_ids: set[str]) -> tuple[str, str | None]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        action = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("response is not one JSON object") from exc
    if not isinstance(action, dict) or set(action) not in (
        {"decision"}, {"decision", "candidate_id"}
    ):
        raise ValueError("response has an unexpected schema")
    decision = str(action.get("decision", ""))
    if decision == "KEEP_ADP" and set(action) == {"decision"}:
        return decision, None
    if decision == "OVERRIDE" and set(action) == {"decision", "candidate_id"}:
        candidate = str(action["candidate_id"])
        if candidate in visible_ids:
            return decision, candidate
    raise ValueError("response is not a legal action")


class ModelOverrideAgent(DraftAgent):
    name = "model_override"

    def __init__(self, sampler: TinkerChatSampler, *, seed_base: int, teacher: Mapping[str, Any]):
        self.sampler = sampler
        self.seed_base = seed_base
        self.teacher = teacher
        self.slot: int | None = None
        self.turn = 0
        self.records: list[dict[str, Any]] = []

    def pick(self, board, my_roster, league, pick_number):
        if self.slot is None:
            self.slot = team_on_the_clock(league, pick_number)
        legal = feasible_players(board, my_roster, league, pick_number)
        candidates = sorted(legal, key=_board_key)[:5]
        adp = candidates[0]
        if len(candidates) == 1:
            return adp.player_id
        state = {
            "board": tuple(board), "roster": tuple(my_roster),
            "pick_number": pick_number, "candidates": tuple(legal),
        }
        masked = {player.player_id: f"B{index + 1:03d}" for index, player in enumerate(candidates)}
        prompt_rows = [{
            "candidate_id": masked[player.player_id],
            "features": decision_features(state, player, league=league, slot=self.slot),
        } for player in candidates]
        x = np.asarray([encode(row["features"]) for row in prompt_rows], dtype=float)
        means = self.teacher["models"]["mean"].predict(x)
        downsides = self.teacher["models"]["downside"].predict(x)
        alternative = max(
            range(1, len(candidates)),
            key=lambda index: (
                float(means[index]), float(downsides[index]), -candidates[index].adp,
            ),
        )
        mean_edge = float(means[alternative] - means[0])
        downside_edge = float(downsides[alternative] - downsides[0])
        policy = self.teacher["policy"]
        teacher_override = (
            mean_edge >= float(policy["mean_margin"])
            and downside_edge >= float(policy["downside_margin"])
        )
        teacher_id = masked[candidates[alternative].player_id] if teacher_override else None
        before = asdict(self.sampler.ledger)
        response = self.sampler.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _visible_prompt(prompt_rows)},
            ],
            max_tokens=48, temperature=0.0, seed=self.seed_base + self.turn,
        )
        after = asdict(self.sampler.ledger)
        valid = True
        try:
            decision, candidate_id = _parse_action(
                response["content"], set(masked.values())
            )
        except ValueError:
            valid = False
            decision, candidate_id = "KEEP_ADP", None
        reverse = {value: key for key, value in masked.items()}
        selected = reverse[candidate_id] if decision == "OVERRIDE" else adp.player_id
        self.records.append({
            "turn": self.turn,
            "overall_pick": pick_number,
            "valid": valid,
            "decision": decision,
            "candidate_id": candidate_id,
            "teacher_decision": "OVERRIDE" if teacher_override else "KEEP_ADP",
            "teacher_candidate_id": teacher_id,
            "teacher_mean_edge": mean_edge,
            "teacher_downside_edge": downside_edge,
            "exact_teacher_action": (
                decision == ("OVERRIDE" if teacher_override else "KEEP_ADP")
                and (not teacher_override or candidate_id == teacher_id)
            ),
            "usage_delta": {key: after[key] - before[key] for key in before},
            "response": response["content"],
        })
        self.turn += 1
        return selected


def _behavior(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    teacher_positive = [row for row in records if row["teacher_decision"] == "OVERRIDE"]
    teacher_negative = [row for row in records if row["teacher_decision"] == "KEEP_ADP"]
    predicted = [row for row in records if row["decision"] == "OVERRIDE"]
    true_positive = [row for row in predicted if row["teacher_decision"] == "OVERRIDE"]
    exact = [row for row in predicted if row["exact_teacher_action"]]
    false_override = [row for row in predicted if row["teacher_decision"] == "KEEP_ADP"]
    precision = len(true_positive) / len(predicted) if predicted else 1.0
    recall = len(true_positive) / len(teacher_positive) if teacher_positive else 0.0
    exact_precision = len(exact) / len(predicted) if predicted else 1.0
    exact_recall = len(exact) / len(teacher_positive) if teacher_positive else 0.0
    return {
        "decisions": len(records),
        "valid_structured_output_rate": sum(bool(row["valid"]) for row in records) / len(records),
        "teacher_override_states": len(teacher_positive),
        "predicted_override_states": len(predicted),
        "teacher_override_precision": precision,
        "teacher_override_recall": recall,
        "exact_teacher_action_precision": exact_precision,
        "exact_teacher_action_recall": exact_recall,
        "exact_teacher_action_f1": (
            2 * exact_precision * exact_recall / (exact_precision + exact_recall)
            if exact_precision + exact_recall else 0.0
        ),
        "false_overrides": len(false_override),
        "false_override_rate_on_teacher_keep_states": (
            len(false_override) / len(teacher_negative) if teacher_negative else 0.0
        ),
        "intervention_rate": len(predicted) / len(records),
    }


def _product(deltas: Sequence[float]) -> dict[str, Any]:
    result = robust_metrics(list(deltas))
    result["catastrophic_drafts_below_minus_100"] = sum(value <= -100 for value in deltas)
    return result


def evaluate_model(
    *, label: str, model_path: str | None, specs: Sequence[tuple[int, int]],
    pool: Sequence[Any], weekly: Mapping[str, Any], league: LeagueConfig,
    adp_points: Mapping[str, float], teacher: Mapping[str, Any],
) -> dict[str, Any]:
    sampler = TinkerChatSampler(
        model_path=model_path, experiment="high-confidence-override-v1"
    )
    episodes = []
    all_records = []
    for episode_index, (slot, seed) in enumerate(specs):
        agent = ModelOverrideAgent(
            sampler, seed_base=20260907 + episode_index * 100, teacher=teacher
        )
        result = DraftSimulator(pool, league).simulate(agent, slot, seed)
        points = realistic_roster_points(result.agent_roster, weekly, league)
        key = f"slot{slot}:seed{seed}"
        episodes.append({
            "episode": key, "points": points, "adp_points": adp_points[key],
            "points_above_adp": points - adp_points[key],
            "decisions": len(agent.records),
        })
        for record in agent.records:
            all_records.append({"episode": key, **record})
    return {
        "label": label,
        "model_path": model_path or "Qwen/Qwen3.5-9B",
        "behavior": _behavior(all_records),
        "product": _product([row["points_above_adp"] for row in episodes]),
        "cost": {**asdict(sampler.ledger), "computed_usd": sampler.ledger.usd},
        "episodes": episodes,
        "records": all_records,
    }


def run() -> dict[str, Any]:
    if RESULT_PATH.exists():
        raise RuntimeError("refusing to overwrite the one-shot 2025 result")
    spec = json.loads(SPEC_PATH.read_text())
    if spec["status"] != "frozen-before-paid-training-and-before-opening-2025":
        raise ValueError("evaluation spec is not frozen")
    summary = json.loads(TRAIN_SUMMARY.read_text())
    if summary.get("status") != "complete":
        raise ValueError("adapter training must be complete before opening 2025")
    adapter_path = str(summary["final_sampler_path"])
    loaded = load_board(2025, "ppr")
    if loaded is None:
        raise ValueError("2025 board/outcomes are unavailable")
    pool, weekly = loaded
    league = LeagueConfig(
        teams=DEFAULT_TEAMS, roster=dict(DEFAULT_ROSTER), scoring=PRESETS["ppr"]
    )
    specs = [(slot, seed) for slot in (1, 6, 12) for seed in range(5)]
    adp_points = {}
    teacher_points = {}
    teacher = load_artifact()
    for slot, seed in specs:
        key = f"slot{slot}:seed{seed}"
        adp = DraftSimulator(pool, league).simulate(AutopickADP(), slot, seed)
        adp_points[key] = realistic_roster_points(adp.agent_roster, weekly, league)
        teacher_agent = RiskAwareRankerAgent(
            teacher["models"],
            mean_margin=teacher["policy"]["mean_margin"],
            downside_margin=teacher["policy"]["downside_margin"],
        )
        teacher_result = DraftSimulator(pool, league).simulate(teacher_agent, slot, seed)
        teacher_points[key] = realistic_roster_points(
            teacher_result.agent_roster, weekly, league
        )
    base = evaluate_model(
        label="base", model_path=None, specs=specs, pool=pool, weekly=weekly,
        league=league, adp_points=adp_points, teacher=teacher,
    )
    adapter = evaluate_model(
        label="adapter", model_path=adapter_path, specs=specs, pool=pool,
        weekly=weekly, league=league, adp_points=adp_points, teacher=teacher,
    )
    teacher_product = _product([
        teacher_points[key] - adp_points[key] for key in sorted(adp_points)
    ])
    gates = {
        "valid_structure": adapter["behavior"]["valid_structured_output_rate"] >= 0.98,
        "override_precision": adapter["behavior"]["teacher_override_precision"] >= 0.50,
        "exact_action_recall": adapter["behavior"]["exact_teacher_action_recall"] >= 0.25,
        "false_override_safety": adapter["behavior"]["false_override_rate_on_teacher_keep_states"] <= 0.02,
        "sparse_intervention": adapter["behavior"]["intervention_rate"] <= 0.10,
        "behavior_improves_over_base": (
            adapter["behavior"]["exact_teacher_action_f1"]
            > base["behavior"]["exact_teacher_action_f1"]
        ),
        "product_safety": (
            adapter["product"]["mean_points_above_adp"] >= -5
            and adapter["product"]["catastrophic_drafts_below_minus_100"]
            <= base["product"]["catastrophic_drafts_below_minus_100"]
        ),
    }
    gates["pass"] = all(gates.values())
    payload = {
        "status": "complete", "completed_at": utc_now(),
        "spec_path": str(SPEC_PATH.relative_to(ROOT)),
        "spec_sha256": sha256_file(SPEC_PATH),
        "training_summary_sha256": sha256_file(TRAIN_SUMMARY),
        "adp": {"mean_points": float(np.mean(list(adp_points.values())))},
        "teacher": {"product": teacher_product},
        "base": base, "adapter": adapter, "gates": gates,
        "incremental_cost_usd": (
            float(summary["cost"]["computed_usd"])
            + float(base["cost"]["computed_usd"])
            + float(adapter["cost"]["computed_usd"])
        ),
        "paired_base_adapter_episode_deltas": [
            {
                "episode": b["episode"],
                "adapter_minus_base_points": a["points"] - b["points"],
            }
            for b, a in zip(base["episodes"], adapter["episodes"], strict=True)
        ],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-one-shot", action="store_true", required=True)
    parser.parse_args()
    report = run()
    print(json.dumps({
        "gates": report["gates"],
        "base_behavior": report["base"]["behavior"],
        "adapter_behavior": report["adapter"]["behavior"],
        "base_product": report["base"]["product"],
        "adapter_product": report["adapter"]["product"],
        "incremental_cost_usd": report["incremental_cost_usd"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
