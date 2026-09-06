"""Build and validate Fantasy Alpha's targeted T1.1 SFT corpus.

T1 taught calibrated forecast prose but contained no model-authored DraftGym
actions: evidence was pre-fetched into user messages.  T1.1 is deliberately a
small, development-only replacement corpus that adds observable draft actions,
tool decisions, post-tool choices, and conservative forecast corrections.

No target in this file uses realized season outcomes.  Draft labels come from
the already-benchmarked market-only ``SurvivalSequencer`` policy plus its
roster/scarcity guardrails.  Forecast labels shrink the old teacher estimate
toward its strictly-prior historical anchor.  The 2025 gate is never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.agents_execution import SurvivalSequencer, survival_probability
from evals.draftbench import DraftPlayer, position_counts
from envs.draftgym import DraftGym
from envs.play_llm import build_messages, parse_action
from training.filter import parse_final_p, uncited_numbers, within_band
from training.tinker_backend import ROOT, TRAIN_SEASONS, load_corpus, sha256_file

CORPUS_PATH = ROOT / "training/datasets/t11_targeted_v1.jsonl"
MANIFEST_PATH = ROOT / "training/datasets/t11_targeted_v1.manifest.json"
DATASET_CARD_PATH = ROOT / "docs/models/fantasy-alpha-t11-dataset-card.md"

TARGET_ROWS = 300
TARGET_FORECAST_ROWS = 120
TARGET_DIRECT_PICK_ROWS = 120
TARGET_TOOL_GROUPS = 30  # one tool action + one post-tool pick each
DATASET_SEED = 20260905
DRAFT_SEASONS = tuple(sorted(TRAIN_SEASONS & set(range(2015, 2023))))
DRAFT_SLOTS = (1, 4, 7, 10, 12)
DRAFT_SEEDS = (3, 11, 29)
FORBIDDEN_DRAFT_INPUT_FIELDS = (
    '"reward"', '"realized', '"outcome"', '"agent_points', '"control_points',
    '"value_ledger"',
)


def _stable_key(value: object) -> str:
    return hashlib.sha256(f"{DATASET_SEED}:{value}".encode()).hexdigest()


def _compact_action(action: Mapping[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def _row(
    *,
    trace_id: str,
    season: int,
    question_id: str,
    messages: Sequence[Mapping[str, str]],
    subtype: str,
    categories: Sequence[str],
    sample_idx: int = 0,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    categories = sorted(set(categories))
    if subtype == "calibration_correction":
        split_family = f"calibration_{(extra_meta or {}).get('source_family', 'unknown')}"
    elif subtype in {"tool_call", "post_tool_pick"}:
        split_family = "draft_tool"
    elif "recovery" in categories:
        split_family = "draft_recovery"
    else:
        split_family = "draft_direct"
    meta = {
        "track": "anonymized",
        "trace_id": trace_id,
        "bench": "draftgym" if subtype != "calibration_correction" else "forecast",
        # ``grouped_development_split`` stratifies on this field.  Keep each
        # forecast family, ordinary draft action, recovery, and complete tool
        # pair represented in the 10% checkpoint-selection slice.
        "family": split_family,
        "season": int(season),
        "question_id": question_id,
        "sample_idx": int(sample_idx),
        "t11_subtype": subtype,
        "categories": categories,
        "label_uses_realized_outcome": False,
    }
    if extra_meta:
        meta.update(extra_meta)
    return {"messages": [dict(m) for m in messages], "meta": meta}


_FORECAST_TEMPLATES = (
    "The strictly-prior historical rate is the strongest anchor. The packet "
    "supports moving {direction}, but not far enough to justify an extreme "
    "forecast. This remains {verbal}.\n\nFINAL PROBABILITY: {p:.2f}",
    "Start with the code-computed base rate and treat the packet as a modest "
    "update. The evidence moves the estimate {direction}; uncertainty remains "
    "material, so a tail probability is not warranted. The outcome is "
    "{verbal}.\n\nFINAL PROBABILITY: {p:.2f}",
    "The evidence does not support overriding the historical grounding with a "
    "high-conviction call. I apply a conservative update {direction} and keep "
    "the estimate in a calibrated range. This is {verbal}.\n\n"
    "FINAL PROBABILITY: {p:.2f}",
    "The packet is informative, but its signals are not strong enough to carry "
    "the estimate far from the prior. After shrinking the directional update "
    "{direction}, I would describe this as {verbal}.\n\n"
    "FINAL PROBABILITY: {p:.2f}",
)


def _target_probability(row: Mapping[str, Any]) -> float:
    """Conservative, outcome-blind update around the strictly-prior anchor."""
    meta = row["meta"]
    anchor = float(meta["anchor_rate"])
    teacher = float(meta["p"])
    # Teacher evidence can move the prior by at most 0.15; retain one third of
    # that move.  A soft 0.08/0.85 rail specifically addresses T1's log-loss
    # failures without collapsing every prediction to 0.5.
    update = max(-0.15, min(0.15, teacher - anchor)) / 3.0
    return round(max(0.08, min(0.85, anchor + update)), 2)


def _verbal(p: float) -> str:
    if p < 0.25:
        return "unlikely"
    if p < 0.60:
        return "uncertain"
    if p < 0.80:
        return "more likely than not"
    return "likely, but not a near-lock"


def _direction(target: float, anchor: float) -> str:
    if target > anchor + 0.015:
        return "slightly upward"
    if target < anchor - 0.015:
        return "slightly downward"
    return "very little away from the anchor"


def build_forecast_rows() -> list[dict[str, Any]]:
    old = [r for r in load_corpus() if r["meta"]["track"] == "anonymized"]
    families = ("full_slate", "bust", "season_threshold", "weekly_h2h")
    out: list[dict[str, Any]] = []
    for family in families:
        candidates = [r for r in old if r["meta"]["family"] == family]
        # One target per question; prioritize teacher estimates farthest from
        # their anchors, then use a stable hash to make ties reproducible.
        unique: dict[tuple[int, str], dict[str, Any]] = {}
        for row in candidates:
            key = (int(row["meta"]["season"]), str(row["meta"]["question_id"]))
            current = unique.get(key)
            distance = abs(float(row["meta"]["p"]) - float(row["meta"]["anchor_rate"]))
            if current is None:
                unique[key] = row
                continue
            old_distance = abs(
                float(current["meta"]["p"]) - float(current["meta"]["anchor_rate"])
            )
            if (distance, _stable_key(row["meta"]["trace_id"])) > (
                old_distance, _stable_key(current["meta"]["trace_id"])
            ):
                unique[key] = row
        ranked = sorted(
            unique.values(),
            key=lambda r: (
                -abs(float(r["meta"]["p"]) - float(r["meta"]["anchor_rate"])),
                _stable_key(r["meta"]["trace_id"]),
            ),
        )[: TARGET_FORECAST_ROWS // len(families)]
        for source in ranked:
            meta = source["meta"]
            target = _target_probability(source)
            anchor = float(meta["anchor_rate"])
            template = _FORECAST_TEMPLATES[
                int(_stable_key(meta["trace_id"]), 16) % len(_FORECAST_TEMPLATES)
            ]
            assistant = template.format(
                direction=_direction(target, anchor), verbal=_verbal(target), p=target
            )
            qid = f"forecast:{meta['season']}:{meta['question_id']}"
            out.append(_row(
                trace_id=f"t11:{qid}",
                season=int(meta["season"]),
                question_id=qid,
                messages=[source["messages"][0], source["messages"][1], {
                    "role": "assistant", "content": assistant,
                }],
                subtype="calibration_correction",
                categories=(
                    "calibrated_probability", "confidence_correction",
                    "contradictory_evidence", "league_arithmetic",
                ),
                extra_meta={
                    "source_trace_id": meta["trace_id"],
                    "source_family": family,
                    "source_teacher_p": float(meta["p"]),
                    "anchor_rate": anchor,
                    "anchor_n": int(meta["anchor_n"]),
                    "target_p": target,
                    "target_formula": "clip(anchor + clip(teacher-anchor,-.15,.15)/3,.08,.85)",
                },
            ))
    if len(out) != TARGET_FORECAST_ROWS:
        raise AssertionError(f"expected {TARGET_FORECAST_ROWS} forecast rows, got {len(out)}")
    return sorted(out, key=lambda r: r["meta"]["trace_id"])


def _masked_id(gym: DraftGym, real_id: str) -> str:
    return gym._to_masked[real_id]  # stable public observation id


def _view(obs: Mapping[str, Any], masked_id: str) -> Mapping[str, Any]:
    return next(x for x in obs["top_available"] if x["player_id"] == masked_id)


def _agent_pick(
    agent: SurvivalSequencer, gym: DraftGym, obs: Mapping[str, Any] | None = None
) -> DraftPlayer:
    offered = tuple(gym._offered(gym.agent_slot))
    real_id = agent.pick(
        offered, tuple(gym._rosters[gym.agent_slot - 1]),
        gym.league, gym._pick_number,
    )
    if obs is not None and _masked_id(gym, real_id) not in {
        x["player_id"] for x in obs["top_available"]
    }:
        visible = {x["player_id"] for x in obs["top_available"]}
        feasible_ids = {p.player_id for p in gym._agent_candidates()}
        visible_board = tuple(
            p for p in offered
            if p.player_id in feasible_ids and _masked_id(gym, p.player_id) in visible
        )
        real_id = agent.pick(
            visible_board, tuple(gym._rosters[gym.agent_slot - 1]),
            gym.league, gym._pick_number,
        )
    return gym._by_id[real_id]


def _tool_opportunity(obs: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    if int(obs["round"]) < 5 or int(obs["lookups_remaining"]) <= 0:
        return False
    adp = max(float(target["adp"]), 1.0)
    uncertain = float(target["prev_season_points"]) <= 0 or (
        float(target["adp_stdev"]) / adp >= 0.12
    )
    return uncertain


def _depth_rank(response: Mapping[str, Any]) -> int | None:
    if not response.get("ok"):
        return None
    result = response.get("result")
    rows = result if isinstance(result, list) else [result]
    ranks = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("unit") != "Offense":
            continue
        value = row.get("depth_rank") or row.get("rank")
        try:
            ranks.append(int(value))
        except (TypeError, ValueError):
            continue
    return min(ranks) if ranks else None


def _draft_categories(
    obs: Mapping[str, Any], target: Mapping[str, Any], *, recovery: bool = False
) -> list[str]:
    cats = {"draft_timing", "league_arithmetic", "mistake_prevention"}
    round_number = int(obs["round"])
    counts = Counter(x["position"] for x in obs["my_roster"])
    required = obs["league"]["roster"]
    if counts[target["position"]] < int(required.get(target["position"], 0)):
        cats.add("roster_construction")
    if round_number >= 11:
        cats.add("late_round_strategy")
    if target["position"] not in {"QB", "TE"} and (
        counts["QB"] >= 1 or counts["TE"] >= 1
    ):
        cats.add("qb_te_guardrail")
    if recovery:
        cats.add("recovery")
        cats.add("roster_construction")
    if not _tool_opportunity(obs, target):
        cats.add("tool_avoidance")
    return sorted(cats)


def _draft_row(
    *, gym: DraftGym, obs: Mapping[str, Any], action: Mapping[str, Any],
    question_id: str, subtype: str, categories: Sequence[str], sample_idx: int = 0,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    messages = build_messages(obs) + [{"role": "assistant", "content": _compact_action(action)}]
    common = {
        "episode": {
            "season": gym.season, "slot": gym.agent_slot, "seed": gym.seed,
            "pick_number": obs["pick_number"], "round": obs["round"],
        },
        "label_policy": "survival_seq_scarcity1.5",
        "input_uses_realized_fields": False,
    }
    if extra_meta:
        common.update(extra_meta)
    return _row(
        trace_id=f"t11:{question_id}:{subtype}", season=gym.season,
        question_id=question_id, messages=messages, subtype=subtype,
        categories=categories, sample_idx=sample_idx, extra_meta=common,
    )


def _collect_direct_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season in DRAFT_SEASONS:
        for slot in DRAFT_SLOTS:
            for seed in DRAFT_SEEDS:
                gym = DraftGym(
                    season=season, preset="ppr", agent_slot=slot, seed=seed,
                    mask_names=True, enable_evidence=False,
                )
                agent = SurvivalSequencer(
                    scarcity_ratio=1.5, name="t11_survival_seq_scarcity1.5"
                )
                obs = gym.reset()
                while not obs["done"]:
                    player = _agent_pick(agent, gym, obs)
                    masked = _masked_id(gym, player.player_id)
                    if masked not in {x["player_id"] for x in obs["top_available"]}:
                        raise AssertionError("expert pick is outside the model-visible candidate set")
                    target = _view(obs, masked)
                    if not _tool_opportunity(obs, target):
                        qid = (
                            f"draft:{season}:slot{slot}:seed{seed}:pick{obs['pick_number']}"
                        )
                        rows.append(_draft_row(
                            gym=gym, obs=obs, action={"pick": masked}, question_id=qid,
                            subtype="direct_pick", categories=_draft_categories(obs, target),
                            extra_meta={
                                "target_survival_to_next_pick": (
                                    round(survival_probability(player, int(obs["pick_number"] + obs["picks_until_next_turn"])), 6)
                                    if obs["picks_until_next_turn"] is not None else None
                                ),
                            },
                        ))
                    obs, _, _, _ = gym.step({"pick": masked})
    return rows


def _collect_recovery_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season in DRAFT_SEASONS:
        for slot in DRAFT_SLOTS:
            for seed in DRAFT_SEEDS:
                gym = DraftGym(
                    season=season, preset="ppr", agent_slot=slot, seed=seed,
                    mask_names=True, enable_evidence=False,
                )
                agent = SurvivalSequencer(
                    scarcity_ratio=1.5, name="t11_recovery_scarcity1.5"
                )
                obs = gym.reset()
                # Deliberately create a legal but weak QB/TE-heavy opening.
                while not obs["done"] and int(obs["round"]) <= 4:
                    _agent_pick(agent, gym, obs)  # initialize policy state; label ignored
                    candidates = list(gym._agent_candidates())
                    counts = position_counts(tuple(gym._rosters[slot - 1]))
                    preferred = "QB" if counts.get("QB", 0) < 2 else "TE"
                    forced = [p for p in candidates if p.position == preferred]
                    player = min(forced or candidates, key=lambda p: (p.adp, p.name))
                    obs, _, _, _ = gym.step({"pick": _masked_id(gym, player.player_id)})
                if obs["done"]:
                    continue
                player = _agent_pick(agent, gym, obs)
                masked = _masked_id(gym, player.player_id)
                if masked not in {x["player_id"] for x in obs["top_available"]}:
                    continue
                target = _view(obs, masked)
                qid = f"recovery:{season}:slot{slot}:seed{seed}:pick{obs['pick_number']}"
                rows.append(_draft_row(
                    gym=gym, obs=obs, action={"pick": masked}, question_id=qid,
                    subtype="direct_pick", categories=_draft_categories(obs, target, recovery=True),
                    extra_meta={"forced_prefix": "two_qb_then_te_heavy_legal_opening"},
                ))
    return rows


def _collect_tool_candidates() -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    groups: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for season in DRAFT_SEASONS:
        for slot in DRAFT_SLOTS:
            for seed in DRAFT_SEEDS:
                gym = DraftGym(
                    season=season, preset="ppr", agent_slot=slot, seed=seed,
                    mask_names=True, enable_evidence=True,
                )
                agent = SurvivalSequencer(
                    scarcity_ratio=1.5, name="t11_tool_scarcity1.5"
                )
                obs = gym.reset()
                while not obs["done"]:
                    player = _agent_pick(agent, gym, obs)
                    masked = _masked_id(gym, player.player_id)
                    visible = {x["player_id"] for x in obs["top_available"]}
                    if masked not in visible:
                        raise AssertionError("expert pick is outside the model-visible candidate set")
                    target = _view(obs, masked)
                    if _tool_opportunity(obs, target):
                        before = obs
                        tool = {
                            "tool": {
                                "name": "depth_chart",
                                "arguments": {"team_or_player": masked},
                            }
                        }
                        after, _, _, _ = gym.step(tool)
                        response = after["tool_results"][-1]["response"]
                        rank = _depth_rank(response)
                        if rank is not None:
                            adjusted = player
                            if rank > 1:
                                feasible_ids = {
                                    p.player_id for p in gym._agent_candidates()
                                }
                                reduced = tuple(
                                    p for p in gym._offered(slot)
                                    if p.player_id != player.player_id
                                    and p.player_id in feasible_ids
                                    and _masked_id(gym, p.player_id) in visible
                                )
                                if reduced:
                                    alternate_id = agent.pick(
                                        reduced, tuple(gym._rosters[slot - 1]), gym.league,
                                        gym._pick_number,
                                    )
                                    alternate = gym._by_id[alternate_id]
                                    if _masked_id(gym, alternate.player_id) in visible:
                                        adjusted = alternate
                            adjusted_masked = _masked_id(gym, adjusted.player_id)
                            qid = (
                                f"tool:{season}:slot{slot}:seed{seed}:pick{before['pick_number']}"
                            )
                            common = [
                                "evidence_tool_use", "tool_precision", "mistake_prevention",
                            ]
                            call_row = _draft_row(
                                gym=gym, obs=before, action=tool, question_id=qid,
                                subtype="tool_call", categories=common,
                                sample_idx=0,
                                extra_meta={
                                    "tool_opportunity": True,
                                    "tool_target": masked,
                                    "expected_depth_rank": rank,
                                },
                            )
                            post_target = _view(after, adjusted_masked)
                            post_row = _draft_row(
                                gym=gym, obs=after, action={"pick": adjusted_masked},
                                question_id=qid, subtype="post_tool_pick",
                                categories=common + _draft_categories(after, post_target),
                                sample_idx=1,
                                extra_meta={
                                    "tool_opportunity": True,
                                    "tool_result_used": True,
                                    "queried_depth_rank": rank,
                                    "changed_pick_after_evidence": adjusted is not player,
                                },
                            )
                            groups.append((call_row, post_row, rank))
                            obs, _, _, _ = gym.step({"pick": adjusted_masked})
                            continue
                    obs, _, _, _ = gym.step({"pick": masked})
    return groups


def build_draft_rows() -> list[dict[str, Any]]:
    recovery = sorted(_collect_recovery_candidates(), key=lambda r: _stable_key(r["meta"]["trace_id"]))
    chosen_recovery = recovery[:24]
    direct = _collect_direct_candidates()
    recovery_ids = {r["meta"]["question_id"] for r in chosen_recovery}
    direct = [r for r in direct if r["meta"]["question_id"] not in recovery_ids]
    # Balance ordinary direct picks across early/middle/late rounds before the
    # stable-hash tie break, then fill any remainder globally.
    buckets: dict[str, list[dict[str, Any]]] = {"early": [], "middle": [], "late": []}
    for row in direct:
        rnd = int(row["meta"]["episode"]["round"])
        buckets["early" if rnd <= 4 else "middle" if rnd <= 10 else "late"].append(row)
    ordinary: list[dict[str, Any]] = []
    per_bucket = (TARGET_DIRECT_PICK_ROWS - len(chosen_recovery)) // 3
    for name in buckets:
        ordinary.extend(sorted(buckets[name], key=lambda r: _stable_key(r["meta"]["trace_id"]))[:per_bucket])
    if len(ordinary) + len(chosen_recovery) < TARGET_DIRECT_PICK_ROWS:
        used = {r["meta"]["trace_id"] for r in ordinary}
        rest = sorted(
            [r for r in direct if r["meta"]["trace_id"] not in used],
            key=lambda r: _stable_key(r["meta"]["trace_id"]),
        )
        ordinary.extend(rest[: TARGET_DIRECT_PICK_ROWS - len(chosen_recovery) - len(ordinary)])

    candidates = _collect_tool_candidates()
    starters = sorted([g for g in candidates if g[2] == 1], key=lambda g: _stable_key(g[0]["meta"]["trace_id"]))
    backups = sorted([g for g in candidates if g[2] > 1], key=lambda g: _stable_key(g[0]["meta"]["trace_id"]))
    n_backup = min(TARGET_TOOL_GROUPS // 2, len(backups))
    selected = backups[:n_backup] + starters[: TARGET_TOOL_GROUPS - n_backup]
    if len(selected) < TARGET_TOOL_GROUPS:
        used = {g[0]["meta"]["trace_id"] for g in selected}
        remaining = sorted(
            [g for g in candidates if g[0]["meta"]["trace_id"] not in used],
            key=lambda g: _stable_key(g[0]["meta"]["trace_id"]),
        )
        selected.extend(remaining[: TARGET_TOOL_GROUPS - len(selected)])
    if len(selected) != TARGET_TOOL_GROUPS:
        raise AssertionError(
            f"needed {TARGET_TOOL_GROUPS} useful tool groups, found {len(selected)}"
        )
    out = chosen_recovery + ordinary + [row for group in selected for row in group[:2]]
    if len(out) != TARGET_DIRECT_PICK_ROWS + 2 * TARGET_TOOL_GROUPS:
        raise AssertionError("draft-row target mismatch")
    return sorted(out, key=lambda r: r["meta"]["trace_id"])


def build_rows() -> list[dict[str, Any]]:
    rows = build_forecast_rows() + build_draft_rows()
    return sorted(rows, key=lambda r: r["meta"]["trace_id"])


def canary_subset(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """24 final-format rows: six forecasts, picks, and six complete tool pairs."""
    by_subtype: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_subtype.setdefault(row["meta"]["t11_subtype"], []).append(row)
    forecasts = sorted(
        by_subtype["calibration_correction"], key=lambda r: _stable_key(r["meta"]["trace_id"])
    )[:6]
    picks = sorted(
        by_subtype["direct_pick"], key=lambda r: _stable_key(r["meta"]["trace_id"])
    )[:6]
    tool_groups: dict[str, list[dict[str, Any]]] = {}
    for subtype in ("tool_call", "post_tool_pick"):
        for row in by_subtype[subtype]:
            tool_groups.setdefault(row["meta"]["question_id"], []).append(row)
    chosen_groups = sorted(tool_groups, key=_stable_key)[:6]
    result = forecasts + picks + [row for key in chosen_groups for row in tool_groups[key]]
    if len(result) != 24:
        raise AssertionError("T1.1 canary must contain exactly 24 rows")
    return sorted(result, key=lambda r: r["meta"]["trace_id"])


def _extract_observation(user: str) -> dict[str, Any]:
    marker = "Draft state (JSON):\n"
    if marker not in user:
        raise ValueError("draft prompt is missing its observation marker")
    tail = user.split(marker, 1)[1]
    return json.JSONDecoder().raw_decode(tail)[0]


def _future_evidence_dates(value: Any, as_of: date) -> list[str]:
    """Return absolute evidence timestamps that are not strictly pre-cutoff."""
    bad: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "published_utc" and isinstance(child, str):
                published = datetime.fromisoformat(child.replace("Z", "+00:00")).date()
                if published >= as_of:
                    bad.append(child)
            elif key == "snapshot" and isinstance(child, str) and re.fullmatch(
                r"\d{4}-\d{2}-\d{2}", child
            ):
                if date.fromisoformat(child) >= as_of:
                    bad.append(child)
            else:
                bad.extend(_future_evidence_dates(child, as_of))
    elif isinstance(value, list):
        for child in value:
            bad.extend(_future_evidence_dates(child, as_of))
    return bad


def _message_fingerprint(row: Mapping[str, Any]) -> str:
    text = "\n".join(str(m["content"]) for m in row["messages"])
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return normalized


def _simhash(text: str) -> int:
    """Compact near-duplicate fingerprint over token trigrams."""
    tokens = re.findall(r"[a-z0-9_.-]+", text)
    shingles = {" ".join(tokens[i:i + 3]) for i in range(max(1, len(tokens) - 2))}
    vector = [0] * 64
    for shingle in shingles:
        bits = int(hashlib.sha256(shingle.encode()).hexdigest()[:16], 16)
        for bit in range(64):
            vector[bit] += 1 if bits & (1 << bit) else -1
    return sum((1 << bit) for bit, value in enumerate(vector) if value >= 0)


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if len(rows) != TARGET_ROWS:
        errors.append(f"expected {TARGET_ROWS} rows, found {len(rows)}")
    trace_ids = [str(r.get("meta", {}).get("trace_id", "")) for r in rows]
    if len(set(trace_ids)) != len(trace_ids) or not all(trace_ids):
        errors.append("trace ids must be present and unique")
    category_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    season_counts: Counter[int] = Counter()
    family_counts: Counter[str] = Counter()
    prompt_hashes: set[str] = set()
    fingerprints: list[tuple[str, int, int]] = []
    tool_groups: Counter[str] = Counter()

    for i, row in enumerate(rows):
        meta = row.get("meta") or {}
        messages = row.get("messages") or []
        tid = str(meta.get("trace_id", f"row-{i}"))
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            errors.append(f"{tid}: expected system/user/assistant messages")
            continue
        if not all(isinstance(m.get("content"), str) and m["content"] for m in messages):
            errors.append(f"{tid}: empty message content")
            continue
        season = int(meta.get("season", -1))
        if season not in TRAIN_SEASONS:
            errors.append(f"{tid}: season {season} is outside the training split")
        if meta.get("label_uses_realized_outcome") is not False:
            errors.append(f"{tid}: label provenance does not exclude realized outcomes")
        subtype = str(meta.get("t11_subtype"))
        subtype_counts[subtype] += 1
        season_counts[season] += 1
        family_counts[str(meta.get("source_family", meta.get("family")))] += 1
        category_counts.update(meta.get("categories") or [])
        user = messages[1]["content"]
        assistant = messages[2]["content"]
        prompt_hash = hashlib.sha256(
            "\n".join(m["content"] for m in messages[:2]).encode()
        ).hexdigest()
        if prompt_hash in prompt_hashes:
            errors.append(f"{tid}: exact duplicate prompt")
        prompt_hashes.add(prompt_hash)
        fingerprint = _message_fingerprint(row)
        fingerprints.append((tid, len(fingerprint), _simhash(fingerprint)))

        if subtype == "calibration_correction":
            p = parse_final_p(assistant)
            if p is None or p < 0.08 or p > 0.85:
                errors.append(f"{tid}: invalid calibrated probability")
            anchor = float(meta.get("anchor_rate", -1))
            if p is not None and not within_band(p, anchor):
                errors.append(f"{tid}: calibrated probability violates grounding band")
            bad = uncited_numbers({"assistant": assistant, "user": user})
            if bad:
                errors.append(f"{tid}: unsupported numeric claims {bad[:3]}")
            continue

        if any(field in user for field in FORBIDDEN_DRAFT_INPUT_FIELDS):
            errors.append(f"{tid}: realized or label field leaked into draft input")
        try:
            obs = _extract_observation(user)
            action = parse_action(assistant)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{tid}: invalid observation/action ({type(exc).__name__})")
            continue
        if int(obs.get("season", -1)) != season or not obs.get("anonymized"):
            errors.append(f"{tid}: draft observation season/anonymization mismatch")
        visible = {str(x["player_id"]) for x in obs.get("top_available", [])}
        if "pick" in action and str(action["pick"]) not in visible:
            errors.append(f"{tid}: pick is not model-visible")
        if "tool" in action:
            tool = action["tool"]
            target = str(tool.get("arguments", {}).get("team_or_player", ""))
            if tool.get("name") != "depth_chart" or target not in visible:
                errors.append(f"{tid}: invalid or unsupported tool label")
            if int(obs.get("lookups_remaining", 0)) <= 0:
                errors.append(f"{tid}: tool label exceeds lookup budget")
            tool_groups[str(meta.get("question_id"))] += 1
        if subtype == "post_tool_pick":
            if not obs.get("tool_results"):
                errors.append(f"{tid}: post-tool trace has no tool result")
            else:
                response = obs["tool_results"][-1].get("response", {})
                if _depth_rank(response) is None:
                    errors.append(f"{tid}: post-tool trace lacks useful structured evidence")
                future = _future_evidence_dates(
                    response, date.fromisoformat(str(obs["evidence_as_of"]))
                )
                if future:
                    errors.append(f"{tid}: evidence is not strictly point-in-time {future[:3]}")
            tool_groups[str(meta.get("question_id"))] += 1

    # Near-duplicate audit: exact prompts are already rejected; this catches
    # effectively identical serialized traces while allowing shared scaffolds.
    near_duplicates: list[tuple[str, str, float]] = []
    for i, (left_id, left_len, left) in enumerate(fingerprints):
        for right_id, right_len, right in fingerprints[i + 1:]:
            if abs(left_len - right_len) > max(left_len, right_len) * 0.01:
                continue
            distance = (left ^ right).bit_count()
            if distance <= 1:
                near_duplicates.append((left_id, right_id, distance))
    if near_duplicates:
        errors.append(f"near-duplicate traces detected: {near_duplicates[:3]}")

    expected_subtypes = {
        "calibration_correction": TARGET_FORECAST_ROWS,
        "direct_pick": TARGET_DIRECT_PICK_ROWS,
        "tool_call": TARGET_TOOL_GROUPS,
        "post_tool_pick": TARGET_TOOL_GROUPS,
    }
    if dict(subtype_counts) != expected_subtypes:
        errors.append(f"subtype coverage mismatch: {dict(subtype_counts)}")
    for group, count in tool_groups.items():
        if count != 2:
            errors.append(f"{group}: tool call/post-result pair is incomplete")
    minimums = {
        "calibrated_probability": 120,
        "confidence_correction": 120,
        "evidence_tool_use": 60,
        "tool_avoidance": 60,
        "draft_timing": 150,
        "mistake_prevention": 180,
        "late_round_strategy": 25,
        "roster_construction": 35,
        "recovery": 20,
        "league_arithmetic": 250,
    }
    for category, minimum in minimums.items():
        if category_counts[category] < minimum:
            errors.append(
                f"category {category} has {category_counts[category]}, needs {minimum}"
            )
    if errors:
        raise ValueError("T1.1 dataset validation failed:\n- " + "\n- ".join(errors))
    return {
        "valid": True,
        "rows": len(rows),
        "subtypes": dict(sorted(subtype_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "seasons": {str(k): v for k, v in sorted(season_counts.items())},
        "families": dict(sorted(family_counts.items())),
        "exact_duplicate_prompts": 0,
        "near_duplicate_traces": 0,
        "tool_groups_complete": len(tool_groups),
        "sealed_seasons_present": [],
        "realized_label_inputs": 0,
    }


def write_dataset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    CORPUS_PATH.write_text(text)
    validation = validate_rows(rows)
    manifest = {
        "protocol": "fantasy-alpha-t11-targeted-sft-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(CORPUS_PATH.relative_to(ROOT)),
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "dataset_seed": DATASET_SEED,
        "target_rows": TARGET_ROWS,
        "source_t1_corpus_sha256": sha256_file(ROOT / "data/processed/sft/t1_corpus_v1.jsonl"),
        "draft_label_policy": "SurvivalSequencer(scarcity_ratio=1.5)",
        "forecast_target_formula": "clip(anchor + clip(teacher-anchor,-.15,.15)/3,.08,.85)",
        "labels_use_realized_outcomes": False,
        "validation": validation,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_dataset_card(manifest)
    return manifest


def write_dataset_card(manifest: Mapping[str, Any]) -> None:
    v = manifest["validation"]
    cats = v["categories"]
    lines = [
        "# Fantasy Alpha T1.1 targeted dataset card",
        "",
        "This 300-trace development-only corpus is a controlled replacement for T1,",
        "not an append-only pile of generic examples. It directly addresses the three",
        "observed T1 gaps: unstable draft choices, overconfident probability tails, and",
        "zero evidence-tool use. All assistant tokens are trainable; system/user context",
        "remains loss-masked by the existing Tinker backend.",
        "",
        "## Composition",
        "",
        "| subtype | rows | observable behavior |",
        "|---|---:|---|",
        f"| calibration correction | {v['subtypes']['calibration_correction']} | conservative, grounded probability |",
        f"| direct draft pick | {v['subtypes']['direct_pick']} | timing, roster and late-round action |",
        f"| evidence-tool call | {v['subtypes']['tool_call']} | query a specific uncertain candidate |",
        f"| post-tool pick | {v['subtypes']['post_tool_pick']} | use structured depth evidence in the next action |",
        "",
        f"Corpus SHA-256: `{manifest['corpus_sha256']}`.",
        "",
        "The draft labels come from the previously benchmarked market-only",
        "`SurvivalSequencer(scarcity_ratio=1.5)` policy. It uses ADP survival to the",
        "next snake turn, positional scarcity, and QB/TE roster guardrails. It does not",
        "see realized outcomes. Forecast targets shrink the old teacher's directional",
        "update toward a strictly-prior historical anchor and cap unsupported tails.",
        "",
        "## Coverage",
        "",
        "| behavior | traces |",
        "|---|---:|",
    ]
    lines.extend(f"| {name.replace('_', ' ')} | {count} |" for name, count in sorted(cats.items()))
    lines += [
        "",
        "## Leakage and quality checks",
        "",
        "- Seasons are limited to the existing training split; 2013, 2018, 2023,",
        "  2024, and 2025 are absent.",
        "- Draft prompts expose only the environment's existing pre-draft observation",
        "  fields. Reward, realized points, outcomes, and verifier data are rejected.",
        "- Tool results are time-gated by `EvidenceStore`; each included call has a",
        "  successful, non-empty structured depth-chart response.",
        "- Picks must be legal, model-visible board IDs. Tool actions must use the",
        "  masked structured interface and have a paired post-result decision.",
        "- Exact and effectively identical prompt/target duplicates are rejected.",
        "- Forecast numeric claims pass the existing cite-or-don't-claim and grounding",
        "  checks. Outcomes are never used to select or construct T1.1 targets.",
        "",
        "## Intended use and limitations",
        "",
        "This corpus tests whether a smaller amount of behavior-specific supervision",
        "beats T1's larger probability-heavy corpus. The draft teacher is a disciplined",
        "rule policy, not an oracle; its historical edge over ADP is small and unresolved.",
        "Tool-use labels encode an explicit product policy (inspect uncertain candidates",
        "with structured depth evidence), not a claim that every successful lookup changes",
        "the final roster. Evaluation therefore reports both task reward and policy-alignment",
        "metrics. The 2025 season remains sealed for a later one-shot gate.",
    ]
    DATASET_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_CARD_PATH.write_text("\n".join(lines) + "\n")


def load_t11_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate_rows(rows)
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else None
    if path == CORPUS_PATH and manifest and sha256_file(path) != manifest["corpus_sha256"]:
        raise ValueError("T1.1 corpus SHA-256 does not match its manifest")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build or validate the frozen T1.1 corpus")
    ap.add_argument("command", choices=("build", "validate"))
    args = ap.parse_args(argv)
    if args.command == "build":
        result = write_dataset(build_rows())
    else:
        rows = load_t11_corpus()
        result = {"corpus_sha256": sha256_file(CORPUS_PATH), "validation": validate_rows(rows)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
