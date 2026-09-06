"""Decision-time guardrails for DraftGym pick and evidence actions."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

DECISIONS = ("ACT_NOW", "USE_TOOL", "WAIT_OR_ABSTAIN")
STRUCTURED_TOOLS = ("depth_chart", "injury_status")


@dataclass(frozen=True)
class SupervisorDecision:
    decision: str
    confidence: float
    reason_code: str
    allowed_tools: tuple[str, ...] = ()
    required_tool: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"invalid supervisor decision: {self.decision}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.decision == "USE_TOOL" and not self.required_tool:
            raise ValueError("USE_TOOL requires a concrete tool call")
        if self.decision != "USE_TOOL" and self.required_tool is not None:
            raise ValueError("only USE_TOOL may include a required tool")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_tools"] = list(self.allowed_tools)
        return payload


class ToolSupervisor(Protocol):
    def decide(
        self, observation: Mapping[str, Any], proposed_action: Mapping[str, Any]
    ) -> SupervisorDecision: ...


def visible_candidate(
    observation: Mapping[str, Any], proposed_action: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    if set(proposed_action) != {"pick"}:
        return None
    player_id = str(proposed_action["pick"])
    return next(
        (
            row for row in observation.get("top_available", [])
            if str(row.get("player_id")) == player_id
        ),
        None,
    )


def expected_tool(candidate: Mapping[str, Any]) -> dict[str, Any]:
    player_id = str(candidate["player_id"])
    if float(candidate.get("prev_season_points") or 0.0) <= 0:
        return {
            "name": "depth_chart",
            "arguments": {"team_or_player": player_id},
        }
    return {
        "name": "injury_status",
        "arguments": {"player": player_id},
    }


def _tool_target(tool: Mapping[str, Any]) -> str:
    args = tool.get("arguments") or {}
    if not isinstance(args, Mapping):
        return ""
    return str(args.get("team_or_player", args.get("player", "")))


def relevant_tool_result(
    observation: Mapping[str, Any], candidate_id: str
) -> tuple[str, Mapping[str, Any] | None]:
    """Return ``success``, ``failure``, or ``missing`` for this candidate."""
    for event in reversed(list(observation.get("tool_results") or [])):
        call = event.get("call") or {}
        if not isinstance(call, Mapping) or _tool_target(call) != candidate_id:
            continue
        response = event.get("response") or {}
        if not isinstance(response, Mapping):
            return "failure", None
        return ("success", response) if response.get("ok") else ("failure", response)
    return "missing", None


def validate_proposed_tool(
    observation: Mapping[str, Any], proposed_action: Mapping[str, Any]
) -> SupervisorDecision:
    tool = proposed_action.get("tool")
    if not isinstance(tool, Mapping):
        return SupervisorDecision(
            "WAIT_OR_ABSTAIN", 1.0, "malformed_action",
        )
    name = str(tool.get("name", ""))
    target = _tool_target(tool)
    visible = {str(row.get("player_id")) for row in observation.get("top_available", [])}
    if name not in STRUCTURED_TOOLS or target not in visible:
        return SupervisorDecision(
            "WAIT_OR_ABSTAIN", 1.0, "invalid_or_irrelevant_tool",
        )
    if int(observation.get("lookups_remaining") or 0) <= 0:
        return SupervisorDecision(
            "WAIT_OR_ABSTAIN", 1.0, "lookup_budget_exhausted",
        )
    return SupervisorDecision(
        "USE_TOOL", 1.0, "valid_model_tool_call", (name,), dict(tool),
    )


class AlwaysActSupervisor:
    def decide(
        self, observation: Mapping[str, Any], proposed_action: Mapping[str, Any]
    ) -> SupervisorDecision:
        del observation, proposed_action
        return SupervisorDecision("ACT_NOW", 1.0, "always_act_baseline")


class AlwaysUseToolSupervisor:
    def decide(
        self, observation: Mapping[str, Any], proposed_action: Mapping[str, Any]
    ) -> SupervisorDecision:
        candidate = visible_candidate(observation, proposed_action)
        if candidate is None or int(observation.get("lookups_remaining") or 0) <= 0:
            return SupervisorDecision(
                "WAIT_OR_ABSTAIN", 1.0, "always_tool_unavailable",
            )
        tool = expected_tool(candidate)
        return SupervisorDecision(
            "USE_TOOL", 1.0, "always_use_tool_baseline",
            (str(tool["name"]),), tool,
        )


class HardCodedToolSupervisor:
    """Precommitted transparent rules from the frozen v1 protocol."""

    RELATIVE_ADP_STDEV_THRESHOLD = 0.15

    def decide(
        self, observation: Mapping[str, Any], proposed_action: Mapping[str, Any]
    ) -> SupervisorDecision:
        if "tool" in proposed_action:
            return validate_proposed_tool(observation, proposed_action)
        candidate = visible_candidate(observation, proposed_action)
        if candidate is None:
            return SupervisorDecision(
                "WAIT_OR_ABSTAIN", 1.0, "illegal_or_missing_candidate",
            )
        candidate_id = str(candidate["player_id"])
        result_status, _ = relevant_tool_result(observation, candidate_id)
        if result_status == "success":
            return SupervisorDecision(
                "ACT_NOW", 1.0, "relevant_evidence_available",
            )
        if result_status == "failure" and int(
            observation.get("lookups_remaining") or 0
        ) <= 0:
            return SupervisorDecision(
                "WAIT_OR_ABSTAIN", 1.0, "evidence_failed_no_budget",
            )

        prev_points = float(candidate.get("prev_season_points") or 0.0)
        adp = max(float(candidate.get("adp") or 0.0), 1.0)
        relative_stdev = float(candidate.get("adp_stdev") or 0.0) / adp
        tool: dict[str, Any] | None = None
        reason = "decision_context_sufficient"
        if prev_points <= 0:
            tool = {
                "name": "depth_chart",
                "arguments": {"team_or_player": candidate_id},
            }
            reason = "missing_depth_chart"
        elif relative_stdev >= self.RELATIVE_ADP_STDEV_THRESHOLD:
            tool = {
                "name": "injury_status",
                "arguments": {"player": candidate_id},
            }
            reason = "high_market_uncertainty"

        if tool is not None:
            if int(observation.get("lookups_remaining") or 0) <= 0:
                return SupervisorDecision(
                    "WAIT_OR_ABSTAIN", 1.0, "required_evidence_no_budget",
                )
            return SupervisorDecision(
                "USE_TOOL", 1.0, reason, (str(tool["name"]),), tool,
            )
        return SupervisorDecision("ACT_NOW", 1.0, reason)


class LearnedToolSupervisor:
    """Reload a fitted narrow classifier without importing training code."""

    def __init__(self, artifact_path: str | Path):
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install fantasy-alpha[baselines]") from exc
        self.artifact_path = Path(artifact_path)
        payload = joblib.load(self.artifact_path)
        if payload.get("protocol") != "fantasy-alpha-learned-tool-supervisor-v1":
            raise ValueError("unexpected learned supervisor artifact protocol")
        self.model = payload["model"]
        self.use_tool_threshold = float(payload["use_tool_threshold"])
        self.wait_threshold = float(payload["wait_threshold"])
        self.classes = tuple(str(value) for value in self.model.classes_)
        self.metadata = dict(payload["metadata"])

    def decide(
        self, observation: Mapping[str, Any], proposed_action: Mapping[str, Any]
    ) -> SupervisorDecision:
        if "tool" in proposed_action:
            return validate_proposed_tool(observation, proposed_action)
        candidate = visible_candidate(observation, proposed_action)
        if candidate is None:
            return SupervisorDecision(
                "WAIT_OR_ABSTAIN", 1.0, "illegal_or_missing_candidate",
            )
        features = decision_features(observation, proposed_action)
        probabilities = self.model.predict_proba([features])[0]
        by_class = dict(zip(self.classes, probabilities, strict=True))
        wait_p = float(by_class.get("WAIT_OR_ABSTAIN", 0.0))
        tool_p = float(by_class.get("USE_TOOL", 0.0))
        if wait_p >= self.wait_threshold:
            decision = "WAIT_OR_ABSTAIN"
            confidence = wait_p
            reason = "learned_unresolved_risk"
            tool = None
        elif tool_p >= self.use_tool_threshold:
            decision = "USE_TOOL"
            confidence = tool_p
            reason = "learned_evidence_required"
            tool = expected_tool(candidate)
        else:
            decision = "ACT_NOW"
            confidence = float(by_class.get("ACT_NOW", 0.0))
            reason = "learned_context_sufficient"
            tool = None
        return SupervisorDecision(
            decision,
            max(0.0, min(1.0, confidence)),
            reason,
            (str(tool["name"]),) if tool else (),
            tool,
        )


def decision_features(
    observation: Mapping[str, Any], proposed_action: Mapping[str, Any]
) -> dict[str, float | str]:
    """Strict decision-time feature allowlist shared by fit and inference."""
    candidate = visible_candidate(observation, proposed_action)
    if candidate is None:
        raise ValueError("learned supervisor requires one visible proposed pick")
    candidate_id = str(candidate["player_id"])
    status, response = relevant_tool_result(observation, candidate_id)
    adp = max(float(candidate.get("adp") or 0.0), 1.0)
    position = str(candidate.get("position") or "UNKNOWN")
    available = list(observation.get("top_available") or [])
    rank = next(
        index for index, row in enumerate(available)
        if str(row.get("player_id")) == candidate_id
    )
    result = (response or {}).get("result")
    return {
        "round": float(observation.get("round") or 0),
        "pick_number": float(observation.get("pick_number") or 0),
        "picks_until_next_turn": float(observation.get("picks_until_next_turn") or 0),
        "roster_size": float(len(observation.get("my_roster") or [])),
        "available_count": float(len(available)),
        "lookups_used": float(observation.get("lookups_used_this_pick") or 0),
        "lookups_remaining": float(observation.get("lookups_remaining") or 0),
        "candidate_adp": adp,
        "candidate_adp_stdev": float(candidate.get("adp_stdev") or 0.0),
        "candidate_relative_adp_stdev": float(candidate.get("adp_stdev") or 0.0) / adp,
        "candidate_prev_season_points": float(candidate.get("prev_season_points") or 0.0),
        "candidate_missing_prior_points": float(
            float(candidate.get("prev_season_points") or 0.0) <= 0
        ),
        "candidate_visible_rank": float(rank),
        "candidate_position": position,
        "result_status": status,
        "result_is_null": float(status == "success" and result is None),
        "result_row_count": float(len(result) if isinstance(result, list) else result is not None),
    }


def decision_from_json(text: str) -> SupervisorDecision:
    """Parse a model-authored supervisor decision for evaluation."""
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("supervisor response must be one JSON object")
    return SupervisorDecision(
        decision=str(payload["decision"]),
        confidence=float(payload.get("confidence", 1.0)),
        reason_code=str(payload.get("reason_code", "model_authored")),
        allowed_tools=tuple(str(value) for value in payload.get("allowed_tools", [])),
        required_tool=dict(payload["required_tool"])
        if payload.get("required_tool") else None,
    )
