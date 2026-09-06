"""Outcome-linked evidence supervisors, isolated from the frozen v1 harness."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from harness.tool_supervisor import (
    SupervisorDecision,
    decision_features,
    expected_tool,
    relevant_tool_result,
    validate_proposed_tool,
    visible_candidate,
)


class AlwaysEvidenceOnceSupervisor:
    """Buy one relevant evidence lookup, then allow the reconsidered pick."""

    def decide(
        self, observation: Mapping[str, Any], proposed_action: Mapping[str, Any]
    ) -> SupervisorDecision:
        candidate = visible_candidate(observation, proposed_action)
        if candidate is None:
            return SupervisorDecision(
                "WAIT_OR_ABSTAIN", 1.0, "always_evidence_invalid_candidate",
            )
        status, _ = relevant_tool_result(observation, str(candidate["player_id"]))
        if status == "success":
            return SupervisorDecision(
                "ACT_NOW", 1.0, "always_evidence_already_available",
            )
        if int(observation.get("lookups_remaining") or 0) <= 0:
            return SupervisorDecision(
                "WAIT_OR_ABSTAIN", 1.0, "always_evidence_unavailable",
            )
        tool = expected_tool(candidate)
        return SupervisorDecision(
            "USE_TOOL", 1.0, "always_evidence_missing",
            (str(tool["name"]),), tool,
        )


class OutcomeValueSupervisor:
    """Use a frozen outcome-linked model to decide whether evidence is valuable."""

    def __init__(self, artifact_path: str | Path):
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install fantasy-alpha[baselines]") from exc
        payload = joblib.load(Path(artifact_path))
        if payload.get("protocol") != "fantasy-alpha-outcome-value-supervisor-v1":
            raise ValueError("unexpected outcome-value supervisor protocol")
        self.model = payload.get("model")
        self.constant_probability = float(payload.get("constant_probability", 0.0))
        self.threshold = float(payload["threshold"])
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
        probability = self.constant_probability
        if self.model is not None:
            classes = [int(value) for value in self.model.classes_]
            probabilities = self.model.predict_proba([features])[0]
            probability = float(
                dict(zip(classes, probabilities, strict=True)).get(1, 0.0)
            )
        if probability >= self.threshold:
            tool = expected_tool(candidate)
            return SupervisorDecision(
                "USE_TOOL", probability, "learned_positive_value_of_information",
                (str(tool["name"]),), tool,
            )
        return SupervisorDecision(
            "ACT_NOW", 1.0 - probability, "learned_low_value_of_information",
        )


class OutcomeValueRuleSupervisor:
    """Transparent outcome-tuned rule frozen before held-out evaluation."""

    def __init__(
        self, *, use_missing_prior: bool, relative_adp_stdev_threshold: float,
    ):
        self.use_missing_prior = bool(use_missing_prior)
        self.relative_adp_stdev_threshold = float(relative_adp_stdev_threshold)

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
        status, _ = relevant_tool_result(observation, str(candidate["player_id"]))
        if status == "success":
            return SupervisorDecision(
                "ACT_NOW", 1.0, "outcome_rule_evidence_available",
            )
        if (
            status == "failure"
            and int(observation.get("lookups_remaining") or 0) <= 0
        ):
            return SupervisorDecision(
                "WAIT_OR_ABSTAIN", 1.0, "outcome_rule_evidence_failed",
            )
        adp = max(float(candidate.get("adp") or 0.0), 1.0)
        relative_stdev = float(candidate.get("adp_stdev") or 0.0) / adp
        missing = float(candidate.get("prev_season_points") or 0.0) <= 0
        intervene = (
            (self.use_missing_prior and missing)
            or relative_stdev >= self.relative_adp_stdev_threshold
        )
        if not intervene:
            return SupervisorDecision(
                "ACT_NOW", 1.0, "outcome_rule_low_expected_value",
            )
        if int(observation.get("lookups_remaining") or 0) <= 0:
            return SupervisorDecision(
                "WAIT_OR_ABSTAIN", 1.0, "outcome_rule_no_budget",
            )
        tool = expected_tool(candidate)
        return SupervisorDecision(
            "USE_TOOL", 1.0, "outcome_rule_positive_expected_value",
            (str(tool["name"]),), tool,
        )
