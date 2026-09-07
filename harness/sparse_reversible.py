"""Sparse, reversible evidence intervention for complete DraftGym trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from harness.tool_supervisor import (
    expected_tool,
    relevant_tool_result,
    visible_candidate,
)


@dataclass(frozen=True)
class InterventionPlan:
    original_action: dict[str, Any]
    original_candidate: dict[str, Any]
    tool: dict[str, Any]
    trigger: dict[str, Any]


@dataclass(frozen=True)
class InterventionResolution:
    final_action: dict[str, Any]
    accepted_revision: bool
    audit: dict[str, Any]


class SparseReversibleController:
    """Permit one early high-uncertainty lookup and revert weak revisions.

    The trigger was selected from development-only matched counterfactuals.  The
    post-evidence gate is deliberately deterministic: a different model action
    is accepted only when the evidence is materially adverse to the original
    candidate and the replacement passes conservative roster/market checks.
    """

    def __init__(
        self,
        *,
        relative_adp_stdev_threshold: float = 0.10,
        first_round: int = 1,
        last_round: int = 4,
        max_adp_reach: float = 18.0,
    ) -> None:
        self.relative_adp_stdev_threshold = float(relative_adp_stdev_threshold)
        self.first_round = int(first_round)
        self.last_round = int(last_round)
        self.max_adp_reach = float(max_adp_reach)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.interventions_used = 0
        self.pending: InterventionPlan | None = None

    def plan(
        self,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
    ) -> InterventionPlan | None:
        if self.interventions_used >= 1 or self.pending is not None:
            return None
        round_number = int(observation.get("round") or 0)
        if not self.first_round <= round_number <= self.last_round:
            return None
        if int(observation.get("lookups_remaining") or 0) <= 0:
            return None
        candidate = visible_candidate(observation, proposed_action)
        if candidate is None:
            return None
        adp = max(float(candidate.get("adp") or 0.0), 1.0)
        relative_stdev = float(candidate.get("adp_stdev") or 0.0) / adp
        if relative_stdev < self.relative_adp_stdev_threshold:
            return None
        tool = expected_tool(candidate)
        plan = InterventionPlan(
            original_action=dict(proposed_action),
            original_candidate=dict(candidate),
            tool=dict(tool),
            trigger={
                "round": round_number,
                "candidate_relative_adp_stdev": relative_stdev,
                "threshold": self.relative_adp_stdev_threshold,
                "reason": "early_high_market_uncertainty",
            },
        )
        self.pending = plan
        self.interventions_used += 1
        return plan

    @staticmethod
    def _adverse_evidence(response: Mapping[str, Any] | None) -> tuple[bool, str]:
        if not response or not response.get("ok"):
            return False, "lookup_failed"
        result = response.get("result")
        if isinstance(result, Mapping):
            report = str(result.get("report_status") or "").strip().lower()
            practice = str(result.get("practice_status") or "").strip().lower()
            injury = str(result.get("primary_injury") or "").strip().lower()
            adverse_reports = {
                "out",
                "doubtful",
                "injured reserve",
                "ir",
                "pup",
                "suspended",
            }
            adverse = report in adverse_reports or (
                "did not participate" in practice
                and injury not in {"", "not injury related"}
            )
            return (
                adverse,
                "adverse_injury_status" if adverse else "non_adverse_injury_status",
            )
        if isinstance(result, list):
            offense_ranks = []
            for row in result:
                if (
                    not isinstance(row, Mapping)
                    or str(row.get("unit", "")).lower() != "offense"
                ):
                    continue
                try:
                    offense_ranks.append(int(str(row.get("depth_rank", "")).strip()))
                except ValueError:
                    continue
            adverse = not offense_ranks or min(offense_ranks) > 1
            return (
                adverse,
                "non_starting_depth_role" if adverse else "starting_depth_role",
            )
        return False, "empty_or_unknown_evidence"

    @staticmethod
    def _position_count(observation: Mapping[str, Any], position: str) -> int:
        return sum(
            str(player.get("position")) == position
            for player in observation.get("my_roster", [])
        )

    def resolve(
        self,
        observation: Mapping[str, Any],
        reconsidered_action: Mapping[str, Any],
    ) -> InterventionResolution:
        if self.pending is None:
            raise RuntimeError("no pending intervention to resolve")
        plan = self.pending
        self.pending = None
        original_id = str(plan.original_action["pick"])
        status, response = relevant_tool_result(observation, original_id)
        adverse, evidence_reason = self._adverse_evidence(response)
        replacement = visible_candidate(observation, reconsidered_action)
        changed = (
            replacement is not None and str(replacement["player_id"]) != original_id
        )
        market_safe = False
        roster_safe = False
        if replacement is not None:
            original_adp = float(plan.original_candidate.get("adp") or 0.0)
            replacement_adp = float(replacement.get("adp") or 0.0)
            market_safe = replacement_adp <= original_adp + self.max_adp_reach
            position = str(replacement.get("position") or "")
            roster_safe = not (
                position in {"QB", "TE"}
                and self._position_count(observation, position) >= 1
            )
        accepted = bool(
            status == "success" and adverse and changed and market_safe and roster_safe
        )
        final = dict(reconsidered_action) if accepted else dict(plan.original_action)
        return InterventionResolution(
            final_action=final,
            accepted_revision=accepted,
            audit={
                "trigger": dict(plan.trigger),
                "tool": dict(plan.tool),
                "tool_status": status,
                "evidence_reason": evidence_reason,
                "evidence_adverse": adverse,
                "original_action": dict(plan.original_action),
                "reconsidered_action": dict(reconsidered_action),
                "final_action": final,
                "action_changed": changed,
                "market_safe": market_safe,
                "roster_safe": roster_safe,
                "accepted_revision": accepted,
            },
        )

    def config(self) -> dict[str, Any]:
        return {
            "relative_adp_stdev_threshold": self.relative_adp_stdev_threshold,
            "first_round": self.first_round,
            "last_round": self.last_round,
            "max_adp_reach": self.max_adp_reach,
        }
