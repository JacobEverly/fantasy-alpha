"""League configuration and draft-order math, parameterized for any format."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from harness.scoring import PRESETS, ScoringConfig, from_research_scoring

NON_STARTING_SLOTS = {"BENCH", "IR", "TAXI"}
FLEX_SLOT = "FLEX"


@dataclass(frozen=True)
class DraftTurn:
    turn_number: int
    picks: tuple[int, ...]
    next_turn_pick: int | None

    @property
    def turn_id(self) -> str:
        picks = "-".join(f"p{p:03d}" for p in self.picks)
        return f"t{self.turn_number:02d}-{picks}"


@dataclass(frozen=True)
class LeagueConfig:
    teams: int
    roster: Mapping[str, int]  # e.g. {"QB":1,"RB":2,"WR":3,"TE":1,"FLEX":2,"BENCH":6}
    scoring: ScoringConfig
    flex_eligible: tuple[str, ...] = ("RB", "WR", "TE")
    rounds: int | None = None  # defaults to sum of roster slots
    draft_slot: int | None = None
    season_weeks: int = 17
    playoff_weeks: tuple[int, ...] = (15, 16, 17)

    def __post_init__(self) -> None:
        if self.teams < 2:
            raise ValueError("teams must be >= 2")
        if any(v < 0 for v in self.roster.values()):
            raise ValueError("roster counts must be >= 0")
        if self.rounds is None:
            object.__setattr__(self, "rounds", sum(self.roster.values()))
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.draft_slot is not None and not 1 <= self.draft_slot <= self.teams:
            raise ValueError("draft_slot must be within 1..teams")
        unknown_flex = [p for p in self.flex_eligible if p in NON_STARTING_SLOTS or p == FLEX_SLOT]
        if unknown_flex:
            raise ValueError(f"flex_eligible cannot contain {unknown_flex}")

    def required_positions(self) -> dict[str, int]:
        return {
            pos: count
            for pos, count in self.roster.items()
            if count > 0 and pos != FLEX_SLOT and pos not in NON_STARTING_SLOTS
        }

    def flex_per_team(self) -> int:
        return int(self.roster.get(FLEX_SLOT, 0))

    def snake_picks(self, slot: int | None = None) -> tuple[int, ...]:
        slot = self.draft_slot if slot is None else slot
        if slot is None:
            raise ValueError("no draft slot set")
        if not 1 <= slot <= self.teams:
            raise ValueError("slot must be within 1..teams")
        picks = []
        for rnd in range(1, self.rounds + 1):
            offset = slot if rnd % 2 == 1 else self.teams - slot + 1
            picks.append((rnd - 1) * self.teams + offset)
        return tuple(picks)

    def draft_turns(self, slot: int | None = None) -> tuple[DraftTurn, ...]:
        """Group your picks into planning turns.

        Two consecutive picks form one turn only when genuinely adjacent —
        fewer than teams/2 opponent picks between them (snake wrap pairs).
        Mid-order slots therefore get singleton turns instead of the fake
        pairs the research repo produced.
        """
        picks = self.snake_picks(slot)
        groups: list[tuple[int, ...]] = []
        i = 0
        while i < len(picks):
            if i + 1 < len(picks) and (picks[i + 1] - picks[i] - 1) < self.teams / 2:
                groups.append((picks[i], picks[i + 1]))
                i += 2
            else:
                groups.append((picks[i],))
                i += 1
        return tuple(
            DraftTurn(
                turn_number=n + 1,
                picks=group,
                next_turn_pick=groups[n + 1][0] if n + 1 < len(groups) else None,
            )
            for n, group in enumerate(groups)
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "LeagueConfig":
        """Product JSON schema: the ~7 manual-entry fields."""
        scoring = d.get("scoring", "half_ppr")
        if isinstance(scoring, str):
            scoring_cfg = PRESETS[scoring]
        elif isinstance(scoring, ScoringConfig):
            scoring_cfg = scoring
        else:
            scoring_cfg = ScoringConfig(
                base=dict(scoring.get("base", scoring)),  # type: ignore[union-attr]
                position_overrides=dict(scoring.get("position_overrides", {})),  # type: ignore[union-attr]
            )
        return cls(
            teams=int(d["teams"]),  # type: ignore[arg-type]
            roster=dict(d["roster"]),  # type: ignore[arg-type]
            scoring=scoring_cfg,
            flex_eligible=tuple(d.get("flex_eligible", ("RB", "WR", "TE"))),  # type: ignore[arg-type]
            rounds=int(d["rounds"]) if d.get("rounds") is not None else None,  # type: ignore[arg-type]
            draft_slot=int(d["draft_slot"]) if d.get("draft_slot") is not None else None,  # type: ignore[arg-type]
        )

    @classmethod
    def from_research_yaml(cls, d: Mapping[str, object]) -> "LeagueConfig":
        """Translate the research repo's config/league.yaml structure."""
        draft = d.get("draft", {})
        return cls(
            teams=int(d["teams"]),  # type: ignore[arg-type]
            roster=dict(d["roster"]),  # type: ignore[arg-type]
            scoring=from_research_scoring(d["scoring"]),  # type: ignore[arg-type]
            flex_eligible=tuple(d.get("flex_eligible", ("RB", "WR", "TE"))),  # type: ignore[arg-type]
            rounds=int(draft.get("rounds")) if isinstance(draft, Mapping) and draft.get("rounds") else None,
            draft_slot=int(draft.get("slot")) if isinstance(draft, Mapping) and draft.get("slot") else None,
        )
