"""ADP -> pick-window mapping (lifted; thresholds are parameters, not magic).

Deliberately maps market price onto concrete picks without inventing survival
probabilities. DraftGym's fitted opponents are the probabilistic layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DraftWindow:
    market_adp: float
    last_pick_before_adp: int | None
    first_pick_after_adp: int | None


def draft_window(adp: float, picks: Sequence[int]) -> DraftWindow:
    before = [p for p in picks if p <= adp]
    after = [p for p in picks if p > adp]
    return DraftWindow(
        market_adp=adp,
        last_pick_before_adp=before[-1] if before else None,
        first_pick_after_adp=after[0] if after else None,
    )


def fall_severity(adp: float, pick: int, modest: float = 3.0, meaningful: float = 10.0) -> str:
    fall = pick - adp
    if fall <= 0:
        return "normal_cost"
    if fall <= modest:
        return "modest_fall"
    if fall <= meaningful:
        return "meaningful_fall"
    return "material_fall"


def recommendation_status(adp: float, current_pick: int, next_pick: int | None) -> str:
    if adp < current_pick:
        return "fall_only"
    if next_pick is None or adp < next_pick:
        return "take_in_this_window"
    return "normally_wait"
