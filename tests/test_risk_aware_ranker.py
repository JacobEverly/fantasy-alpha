import pytest

from training.risk_aware_ranker import _selection_key, replay


def candidate(*, gate, catastrophes, robust, trimmed, wins, losses, rate):
    return {
        "passes_headroom_gate": gate,
        "catastrophic_drafts_below_minus_100": catastrophes,
        "mean_without_largest_gain": robust,
        "trimmed_mean_10pct": trimmed,
        "wins": wins,
        "losses": losses,
        "deviation_rate": rate,
    }


def test_selection_prioritizes_gate_then_downside():
    safe = candidate(
        gate=True, catastrophes=1, robust=2, trimmed=3, wins=5, losses=3, rate=0.1
    )
    flashy = candidate(
        gate=True, catastrophes=4, robust=100, trimmed=100, wins=10, losses=1, rate=0.8
    )
    failed = candidate(
        gate=False, catastrophes=0, robust=100, trimmed=100, wins=10, losses=1, rate=0.0
    )
    assert _selection_key(safe) > _selection_key(flashy)
    assert _selection_key(safe) > _selection_key(failed)


def test_development_replay_refuses_sealed_seasons_before_loading_data():
    with pytest.raises(ValueError, match="sealed"):
        replay({}, [2013], mean_margin=0, downside_margin=0)
