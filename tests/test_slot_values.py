"""Slot-value curve: monotonicity, timing-score economics, survival edge cases.

Design-doc reference: docs/breakoutbench-design.md section 3, item 5.
"""
import math

import pytest

from evals.slot_values import (
    ADP_DIR,
    CSV_PATH,
    FORMATS,
    MAX_PICK,
    _pava_nonincreasing,
    slot_expected_vorp,
    smoothed_curve,
    survival_probability,
    timing_score,
)

HAVE_DATA = ADP_DIR.exists() and CSV_PATH.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ snapshots + built CSV")


# ---------------------------------------------------------------------------
# survival_probability (pure function)


def test_survival_at_adp_is_half():
    assert survival_probability(90, 90, 8) == pytest.approx(0.5)


def test_survival_monotone_decreasing_in_act_pick():
    vals = [survival_probability(p, 90, 8) for p in range(1, 181, 5)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_survival_extremes():
    assert survival_probability(1, 200, 15) > 0.999
    assert survival_probability(150, 20, 5) < 1e-6


def test_survival_stdev_floor():
    # stdev 0 (or negative/garbage-small) must not divide by zero: floored to 1.0
    assert survival_probability(89, 90, 0.0) == pytest.approx(
        survival_probability(89, 90, 1.0)
    )
    assert 0.0 < survival_probability(91, 90, 0.2) < 0.5
    assert math.isfinite(survival_probability(90, 90, -3))


# ---------------------------------------------------------------------------
# PAVA / smoothed curve (pure functions)


def test_pava_nonincreasing_output():
    fitted = _pava_nonincreasing([5.0, 7.0, 3.0, 4.0, 1.0], [1, 1, 1, 1, 1])
    assert all(a >= b for a, b in zip(fitted, fitted[1:]))
    # weighted mean preserved
    assert sum(fitted) == pytest.approx(20.0)


def test_pava_already_monotone_is_identity():
    y = [10.0, 8.0, 8.0, 2.0]
    assert _pava_nonincreasing(y, [1, 2, 1, 1]) == pytest.approx(y)


def test_smoothed_curve_fills_gaps_and_floors_at_zero():
    obs = [(1, 100.0), (2, 90.0), (50, 5.0), (120, -30.0)]
    curve = smoothed_curve(obs)
    assert set(curve) == set(range(1, MAX_PICK + 1))
    vals = [curve[p] for p in range(1, MAX_PICK + 1)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert curve[25] == curve[2]  # gap carries previous fitted value
    assert curve[MAX_PICK] == 0.0  # price floor: opportunity cost never negative


# ---------------------------------------------------------------------------
# Built curves (require data)


@needs_data
def test_curve_monotone_nonincreasing_every_format():
    for fmt in FORMATS:
        vals = [slot_expected_vorp(p, fmt) for p in range(1, MAX_PICK + 1)]
        assert all(a >= b for a, b in zip(vals, vals[1:])), fmt


@needs_data
def test_early_picks_dominate_round_eight():
    for fmt in FORMATS:
        assert slot_expected_vorp(2, fmt) > slot_expected_vorp(90, fmt) + 50, fmt


@needs_data
def test_slot_expected_vorp_clamps_out_of_range():
    for fmt in FORMATS:
        assert slot_expected_vorp(0, fmt) == slot_expected_vorp(1, fmt)
        assert slot_expected_vorp(500, fmt) == slot_expected_vorp(MAX_PICK, fmt)


@needs_data
def test_unknown_format_raises():
    with pytest.raises(KeyError):
        slot_expected_vorp(1, "superflex-dynasty")


# ---------------------------------------------------------------------------
# timing_score economics (Jacob's worked example)


@needs_data
def test_timing_score_round7_action_wins():
    # 8th-round-ADP player (adp 90, stdev 8) who realized VORP 120
    at_1 = timing_score(1, 90, 8, 120, "ppr")
    at_78 = timing_score(78, 90, 8, 120, "ppr")
    at_100 = timing_score(100, 90, 8, 120, "ppr")
    assert at_78 > at_1  # reaching to pick 1 pays a first-round opportunity cost
    assert at_78 > at_100  # after ADP, survival collapses


@needs_data
def test_timing_score_undrafted_flier_is_pure_capture():
    # last-round slot prices ~zero -> score ~= survival * realized_vorp
    score = timing_score(175, 200, 15, 80, "ppr")
    assert score == pytest.approx(survival_probability(175, 200, 15) * 80, abs=1.0)


# ---------------------------------------------------------------------------
# Leakage-by-design documentation guard


def test_same_season_join_is_documented_as_scoring_not_feature():
    """The module deliberately joins season-N ADP to season-N outcomes.

    That is correct for a scoring curve (price axis used after outcomes are
    known) and would be leakage anywhere on the packet/feature side. The
    docstring must keep that distinction explicit so nobody lifts this into
    packet data.
    """
    import evals.slot_values as sv

    doc = sv.__doc__.lower()
    assert "same-season" in doc
    assert "scoring curve" in doc
    assert "never" in doc and "feature" in doc
