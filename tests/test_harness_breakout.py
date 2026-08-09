"""Leakage guard + cohort math for the BreakoutBench grounding stage."""
import pytest

from evals.harness_breakout import grounding_for, past_rows


def row(season, pos="WR", rank=45, breakout=False, pid="x"):
    return {"season": season, "player_id": pid, "position": pos,
            "adp_pos_rank": rank, "breakout": breakout}


CAND = {"position": "WR", "adp_pos_rank": 45,
        "prior_season": {"games": 14, "total": 120.0, "ppg": 8.6, "pos_rank": 50}}


def test_past_rows_excludes_eval_and_future_seasons():
    rows = [row(2016), row(2018), row(2019), row(2020), row(2024)]
    past = past_rows(rows, 2019)
    assert {r["season"] for r in past} == {2016, 2018}


def test_grounding_rejects_future_rows():
    with pytest.raises(AssertionError, match="leakage"):
        grounding_for(CAND, [row(2018), row(2019)], {}, 2019)
    with pytest.raises(AssertionError, match="leakage"):
        grounding_for(CAND, [row(2025, breakout=True)], {}, 2019)


def test_future_data_cannot_move_rates():
    """A future breakout row must be filtered out before grounding, so rates
    computed via past_rows are identical with or without it."""
    base = [row(2016, rank=45, breakout=True), row(2017, rank=45, breakout=False)]
    poisoned = base + [row(2019, rank=45, breakout=True), row(2022, rank=45, breakout=True)]
    g_base = grounding_for(CAND, past_rows(base, 2019), {}, 2019)
    g_pois = grounding_for(CAND, past_rows(poisoned, 2019), {}, 2019)
    assert g_base == g_pois
    assert g_base["cohort_rate"] == 0.5 and g_base["cohort_n"] == 2


def test_cohort_math_hand_built():
    rows = [
        row(2016, rank=37, breakout=True,  pid="a"),   # |37-45|=8 -> in cohort
        row(2016, rank=53, breakout=False, pid="b"),   # 8 -> in
        row(2017, rank=54, breakout=False, pid="c"),   # 9 -> out
        row(2017, rank=45, breakout=False, pid="d"),   # 0 -> in
        row(2016, rank=45, breakout=True,  pid="e", pos="RB"),  # wrong position -> out
    ]
    g = grounding_for(CAND, rows, {}, 2019)
    assert g["cohort_n"] == 3
    assert g["cohort_rate"] == pytest.approx(1 / 3, abs=1e-3)
    # position base rate: 4 WR rows, 1 breakout
    assert g["position_base_n"] == 4
    assert g["position_base_rate"] == 0.25


def test_prior_rank_cohort():
    rows = [
        row(2016, rank=41, breakout=True,  pid="a"),
        row(2016, rank=60, breakout=False, pid="b"),
        row(2017, rank=42, breakout=False, pid="c"),
    ]
    # prior-season (season-1) finish ranks for the past candidates
    prior_ranks = {("a", 2015): 40,   # |40-50|=10 -> in
                   ("b", 2015): 61,   # 11 -> out
                   ("c", 2016): 55}   # 5 -> in
    g = grounding_for(CAND, rows, prior_ranks, 2019)
    assert g["prior_rank_cohort_n"] == 2
    assert g["prior_rank_cohort_rate"] == 0.5


def test_prior_rank_cohort_skipped_without_prior_season():
    cand = {"position": "WR", "adp_pos_rank": 45, "prior_season": None}
    g = grounding_for(cand, [row(2016, breakout=True)], {("x", 2015): 50}, 2019)
    assert g["prior_rank_cohort_rate"] is None and g["prior_rank_cohort_n"] == 0
