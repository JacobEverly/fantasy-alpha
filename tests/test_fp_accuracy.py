import pytest

from evals.fp_accuracy import (
    ADP_DIR,
    LABELS,
    POOL_N,
    POSITIONS,
    actual_position_ranks,
    adp_rankings,
    avg3_rankings,
    build_slot_curve,
    consensus_order,
    momentum_rankings,
    multiplier,
    score_overall,
    score_position_core,
    score_rankings,
    slot_points,
)

HAVE_DATA = ADP_DIR.exists() and (LABELS / "season_points.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ snapshots")


# ---------------------------------------------------------------------------
# Multiplier (their step-4 weighting)


def test_multiplier_boundaries_and_interpolation():
    # QB: maxrank 18, minrank 30
    assert multiplier(1, "QB") == 1.0
    assert multiplier(18, "QB") == 1.0
    assert multiplier(30, "QB") == 0.5
    assert multiplier(999, "QB") == 0.5
    assert multiplier(24, "QB") == pytest.approx(0.75)  # exact midpoint
    # TE: maxrank 18, minrank 24
    assert multiplier(21, "TE") == pytest.approx(0.75)
    assert multiplier(19, "TE") == pytest.approx(1.0 - 0.5 / 6)


# ---------------------------------------------------------------------------
# Slot curve


def _row(season, position, rank, pts, player="X"):
    return {
        "season": str(season),
        "position": position,
        "player": player,
        "pos_season_rank_ppr": str(rank),
        "total_ppr": str(pts),
    }


def test_slot_curve_averages_trailing_three_seasons():
    rows = [
        _row(2017, "RB", 1, 300.0),
        _row(2018, "RB", 1, 320.0),
        _row(2019, "RB", 1, 340.0),
        _row(2019, "RB", 2, 250.0),  # rank 2 only reached in one season
    ]
    curve = build_slot_curve(rows, 2020, "RB")
    assert curve[1] == pytest.approx(320.0)
    assert curve[2] == pytest.approx(250.0)


def test_slot_curve_never_uses_evaluated_or_future_seasons():
    base = [_row(s, "RB", 1, 300.0) for s in (2017, 2018, 2019)]
    leaky = base + [
        _row(2020, "RB", 1, 999.0),  # the evaluated season itself
        _row(2021, "RB", 1, 999.0),  # the future
        _row(2016, "RB", 1, 999.0),  # older than the 3-yr window
    ]
    assert build_slot_curve(leaky, 2020, "RB") == build_slot_curve(base, 2020, "RB")


def test_slot_curve_empty_window_raises():
    with pytest.raises(ValueError):
        build_slot_curve([_row(2010, "RB", 1, 300.0)], 2020, "RB")


def test_slot_points_clamps_deep_ranks():
    curve = {1: 300.0, 2: 200.0, 3: 100.0}
    assert slot_points(curve, 2) == 200.0
    assert slot_points(curve, 50) == 100.0  # beyond deepest slot -> clamp
    with pytest.raises(ValueError):
        slot_points(curve, 0)
    with pytest.raises(ValueError):
        slot_points({}, 1)


# ---------------------------------------------------------------------------
# Scoring core on hand-built cases.
# Curve: rank r -> 100 - 10*(r-1) for ranks 1..9 (so 1 rank slot = 10 pts).


CURVE = {r: 100.0 - 10.0 * (r - 1) for r in range(1, 10)}


def test_perfect_expert_scores_zero():
    consensus = ["A", "B", "C", "D"]
    actual = {"a": 1, "b": 2, "c": 3, "d": 4}
    score = score_position_core(["A", "B", "C", "D"], consensus, actual, CURVE, "QB", pool_n=3)
    assert score.accuracy_gap == 0.0
    assert score.penalty_total == 0.0
    # pool = consensus top 3 UNION actual top 3 = {a, b, c}
    assert score.pool_size == 3


def test_gap_formula_by_hand():
    # Expert flips A and B; consensus agrees with actual order.
    consensus = ["A", "B", "C"]
    actual = {"a": 1, "b": 2, "c": 3}
    score = score_position_core(["B", "A", "C"], consensus, actual, CURVE, "QB", pool_n=3)
    # A: expert 2 -> 90 vs actual 1 -> 100: gap 10. B symmetric. C exact.
    # Consensus ranks 1..3 are all <= QB maxrank 18 -> weight 1.0.
    assert score.accuracy_gap == pytest.approx(20.0)
    by_key = {p.key: p for p in score.players}
    assert by_key["a"].gap == pytest.approx(10.0)
    assert by_key["a"].weight == 1.0
    assert by_key["c"].gap == 0.0


def test_pool_is_union_of_consensus_and_actual_top_n():
    # E busted out of the top 3 actually; F broke in. Both must be evaluated.
    consensus = ["A", "B", "E", "F"]  # E is consensus #3, F is consensus #4
    actual = {"a": 1, "b": 2, "f": 3, "e": 4}
    score = score_position_core(["A", "B", "E", "F"], consensus, actual, CURVE, "QB", pool_n=3)
    keys = {p.key for p in score.players}
    assert keys == {"a", "b", "e", "f"}  # union: 4 players from an N of 3


def test_unranked_consensus_pool_player_gets_last_plus_one():
    consensus = ["A", "B", "C"]
    actual = {"a": 1, "b": 2, "c": 3}
    score = score_position_core(["A", "B"], consensus, actual, CURVE, "QB", pool_n=3)
    c = next(p for p in score.players if p.key == "c")
    assert not c.ranked_by_expert
    assert c.expert_rank == 3  # expert ranked 2 players -> last + 1
    assert c.gap == pytest.approx(0.0)  # 3-slot projection vs actual rank 3


def test_unranked_actual_only_player_gets_worse_of_the_two():
    # F qualifies via actual finish only (consensus rank 6 > pool_n 3).
    consensus = ["A", "B", "C", "D", "E", "F"]
    actual = {"a": 1, "b": 2, "f": 3, "c": 5, "d": 6, "e": 7}
    score = score_position_core(["A", "B", "C"], consensus, actual, CURVE, "QB", pool_n=3)
    f = next(p for p in score.players if p.key == "f")
    # worse of consensus+1 (7) and expert-last+1 (4) -> 7 (the Mostert rule)
    assert f.expert_rank == 7
    # and a deep-ranking expert is not punished below consensus+1:
    deep = score_position_core(
        ["A", "B", "C", "D", "E"], consensus, actual, CURVE, "QB", pool_n=3
    )
    f2 = next(p for p in deep.players if p.key == "f")
    assert f2.expert_rank == 7  # max(consensus 6 + 1, expert-last 5 + 1)


def test_out_of_pool_penalty_only_when_worse_than_consensus():
    # Z is in neither pool half (consensus 5, actual rank 9). Expert ranks
    # Z #1 — far worse than consensus -> penalized by the excess gap.
    consensus = ["A", "B", "C", "D", "Z"]
    actual = {"a": 1, "b": 2, "c": 3, "d": 4, "z": 9}
    score = score_position_core(["Z", "A", "B"], consensus, actual, CURVE, "QB", pool_n=3)
    assert len(score.penalties) == 1
    pen = score.penalties[0]
    assert pen.key == "z"
    # expert gap |100-20|=80, consensus gap |60-20|=40 -> penalty 40 * weight 1.0
    assert pen.penalty == pytest.approx(40.0)
    assert score.penalty_total == pytest.approx(40.0)
    # An expert who agrees with consensus on Z draws no penalty.
    agree = score_position_core(["A", "B", "C", "D", "Z"], consensus, actual, CURVE, "QB", pool_n=3)
    assert agree.penalty_total == 0.0


def test_name_normalization_in_rankings():
    consensus = ["Mike Vick", "B", "C"]
    actual = {"mikevick": 1, "b": 2, "c": 3}  # norm_name canonical form
    score = score_position_core(
        ["Michael Vick Jr.", "B", "C"], consensus, actual, CURVE, "QB", pool_n=3
    )
    assert score.accuracy_gap == 0.0


def test_invalid_position_rejected():
    with pytest.raises(ValueError):
        score_position_core([], [], {}, CURVE, "K")


# ---------------------------------------------------------------------------
# Data-backed behavior


@needs_data
def test_holdout_season_guarded():
    with pytest.raises(ValueError, match="holdout"):
        score_rankings(["Someone"], 2025, "RB")


@needs_data
def test_adp_contender_scores_all_positions_2020():
    by_pos = {pos: adp_rankings(2020, pos) for pos in POSITIONS}
    overall = score_overall(by_pos, 2020)
    assert overall.accuracy_gap > 0
    for pos in POSITIONS:
        s = overall.by_position[pos]
        assert s.pool_size >= POOL_N[pos]  # union can only add players
        assert s.penalty_total == 0.0  # ADP == the consensus proxy: no penalties
        assert s.accuracy_gap == pytest.approx(
            sum(p.weighted_gap for p in s.players), abs=0.01
        )


@needs_data
def test_hindsight_rankings_beat_every_reference_contender():
    # Ranking players by their realized finish is the best possible list.
    for pos in ("RB", "WR"):
        actual = actual_position_ranks(2019, pos)
        hindsight = [k for k, _ in sorted(actual.items(), key=lambda kv: kv[1])]
        best = score_rankings(hindsight, 2019, pos).accuracy_gap
        for fn in (adp_rankings, momentum_rankings, avg3_rankings):
            assert best < score_rankings(fn(2019, pos), 2019, pos).accuracy_gap


@needs_data
def test_reference_contenders_leak_free_inputs():
    # Momentum and avg3 lists for season S must not require season-S data:
    # every name they rank must come from seasons < S.
    prior = {k for s in (2017, 2018, 2019) for k in actual_position_ranks(s, "RB")}
    from evals.names import norm_name

    for fn in (momentum_rankings, avg3_rankings):
        assert {norm_name(n) for n in fn(2020, "RB")} <= prior


@needs_data
def test_consensus_order_is_position_filtered_and_sorted():
    order = consensus_order(2020, "RB")
    assert len(order) >= POOL_N["RB"]
    assert "Christian McCaffrey" == order[0]
