import json
import math

import pytest

from evals.calibbench import (
    ADP_DIR,
    BASE_RATE_SEASONS,
    H2H_MIN_GAMES,
    H2H_MIN_PPG,
    H2H_PPG_TOLERANCE,
    H2H_WEEKS,
    HOLDOUT_SEASON,
    LABELS,
    _weekly_by_season,
    brier,
    build,
    build_h2h_questions,
    build_question_set,
    ece,
    family_base_rates,
    log_loss,
    main,
    reliability_table,
    score_answers,
    season_to_date,
)

HAVE_DATA = ADP_DIR.exists() and (LABELS / "weekly_points.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ snapshots")


# ---------------------------------------------------------------------------
# Scoring rules (hand-built cases)


def test_brier_hand_example():
    assert brier([(0.8, 1), (0.3, 0)]) == pytest.approx((0.04 + 0.09) / 2)
    assert brier([(0.5, 1), (0.5, 0)]) == 0.25  # coin on any outcome
    assert brier([(1.0, 1), (0.0, 0)]) == 0.0  # oracle


def test_log_loss_hand_example():
    assert log_loss([(0.8, 1), (0.7, 0)]) == pytest.approx(
        (-math.log(0.8) - math.log(0.3)) / 2
    )
    assert log_loss([(0.5, 1), (0.5, 0)]) == pytest.approx(math.log(2))
    # confident wrong answers are clamped, not infinite
    assert log_loss([(1.0, 0)]) < 40


def test_reliability_table_and_ece_hand_example():
    # p=0.8 (bin 8) hits, p=0.3 (bin 3) misses, p=1.0 lands in the TOP bin.
    pairs = [(0.8, 1), (0.3, 0), (1.0, 1)]
    table = reliability_table(pairs)
    assert len(table) == 10
    top = table[9]
    assert (top["n"], top["mean_p"], top["freq"]) == (1, 1.0, 1.0)
    assert table[8] == {"bin": "[0.8,0.9)", "n": 1, "mean_p": 0.8, "freq": 1.0}
    assert table[0]["n"] == 0 and table[0]["mean_p"] is None
    # ECE = 1/3*|0.3-0| + 1/3*|0.8-1| + 1/3*|1.0-1.0|
    assert ece(pairs) == pytest.approx((0.3 + 0.2 + 0.0) / 3)
    # a perfectly calibrated coin has ECE 0 even though Brier is 0.25
    assert ece([(0.5, 1), (0.5, 0)]) == pytest.approx(0.0)


def test_score_answers_validation():
    key = {"questions": {
        "q1": {"anon_id": "Q0000", "family": "weekly_h2h", "outcome": 1},
        "q2": {"anon_id": "Q0001", "family": "weekly_h2h", "outcome": 0},
    }}
    # anon and named ids are interchangeable and score identically
    results = score_answers(
        [{"question_id": "Q0000", "p": 0.9}, {"question_id": "q2", "p": 0.1}], key
    )
    assert results["overall"]["n"] == 2
    assert results["overall"]["brier"] == pytest.approx(0.01)
    with pytest.raises(KeyError, match="unknown question_id"):
        score_answers([{"question_id": "nope", "p": 0.5}], key)
    with pytest.raises(ValueError, match="duplicate"):
        score_answers([{"question_id": "q1", "p": 0.5}, {"question_id": "Q0000", "p": 0.5}], key)
    with pytest.raises(ValueError, match="out of"):
        score_answers([{"question_id": "q1", "p": 1.5}], key)


# ---------------------------------------------------------------------------
# Season-to-date: only weeks strictly before W


def test_season_to_date_uses_only_prior_weeks():
    weeks = {1: 10.0, 2: 20.0, 4: 30.0, 5: 99.0, 6: 99.0}
    games, total, ppg = season_to_date(weeks, upto_week=5)  # week 5+ excluded
    assert (games, total, ppg) == (3, 60.0, 20.0)
    assert season_to_date(weeks, upto_week=1) == (0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Holdout guard


def test_holdout_season_refused_without_flag():
    for command in ("build", "score", "baseline"):
        argv = [command, "--season", str(HOLDOUT_SEASON), "--format", "ppr"]
        if command == "score":
            argv += ["--answers", "whatever.json"]
        with pytest.raises(SystemExit):
            main(argv)


def test_base_rate_seasons_never_include_holdout_or_2024():
    # base rates stop at 2023 so they cannot peek at 2024-2025 outcomes either
    assert max(BASE_RATE_SEASONS) == 2023
    assert HOLDOUT_SEASON not in BASE_RATE_SEASONS


# ---------------------------------------------------------------------------
# Real-data question generation


@needs_data
def test_build_is_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    paths_a = build(2019, "ppr", out_dir=a)
    paths_b = build(2019, "ppr", out_dir=b)
    for pa, pb in zip(paths_a, paths_b):
        assert pa.read_bytes() == pb.read_bytes()
    named_a, anon_a, key_a = build_question_set(2019, "ppr")
    named_b, anon_b, key_b = build_question_set(2019, "ppr")
    assert (named_a, anon_a, key_a) == (named_b, anon_b, key_b)


@needs_data
def test_question_files_leak_no_outcomes(tmp_path):
    named_path, anon_path, key_path = build(2019, "ppr", out_dir=tmp_path)
    named_text = named_path.read_text()
    anon_text = anon_path.read_text()
    assert '"outcome"' not in named_text
    assert '"outcome"' not in anon_text
    key = json.loads(key_path.read_text())
    assert all("outcome" in q for q in key["questions"].values())
    # the anonymized file carries no names and no season
    anon = json.loads(anon_text)
    assert anon.get("season_masked") is True and "season" not in anon
    names = {q.get("player") or q.get("player_a") for q in key["questions"].values()}
    for name in names:
        assert name not in anon_text
    # named and anon variants ask the same questions under different ids
    named = json.loads(named_path.read_text())
    anon_ids = {key["questions"][q["question_id"]]["anon_id"] for q in named["questions"]}
    assert anon_ids == {q["question_id"] for q in anon["questions"]}


@needs_data
def test_h2h_constraints_and_no_future_leakage():
    questions = build_h2h_questions(2019, "ppr")
    assert len(questions) == 300  # H2H_DEFAULT_PAIRS raised 50 -> 300 for power
    players = _weekly_by_season()[2019]
    seen_pairs = set()
    for q in questions:
        packet = q["packet"]
        week = packet["week"]
        assert week in H2H_WEEKS
        _, _, _, pid_a, pid_b = q["question_id"].split(":")
        assert (week, pid_a) not in seen_pairs and (week, pid_b) not in seen_pairs
        seen_pairs.update(((week, pid_a), (week, pid_b)))
        for pid, side in ((pid_a, "a"), (pid_b, "b")):
            info = players[pid]
            assert info["position"] == packet["position"]
            assert week in info["weeks"]  # both actually played week W
            weeks = {w: pts["ppr"] for w, pts in info["weeks"].items()}
            games, total, ppg = season_to_date(weeks, week)  # strictly < W
            assert packet[side] == {"games": games, "total": total, "ppg": ppg}
            assert games >= H2H_MIN_GAMES and ppg >= H2H_MIN_PPG
        ppg_a, ppg_b = packet["a"]["ppg"], packet["b"]["ppg"]
        assert abs(ppg_a - ppg_b) <= H2H_PPG_TOLERANCE * max(ppg_a, ppg_b) + 1e-9


@needs_data
def test_season_threshold_outcomes_and_packets():
    named, _, key = build_question_set(2019, "ppr")
    season_qs = [q for q in named["questions"] if q["family"] == "season_threshold"]
    assert len(season_qs) > 100
    # positives are bounded by 12+12+24+24 threshold slots
    positives = sum(q["outcome"] for q in key["questions"].values()
                    if q["family"] == "season_threshold")
    assert 0 < positives <= 72
    for q in season_qs:
        packet = q["packet"]
        assert packet["threshold"] == (12 if packet["position"] in ("QB", "TE") else 24)
        # as-of discipline: history blocks are prior seasons only
        for block in (packet["prior_season"], packet["two_seasons_ago"]):
            if block is not None:
                assert set(block) == {"games", "total", "ppg", "pos_rank"}
    # spot-check a known outcome: Lamar Jackson finished QB1 in 2019
    lamar = next(qid for qid, q in key["questions"].items()
                 if q.get("player") == "Lamar Jackson")
    assert key["questions"][lamar]["outcome"] == 1


@needs_data
def test_base_rates_exclude_scored_season():
    rates, seasons_used = family_base_rates("ppr", exclude_season=2019)
    assert 2019 not in seasons_used
    assert HOLDOUT_SEASON not in seasons_used
    assert set(seasons_used) <= set(BASE_RATE_SEASONS)
    assert 0.2 < rates["season_threshold"] < 0.6
    assert 0.4 < rates["weekly_h2h"] < 0.6


@needs_data
def test_baseline_scoring_semantics():
    from evals.calibbench import run_baselines

    results = run_baselines(2019, "ppr", verbose=False)
    # coin Brier is exactly 0.25 on 0/1 outcomes, log-loss exactly ln 2
    for family in ("season_threshold", "weekly_h2h", "overall"):
        assert results["coin"][family]["brier"] == 0.25
        assert results["coin"][family]["log_loss"] == pytest.approx(math.log(2), abs=1e-4)
    # base rate can't be worse than coin on Brier for the season family
    # (base rate ~0.38 vs realized ~0.38 — closer than 0.5 by construction)
    assert (results["base_rate"]["season_threshold"]["brier"]
            < results["coin"]["season_threshold"]["brier"])
