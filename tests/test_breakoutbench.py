"""Leakage, gate, bootstrap-determinism, and no-peek guards for BreakoutBench v0.3."""
import json

import pytest

from evals.breakoutbench import (
    ADP_DIR,
    ADP_GATE,
    BUST_ADP_GATE,
    FAMILIES,
    HOLDOUT_SEASON,
    LABELS,
    OU_RANK_HI,
    OU_RANK_LO,
    bootstrap_cis,
    build_question_set,
    cell_aggregates,
    family_base_rates,
    family_of,
    load_answer_files,
    main,
    outcome_for,
    score_run,
    season_formats,
)

HAVE_DATA = ADP_DIR.exists() and (LABELS / "breakouts.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ snapshots")


# ---------------------------------------------------------------------------
# Family membership + outcomes (hand-built)


def test_family_of_gates():
    assert family_of("WR", 41, 20) == ["full_slate", "over_under"]
    assert family_of("WR", 40, 20) == ["over_under"]  # not beyond the WR gate
    assert family_of("QB", 19, None) == ["full_slate"]  # no finish -> no over_under
    assert family_of("RB", 12, 5) == ["bust"]
    assert family_of("TE", 13, 30) == ["over_under"]
    assert family_of("WR", 46, 50) == ["full_slate"]  # beyond the o/u band
    assert family_of("RB", 13, None) == []


def test_outcome_definitions():
    row = {"breakout": True, "bust": False, "adp_pos_rank": 30, "finish_pos_rank": 12}
    assert outcome_for("full_slate", row) == 1
    assert outcome_for("bust", row) == 0
    assert outcome_for("over_under", row) == 1  # 12 beats 30
    assert outcome_for("over_under", {**row, "finish_pos_rank": 30}) == 0  # tie = not better


# ---------------------------------------------------------------------------
# Holdout guard


def test_holdout_season_refused():
    with pytest.raises(SystemExit):
        main(["build", "--season", str(HOLDOUT_SEASON)])
    with pytest.raises(SystemExit):
        main(["score", f"answers_{HOLDOUT_SEASON}_ppr.json"])
    with pytest.raises(AssertionError):
        build_question_set(HOLDOUT_SEASON, "ppr")


@needs_data
def test_season_formats_exclude_holdout():
    sf = season_formats()
    assert all(s < HOLDOUT_SEASON for s, _ in sf)
    assert (2019, "ppr") in sf and (2019, "standard") in sf
    assert (2015, "half_ppr") not in sf  # half-ppr ADP starts 2018


# ---------------------------------------------------------------------------
# No leakage in question files (all three families)


@needs_data
def test_question_payload_leaks_no_outcomes_or_names():
    qpayload, kpayload = build_question_set(2019, "ppr")
    text = json.dumps(qpayload)
    assert '"outcome"' not in text and '"finish_pos_rank"' not in text
    assert qpayload["season_masked"] is True and "season" not in qpayload
    # every family present, and no player name from the key appears anywhere
    fams = {q["family"] for q in qpayload["questions"]}
    assert fams == set(FAMILIES)
    for q in kpayload["questions"].values():
        assert q["player"] not in text
    # packet history blocks are strictly prior-season shaped (no target-season stats)
    for q in qpayload["questions"]:
        packet = q["packet"]
        assert set(packet) == {"position", "adp_overall", "adp_pos_rank", "adp_stdev",
                               "seasons_of_data", "years_since_first_season",
                               "prior_season", "two_seasons_ago"}
        for block in (packet["prior_season"], packet["two_seasons_ago"]):
            if block is not None:
                assert set(block) == {"games", "total", "ppg", "pos_rank"}


@needs_data
def test_gate_correctness_all_families():
    qpayload, kpayload = build_question_set(2019, "ppr")
    key = kpayload["questions"]
    for q in qpayload["questions"]:
        rank = q["packet"]["adp_pos_rank"]
        pos = q["packet"]["position"]
        k = key[q["question_id"]]
        assert k["adp_pos_rank"] == rank
        if q["family"] == "full_slate":
            assert rank > ADP_GATE[pos]
        elif q["family"] == "bust":
            assert rank <= BUST_ADP_GATE
        elif q["family"] == "over_under":
            assert OU_RANK_LO <= rank <= OU_RANK_HI
            assert k["finish_pos_rank"] is not None
            assert k["outcome"] == int(k["finish_pos_rank"] < rank)


@needs_data
def test_build_is_deterministic():
    assert build_question_set(2019, "ppr") == build_question_set(2019, "ppr")


# ---------------------------------------------------------------------------
# Base-rate no-peek


@needs_data
def test_base_rates_use_only_prior_seasons():
    rates, seasons_used = family_base_rates(before_season=2018)
    assert seasons_used and max(seasons_used) < 2018 and min(seasons_used) >= 2011
    assert set(rates) == set(FAMILIES)
    for fam, rate in rates.items():
        assert 0.0 < rate < 1.0
    # breakouts are rare, busts common-ish, over/under below a coin flip
    assert rates["full_slate"] < 0.25
    assert rates["over_under"] < 0.5


@needs_data
def test_base_rates_shift_with_cutoff():
    r15, used15 = family_base_rates(before_season=2015)
    r24, used24 = family_base_rates(before_season=2024)
    assert set(used15) < set(used24)
    assert r15 != r24  # more seasons move the rates


# ---------------------------------------------------------------------------
# Answers loading + bootstrap


@needs_data
def test_load_answers_skips_bad_entries(tmp_path):
    qpayload, kpayload = build_question_set(2019, "ppr")
    qids = list(kpayload["questions"])
    answers = [
        {"question_id": qids[0], "p": 0.4},
        {"question_id": qids[0], "p": 0.9},   # duplicate -> skipped, first kept
        {"question_id": "NOPE", "p": 0.5},    # unknown id -> skipped
        {"question_id": qids[1], "p": 1.5},   # out of range -> skipped
        {"question_id": qids[2], "p": 0.2},
    ]
    path = tmp_path / "run_2019_ppr.json"
    path.write_text(json.dumps(answers))
    loaded = load_answer_files([path])
    assert loaded["answered"] == 2 and loaded["skipped"] == 3
    triples = [t for fam in loaded["cells"][(2019, "ppr")]["pairs"].values() for t in fam]
    kept = {qid: p for p, _y, qid in triples}
    assert kept[qids[0]] == 0.4


@needs_data
def test_bootstrap_deterministic_under_seed(tmp_path):
    # base-rate answers over two seasons, tiny rep count for speed
    paths = []
    for season in (2018, 2019):
        _, kpayload = build_question_set(season, "ppr")
        rates, _ = family_base_rates(before_season=season)
        answers = [{"question_id": qid, "p": rates[q["family"]]}
                   for qid, q in kpayload["questions"].items()]
        p = tmp_path / f"br_{season}_ppr.json"
        p.write_text(json.dumps(answers))
        paths.append(p)
    a = score_run(paths, reps=200, seed=7)
    b = score_run(paths, reps=200, seed=7)
    assert a == b
    c = score_run(paths, reps=200, seed=8)
    assert c["pooled"] == a["pooled"]  # point estimates don't depend on the seed
    lo, hi = a["ci"]["pooled_brier"]
    assert lo <= a["pooled"]["pooled_brier"] <= hi


def test_bootstrap_cis_hand_built():
    # one season only -> every resample is that season -> degenerate CI
    agg = cell_aggregates({(2019, "ppr"): {
        "pairs": {"full_slate": [(0.9, 1, "B0000"), (0.1, 0, "B0001")]},
        "n_total": 2, "n_family_total": {"full_slate": 2}, "slate_base": 0.5,
    }})
    cis = bootstrap_cis(agg, reps=50, seed=1)
    lo, hi = cis["pooled_brier"]
    assert lo == hi == pytest.approx(0.01)
