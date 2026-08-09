"""Guards for the enriched-packet (v2) experiment.

1. v2 slate leakage — no outcome fields, no player names, enrichment values
   traceable to data/processed/packet_features only;
2. determinism — v2 builds and GBDT-v2 folds reproduce exactly under a seed;
3. coverage-status propagation — missing != zero survives the join and the
   GBDT encoding (NaN + indicator), including the pre-coverage news seasons.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.breakoutbench import (  # noqa: E402
    ADP_DIR,
    LABELS,
    PACKET_FEATURES_DIR,
    V2_FIELDS,
    V2_SEASON_FLOOR,
    V2_STATUSES,
    build_question_set,
    enrichment_for,
    main,
)
from evals.gbdt_baseline import (  # noqa: E402
    FEATURE_NAMES,
    V2_FEATURE_NAMES,
    V2_ORDER,
    encode_v2,
)
from evals.names import norm_name  # noqa: E402

HAVE_DATA = (ADP_DIR.exists() and (LABELS / "breakouts.csv").exists()
             and (PACKET_FEATURES_DIR / "features_2019.csv").exists())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ snapshots")

SEASON, PRESET = 2019, "ppr"


# ---------------------------------------------------------------------------
# v2 slate leakage


@needs_data
def test_v2_payload_has_no_outcomes_or_names():
    qpayload, kpayload = build_question_set(SEASON, PRESET, "v2")
    text = json.dumps(qpayload)
    assert '"outcome"' not in text and '"finish_pos_rank"' not in text
    assert qpayload["season_masked"] is True and "season" not in qpayload
    assert qpayload["packet"] == "v2"
    for q in kpayload["questions"].values():
        assert q["player"] not in text  # anonymity holds with enrichment


@needs_data
def test_v2_enrichment_fields_come_only_from_packet_features():
    """Every enrichment key is a declared v2 field/status, and every non-null
    value matches the packet_features CSV row for that candidate."""
    qpayload, kpayload = build_question_set(SEASON, PRESET, "v2")
    by_name: dict[tuple, dict] = {}
    with open(PACKET_FEATURES_DIR / f"features_{SEASON}.csv", newline="") as f:
        for r in csv.DictReader(f):
            by_name.setdefault((norm_name(r["player"]), r["position"]), r)
    checked = 0
    for q in qpayload["questions"]:
        enr = q["packet"]["enrichment"]
        assert set(enr) == set(V2_FIELDS) | set(V2_STATUSES)
        player = kpayload["questions"][q["question_id"]]["player"]
        row = by_name.get((norm_name(player), q["packet"]["position"]))
        if row is None:
            assert enr["news_status"] == "no_feature_row"
            continue
        for field, cast in V2_FIELDS.items():
            if enr[field] is not None and row[field] != "":
                assert enr[field] == pytest.approx(cast(float(row[field])))
                checked += 1
    assert checked > 100  # the join actually landed


@needs_data
def test_v2_anon_ids_and_key_match_v1():
    """v2 must be the SAME questions with richer packets — identical key."""
    q1, k1 = build_question_set(SEASON, PRESET, "v1")
    q2, k2 = build_question_set(SEASON, PRESET, "v2")
    assert k1["questions"] == k2["questions"]
    for a, b in zip(q1["questions"], q2["questions"]):
        assert a["question_id"] == b["question_id"]
        assert a["family"] == b["family"]
        assert {k: v for k, v in b["packet"].items() if k != "enrichment"} == a["packet"]


def test_v2_refuses_pre_coverage_seasons():
    with pytest.raises(AssertionError):
        build_question_set(V2_SEASON_FLOOR - 1, "ppr", "v2")
    with pytest.raises(SystemExit):
        main(["build", "--season", "2015", "--packet", "v2"])


# ---------------------------------------------------------------------------
# determinism


@needs_data
def test_v2_build_deterministic():
    assert build_question_set(SEASON, PRESET, "v2") == \
        build_question_set(SEASON, PRESET, "v2")


@needs_data
def test_gbdt_v2_deterministic_and_leak_free():
    from evals.gbdt_baseline import load_data, train_predict_v2

    data = load_data("ppr", "ppr", seasons=tuple(range(2015, 2017)))
    a, _, xa, _ = train_predict_v2(data, 2016, "ppr", seed=123)
    b, _, xb, _ = train_predict_v2(data, 2016, "ppr", seed=123)
    assert [r["p_breakout"] for r in a] == [r["p_breakout"] for r in b]
    assert ((xa == xb) | (xa != xa)).all()  # equal or both-NaN
    # leakage: v2 training frames stop strictly before the target season —
    # the 2016 frame is exactly the 2015 candidate slate, nothing newer
    from evals.gbdt_baseline import build_candidates_v2, training_frame_v2
    x, y = training_frame_v2(data, 2016, "ppr", train_start=2015)
    assert len(x) == len(y) == len(build_candidates_v2(data, 2015, "ppr"))


# ---------------------------------------------------------------------------
# coverage-status propagation (missing != zero)


def test_encode_v2_missing_is_nan_plus_indicator():
    features = {"position": "WR", "adp_overall": 100.0, "adp_pos_rank": 45,
                "adp_stdev": 3.0, "seasons_of_data": 0,
                "years_since_first_season": 0,
                "prior_season": None, "two_seasons_ago": None}
    enrich = {k: None for k in V2_FIELDS}
    enrich.update({"usage_status": "no_prior_season", "news_status": "pre_coverage"})
    row = encode_v2(features, enrich)
    assert len(row) == len(V2_FEATURE_NAMES)
    base = len(FEATURE_NAMES)
    for i, field in enumerate(V2_ORDER):
        assert math.isnan(row[base + 2 * i]), field       # value = NaN, not 0
        assert row[base + 2 * i + 1] == 1.0, field        # indicator set
    # and a present value is passed through with indicator cleared
    enrich["n_news_90d"] = 7
    row = encode_v2(features, enrich)
    j = base + 2 * list(V2_ORDER).index("n_news_90d")
    assert row[j] == 7.0 and row[j + 1] == 0.0


@needs_data
def test_pre_coverage_news_stays_null_not_zero():
    # 2015 features exist but news is pre-coverage: counts must be None
    enr = enrichment_for(2015, "Antonio Brown", "WR", None)
    assert enr["news_status"] == "pre_coverage"
    assert enr["n_news_90d"] is None and enr["hype_flags"] is None
    # a covered season for the same player yields real counts
    enr19 = enrichment_for(2019, "Antonio Brown", "WR", None)
    assert enr19["news_status"] == "ok" and enr19["n_news_90d"] is not None


@needs_data
def test_unmatched_candidate_gets_no_feature_row_status():
    enr = enrichment_for(2019, "Nonexistent Player XYZ", "WR", None)
    assert all(enr[f] is None for f in V2_FIELDS)
    assert enr["usage_status"] == enr["news_status"] == "no_feature_row"
