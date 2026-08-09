"""Guards for the Vegas-vs-ADP packet test: draft-day legality of the vegas
join, anonymization safety of the v3 packet, v3 encoding, and the stratified
/ bootstrap helpers."""
import csv
import json
import math

import pytest

from evals.breakoutbench import (
    V3_FIELDS,
    V3_STATUS,
    VEGAS_TABLE,
    VEGAS_TEAM_NORMALIZE,
    _team_vegas_index,
    build_question_set,
    vegas_for,
)
from evals.gbdt_baseline import (
    FEATURE_NAMES,
    V2_FEATURE_NAMES,
    V3_FEATURE_NAMES,
    encode_v3,
)
from evals.vegas_packet_test import (
    SEASONS,
    adp_band,
    family_candidates,
    family_paired_delta,
    tercile_split,
)

HAVE_DATA = VEGAS_TABLE.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ snapshots")


# ---------------------------------------------------------------------------
# Draft-day legality of the vegas join


@needs_data
def test_vegas_index_loads_only_draft_day_legal_columns():
    idx = _team_vegas_index()
    row = next(iter(idx.values()))
    # the retrospective season-S mean must never be loaded here
    assert set(row) == {"week1_implied_total", "prior_season_mean_implied_total"}
    assert "mean_implied_total" not in row


@needs_data
def test_vegas_index_matches_source_table():
    with open(VEGAS_TABLE, newline="") as f:
        rows = {(int(r["season"]), r["team"]): r for r in csv.DictReader(f)}
    idx = _team_vegas_index()
    probe = rows[(2019, "KC")]
    assert idx[(2019, "KC")]["week1_implied_total"] == float(
        probe["week1_implied_total"])
    assert idx[(2019, "KC")]["prior_season_mean_implied_total"] == float(
        probe["prior_season_mean_implied_total"])


@needs_data
def test_vegas_for_joins_and_normalizes_era_codes():
    """A candidate whose S-1 depth team carries an era-native code (SD in the
    2015 depth chart) must join the current-franchise vegas row (LAC)."""
    from evals.breakoutbench import PACKET_FEATURES_DIR
    with open(PACKET_FEATURES_DIR / "features_2016.csv", newline="") as f:
        sd = next(r for r in csv.DictReader(f)
                  if r["team_s1_end"] == "SD" and not r["team_asof"])
    got = vegas_for(2016, sd["player"], sd["position"], sd["player_id"])
    assert got[V3_STATUS] == "ok"
    expected = _team_vegas_index()[(2016, VEGAS_TEAM_NORMALIZE["SD"])]
    assert got["team_week1_implied_total"] == expected["week1_implied_total"]
    assert got["team_prior_season_mean_implied_total"] == \
        expected["prior_season_mean_implied_total"]


@needs_data
def test_vegas_for_is_anonymization_safe_and_statused():
    got = vegas_for(2019, "Austin Ekeler", "RB", None)
    assert set(got) == set(V3_FIELDS) | {V3_STATUS}  # no team code emitted
    missing = vegas_for(2019, "No Such Player Xyz", "WR", None)
    assert missing[V3_STATUS] == "no_feature_row"
    assert all(missing[k] is None for k in V3_FIELDS)


# ---------------------------------------------------------------------------
# v3 packet build (anonymized track)


@needs_data
def test_v3_question_set_shares_key_and_adds_only_vegas_block():
    q1, k1 = build_question_set(2019, "ppr", "v1")
    q3, k3 = build_question_set(2019, "ppr", "v3")
    assert k1 == k3  # anon id -> outcome map identical across packet versions
    assert q3["packet"] == "v3"
    text = json.dumps(q3)
    assert '"outcome"' not in text and '"finish_pos_rank"' not in text
    assert "team_asof" not in text and "team_s1_end" not in text
    for q in q3["questions"]:
        assert set(q["packet"]) == set(q1["questions"][0]["packet"]) | \
            {"enrichment", "team_vegas"}
        tv = q["packet"]["team_vegas"]
        assert set(tv) == set(V3_FIELDS) | {V3_STATUS}
    for q in k3["questions"].values():
        assert q["player"] not in text


def test_v3_packets_refuse_pre_coverage_and_holdout_seasons():
    with pytest.raises(AssertionError):
        build_question_set(2015, "ppr", "v3")
    with pytest.raises(AssertionError):
        build_question_set(2025, "ppr", "v3")


# ---------------------------------------------------------------------------
# v3 encoding


def test_encode_v3_layout_and_missing_encoding():
    assert V3_FEATURE_NAMES[:len(V2_FEATURE_NAMES)] == V2_FEATURE_NAMES
    assert V3_FEATURE_NAMES[len(V2_FEATURE_NAMES):] == [
        "team_week1_implied_total", "team_week1_implied_total_missing",
        "team_prior_season_mean_implied_total",
        "team_prior_season_mean_implied_total_missing"]
    features = {"position": "WR", "adp_overall": 120.0, "adp_pos_rank": 45,
                "adp_stdev": 3.0, "seasons_of_data": 0,
                "years_since_first_season": 0,
                "prior_season": None, "two_seasons_ago": None}
    enrichment = {k: None for k in V2_FEATURE_NAMES[len(FEATURE_NAMES)::2]}
    row = encode_v3(features, enrichment,
                    {"team_week1_implied_total": 24.5,
                     "team_prior_season_mean_implied_total": None})
    assert len(row) == len(V3_FEATURE_NAMES)
    assert row[-4:-2] == [24.5, 0.0]           # present -> value + 0 flag
    assert math.isnan(row[-2]) and row[-1] == 1.0  # absent -> NaN + 1 flag


# ---------------------------------------------------------------------------
# study helpers


def test_seasons_exclude_holdout():
    assert 2025 not in SEASONS and max(SEASONS) == 2024


def test_adp_band_matches_source_alpha_pattern():
    assert adp_band("WR", 41) == "near_gate"   # +1 beyond the WR gate (40)
    assert adp_band("WR", 52) == "near_gate"   # +12
    assert adp_band("WR", 53) == "mid"         # +13
    assert adp_band("QB", 48) == "mid"         # +30 beyond the QB gate (18)
    assert adp_band("QB", 49) == "deep"        # +31
    assert adp_band("TE", 60) == "deep"


def test_tercile_split_thirds_and_ties():
    assert tercile_split([1, 2, 3, 4, 5, 6]) == [0, 0, 1, 1, 2, 2]
    assert tercile_split([6, 5, 4, 3, 2, 1]) == [2, 2, 1, 1, 0, 0]
    assert tercile_split([5.0, 5.0, 5.0]) == [0, 1, 2]  # ties by input order
    assert tercile_split([]) == []


def test_family_paired_delta_deterministic_and_signed():
    import numpy as np
    a = {"per_season": {s: {"sq_err": np.full(4, 0.30)} for s in (2016, 2017)},
         "brier": 0.30}
    b = {"per_season": {s: {"sq_err": np.full(4, 0.20)} for s in (2016, 2017)},
         "brier": 0.20}
    d = family_paired_delta(a, b, draws=200, seed=7)
    assert d["brier_delta"] == pytest.approx(0.10)
    lo, hi = d["brier_delta_ci"]
    assert lo == pytest.approx(0.10) and hi == pytest.approx(0.10)
    assert family_paired_delta(a, b, draws=200, seed=7) == d


@needs_data
def test_family_candidates_respect_gates_and_carry_vegas():
    busts = family_candidates(2019, "bust")
    assert busts and all(c["features"]["adp_pos_rank"] <= 12 for c in busts)
    ous = family_candidates(2019, "over_under")
    assert ous and all(13 <= c["features"]["adp_pos_rank"] <= 45 for c in ous)
    for c in busts + ous:
        assert set(c["vegas"]) == set(V3_FIELDS) | {V3_STATUS}
        assert c["outcome"] in (0, 1)
