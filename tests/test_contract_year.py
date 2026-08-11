"""Unit tests for evals/contract_year_test.py derivation + join + study math.

All offline: raw contract rows are synthetic dicts, no files or network.
"""
import csv
from pathlib import Path

import pytest

from evals.contract_year_test import (
    STATUS_COLUMNS,
    derive_status,
    flag_cells,
    load_status_index,
    pooled_row,
    select_contract,
    status_for,
    status_row,
    write_status,
)


def _raw(**kw):
    row = {
        "player": "Test Player", "position": "RB", "team": "Giants",
        "otc_id": "100", "gsis_id": "00-0000100", "year_signed": 2020,
        "years": 4, "value": 30.0, "apy": 7.5, "apy_cap_pct": "0.04",
        "contract_type": "Drafted", "option_through": None,
        "draft_year": 2020,
    }
    row.update(kw)
    return row


# --------------------------------------------------------------- selection

def test_final_scheduled_year_is_contract_year():
    deal = [_raw()]  # 2020-2023
    for season, expect_cy, expect_left in ((2020, False, 4), (2022, False, 2),
                                           (2023, True, 1)):
        sel = select_contract(deal, season)
        r = status_row(sel, season)
        assert r["in_contract_year"] is expect_cy
        assert r["years_remaining"] == expect_left


def test_no_contract_on_file_before_signing_and_after_expiry():
    deal = [_raw()]
    assert select_contract(deal, 2019) is None  # as-of: future deal invisible
    assert select_contract(deal, 2024) is None  # expired, nothing newer


def test_extension_supersedes_from_its_signing_year():
    deals = [_raw(), _raw(year_signed=2022, years=3, value=45.0,
                          contract_type="Extension")]
    # 2021: still the rookie deal; 2022: the extension (signed that year)
    assert select_contract(deals, 2021)["contract_type"] == "Drafted"
    r = status_row(select_contract(deals, 2022), 2022)
    assert r["contract_type"] == "Extension"
    assert r["in_contract_year"] is False  # runs through 2024
    assert r["signed_recently"] is True    # the year-granularity ambiguity flag
    assert r["rookie_deal"] is False


def test_same_year_tiebreak_prefers_real_deal_over_practice_deal():
    deals = [_raw(year_signed=2022, years=3, value=45.0, contract_type="UFA"),
             _raw(year_signed=2022, years=1, value=0.2,
                  contract_type="Practice")]
    assert select_contract(deals, 2022)["contract_type"] == "UFA"


def test_fifth_year_option_extends_round1_rookie_deal():
    deal = [_raw(option_through=2024)]  # 2020-2023 + exercised option 2024
    y4 = status_row(select_contract(deal, 2023), 2023)
    assert y4["in_contract_year"] is False and y4["years_remaining"] == 2
    y5 = status_row(select_contract(deal, 2024), 2024)
    assert y5["in_contract_year"] is True
    assert y5["fifth_year_option"] is True


def test_franchise_tag_is_a_contract_year():
    tag = [_raw(year_signed=2023, years=1, value=10.1,
                contract_type="Franchise", draft_year=2018)]
    r = status_row(select_contract(tag, 2023), 2023)
    assert r["in_contract_year"] is True
    assert r["rookie_deal"] is False


def test_rookie_deal_flag_falls_back_to_draft_year_when_type_unaligned():
    r = status_row(select_contract([_raw(contract_type="")], 2021), 2021)
    assert r["rookie_deal"] is True
    r = status_row(select_contract(
        [_raw(contract_type="", draft_year=2016)], 2021), 2021)
    assert r["rookie_deal"] is False


def test_derive_status_asof_future_signing_never_leaks_backward():
    """A 2023 extension must not change any season<=2022 row."""
    base = derive_status([_raw()], seasons=(2021, 2022, 2023))
    extended = derive_status(
        [_raw(), _raw(year_signed=2023, years=3, value=60.0,
                      contract_type="Extension")],
        seasons=(2021, 2022, 2023))
    by_season = lambda rows: {r["season"]: r for r in rows}  # noqa: E731
    b, e = by_season(base), by_season(extended)
    assert b[2021] == e[2021] and b[2022] == e[2022]
    assert b[2023]["in_contract_year"] is True      # rookie deal final year
    assert e[2023]["in_contract_year"] is False     # superseded by extension


def test_derive_status_emits_one_row_per_covered_season():
    rows = derive_status([_raw()], seasons=(2019, 2020, 2021, 2022, 2023,
                                            2024))
    assert [r["season"] for r in rows] == [2020, 2021, 2022, 2023]
    assert set(STATUS_COLUMNS) == set(rows[0])


# ------------------------------------------------------------ join lookup

def _index(tmp_path: Path):
    rows = derive_status(
        [_raw(), _raw(otc_id="200", gsis_id="", player="Fully Back",
                      position="FB", years=2, contract_type="UFA",
                      draft_year=2016)],
        seasons=(2020, 2021))
    path = tmp_path / "contract_status.csv"
    write_status(rows, path)
    return load_status_index(path)


def test_status_for_matches_gsis_then_name(tmp_path):
    idx = _index(tmp_path)
    hit = status_for(idx, 2020, "Someone Renamed", "RB", "00-0000100")
    assert hit is not None and hit["years_remaining"] == 4
    # name+position fallback (no gsis in the table for the FB), FB -> RB
    hit = status_for(idx, 2021, "Fully Back", "RB", "00-0999999")
    assert hit is not None and hit["in_contract_year"] is True
    assert status_for(idx, 2021, "Unknown Player", "WR", None) is None
    # season is part of the key: no 2022 rows exist
    assert status_for(idx, 2022, "Test Player", "RB", "00-0000100") is None


# ------------------------------------------------------------- study math

def _pop(per_season_rows):
    return {s: rows for s, rows in per_season_rows.items()}


def test_flag_cells_excludes_unmatched_and_counts_flags(tmp_path):
    idx = _index(tmp_path)
    pop = _pop({2020: [
        {"season": 2020, "player": "Test Player", "player_id": "00-0000100",
         "position": "RB", "breakout": True},
        {"season": 2020, "player": "Fully Back", "player_id": "",
         "position": "RB", "breakout": False},
        {"season": 2020, "player": "Not In Table", "player_id": "",
         "position": "WR", "breakout": True},
    ]})
    per_season, cov = flag_cells(pop, idx, "rookie_deal", "breakout")
    assert cov == {"n": 3, "matched": 2, "flag_true": 1,
                   "ambiguous": 2}  # both matched deals signed in 2020
    assert per_season[2020] == {"k_c": 1, "n_c": 1, "k_u": 0, "n_u": 1}


def test_pooled_row_lift_and_ci():
    per_season = {s: {"k_c": 3, "n_c": 10, "k_u": 10, "n_u": 100}
                  for s in range(2011, 2019)}
    r = pooled_row(per_season, reps=500, seed=1)
    assert r["n_flag"] == 80 and r["n_noflag"] == 800
    assert r["rate_flag"] == pytest.approx(0.3)
    assert r["lift"] == pytest.approx(3.0)
    lo, hi = r["ci"]
    assert lo == pytest.approx(3.0) and hi == pytest.approx(3.0)


def test_pooled_row_degenerate_flag_group():
    per_season = {2020: {"k_c": 0, "n_c": 0, "k_u": 5, "n_u": 50}}
    r = pooled_row(per_season, reps=200, seed=1)
    assert r["lift"] is None
