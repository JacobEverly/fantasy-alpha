"""Tests for scripts/collect_play_callers.py: team normalization, as-of
discipline (opener vs midseason), seed integration, and the coverage floor
on the shipped table."""
import csv
from pathlib import Path

import pytest

from scripts.collect_play_callers import (
    COLUMNS, NORMALIZE_TEAM, OUT_CSV, SEASONS, TEAMS, assemble, coverage,
    franchise_name, norm_coach, page_title, parse_page, staff_section_ocs,
)

# ---------------------------------------------------------------------------
# Team normalization must match the vegas_features join convention

def test_normalization_matches_vegas_features():
    from evals.vegas_features import NORMALIZE_TEAM as VEGAS
    for src, dst in VEGAS.items():
        assert NORMALIZE_TEAM.get(src) == dst
    # ADP files also carry LAR for the Rams; must land on the vegas code
    assert NORMALIZE_TEAM["LAR"] == "LA"


def test_team_codes_are_franchise_normalized():
    assert len(TEAMS) == 32
    for bad in ("SD", "STL", "OAK", "JAC", "LAR"):
        assert bad not in TEAMS


def test_franchise_names_across_relocations():
    assert franchise_name("LA", 2015) == "St. Louis Rams"
    assert franchise_name("LA", 2016) == "Los Angeles Rams"
    assert franchise_name("LAC", 2016) == "San Diego Chargers"
    assert franchise_name("LAC", 2017) == "Los Angeles Chargers"
    assert franchise_name("LV", 2019) == "Oakland Raiders"
    assert franchise_name("LV", 2020) == "Las Vegas Raiders"
    assert franchise_name("WAS", 2019) == "Washington Redskins"
    assert franchise_name("WAS", 2020) == "Washington Football Team"
    assert franchise_name("WAS", 2022) == "Washington Commanders"
    assert page_title("LA", 2008) == "2008 St. Louis Rams season"


# ---------------------------------------------------------------------------
# As-of discipline: the opener is the pre-Sept-1-knowable staff; midseason
# changes are flagged separately and never leak into the opener columns.

INFOBOX_MIDSEASON = """{{Infobox NFL team season
| coach = [[Old Guy]] (fired October 5)<br>[[Interim Guy]] (interim)
| off_coach = [[Opener OC]]
}}"""


def test_opener_is_first_listed_and_midseason_flagged():
    p = parse_page(INFOBOX_MIDSEASON)
    assert p["head_coach"] == "Old Guy"          # opener, not the interim
    assert p["midseason_hc_change"] == 1
    assert p["offensive_coordinator"] == "Opener OC"
    assert p["midseason_oc_change"] == 0


def test_clean_offseason_hire_not_flagged():
    p = parse_page("{{Infobox NFL team season\n| coach = [[New Hire]]\n"
                   "| off_coach = [[Fresh OC]]\n}}")
    assert p["head_coach"] == "New Hire"
    assert p["midseason_hc_change"] == 0
    assert p["midseason_oc_change"] == 0


STAFF_BODY = """{{Infobox NFL team season
| coach = [[Head Guy]]
}}
==Staff==
|Offensive Coaches=
* Offensive coordinator – [[End Season OC]]
* Assistant offensive coordinator – [[Helper Guy]]
Later the team fired offensive coordinator [[Opener OC]] in week 8.
"""


def test_staff_fallback_prefers_prose_fired_opener():
    """Staff lists show END-of-season staff; a prose-recorded midseason
    firing makes the fired coach the opener (as-of discipline)."""
    names, midseason = staff_section_ocs(STAFF_BODY)
    assert names[0] == "Opener OC"
    assert "End Season OC" in names
    assert "Helper Guy" not in names   # assistant-OC excluded
    assert midseason is True
    p = parse_page(STAFF_BODY)
    assert p["offensive_coordinator"] == "Opener OC"
    assert p["midseason_oc_change"] == 1


def test_staff_fallback_assistant_head_coach_slash_oc_is_oc():
    names, _ = staff_section_ocs(
        "* Assistant head coach/offensive coordinator – [[Real OC]]\n")
    assert names == ["Real OC"]


def test_interim_oc_never_opener():
    names, mid = staff_section_ocs(
        "*Offensive coordinator - [[Luke Getsy]] ''(fired on November 5)''\n"
        "*Interim offensive coordinator - [[Scott Turner]]\n")
    assert names[0] == "Luke Getsy"
    assert mid is True


# ---------------------------------------------------------------------------
# changed flag + seed integration on synthetic pages

def _pages(spec):
    """spec: {(season, team): (hc, oc)} -> fake wikitext pages."""
    out = {}
    for (season, team), (hc, oc) in spec.items():
        oc_line = f"| off_coach = [[{oc}]]\n" if oc else ""
        out[page_title(team, season)] = (
            f"{{{{Infobox NFL team season\n| coach = [[{hc}]]\n{oc_line}}}}}")
    return out


def test_changed_flag_and_hc_fallback():
    pages = _pages({(2019, "KC"): ("Andy Reid", "Eric Bieniemy"),
                    (2020, "KC"): ("Andy Reid", "Eric Bieniemy"),
                    (2021, "KC"): ("Andy Reid", "New OC"),
                    (2022, "KC"): ("Andy Reid", None)})
    rows = {(r["season"], r["team"]): r for r in assemble(pages, seed={})}
    assert rows[(2020, "KC")]["changed_play_caller"] == 0
    assert rows[(2021, "KC")]["changed_play_caller"] == 1
    # no OC listed -> HC is the recorded play caller, still ambiguous
    r22 = rows[(2022, "KC")]
    assert r22["play_caller"] == "Andy Reid"
    assert r22["play_caller_source"] == "wikipedia_hc_no_oc"
    assert r22["play_caller_ambiguous"] == 1
    assert r22["changed_play_caller"] == 1  # New OC -> Reid counts as change
    # 2019 has no in-table prior season -> flag empty, never guessed
    assert rows[(2019, "KC")]["changed_play_caller"] == ""
    # missing page -> empty row, no invented names
    assert rows[(2019, "SF")]["play_caller"] == ""
    assert rows[(2019, "SF")]["play_caller_source"] == "missing"


def test_seed_fact_pins_play_caller_unambiguously():
    pages = _pages({(2023, "MIA"): ("Mike McDaniel", "Frank Smith")})
    seed = {(2023, "MIA"): {"coach": "Mike McDaniel", "season": "2023",
                            "team": "MIA", "play_calling_weight": "1.0",
                            "role": "head_coach_play_caller"}}
    r = [x for x in assemble(pages, seed)
         if (x["season"], x["team"]) == (2023, "MIA")][0]
    assert r["play_caller"] == "Mike McDaniel"
    assert r["play_caller_source"] == "seed_fact"
    assert r["play_caller_ambiguous"] == 0


def test_norm_coach():
    assert norm_coach("Bill  O'Brien ") == norm_coach("bill o'brien")
    assert norm_coach("A.J. Smith") == norm_coach("AJ Smith")


# ---------------------------------------------------------------------------
# Coverage floor on the shipped table

@pytest.mark.skipif(not OUT_CSV.exists(), reason="table not built yet")
def test_shipped_table_coverage_floor():
    with open(OUT_CSV, newline="") as f:
        rows = [{**r, "season": int(r["season"])} for r in csv.DictReader(f)]
    assert [c for c in rows[0]] == COLUMNS
    assert len(rows) == 32 * len(list(SEASONS))
    flagged = [r for r in rows if r["season"] >= 2010
               and r["play_caller"] and r["changed_play_caller"] != ""]
    pop = [r for r in rows if r["season"] >= 2010]
    assert len(flagged) / len(pop) >= 0.90
    teams = {r["team"] for r in rows}
    assert teams == set(TEAMS)
