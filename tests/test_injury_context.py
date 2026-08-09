"""Lexicon, attribution, and face-validity guards for injury-context labels."""
import csv

import pytest

from evals.build_injury_context import (
    ACUTE,
    INJ_DIR,
    LABELS,
    OTHER,
    SOFT,
    attribute_season,
    classify,
)

HAVE_DATA = INJ_DIR.exists() and (LABELS / "injury_context.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ snapshots")


# ---------------------------------------------------------------------------
# Lexicon (hand-built strings)


def test_acute_keywords():
    for text in ("Achilles", "Torn ACL", "MCL", "ankle fracture", "Broken Collarbone",
                 "dislocated shoulder", "Lisfranc", "Patellar Tendon", "Fibula",
                 "ruptured tendon", "Sternoclavicular"):
        assert classify(text) == (ACUTE, "acute"), text


def test_soft_tissue_keywords():
    for text in ("Hamstring", "right groin", "Calf", "Quadricep", "soft tissue",
                 "hamstring tightness", "Hip Flexor", "Oblique", "Pectoral"):
        assert classify(text) == (SOFT, "soft"), text


def test_ambiguous_body_part_is_other_not_acute():
    # "Knee" alone is OTHER — acute only with torn/acl/etc. context
    assert classify("Knee") == (OTHER, "body")
    assert classify("Ankle") == (OTHER, "body")
    assert classify("Shoulder") == (OTHER, "body")
    assert classify("Knee (torn ACL)") == (ACUTE, "acute")


def test_multi_injury_string_resolves_most_severe():
    assert classify("Knee, Achilles") == (ACUTE, "acute")
    assert classify("Toe, Achilles, Hip") == (ACUTE, "acute")
    assert classify("Knee, Hamstring") == (SOFT, "soft")


def test_other_subtags():
    assert classify("Concussion") == (OTHER, "concussion")
    assert classify("Illness") == (OTHER, "illness")
    assert classify("Not Injury Related - Personal Matter") == (OTHER, "personal")


def test_excluded_designations():
    for text in ("", "Not Injury Related", "Not Injury Related - Resting Player",
                 "Coach's Decision", "not injury related -- resting veteran",
                 "Not Injury Related - Travel", "--"):
        assert classify(text) is None, repr(text)


def test_unmatched_falls_to_other_and_is_flagged():
    assert classify("Arrhythmia") == (OTHER, "unmatched")


# ---------------------------------------------------------------------------
# Attribution on a synthetic season (17 team games, weeks 1-18, bye week 9)

TEAM_WEEKS = set(range(1, 19)) - {9}


def _des(week, text):
    cls, subtag = classify(text)
    return {"week": week, "cls": cls, "subtag": subtag, "text": text}


def test_attribution_carries_designation_until_next_played_game():
    # hamstring in week 3, misses 3-5, returns week 6, plays out the season
    played = TEAM_WEEKS - {3, 4, 5}
    out = attribute_season(played, TEAM_WEEKS, [_des(3, "Hamstring")])
    assert out["missed_soft"] == 3
    assert out["missed_acute"] == out["missed_other"] == 0
    assert not out["season_ending_acute"]
    assert out["games_played"] == 14


def test_return_to_play_clears_designation():
    # missed week 3 (hamstring), played 4-10, missed 11 with no report: unattributed
    played = TEAM_WEEKS - {3, 11}
    out = attribute_season(played, TEAM_WEEKS, [_des(3, "Hamstring")])
    assert out["missed_soft"] == 1
    assert out["missed_other"] == 0


def test_missed_week_attributes_to_most_recent_designation():
    # questionable knee week 2 (played through), achilles week 5, misses 5-18
    played = set(range(1, 5))
    out = attribute_season(played, TEAM_WEEKS,
                           [_des(2, "Knee"), _des(5, "Achilles")])
    assert out["missed_acute"] == 13  # weeks 5-18 minus bye
    assert out["missed_other"] == 0
    assert out["season_ending_acute"]
    assert out["primary_injury_text_sample"] == "Achilles"


def test_season_ending_acute_needs_three_remaining_team_games():
    # achilles designation in week 17: only one team game remains -> no flag
    played = set(range(1, 17)) - {9}
    out = attribute_season(played, TEAM_WEEKS, [_des(17, "Achilles")])
    assert out["missed_acute"] == 2 and not out["season_ending_acute"]


def test_structural_body_part_escalates_only_when_season_ending():
    # "Knee" ending the season -> escalated ACUTE; mid-season "Knee" stays OTHER
    played = set(range(1, 8))
    ending = attribute_season(played, TEAM_WEEKS, [_des(8, "Knee")])
    assert ending["season_ending_acute"] and ending["missed_acute"] == 10
    mid = attribute_season(TEAM_WEEKS - {4, 5}, TEAM_WEEKS, [_des(4, "Knee")])
    assert mid["missed_other"] == 2 and not mid["season_ending_acute"]


def test_soft_tissue_season_ender_stays_soft():
    played = set(range(1, 8))
    out = attribute_season(played, TEAM_WEEKS, [_des(8, "Hamstring")])
    assert out["missed_soft"] == 10 and not out["season_ending_acute"]


def test_vanish_without_any_designation():
    # played weeks 1-2 then straight to IR, no report rows ever
    out = attribute_season({1, 2}, TEAM_WEEKS, [])
    assert out["vanished_unattributed"]
    assert out["missed_acute"] == out["missed_soft"] == out["missed_other"] == 0
    covered = attribute_season({1, 2}, TEAM_WEEKS, [_des(3, "Knee")])
    assert not covered["vanished_unattributed"]


def test_healthy_scratch_weeks_not_attributed():
    out = attribute_season(TEAM_WEEKS - {6, 7}, TEAM_WEEKS, [])
    assert out["missed_acute"] == out["missed_soft"] == out["missed_other"] == 0
    assert not out["season_ending_acute"] and not out["vanished_unattributed"]


# ---------------------------------------------------------------------------
# Face validity against the on-disk labels


@needs_data
def test_aaron_rodgers_2023_achilles():
    # Played week 1 only; "Achilles, Out" rows appear weeks 13-18 when his
    # practice window opened. Deterministic given the pinned snapshot.
    with open(LABELS / "injury_context.csv", newline="") as f:
        rows = {(r["season"], r["player_id"]): r for r in csv.DictReader(f)}
    r = rows[("2023", "00-0023459")]
    assert r["player"] == "Aaron Rodgers" and r["position"] == "QB"
    assert r["games_played"] == "1"
    assert int(r["missed_acute"]) >= 5
    assert r["season_ending_acute"] == "True"
    assert "achil" in r["primary_injury_text_sample"].lower()


@needs_data
def test_output_schema_and_coverage():
    with open(LABELS / "injury_context.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "season", "player_id", "player", "position", "games_played",
            "missed_acute", "missed_soft", "missed_other",
            "season_ending_acute", "primary_injury_text_sample"]
        rows = list(reader)
    seasons = {r["season"] for r in rows}
    assert {str(y) for y in range(2011, 2026)} <= seasons
    assert all(r["position"] in {"QB", "RB", "WR", "TE"} for r in rows)
