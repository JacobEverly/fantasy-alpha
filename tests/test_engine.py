import pytest

from harness.league import LeagueConfig
from harness.scoring import PRESETS, ScoringConfig, from_research_scoring
from harness.valuation import ProjectedPlayer, assign_starters, value_over_replacement
from harness.windows import draft_window, fall_severity, recommendation_status

RESEARCH_LEAGUE = {
    "teams": 10,
    "roster": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2, "BENCH": 6},
    "flex_eligible": ["RB", "WR", "TE"],
    "scoring": {
        "passing_yard": 0.04,
        "passing_td": 4.0,
        "interception": -1.0,
        "rushing_yard": 0.1,
        "rushing_td": 6.0,
        "reception": 0.5,
        "receiving_yard": 0.1,
        "receiving_td": 6.0,
        "fumble_lost": -2.0,
    },
    "draft": {"slot": 2, "rounds": 15},
}


def league(slot=2, teams=10, rounds=15):
    return LeagueConfig(
        teams=teams,
        roster={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2, "BENCH": 6},
        scoring=PRESETS["half_ppr"],
        draft_slot=slot,
        rounds=rounds,
    )


def test_snake_picks_matches_research_league():
    assert league().snake_picks() == (2, 19, 22, 39, 42, 59, 62, 79, 82, 99, 102, 119, 122, 139, 142)


def test_turns_slot2_pairs_after_first_pick():
    turns = league().draft_turns()
    assert turns[0].picks == (2,)
    assert turns[1].picks == (19, 22)
    assert turns[1].next_turn_pick == 39
    assert turns[-1].next_turn_pick is None


def test_turns_slot1_wraps_pair():
    turns = league(slot=1).draft_turns()
    assert turns[1].picks == (20, 21)


def test_turns_mid_slot_stays_singletons():
    # Slot 5 of 10: picks are 9 and 11 apart — the research repo faked pairs here.
    turns = league(slot=5).draft_turns()
    assert all(len(t.picks) == 1 for t in turns)


def test_turns_even_rounds_do_not_crash():
    turns = league(slot=7, teams=12, rounds=16).draft_turns()
    assert sum(len(t.picks) for t in turns) == 16


def test_research_yaml_translation_and_scoring():
    cfg = LeagueConfig.from_research_yaml(RESEARCH_LEAGUE)
    assert cfg.snake_picks()[0] == 2
    stats = {
        "passing_yards": 250,
        "passing_tds": 2,
        "passing_interceptions": 1,
        "rushing_yards": 40,
        "receptions": 5,
        "receiving_yards": 60,
        "receiving_tds": 1,
        "fumbles_lost": 1,
    }
    # 10 + 8 - 1 + 4 + 2.5 + 6 + 6 - 2 = 33.5
    assert cfg.scoring.score(stats) == 33.5


def test_scoring_from_nflverse_row_strings_and_derived():
    row = {
        "rushing_yards": "112",
        "rushing_tds": "1",
        "receptions": "4",
        "receiving_yards": "38",
        "sack_fumbles_lost": "",
        "rushing_fumbles_lost": "1",
        "receiving_fumbles_lost": "0",
    }
    # 11.2 + 6 + 4 + 3.8 - 2 = 23.0 in full PPR
    assert PRESETS["ppr"].score(row) == 23.0


def test_position_override_te_premium():
    cfg = ScoringConfig(base=dict(PRESETS["ppr"].base), position_overrides={"TE": {"receptions": 1.5}})
    stats = {"receptions": 10}
    assert cfg.score(stats, "WR") == 10.0
    assert cfg.score(stats, "TE") == 15.0


def test_valuation_signed_vorp_and_degradation():
    lg = LeagueConfig(
        teams=2,
        roster={"QB": 1, "RB": 1, "FLEX": 1, "BENCH": 1},
        scoring=PRESETS["half_ppr"],
        flex_eligible=("RB", "WR"),
        rounds=4,
    )
    players = [
        ProjectedPlayer("q1", "QB One", "QB", 300),
        ProjectedPlayer("q2", "QB Two", "QB", 250),
        ProjectedPlayer("q3", "QB Three", "QB", 200),
        ProjectedPlayer("r1", "RB One", "RB", 220),
        ProjectedPlayer("r2", "RB Two", "RB", 180),
        ProjectedPlayer("r3", "RB Three", "RB", 150),
        ProjectedPlayer("w1", "WR One", "WR", 190),
        ProjectedPlayer("w2", "WR Two", "WR", 100),
    ]
    a = assign_starters(players, lg)
    # 2 QB start, replacement is q3; 2 RB start, flex takes w1 and r3, replacement RB is next unselected
    assert a.replacement_points["QB"] == 200
    assert [p.player_id for p in a.flex_starters] == ["w1", "r3"]
    assert value_over_replacement(players[0], a) == 100
    assert value_over_replacement(players[5], a) < 0 or players[5].player_id in {p.player_id for p in a.flex_starters}
    # thin pool: no TE anywhere -> degrade, don't raise
    lg2 = LeagueConfig(
        teams=2,
        roster={"TE": 1, "BENCH": 1},
        scoring=PRESETS["half_ppr"],
        flex_eligible=(),
        rounds=2,
    )
    a2 = assign_starters(players, lg2)
    assert a2.coverage_status.get("TE") == "pool_exhausted"


def test_windows():
    picks = (2, 19, 22, 39, 42)
    w = draft_window(30.5, picks)
    assert w.last_pick_before_adp == 22
    assert w.first_pick_after_adp == 39
    assert fall_severity(30.5, 39) == "meaningful_fall"
    assert recommendation_status(30.5, 22, 39) == "take_in_this_window"
    assert recommendation_status(18.0, 22, 39) == "fall_only"
    assert recommendation_status(60.0, 22, 39) == "normally_wait"
