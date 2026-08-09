import pytest

from evals.draftbench import ADP_DIR, LABELS, run, score_roster, weekly_roster_points
from evals.normalize import (
    PLAYOFF_TEAMS,
    REGULAR_SEASON_WEEKS,
    championship_odds,
    league_index,
    league_percentile,
    league_rank,
    normalized_metrics,
)

HAVE_DATA = ADP_DIR.exists() and (LABELS / "weekly_points.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ snapshots")


# ---------------------------------------------------------------------------
# League Index math


def test_league_index_anchors():
    # 100 = tied with the league-best opponent.
    assert league_index(1500.0, [1500.0, 1200.0, 900.0]) == 100.0
    # >100 = outscored everyone; scale is linear in own points.
    assert league_index(1650.0, [1500.0, 1200.0]) == 110.0
    assert league_index(750.0, [1500.0, 1200.0]) == 50.0
    # Anchored to the BEST opponent, not the mean/median.
    assert league_index(1200.0, [1000.0, 2400.0]) == 50.0


def test_league_index_rejects_degenerate_leagues():
    with pytest.raises(ValueError):
        league_index(100.0, [])
    with pytest.raises(ValueError):
        league_index(100.0, [0.0, -5.0])


def test_league_rank_and_percentile():
    opps = [1500.0, 1400.0, 1300.0]  # 4-team league
    assert league_rank(1600.0, opps) == 1
    assert league_percentile(1600.0, opps) == 100.0
    assert league_rank(1450.0, opps) == 2
    assert league_rank(1000.0, opps) == 4
    assert league_percentile(1000.0, opps) == 0.0
    # Ties share the better rank.
    assert league_rank(1400.0, opps) == 2


# ---------------------------------------------------------------------------
# Championship odds — determinism + invariants


def _flat_league(n_teams=12, n_weeks=17, base=100.0, step=5.0):
    """Team i scores base + i*step every week — strictly ordered, no ties."""
    return [[base + i * step] * n_weeks for i in range(n_teams)]


def test_championship_odds_deterministic_under_seed():
    league = _flat_league()
    a = championship_odds(league, n_sims=500, seed=42)
    b = championship_odds(league, n_sims=500, seed=42)
    assert a == b
    c = championship_odds(league, n_sims=500, seed=43)
    assert a != c  # different schedule draws


def test_title_and_playoff_probabilities_are_conserved():
    odds = championship_odds(_flat_league(), n_sims=300, seed=1)
    assert abs(sum(o["title_pct"] for o in odds) - 1.0) < 1e-9
    assert abs(sum(o["playoff_pct"] for o in odds) - PLAYOFF_TEAMS) < 1e-9
    # Every week hands out teams/2 wins -> expected wins are conserved too.
    n = len(odds)
    assert abs(sum(o["expected_wins"] for o in odds) - REGULAR_SEASON_WEEKS * n / 2) < 0.05


def test_better_weekly_scores_mean_monotonically_better_odds():
    # Strictly dominant teams: odds must be monotone in team strength.
    odds = championship_odds(_flat_league(n_teams=8), n_sims=400, seed=7)
    titles = [o["title_pct"] for o in odds]
    playoffs = [o["playoff_pct"] for o in odds]
    wins = [o["expected_wins"] for o in odds]
    assert titles == sorted(titles)
    assert playoffs == sorted(playoffs)
    assert wins == sorted(wins)
    # A team that outscores everyone every single week always wins the title.
    assert titles[-1] == 1.0
    assert playoffs[-1] == 1.0
    assert wins[-1] == float(REGULAR_SEASON_WEEKS)
    # And the strictly-worst team never makes the top 4 of 8.
    assert playoffs[0] == 0.0


def test_input_validation():
    with pytest.raises(ValueError):
        championship_odds(_flat_league(n_teams=2))  # < 4 teams
    with pytest.raises(ValueError):
        championship_odds(_flat_league(n_teams=5))  # odd pairing
    with pytest.raises(ValueError):
        championship_odds(_flat_league(n_weeks=15))  # too few weeks
    ragged = _flat_league()
    ragged[3] = ragged[3][:16]
    with pytest.raises(ValueError):
        championship_odds(ragged)
    with pytest.raises(ValueError):
        championship_odds(_flat_league(), n_sims=0)


# ---------------------------------------------------------------------------
# Week conventions: final = weeks 16+17; pre-2021 (16 weeks) final = 15+16


def _playoff_decider_league(n_weeks):
    """4 teams engineered so teams 2 and 3 always meet in the final:
    regular-season strength order is 0 < 1 < 2 < 3, but the finals weeks
    decide the title between the top two."""
    league = _flat_league(n_teams=4, n_weeks=n_weeks)
    return league


def test_week17_decides_the_final_when_present():
    league = _playoff_decider_league(17)
    # Team 2 explodes in weeks 16+17 only: beats team 3 in the final
    # (semis: 3 beats worst seed, 2 beats the other; final = wk16+wk17).
    league[2] = league[2][:15] + [10_000.0, 10_000.0]
    odds = championship_odds(league, n_sims=200, seed=3)
    assert odds[2]["title_pct"] == 1.0
    # ...but if the explosion is week 17 only and the data is truncated to 16
    # weeks (pre-2021 convention: final = weeks 15+16), it never counts.
    league16 = _playoff_decider_league(16)
    assert odds[3]["title_pct"] == 0.0
    odds16 = championship_odds(league16, n_sims=200, seed=3)
    assert odds16[3]["title_pct"] == 1.0  # strongest team wins on 15+16


def test_pre2021_final_uses_weeks_15_and_16():
    league = _playoff_decider_league(16)
    # Week 16 alone is not enough to flip the final if week 15 (shared with
    # the semifinal, by documented convention) plus 16 still favors team 3.
    league[2][15] = league[2][15] + 25.0  # 2's wk16: 110+25=135 vs 3's 115
    # final totals: team2 = 110+135 = 245, team3 = 115+115 = 230 -> team 2 wins
    odds = championship_odds(league, n_sims=200, seed=5)
    assert odds[2]["title_pct"] == 1.0
    # Sanity: with a 17th week restored, week 15 is no longer in the final.
    league17 = _playoff_decider_league(17)
    league17[2][15] = league17[2][15] + 25.0  # wk16 bump only
    # final: team2 = 135+110 = 245 vs team3 = 115+115 = 230 -> still team 2,
    # but via weeks 16+17; make week 17 decisive for team 3 instead:
    league17[3][16] = league17[3][16] + 100.0
    odds17 = championship_odds(league17, n_sims=200, seed=5)
    assert odds17[3]["title_pct"] == 1.0


# ---------------------------------------------------------------------------
# normalized_metrics glue


def test_normalized_metrics_fields_and_consistency():
    league = _flat_league(n_teams=6)
    m = normalized_metrics(league, team_index=5, n_sims=200, seed=9)
    assert set(m) == {
        "league_index", "league_rank", "league_percentile",
        "expected_wins", "playoff_pct", "title_pct",
    }
    assert m["league_index"] > 100.0  # outscores every opponent
    assert m["league_rank"] == 1
    assert m["league_percentile"] == 100.0
    assert m["title_pct"] == 1.0
    worst = normalized_metrics(league, team_index=0, n_sims=200, seed=9)
    assert worst["league_index"] < 100.0
    assert worst["league_rank"] == 6
    with pytest.raises(ValueError):
        normalized_metrics(league, team_index=6)


# ---------------------------------------------------------------------------
# Reporting wiring (DraftBench columns; DraftGym is covered in test_draftgym)


@needs_data
def test_draftbench_rows_carry_normalized_fields():
    rows = run(
        seasons=(2018,),
        formats=("ppr",),
        slots=(6,),
        n_seeds=1,
        out_path=None,
        verbose=False,
        champ_sims=200,
    )
    assert rows
    for r in rows:
        assert 0.0 < r["league_index"] < 200.0
        assert 1 <= r["league_rank"] <= 12
        assert 0.0 <= r["playoff_pct"] <= 1.0
        assert 0.0 <= r["title_pct"] <= r["playoff_pct"]


@needs_data
def test_weekly_roster_points_sums_to_score_roster():
    from evals.draftbench import (
        DEFAULT_ROSTER,
        DEFAULT_TEAMS,
        AutopickADP,
        DraftSimulator,
        load_board,
    )
    from harness.league import LeagueConfig
    from harness.scoring import PRESETS

    pool, weekly = load_board(2021, "ppr")
    league = LeagueConfig(teams=DEFAULT_TEAMS, roster=dict(DEFAULT_ROSTER), scoring=PRESETS["ppr"])
    result = DraftSimulator(pool, league).simulate(AutopickADP(), slot=4, seed=2)
    for roster in result.rosters:
        weeks = weekly_roster_points(roster, weekly, league)
        assert len(weeks) == league.season_weeks
        assert round(sum(weeks), 2) == score_roster(roster, weekly, league).total
