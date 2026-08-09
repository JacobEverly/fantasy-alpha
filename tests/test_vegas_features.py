"""Unit tests for evals/vegas_features.py (synthetic data, no real files)."""
import math

import pytest

from evals import vegas_features as vf


def _game(season, week, home, away, spread, total, game_type="REG"):
    return {
        "game_id": f"{season}_{week:02d}_{away}_{home}",
        "season": season, "week": week, "gameday": f"{season}-09-10",
        "home_team": home, "away_team": away,
        "spread_line": spread, "total_line": total,
    }


# ------------------------------------------------------- implied totals math

def test_implied_totals_home_away_split():
    # home favored by 3, total 44 -> home 23.5, away 20.5
    rows = vf.implied_team_totals([_game(2020, 1, "KC", "HOU", 3.0, 44.0)])
    by_team = {r["team"]: r for r in rows}
    assert by_team["KC"]["implied_total"] == pytest.approx(23.5)
    assert by_team["HOU"]["implied_total"] == pytest.approx(20.5)
    assert by_team["KC"]["is_home"] == 1 and by_team["HOU"]["is_home"] == 0
    assert by_team["KC"]["opponent"] == "HOU"
    # implied totals always sum back to the game total
    assert by_team["KC"]["implied_total"] + by_team["HOU"]["implied_total"] \
        == pytest.approx(44.0)


def test_implied_totals_road_favorite_and_spread_sign():
    # spread_line = -7 means home is a 7-point underdog
    rows = vf.implied_team_totals([_game(2021, 5, "NYJ", "NE", -7.0, 40.0)])
    by_team = {r["team"]: r for r in rows}
    assert by_team["NE"]["implied_total"] == pytest.approx(23.5)
    assert by_team["NYJ"]["implied_total"] == pytest.approx(16.5)
    # team_spread: negative = favored (sportsbook display convention)
    assert by_team["NE"]["team_spread"] == pytest.approx(-7.0)
    assert by_team["NYJ"]["team_spread"] == pytest.approx(7.0)


# ------------------------------------------------------------- aggregations

def _weekly_fixture():
    games = [
        _game(2019, 1, "KC", "HOU", 3.0, 44.0),   # KC 23.5, HOU 20.5
        _game(2019, 2, "KC", "DEN", 7.0, 50.0),   # KC 28.5, DEN 21.5
        _game(2020, 1, "KC", "DEN", 9.0, 47.0),   # KC 28.0, DEN 19.0
    ]
    return vf.implied_team_totals(games)


def test_season_mean_is_retrospective_average():
    means = vf.season_mean_implied(_weekly_fixture())
    assert means[(2019, "KC")]["mean_implied_total"] == pytest.approx(26.0)
    assert means[(2019, "KC")]["games_with_lines"] == 2
    assert means[(2019, "HOU")]["mean_implied_total"] == pytest.approx(20.5)
    assert means[(2020, "KC")]["games_with_lines"] == 1


def test_draft_day_features_use_only_legal_inputs():
    weekly = _weekly_fixture()
    feats = vf.draft_day_features(weekly, vf.season_mean_implied(weekly))
    kc_2020 = feats[(2020, "KC")]
    # week-1 line of season 2020 itself: legal (posted pre-draft)
    assert kc_2020["week1_implied_total"] == pytest.approx(28.0)
    # prior season (2019) mean: legal
    assert kc_2020["prior_season_mean_implied_total"] == pytest.approx(26.0)
    # 2019 has no 2018 data -> prior is None, week1 still present
    kc_2019 = feats[(2019, "KC")]
    assert kc_2019["prior_season_mean_implied_total"] is None
    assert kc_2019["week1_implied_total"] == pytest.approx(23.5)
    # HOU played week 1 only in 2019; no 2020 rows -> no (2020, HOU) week1
    assert feats.get((2020, "HOU"), {}).get("week1_implied_total") is None


def test_team_codes_normalized_to_current_franchise(tmp_path):
    csv_text = (
        "game_id,season,game_type,week,gameday,home_team,away_team,"
        "home_score,away_score,spread_line,total_line\n"
        "2010_01_SD_OAK,2010,REG,1,2010-09-12,SD,OAK,24,20,7,44\n"
        "2010_02_STL_JAC,2010,REG,2,2010-09-19,STL,JAC,10,20,-3,42\n"
        "2010_19_SD_NYJ,2010,WC,19,2011-01-08,SD,NYJ,24,20,7,44\n"   # playoffs: dropped
        "2010_03_KC_CHI,2010,REG,3,2010-09-26,KC,CHI,20,17,,\n")     # no line: dropped
    p = tmp_path / "games_2026-08-08.csv"
    p.write_text(csv_text)
    games = vf.load_games(p)
    assert len(games) == 2  # REG with lines only
    teams = {g["home_team"] for g in games} | {g["away_team"] for g in games}
    assert teams == {"LAC", "LV", "LA", "JAX"}


# ------------------------------------------------------------- study wiring

def test_build_study_rows_walk_forward_no_leakage():
    weekly = _weekly_fixture()
    fantasy = {(2019, "KC"): 1400.0, (2020, "KC"): 1500.0,
               (2019, "DEN"): 1000.0, (2020, "DEN"): 990.0}
    tds = {(2020, "KC"): 55, (2020, "DEN"): 30}
    rows = vf.build_study_rows(weekly=weekly, fantasy=fantasy, tds=tds,
                               last_season=2024)
    # only (2020, KC) has all of: week1(2020), prior mean(2019), naive(2019),
    # outcomes(2020). DEN lacks 2020 week-1... actually DEN played week 1 2020.
    by_key = {(r["season"], r["team"]): r for r in rows}
    kc = by_key[(2020, "KC")]
    assert kc["week1_implied"] == pytest.approx(28.0)        # season-2020 week 1
    assert kc["prior_mean_implied"] == pytest.approx(26.0)   # season-2019 mean
    assert kc["naive_prior_points"] == pytest.approx(1400.0)  # 2019 realized
    assert kc["out_points"] == pytest.approx(1500.0)          # 2020 realized
    den = by_key[(2020, "DEN")]
    assert den["prior_mean_implied"] == pytest.approx(21.5)
    # no row may pair a season-S feature with a season-S-1 outcome
    assert (2019, "KC") not in by_key  # no 2018 features exist


def test_build_study_rows_excludes_holdout_2025():
    games = [_game(2024, 1, "KC", "DEN", 7.0, 50.0),
             _game(2025, 1, "KC", "DEN", 6.0, 49.0)]
    weekly = vf.implied_team_totals(games)
    fantasy = {(2024, "KC"): 1400.0, (2025, "KC"): 1500.0,
               (2024, "DEN"): 900.0, (2025, "DEN"): 950.0}
    tds = {(2025, "KC"): 50, (2025, "DEN"): 30}
    rows = vf.build_study_rows(weekly=weekly, fantasy=fantasy, tds=tds,
                               last_season=2030)
    assert all(r["season"] != vf.HOLDOUT_SEASON for r in rows)
    assert rows == []  # the only candidate season was the holdout


# --------------------------------------------------------------- statistics

def test_pearson_and_residualize():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert vf._pearson(xs, [2.0, 4.0, 6.0, 8.0]) == pytest.approx(1.0)
    assert vf._pearson(xs, [8.0, 6.0, 4.0, 2.0]) == pytest.approx(-1.0)
    assert math.isnan(vf._pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    resid = vf._residualize([2.0, 4.0, 6.0, 8.0], xs)
    assert all(abs(r) < 1e-12 for r in resid)  # perfect fit -> zero residuals


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    pairs = [(float(i), float(i) + ((-1) ** i) * 2.0) for i in range(30)]
    a = vf._bootstrap_ci(pairs, n_boot=200)
    b = vf._bootstrap_ci(pairs, n_boot=200)
    assert a == b  # seeded
    point, lo, hi = a
    assert lo <= point <= hi


def test_run_study_shape_on_synthetic_rows():
    rng_rows = []
    for season in range(2000, 2024):
        for i, team in enumerate(["KC", "DEN", "LAC", "LV"]):
            base = 20.0 + i
            rng_rows.append({
                "season": season, "team": team,
                "week1_implied": base + 0.1 * season % 3,
                "prior_mean_implied": base,
                "naive_prior_points": 50.0 * base,
                "out_points": 51.0 * base + (season % 5),
                "out_tds": base,
            })
    res = vf.run_study(rows=rng_rows, n_boot=50)
    assert res["n_rows"] == len(rng_rows)
    assert set(res["eras"]) == {"2000-2007", "2008-2015", "2016-2024", "2000-2024"}
    block = res["eras"]["2000-2024"]["out_points"]
    for key in ("prior_mean_implied", "week1_implied", "naive_prior_points",
                "prior_mean_implied_incremental", "week1_implied_incremental"):
        point, lo, hi = block[key]
        assert lo <= hi
