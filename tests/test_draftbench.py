import pytest

from evals.draftbench import (
    ADP_DIR,
    DEFAULT_ROSTER,
    LABELS,
    AutopickADP,
    BestBallADP,
    DraftAgent,
    DraftPlayer,
    DraftSimulator,
    GreedyVORP,
    adp_sigma,
    can_add,
    load_board,
    optimal_lineup_points,
    picks_left,
    position_counts,
    run,
    score_roster,
    starter_deficit,
    startable_count,
    team_on_the_clock,
    _bench_capacity,
)
from harness.league import LeagueConfig
from harness.scoring import PRESETS

HAVE_DATA = ADP_DIR.exists() and (LABELS / "weekly_points.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ snapshots")


def small_league(**overrides):
    cfg = dict(
        teams=4,
        roster={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BENCH": 2},  # 9 rounds
        scoring=PRESETS["ppr"],
    )
    cfg.update(overrides)
    return LeagueConfig(**cfg)


def mk_player(pid, position, adp, stdev=2.0, prev_points=0.0, name=None):
    return DraftPlayer(
        player_id=str(pid),
        name=name or f"P{pid}",
        position=position,
        team="",
        adp=float(adp),
        stdev=stdev,
        times_drafted=100,
        prev_points=prev_points,
        nflverse_id=None,
    )


def synthetic_pool(n_qb=8, n_rb=14, n_wr=14, n_te=6):
    """QB/TE supply is tight-ish relative to a 4-team league; ADP interleaved."""
    pool, adp = [], 1.0
    specs = [("QB", n_qb), ("RB", n_rb), ("WR", n_wr), ("TE", n_te)]
    remaining = {pos: n for pos, n in specs}
    pid = 0
    while any(remaining.values()):
        for pos, _ in specs:
            if remaining[pos]:
                remaining[pos] -= 1
                pid += 1
                pool.append(mk_player(pid, pos, adp, prev_points=400.0 - adp))
                adp += 1.0
    return pool


# ---------------------------------------------------------------------------
# Lineup setter


def test_optimal_lineup_hand_example():
    lg = LeagueConfig(
        teams=2,
        roster={"QB": 1, "RB": 1, "FLEX": 1, "BENCH": 1},
        scoring=PRESETS["ppr"],
        flex_eligible=("RB", "WR"),
        season_weeks=2,
        playoff_weeks=(2,),
    )
    roster = (
        mk_player("a", "QB", 1),
        mk_player("b", "RB", 2),
        mk_player("c", "WR", 3),
        mk_player("d", "RB", 4),
    )
    weekly = {
        "a": {1: 20.0, 2: -3.0},  # negative week still starts (only QB rostered)
        "b": {1: 5.0},  # missing week 2 -> scores 0
        "c": {1: 7.0, 2: 4.0},
        "d": {2: 9.0},
    }
    # wk1: QB 20 + RB best(5, 0) + FLEX best remaining(7, 0) = 32
    assert optimal_lineup_points([("QB", 20.0), ("RB", 5.0), ("WR", 7.0), ("RB", 0.0)], lg) == 32.0
    # wk2: QB -3 + RB 9 + FLEX best(0, 4) = 10
    assert optimal_lineup_points([("QB", -3.0), ("RB", 0.0), ("WR", 4.0), ("RB", 9.0)], lg) == 10.0
    score = score_roster(roster, weekly, lg)
    assert score.total == 42.0
    assert score.playoff == 10.0


def test_optimal_lineup_flex_spill_and_empty_slot():
    lg = LeagueConfig(
        teams=2,
        roster={"RB": 1, "TE": 1, "FLEX": 1, "BENCH": 1},
        scoring=PRESETS["ppr"],
        flex_eligible=("RB", "WR"),
    )
    # No TE rostered -> TE slot scores 0; second RB spills into FLEX over the WR.
    assert optimal_lineup_points([("RB", 10.0), ("RB", 8.0), ("WR", 6.0)], lg) == 18.0


# ---------------------------------------------------------------------------
# Draft mechanics


def test_snake_order():
    lg = small_league(teams=12, roster=dict(DEFAULT_ROSTER))
    assert [team_on_the_clock(lg, p) for p in (1, 12, 13, 24, 25)] == [1, 12, 12, 1, 1]


def test_picks_left_respects_board_exhaustion():
    lg = LeagueConfig(teams=2, roster={"QB": 1, "RB": 1, "BENCH": 1}, scoring=PRESETS["ppr"])
    # team 1 schedule with 3 rounds: picks 1, 4, 5
    assert picks_left(lg, 1, board_size=100) == 3
    assert picks_left(lg, 1, board_size=2) == 1  # picks 4 and 5 never happen
    assert picks_left(lg, 4, board_size=1) == 1


def test_can_add_forces_required_slots():
    lg = LeagueConfig(teams=2, roster={"QB": 1, "RB": 1, "BENCH": 1}, scoring=PRESETS["ppr"])
    counts: dict[str, int] = {}
    # 2 picks left, QB and RB both unfilled: any extra position is infeasible.
    assert can_add("QB", counts, lg, picks_left=2)
    assert can_add("RB", counts, lg, picks_left=2)
    assert not can_add("WR", counts, lg, picks_left=2)


def test_simulator_determinism():
    league = small_league()
    pool = synthetic_pool()
    a = DraftSimulator(pool, league).simulate(AutopickADP(), slot=2, seed=7)
    b = DraftSimulator(pool, league).simulate(AutopickADP(), slot=2, seed=7)
    assert a.picks == b.picks
    assert a.rosters == b.rosters
    c = DraftSimulator(pool, league).simulate(AutopickADP(), slot=2, seed=8)
    assert a.picks != c.picks


def test_opponents_always_produce_legal_rosters_synthetic():
    league = small_league()
    pool = synthetic_pool()  # 42 players > 36 picks, but QB/TE are scarce
    sim = DraftSimulator(pool, league)
    for seed in range(10):
        for slot in (1, 3):
            result = sim.simulate(AutopickADP(), slot, seed)
            for roster in result.rosters:
                counts = position_counts(roster)
                assert starter_deficit(counts, league) == 0
                assert len(roster) <= league.rounds
                assert len(roster) - startable_count(counts, league) <= _bench_capacity(league)


def test_illegal_agent_pick_raises():
    class Stubborn(DraftAgent):
        name = "stubborn"

        def pick(self, board, my_roster, league, pick_number):
            return "not-a-player"

    sim = DraftSimulator(synthetic_pool(), small_league())
    with pytest.raises(ValueError, match="illegal pick"):
        sim.simulate(Stubborn(), slot=1, seed=0)


def test_bestball_waits_on_qb_te():
    league = small_league()
    pool = synthetic_pool()  # QBs and TEs hold the very best ADPs
    result = DraftSimulator(pool, league).simulate(BestBallADP(), slot=1, seed=3)
    roster = result.agent_roster
    assert all(p.position not in {"QB", "TE"} for p in roster[:4])
    counts = position_counts(roster)
    assert counts.get("QB", 0) >= 1 and counts.get("TE", 0) >= 1  # still fills starters
    control = DraftSimulator(pool, league).simulate(AutopickADP(), slot=1, seed=3)
    assert control.agent_roster[0].position == "QB"  # market baseline takes ADP #1


def test_greedy_vorp_prefers_projected_value():
    league = small_league()
    pool = synthetic_pool()
    # Give a late-ADP RB a monster previous season: GreedyVORP should grab
    # him first while AutopickADP follows ADP.
    stud = mk_player("stud", "RB", adp=30.0, prev_points=999.0, name="Stud")
    pool = pool + [stud]
    greedy = DraftSimulator(pool, league).simulate(GreedyVORP(), slot=1, seed=0)
    assert greedy.agent_roster[0].player_id == "stud"
    control = DraftSimulator(pool, league).simulate(AutopickADP(), slot=1, seed=0)
    assert control.agent_roster[0].player_id != "stud"


def test_adp_sigma_fallback():
    assert adp_sigma(mk_player(1, "RB", 100.0, stdev=7.5)) == 7.5
    assert adp_sigma(mk_player(2, "RB", 100.0, stdev=None)) == 15.0
    assert adp_sigma(mk_player(3, "RB", 2.0, stdev=0.0)) == 1.0  # floor


# ---------------------------------------------------------------------------
# Real-data integration


@needs_data
def test_board_joins_to_realized_outcomes():
    loaded = load_board(2023, "ppr")
    assert loaded is not None
    pool, weekly = loaded
    assert len(pool) > 150
    matched = sum(1 for p in pool if p.nflverse_id)
    assert matched / len(pool) > 0.9
    jefferson = next(p for p in pool if p.name == "Justin Jefferson")
    assert jefferson.prev_points > 300  # monster 2022 PPR season
    best_realized = max(sum(w.values()) for w in weekly.values())
    assert best_realized > 250  # someone always has a big realized season


@needs_data
def test_opponents_always_produce_legal_rosters_real_boards():
    # 2022 half-ppr is the nastiest board on disk: 117 players for 180 picks
    # and exactly 12 TEs for 12 required TE slots.
    for season, preset in ((2022, "half_ppr"), (2016, "standard")):
        pool, _ = load_board(season, preset)
        league = LeagueConfig(teams=12, roster=dict(DEFAULT_ROSTER), scoring=PRESETS[preset])
        sim = DraftSimulator(pool, league)
        for factory in (AutopickADP, GreedyVORP):
            for seed in range(3):
                result = sim.simulate(factory(), slot=5, seed=seed)
                for roster in result.rosters:
                    counts = position_counts(roster)
                    assert starter_deficit(counts, league) == 0
                    assert len(roster) <= league.rounds


@needs_data
def test_run_small_grid_and_control_semantics():
    rows = run(
        seasons=(2018, 2022),
        formats=("ppr",),
        slots=(2, 9),
        n_seeds=2,
        agents={"autopick_adp": AutopickADP, "greedy_vorp": GreedyVORP},
        out_path=None,
        verbose=False,
    )
    autopick = [r for r in rows if r["agent"] == "autopick_adp"]
    greedy = [r for r in rows if r["agent"] == "greedy_vorp"]
    assert len(autopick) == len(greedy) == 8
    # The control IS AutopickADP in the same seat/seed: its delta is 0 exactly.
    assert all(r["points_above_control"] == 0.0 for r in autopick)
    assert all(r["control_points"] == a["points"] for r, a in zip(greedy, autopick))
    # Realized totals should look like fantasy seasons, not garbage.
    assert all(800 < r["points"] < 3000 for r in rows)

    # Recorded honest result (2026-08-08): GreedyVORP does NOT beat the
    # market. Last-season-points is a bad projection (ages, injuries, role
    # changes, rookies pinned at 0), so ADP autopick wins: -141.5 +/- 271.2
    # points over the full 2015-2024 default grid (n=405), and mean -337.1 on
    # this small deterministic grid. Pinned as a direction regression, not as
    # the desired outcome — beating the market is the actual product bar.
    import statistics

    assert statistics.mean(r["points_above_control"] for r in greedy) < 0
