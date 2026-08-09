import pytest

from evals.agents_execution import (
    EXECUTION_AGENTS,
    SurvivalSequencer,
    SurvivalSequencerVORP,
    bootstrap_ci,
    survival_probability,
)
from evals.draftbench import (
    ADP_DIR,
    DEFAULT_ROSTER,
    LABELS,
    AutopickADP,
    DraftPlayer,
    DraftSimulator,
    load_board,
    position_counts,
    starter_deficit,
    startable_count,
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
    """ADP-interleaved pool: QBs and TEs hold top-of-board ADPs early."""
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
# survival math


def test_survival_probability_shape():
    # ADP far before my next pick -> almost surely gone; far after -> survives.
    assert survival_probability(mk_player(1, "RB", adp=2.0, stdev=1.0), pick=20) < 0.01
    assert survival_probability(mk_player(2, "RB", adp=40.0, stdev=1.0), pick=20) > 0.99
    # At exactly ADP the CDF is 0.5.
    assert survival_probability(mk_player(3, "RB", adp=20.0, stdev=5.0), pick=20) == pytest.approx(0.5)
    # stdev floored at 1 (0.1 would make survival ~0 here; floor keeps it sane).
    tight = survival_probability(mk_player(4, "RB", adp=19.0, stdev=0.1), pick=20)
    assert 0.1 < tight < 0.2  # z = (20-19)/1 -> 1 - Phi(1) ~ 0.159


def test_bootstrap_ci_signs():
    lo, hi = bootstrap_ci([10.0, 12.0, 11.0, 9.0, 13.0], n_boot=2000, seed=1)
    assert lo > 0 and hi > lo
    lo, hi = bootstrap_ci([-10.0, -12.0, -11.0, -9.0, -13.0], n_boot=2000, seed=1)
    assert hi < 0


# ---------------------------------------------------------------------------
# guardrail invariants


def test_never_second_qb_or_te_before_core_starters():
    league = small_league()
    pool = synthetic_pool()
    for factory in (SurvivalSequencer, SurvivalSequencerVORP):
        for slot in (1, 2, 4):
            for seed in (0, 3, 7):
                result = DraftSimulator(pool, league).simulate(factory(), slot, seed)
                roster = result.agent_roster
                required = league.required_positions()
                for k in range(1, len(roster) + 1):
                    counts = position_counts(roster[:k])
                    for pos in ("QB", "TE"):
                        if counts.get(pos, 0) >= 2:
                            # at the moment the 2nd was taken, RB/WR required and
                            # FLEX must have been coverable by the prior picks
                            prior = position_counts(roster[: k - 1]) if roster[k - 1].position == pos else counts
                            core_missing = sum(
                                max(0, required.get(q, 0) - prior.get(q, 0))
                                for q in ("RB", "WR")
                            )
                            flex_extra = sum(
                                max(0, prior.get(q, 0) - required.get(q, 0))
                                for q in league.flex_eligible
                            )
                            assert core_missing == 0
                            assert flex_extra >= league.flex_per_team()


def test_no_qb_te_before_round_5_without_scarcity():
    # Plentiful QB/TE supply -> scarcity never triggers early -> rounds 1-4
    # must be RB/WR even though QBs/TEs hold the very best ADPs.
    league = small_league()
    pool = synthetic_pool(n_qb=10, n_rb=16, n_wr=16, n_te=10)
    for slot in (1, 3):
        result = DraftSimulator(pool, league).simulate(SurvivalSequencer(), slot, seed=1)
        assert all(p.position not in {"QB", "TE"} for p in result.agent_roster[:4])
        counts = position_counts(result.agent_roster)
        assert counts.get("QB", 0) >= 1 and counts.get("TE", 0) >= 1  # still legal


def test_scarcity_overrides_early_te_ban():
    # Exactly teams TEs on the whole board, all inside the depth cutoff:
    # TE demand (4 required) == quality supply (4) -> scarce from pick one,
    # so the round-5 ban is waived and the agent may take a TE early.
    league = small_league()
    pool = (
        [mk_player(f"te{i}", "TE", adp=5.0 + i) for i in range(4)]
        + [mk_player(f"rb{i}", "RB", adp=10.0 + i) for i in range(14)]
        + [mk_player(f"wr{i}", "WR", adp=30.0 + i) for i in range(14)]
        + [mk_player(f"qb{i}", "QB", adp=50.0 + i) for i in range(8)]
    )
    agent = SurvivalSequencer()
    result = DraftSimulator(pool, league).simulate(agent, slot=1, seed=0)
    te_rounds = [i + 1 for i, p in enumerate(result.agent_roster) if p.position == "TE"]
    assert te_rounds and te_rounds[0] < 5


# ---------------------------------------------------------------------------
# survival-band behavior on a hand-built board


def test_takes_the_urgent_player_over_a_safer_better_priced_one():
    """teams=4, slot 1: picks 1 and 8. Candidate A (adp 6, stdev 50) is the
    top of the market but a coin flip to survive to pick 8 (band 1);
    candidate B (adp 7, stdev 1) is almost surely gone (band 0). The
    sequencer must take B now, then still get A on the way back."""
    league = small_league()
    a = mk_player("safe", "RB", adp=6.0, stdev=50.0)
    b = mk_player("gone", "RB", adp=7.0, stdev=1.0)
    filler = (
        [mk_player(f"wr{i}", "WR", adp=20.0 + i) for i in range(12)]
        + [mk_player(f"rb{i}", "RB", adp=40.0 + i) for i in range(10)]
        + [mk_player(f"qb{i}", "QB", adp=60.0 + i) for i in range(6)]
        + [mk_player(f"te{i}", "TE", adp=70.0 + i) for i in range(6)]
    )
    pool = [a, b] + filler
    assert survival_probability(b, 8) < 0.40  # band 0
    assert 0.40 <= survival_probability(a, 8) <= 0.70  # band 1

    result = DraftSimulator(pool, league).simulate(SurvivalSequencer(), slot=1, seed=0)
    assert result.agent_roster[0].player_id == "gone"
    # The market baseline takes the lower ADP first — behavior differs.
    control = DraftSimulator(pool, league).simulate(AutopickADP(), slot=1, seed=0)
    assert control.agent_roster[0].player_id == "safe"


def test_passes_on_players_who_safely_survive():
    """At pick 1 (next pick 8) every RB with adp >= 12/stdev 1 survives with
    p > 0.99 (band 2), while a scarce-position WR tier is evaporating: the
    sequencer spends the pick on the scarce WR, not the better-priced RB."""
    league = small_league()
    # 4 quality WRs for 8 league-wide WR required slots -> WR scarce.
    wrs = [mk_player(f"wr{i}", "WR", adp=12.0 + i, stdev=1.0) for i in range(4)]
    rbs = [mk_player(f"rb{i}", "RB", adp=16.0 + i, stdev=1.0) for i in range(14)]
    late_wr = [mk_player(f"lwr{i}", "WR", adp=200.0 + i, stdev=1.0) for i in range(6)]
    qb_te = [mk_player(f"qb{i}", "QB", adp=60.0 + i) for i in range(6)] + [
        mk_player(f"te{i}", "TE", adp=70.0 + i) for i in range(6)
    ]
    pool = wrs + rbs + late_wr + qb_te
    agent = SurvivalSequencer()
    result = DraftSimulator(pool, league).simulate(agent, slot=1, seed=0)
    # rb0 has the best ADP among non-WR but safely survives; WR tier is scarce.
    assert result.agent_roster[0].position == "WR"


def test_vorp_variant_orders_by_projection_not_adp():
    league = small_league()
    pool = synthetic_pool()
    # Late-ADP RB with a monster previous season, and urgent (gone by next pick).
    stud = mk_player("stud", "RB", adp=7.5, stdev=1.0, prev_points=999.0, name="Stud")
    pool = pool + [stud]
    result = DraftSimulator(pool, league).simulate(SurvivalSequencerVORP(), slot=1, seed=0)
    assert result.agent_roster[0].player_id == "stud"


# ---------------------------------------------------------------------------
# determinism + config plumbing


def test_determinism_same_seed_same_draft():
    league = small_league()
    pool = synthetic_pool()
    for key, factory in EXECUTION_AGENTS.items():
        a = DraftSimulator(pool, league).simulate(factory(), slot=2, seed=5)
        b = DraftSimulator(pool, league).simulate(factory(), slot=2, seed=5)
        assert a.picks == b.picks, key
        assert a.rosters == b.rosters, key


def test_config_registry_names_match():
    for key, factory in EXECUTION_AGENTS.items():
        assert factory().name == key


def test_bad_bands_rejected():
    with pytest.raises(ValueError):
        SurvivalSequencer(take_band=0.8, pass_band=0.4)


# ---------------------------------------------------------------------------
# real-data integration: legal rosters on the nastiest board on disk


@needs_data
def test_legal_rosters_on_nastiest_real_boards():
    # 2022 half-ppr: 117 players for 180 picks, exactly 12 TEs for 12 slots.
    for season, preset in ((2022, "half_ppr"), (2016, "standard")):
        pool, _ = load_board(season, preset)
        league = LeagueConfig(teams=12, roster=dict(DEFAULT_ROSTER), scoring=PRESETS[preset])
        sim = DraftSimulator(pool, league)
        for key, factory in EXECUTION_AGENTS.items():
            for seed in range(3):
                result = sim.simulate(factory(), slot=5, seed=seed)
                for roster in result.rosters:
                    counts = position_counts(roster)
                    assert starter_deficit(counts, league) == 0, (season, preset, key)
                    assert len(roster) <= league.rounds
                    assert len(roster) - startable_count(counts, league) <= _bench_capacity(league)
