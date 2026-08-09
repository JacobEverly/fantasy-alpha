"""Tests for evals/value_attribution.py — the value ledger.

Covers: per-pick capture math on hand-built cases, ledger aggregation,
autopick-relative deltas, injury flagging, determinism, and the DraftGym
telemetry surface (reward math untouched)."""

import pytest

from evals.draftbench import ADP_DIR, LABELS, AutopickADP, DraftSimulator
from evals.value_attribution import (
    aggregate_ledgers,
    build_ledger,
    collect_grid_ledgers,
    ledger_from_result,
    market_delta,
    realized_season_vorp,
    value_ledger_rows,
    _percentile,
)
from harness.league import LeagueConfig
from harness.scoring import PRESETS
from test_draftbench import mk_player, small_league, synthetic_pool

HAVE_DATA = (
    ADP_DIR.exists()
    and (LABELS / "season_points.csv").exists()
    and (LABELS / "slot_values.csv").exists()
)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ snapshots")


# ---------------------------------------------------------------------------
# Capture math (hand-built, no disk)


def _hand_ledger():
    """3 picks, hand-checkable curve/vorp: capture = vorp - cost."""
    picks = [
        (25, mk_player("a", "RB", adp=30.0)),   # round 3 in a 12-team league
        (1, mk_player("b", "WR", adp=1.0)),     # round 1
        (170, mk_player("c", "TE", adp=180.0)),  # round 15
    ]
    vorp = {"a": 110.0, "b": 90.0, "c": 62.0}
    cost = {25: 55.0, 1: 120.0, 170: 0.0}
    return build_ledger(
        picks,
        teams=12,
        realized_vorp=lambda p: vorp[p.player_id],
        slot_cost=cost.__getitem__,
        injury_flag=lambda p: p.player_id == "b",
    )


def test_capture_math_hand_case():
    rows = _hand_ledger()
    assert [r["pick_number"] for r in rows] == [1, 25, 170]  # sorted by pick
    by_id = {r["player_id"]: r for r in rows}
    assert by_id["a"]["capture"] == pytest.approx(110.0 - 55.0)
    assert by_id["b"]["capture"] == pytest.approx(90.0 - 120.0)  # negative: overpaid
    assert by_id["c"]["capture"] == pytest.approx(62.0)  # free slot -> pure capture
    assert by_id["a"]["round"] == 3
    assert by_id["b"]["round"] == 1
    assert by_id["c"]["round"] == 15
    assert all(r["basis"] == "hindsight_season_total" for r in rows)


def test_injury_flagging_in_rows():
    rows = {r["player_id"]: r for r in _hand_ledger()}
    assert rows["b"]["season_ending_acute"] is True
    assert rows["a"]["season_ending_acute"] is False
    # no injury_fn -> all False
    plain = build_ledger(
        [(1, mk_player("x", "RB", adp=1.0))],
        teams=12,
        realized_vorp=lambda p: 10.0,
        slot_cost=lambda n: 5.0,
    )
    assert plain[0]["season_ending_acute"] is False


def test_percentile_helper():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(vals, 0.5) == pytest.approx(30.0)
    assert _percentile(vals, 0.10) == pytest.approx(14.0)  # inclusive interpolation
    assert _percentile(vals, 0.90) == pytest.approx(46.0)
    assert _percentile([7.0], 0.9) == 7.0
    with pytest.raises(ValueError):
        _percentile([], 0.5)


# ---------------------------------------------------------------------------
# Aggregation


def _row(capture, rnd=1, pos="RB", flagged=False):
    return {
        "capture": capture, "round": rnd, "position": pos,
        "season_ending_acute": flagged,
    }


def test_aggregate_tier_rates_and_consistency():
    # draft 1: two +50s (one also +100); draft 2: none; draft 3: one +25 only
    drafts = [
        [_row(120.0), _row(55.0), _row(-30.0)],
        [_row(10.0), _row(-80.0)],
        [_row(30.0), _row(0.0)],
    ]
    agg = aggregate_ledgers(drafts)
    assert agg["n_drafts"] == 3 and agg["n_picks"] == 7
    assert agg["tier_rates_per_draft"][25] == pytest.approx(1.0, abs=1e-3)  # 120, 55, 30
    assert agg["tier_rates_per_draft"][50] == pytest.approx(2 / 3, abs=1e-3)
    assert agg["tier_rates_per_draft"][100] == pytest.approx(1 / 3, abs=1e-3)
    assert agg["consistency"]["ge1_plus50"] == pytest.approx(1 / 3, abs=1e-3)
    assert agg["consistency"]["ge2_plus50"] == pytest.approx(1 / 3, abs=1e-3)
    assert agg["capture_median"] == pytest.approx(10.0)


def test_aggregate_buckets_positions_and_injury_caveat():
    drafts = [[
        _row(60.0, rnd=2, pos="RB", flagged=True),   # +50 on flagged player
        _row(70.0, rnd=8, pos="WR"),
        _row(-55.0, rnd=1, pos="RB", flagged=True),  # -50 loss, injury luck
        _row(-60.0, rnd=5, pos="TE"),
    ]]
    agg = aggregate_ledgers(drafts)
    assert agg["by_round_bucket"]["1-3"]["plus50"] == 1
    assert agg["by_round_bucket"]["7-9"]["plus50"] == 1
    assert agg["by_round_bucket"]["4-6"]["plus50"] == 0
    assert agg["by_position"]["RB"]["n_picks"] == 2
    assert agg["by_position"]["WR"]["plus50"] == 1
    c = agg["injury_caveat"]
    assert c["plus50_on_season_ending_acute"] == 1 and c["plus50_total"] == 2
    assert c["minus50_on_season_ending_acute"] == 1 and c["minus50_total"] == 2


def test_aggregate_rejects_empty():
    with pytest.raises(ValueError):
        aggregate_ledgers([])


# ---------------------------------------------------------------------------
# Autopick-relative deltas


def test_market_delta_hand_case():
    agent = [[_row(60.0), _row(55.0)], [_row(70.0), _row(80.0), _row(90.0)]]
    control = [[_row(60.0)], [_row(70.0)]]  # 1 each
    d = market_delta(agent, control)
    assert d["mean_delta"] == pytest.approx((1 + 2) / 2)
    lo, hi = d["ci95"]
    assert lo <= d["mean_delta"] <= hi
    assert d["verdict"] == "more" and d["n"] == 2 and d["tier"] == 50


def test_market_delta_zero_and_determinism():
    drafts = [[_row(60.0)], [_row(10.0)], [_row(120.0)]]
    d = market_delta(drafts, drafts)  # policy IS the market -> delta 0
    assert d["mean_delta"] == 0.0 and d["ci95"] == (0.0, 0.0)
    a = [[_row(60.0)], [_row(60.0)], [_row(-5.0)]]
    c = [[_row(-5.0)], [_row(60.0)], [_row(60.0)]]
    assert market_delta(a, c) == market_delta(a, c)  # seed-deterministic CI


def test_market_delta_requires_pairing():
    with pytest.raises(ValueError):
        market_delta([[_row(60.0)]], [])


# ---------------------------------------------------------------------------
# Disk-backed machinery (real boards + labels)


@needs_data
def test_realized_vorp_matches_slot_values_machinery():
    """Realized VORP = season_points total - the canonical-league replacement
    level slot_values.replacement_levels computes on the same rows."""
    import csv as _csv

    from evals.draftbench import DraftPlayer, _season_rows
    from evals.slot_values import replacement_levels

    season, preset = 2019, "ppr"
    rows = _season_rows()[season]
    repl = replacement_levels(rows, preset, qb_slots=1)
    row = max(rows, key=lambda r: float(r["total_ppr"]) if r["position"] == "RB" else -1)
    player = DraftPlayer(
        player_id="x", name=row["player"], position="RB", team="", adp=1.0,
        stdev=1.0, times_drafted=1, prev_points=0.0, nflverse_id=row["player_id"],
    )
    got = realized_season_vorp(player, season, preset)
    assert got == pytest.approx(float(row["total_ppr"]) - repl["RB"], abs=0.02)
    # unmatched board player: 0 points -> full negative replacement cost
    ghost = DraftPlayer(
        player_id="g", name="Ghost", position="RB", team="", adp=200.0,
        stdev=1.0, times_drafted=1, prev_points=0.0, nflverse_id=None,
    )
    assert realized_season_vorp(ghost, season, preset) == pytest.approx(-repl["RB"], abs=0.02)


@needs_data
def test_ledger_determinism_on_real_cell():
    from evals.draftbench import load_board

    pool, _weekly = load_board(2019, "ppr")
    league = LeagueConfig(
        teams=12,
        roster={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2, "BENCH": 6},
        scoring=PRESETS["ppr"],
    )
    by_id = {p.player_id: p for p in pool}
    sim = DraftSimulator(pool, league)

    def one():
        result = sim.simulate(AutopickADP(), 6, 3)
        return result, ledger_from_result(result, 2019, "ppr", league, by_id)

    (result, a), (_, b) = one(), one()
    assert a == b
    # one row per agent pick (thin QB/RB/WR/TE boards can end drafts early)
    assert len(a) == len(result.agent_roster) >= 13
    assert [r["pick_number"] for r in a] == sorted(r["pick_number"] for r in a)
    # capture identity holds on every row
    for r in a:
        assert r["capture"] == pytest.approx(r["realized_vorp"] - r["slot_cost"], abs=0.02)


@needs_data
def test_draftbench_on_result_hook_reporting_only():
    from evals.draftbench import BASELINE_AGENTS, run

    calls = []
    baseline = run(
        seasons=[2019], formats=["ppr"], slots=[1], n_seeds=1,
        agents={"autopick_adp": BASELINE_AGENTS["autopick_adp"]},
        out_path=None, verbose=False, champ_sims=1,
    )
    rows = run(
        seasons=[2019], formats=["ppr"], slots=[1], n_seeds=1,
        agents={"autopick_adp": BASELINE_AGENTS["autopick_adp"]},
        out_path=None, verbose=False, champ_sims=1,
        on_result=lambda *args: calls.append(args),
    )
    assert len(calls) == 1
    agent_name, season, preset, slot, seed, result, control, by_id, league = calls[0]
    assert (agent_name, season, preset, slot, seed) == ("autopick_adp", 2019, "ppr", 1, 0)
    assert result is control  # autopick IS the control
    ledger = ledger_from_result(result, season, preset, league, by_id)
    assert len(ledger) == len(result.agent_roster) >= 13
    # hook never changes the benchmark rows
    assert rows == baseline


@needs_data
def test_grid_collector_pairs_agent_and_control():
    ledgers = collect_grid_ledgers(n_seeds=1, verbose=False)
    assert "autopick_adp" in ledgers and "survival_seq" in ledgers
    ap = ledgers["autopick_adp"]
    assert ap["drafts"] == ap["controls"]  # market delta 0 by construction
    d = market_delta(ap["drafts"], ap["controls"])
    assert d["mean_delta"] == 0.0
    ss = ledgers["survival_seq"]
    assert len(ss["drafts"]) == len(ss["controls"]) == len(ss["cells"])
    assert ss["cells"] == ap["cells"]  # same grid cells, same order


# ---------------------------------------------------------------------------
# DraftGym telemetry (reward math untouched)


@needs_data
def test_draftgym_value_ledger_telemetry():
    from envs.draftgym import DraftGym
    from evals.draftbench import _board_key

    gym = DraftGym(season=2019, preset="ppr", agent_slot=4, seed=2, enable_evidence=False)
    obs = gym.reset()
    done, reward, info = False, 0.0, {}
    while not done:
        pick = min(gym._agent_candidates(), key=_board_key).player_id
        obs, reward, done, info = gym.step({"pick": pick})

    ledger = info["value_ledger"]
    agent_picks = [p for p in info["picks"] if p["team"] == 4]
    assert isinstance(ledger, list) and len(ledger) == len(agent_picks) >= 13
    assert [r["pick_number"] for r in ledger] == [p["pick_number"] for p in agent_picks]
    for r in ledger:
        assert set(r) >= {
            "pick_number", "round", "player_id", "name", "position", "adp",
            "realized_vorp", "slot_cost", "capture", "season_ending_acute", "basis",
        }
        assert r["capture"] == pytest.approx(r["realized_vorp"] - r["slot_cost"], abs=0.02)
    # reward math untouched: autopick play still rewards 0 by construction,
    # and the reward is exactly the realistic-points difference
    assert reward == pytest.approx(0.0, abs=0.01)
    assert info["reward"] == pytest.approx(
        info["agent_points_realistic"] - info["control_points_realistic"], abs=0.01
    )
    # telemetry matches the standalone builder for the same picks
    rebuilt = value_ledger_rows(
        [(p["pick_number"], gym._by_id[p["player_id"]]) for p in agent_picks],
        2019, "ppr", teams=gym.league.teams,
    )
    assert rebuilt == ledger


def test_draftgym_value_ledger_synthetic_pool_never_breaks_reward():
    """Synthetic pools (no nflverse ids) must not crash the terminal step:
    the ledger is telemetry — reward stays intact regardless."""
    from envs.draftgym import DraftGym
    from evals.draftbench import _board_key

    league = small_league()
    pool = synthetic_pool()
    weekly = {
        p.player_id: {w: max(0.0, p.prev_points / 20) for w in range(1, 18)}
        for p in pool
    }
    gym = DraftGym(
        season=2019, preset="ppr", league=league, agent_slot=2, seed=7,
        enable_evidence=False, pool=pool, weekly=weekly,
    )
    gym.reset()
    done, reward, info = False, 0.0, {}
    while not done:
        pick = min(gym._agent_candidates(), key=_board_key).player_id
        _, reward, done, info = gym.step({"pick": pick})
    assert reward == pytest.approx(0.0, abs=0.01)  # autopick control, unchanged
    ledger = info["value_ledger"]
    if HAVE_DATA:
        assert isinstance(ledger, list) and len(ledger) == len(info["agent_roster"])
        # no realized data for synthetic ids -> vorp is minus replacement
        assert all(r["realized_vorp"] <= 0.0 for r in ledger)
    else:
        assert ledger is None
