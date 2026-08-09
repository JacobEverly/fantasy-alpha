import json

import pytest

from evals.draftbench import (
    ADP_DIR,
    LABELS,
    AutopickADP,
    DraftSimulator,
    _bench_capacity,
    _board_key,
    position_counts,
    startable_count,
    starter_deficit,
)
from envs.draftgym import (
    HOLDOUT_SEASON,
    OBS_PLAYER_FIELDS,
    DraftGym,
    hindsight_roster_points,
    realistic_roster_points,
)
from test_draftbench import mk_player, small_league, synthetic_pool

HAVE_DATA = ADP_DIR.exists() and (LABELS / "weekly_points.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ snapshots")


def synthetic_gym(**overrides):
    """A DraftGym over the synthetic 4-team pool — no disk, no evidence."""
    league = small_league()
    pool = synthetic_pool()
    weekly = {
        p.player_id: {w: max(0.0, p.prev_points / 20 + (int(p.player_id) % 5) - 2) for w in range(1, 18)}
        for p in pool
    }
    kw = dict(
        season=2019,
        preset="ppr",
        league=league,
        agent_slot=2,
        seed=7,
        enable_evidence=False,
        pool=pool,
        weekly=weekly,
    )
    kw.update(overrides)
    return DraftGym(**kw)


def autopick_step(gym):
    """The exact AutopickADP choice on the gym's current offered board."""
    return {"pick": min(gym._agent_candidates(), key=_board_key).player_id}


def play_autopick(gym):
    obs = gym.reset()
    reward, done, info = 0.0, False, {}
    while not done:
        obs, reward, done, info = gym.step(autopick_step(gym))
    return obs, reward, info


# ---------------------------------------------------------------------------
# Determinism


def test_determinism_same_seed_same_episode():
    _, r1, i1 = play_autopick(synthetic_gym())
    _, r2, i2 = play_autopick(synthetic_gym())
    assert i1["picks"] == i2["picks"]
    assert r1 == r2
    _, _, i3 = play_autopick(synthetic_gym(seed=8))
    assert i1["picks"] != i3["picks"]


def test_reset_replays_identically():
    gym = synthetic_gym()
    _, r1, i1 = play_autopick(gym)
    obs = gym.reset()
    assert not obs["done"]
    _, r2, i2 = play_autopick(gym)  # includes internal reset state
    assert i1["picks"] == i2["picks"] and r1 == r2


# ---------------------------------------------------------------------------
# Reward semantics: autopick == control -> reward exactly 0


def test_autopick_policy_reward_is_zero_synthetic():
    _, reward, info = play_autopick(synthetic_gym())
    assert reward == 0.0
    assert info["agent_points_realistic"] == info["control_points_realistic"]


@needs_data
def test_autopick_policy_reward_is_zero_real_board():
    for season, slot, seed in ((2018, 1, 0), (2021, 7, 3)):
        gym = DraftGym(season=season, preset="ppr", agent_slot=slot, seed=seed, enable_evidence=False)
        _, reward, info = play_autopick(gym)
        assert reward == 0.0, f"{season}/{slot}/{seed}: {reward}"
        # And the gym's opponent stream matches DraftSimulator's exactly.
        control = DraftSimulator(gym.pool, gym.league).simulate(AutopickADP(), slot, seed)
        assert [p["player_id"] for p in info["picks"]] == [r.player_id for r in control.picks]


# ---------------------------------------------------------------------------
# Realistic manager policy vs hindsight (the risk-register #7 ordering)


def test_realistic_never_beats_hindsight_synthetic():
    gym = synthetic_gym()
    play_autopick(gym)
    roster = tuple(gym._rosters[gym.agent_slot - 1])
    real = realistic_roster_points(roster, gym.weekly, gym.league)
    hind = hindsight_roster_points(roster, gym.weekly, gym.league)
    assert real <= hind + 1e-9


@needs_data
def test_realistic_never_beats_hindsight_real_rosters():
    gym = DraftGym(season=2021, preset="ppr", agent_slot=4, seed=11, enable_evidence=False)
    play_autopick(gym)
    for roster in gym._rosters:  # all 12 rosters, not just the agent's
        real = realistic_roster_points(tuple(roster), gym.weekly, gym.league)
        hind = hindsight_roster_points(tuple(roster), gym.weekly, gym.league)
        assert real <= hind + 1e-9
        assert real > 0  # a drafted roster scores something under a real manager


def test_realistic_manager_cannot_see_the_future():
    """A one-week boom buried behind a steady starter is NOT harvested."""
    league = small_league(
        teams=2,
        roster={"RB": 1, "BENCH": 1},
        flex_eligible=("RB",),
    )
    steady = mk_player("steady", "RB", adp=1.0)
    boom = mk_player("boom", "RB", adp=50.0)
    roster = (steady, boom)
    weekly = {
        "steady": {w: 10.0 for w in range(1, 18)},
        "boom": {5: 60.0},  # single boom week, 0 ppg before it
    }
    real = realistic_roster_points(roster, weekly, league)
    hind = hindsight_roster_points(roster, weekly, league)
    # Manager: ADP lineup wk1-5 (misses the boom), chases it wk6 (scores 0),
    # benches it again wk7+ as its per-week average decays. 16 * 10 = 160.
    assert real == 16 * 10.0
    assert hind == 16 * 10.0 + 60.0  # hindsight starts the boom week
    assert real < hind


# ---------------------------------------------------------------------------
# Leakage guards


def test_observation_player_fields_whitelist():
    gym = synthetic_gym()
    obs = gym.reset()
    for view in obs["top_available"] + obs["my_roster"]:
        assert tuple(sorted(view)) == tuple(sorted(OBS_PLAYER_FIELDS))


@needs_data
def test_board_features_never_contain_eval_season_outcomes():
    """prev_season_points must be the PRIOR season's line; the episode
    season's realized totals must not appear anywhere in the observation."""
    gym = DraftGym(season=2021, preset="ppr", agent_slot=1, seed=0, enable_evidence=False)
    obs = gym.reset()
    rendered = obs.render_json()
    assert "nflverse_id" not in rendered
    assert "pts_ppr" not in rendered and "weekly" not in rendered
    # Spot check: a board player's realized 2021 total is not their obs value.
    for view in obs["top_available"]:
        p = gym._by_id[view["player_id"]]
        realized_2021 = round(sum(gym.weekly.get(p.player_id, {}).values()), 2)
        if realized_2021 > 0:
            assert view["prev_season_points"] != pytest.approx(realized_2021, abs=0.01) or (
                view["prev_season_points"] == 0.0
            )


def test_poisoned_future_doc_never_surfaces(tmp_path):
    """Evidence tools with a future-dated doc in the index: the time gate must
    keep it out of tool results (poison test, risk register #4/#7)."""
    idx = tmp_path / "evidence_index"
    (idx / "players").mkdir(parents=True)
    (idx / "docs").mkdir()
    pid = "9999"
    (idx / "players.json").write_text(json.dumps({
        "players": {pid: {"player_id": pid, "name": "Test Player", "position": "RB"}},
        "by_norm_name": {"testplayer": [pid]},  # evals.names.norm_name strips spaces
    }))
    past = {"player_id": pid, "player_name": "Test Player", "position": "RB",
            "published_ms": 1230000000000, "published_utc": "2008-12-23T00:00:00Z",
            "title": "camp note", "text": "SAFE_PAST_DOC camp battle", "source": "test"}
    future = {"player_id": pid, "player_name": "Test Player", "position": "RB",
              "published_ms": 1640000000000, "published_utc": "2021-12-20T00:00:00Z",
              "title": "season result", "text": "POISON_FUTURE_DOC wins 2021 fantasy title", "source": "test"}
    # ascending published_ms within files (store relies on it for early exit)
    lines = json.dumps(past) + "\n" + json.dumps(future) + "\n"
    (idx / "players" / f"{pid}.jsonl").write_text(lines)
    (idx / "docs" / "docs_2021.jsonl").write_text(lines)

    gym = synthetic_gym(season=2021, evidence_index_dir=idx, enable_evidence=True)
    gym.reset()
    for call in (
        {"name": "player_news", "arguments": {"player": "Test Player"}},
        {"name": "search", "arguments": {"query": "POISON_FUTURE_DOC"}},
        {"name": "search", "arguments": {"query": "fantasy title"}},
    ):
        obs, _, _, _ = gym.step({"tool": call})
        # check responses only — the agent's own query text echoes in "call"
        responses = json.dumps([t["response"] for t in obs["tool_results"]])
        assert "POISON_FUTURE_DOC" not in responses, call
        assert "2021-12-20" not in responses, call
    # the gate is time-based, not text-based: the past doc IS retrievable
    obs, _, _, _ = gym.step({"tool": {"name": "player_news", "arguments": {"player": "Test Player"}}})
    assert "SAFE_PAST_DOC" in obs.render_json()


def test_holdout_season_refused():
    with pytest.raises(ValueError, match="holdout"):
        DraftGym(season=HOLDOUT_SEASON, preset="ppr")


def test_adapter_split_discipline():
    from envs.verifiers_v1_adapter import TRAIN_SEASONS, VAL_SEASONS, episode_rows

    assert HOLDOUT_SEASON not in TRAIN_SEASONS + VAL_SEASONS
    assert not set(TRAIN_SEASONS) & set(VAL_SEASONS)
    train = {r["season"] for r in episode_rows(split="train", presets=("ppr",), slots=(1,), n_seeds=1)}
    assert train == set(TRAIN_SEASONS)
    with pytest.raises(ValueError):
        list(episode_rows(split="holdout"))


# ---------------------------------------------------------------------------
# Tool-cap enforcement


def test_tool_cap_enforced_and_counted():
    gym = synthetic_gym(evidence_index_dir="/nonexistent", enable_evidence=True)
    gym.reset()
    cap = gym.max_lookups_per_pick
    for i in range(cap + 2):
        obs, reward, done, info = gym.step({"tool": {"name": "search", "arguments": {"query": "x"}}})
        assert reward == 0.0 and not done
    assert info["total_lookups"] == cap  # over-cap calls never dispatch
    assert info["cap_hits"] == 2
    last = obs["tool_results"][-1]["response"]
    assert last["ok"] is False and "cap" in last["error"]
    # a pick resets the per-pick counter
    gym.step(autopick_step(gym))
    obs, *_ = gym.step({"tool": {"name": "search", "arguments": {"query": "y"}}})
    assert obs["lookups_used_this_pick"] == 1


def test_tool_results_cleared_after_pick():
    gym = synthetic_gym(enable_evidence=False)
    gym.reset()
    obs, *_ = gym.step({"tool": {"name": "search", "arguments": {"query": "x"}}})
    assert len(obs["tool_results"]) == 1
    obs, _, done, _ = gym.step(autopick_step(gym))
    if not done:
        assert obs["tool_results"] == []


# ---------------------------------------------------------------------------
# Legality


def test_illegal_actions_raise():
    gym = synthetic_gym()
    gym.reset()
    with pytest.raises(ValueError, match="illegal pick"):
        gym.step({"pick": "not-a-player"})
    with pytest.raises(ValueError, match="pick.*or.*tool|'pick'"):
        gym.step({"nonsense": 1})
    gym.step(autopick_step(gym))


def test_all_rosters_legal_after_episode():
    for seed in range(5):
        gym = synthetic_gym(seed=seed)
        obs = gym.reset()
        done = False
        while not done:
            # adversarial-ish agent: always takes the LAST feasible candidate
            pick = max(gym._agent_candidates(), key=_board_key).player_id
            obs, _, done, info = gym.step({"pick": pick})
        league = gym.league
        for roster in gym._rosters:
            counts = position_counts(roster)
            assert starter_deficit(counts, league) == 0
            assert len(roster) <= league.rounds
            assert len(roster) - startable_count(counts, league) <= _bench_capacity(league)


def test_step_after_done_raises():
    gym = synthetic_gym()
    play_autopick(gym)
    with pytest.raises(RuntimeError, match="done"):
        gym.step({"pick": "1"})


# ---------------------------------------------------------------------------
# Observation shape / rendering


def test_observation_renders_and_has_core_fields():
    gym = synthetic_gym()
    obs = gym.reset()
    blob = json.loads(obs.render_json())
    for key in (
        "season", "format", "league", "pick_number", "picks_until_next_turn",
        "my_roster", "top_available", "lookups_remaining", "tool_results",
        "feasible_positions", "evidence_as_of", "done",
    ):
        assert key in blob, key
    assert blob["league"]["teams"] == gym.league.teams
    assert blob["evidence_as_of"] == "2019-09-01"
    assert 1 <= len(blob["top_available"]) <= 25
    # rationale channel is recorded
    pick = autopick_step(gym)
    gym.step({**pick, "rationale": "value at the turn"})
    done = False
    while not done:
        _, _, done, info = gym.step(autopick_step(gym))
    assert "value at the turn" in info["rationales"].values()


# ---------------------------------------------------------------------------
# Anonymized (masked) mode — memorization counter, risk register #8


def test_masked_mode_determinism_and_reward_parity():
    """Masking changes only the observation surface: same seed, same picks,
    same reward as the named episode for an identity-blind policy."""
    _, r_named, i_named = play_autopick(synthetic_gym())
    gym = synthetic_gym(mask_names=True)
    obs = gym.reset()
    done = False
    while not done:
        # pick through the MASKED id of the autopick choice
        real = min(gym._agent_candidates(), key=_board_key).player_id
        obs, r_masked, done, i_masked = gym.step({"pick": gym._to_masked[real]})
    assert i_named["picks"] == i_masked["picks"]
    assert r_named == r_masked == 0.0


def test_masked_observation_hides_names():
    import re

    gym = synthetic_gym(mask_names=True)
    obs = gym.reset()
    for view in obs["top_available"] + obs["my_roster"]:
        assert re.fullmatch(r"B\d{3}", view["player_id"])
        assert view["name"] == view["player_id"]
        assert tuple(sorted(view)) == tuple(sorted(OBS_PLAYER_FIELDS))
    # ids are stable within the episode and per-board deterministic
    again = synthetic_gym(mask_names=True)
    again.reset()
    assert gym._to_masked == again._to_masked


@needs_data
def test_masked_real_board_leaks_no_names():
    gym = DraftGym(season=2021, preset="ppr", agent_slot=1, seed=0,
                   enable_evidence=False, mask_names=True)
    obs = gym.reset()
    rendered = obs.render_json()
    for p in gym.pool[:50]:  # the whole visible top of the board
        assert p.name not in rendered


def test_masked_mode_tool_surface(tmp_path):
    """player_news/search disabled; structured lookups re-key to board ids;
    errors never echo a real name."""
    idx = tmp_path / "evidence_index"
    (idx / "players").mkdir(parents=True)
    (idx / "docs").mkdir()
    (idx / "players.json").write_text(json.dumps({"players": {}, "by_norm_name": {}}))

    gym = synthetic_gym(mask_names=True, evidence_index_dir=idx, enable_evidence=True)
    gym.reset()
    obs, *_ = gym.step({"tool": {"name": "player_news", "arguments": {"player": "B001"}}})
    resp = obs["tool_results"][-1]["response"]
    assert resp["ok"] is False and "anonymized" in resp["error"]
    obs, *_ = gym.step({"tool": {"name": "search", "arguments": {"query": "camp"}}})
    assert obs["tool_results"][-1]["response"]["ok"] is False
    # structured lookup with a non-board key is rejected without a name echo
    obs, *_ = gym.step({"tool": {"name": "injury_status", "arguments": {"player": "Tom Brady"}}})
    resp = obs["tool_results"][-1]["response"]
    assert resp["ok"] is False and "Tom Brady" not in json.dumps(resp)
    # structured lookup with a board id: dispatches on the real name but the
    # envelope (here: empty index -> sanitized failure or masked rows) must
    # not contain the real synthetic name
    obs, *_ = gym.step({"tool": {"name": "injury_status", "arguments": {"player": "B001"}}})
    real_name = gym._by_id[gym._from_masked["B001"]].name
    assert real_name not in json.dumps(obs["tool_results"][-1])


def test_masked_mode_legal_rosters():
    gym = synthetic_gym(mask_names=True, seed=3)
    obs = gym.reset()
    done = False
    while not done:
        pick = obs["top_available"][0]["player_id"]  # masked id from the obs
        try:
            obs, _, done, _ = gym.step({"pick": pick})
        except ValueError:
            real = min(gym._agent_candidates(), key=_board_key).player_id
            obs, _, done, _ = gym.step({"pick": gym._to_masked[real]})
    for roster in gym._rosters:
        counts = position_counts(roster)
        assert starter_deficit(counts, gym.league) == 0


# ---------------------------------------------------------------------------
# play_llm parsing helpers (no network)


def test_parse_action_roundtrips():
    from envs.play_llm import parse_action

    assert parse_action('{"pick": "123"}') == {"pick": "123"}
    assert parse_action('noise {"pick": 42, "rationale": "why"} trailing') == {
        "pick": "42", "rationale": "why",
    }
    tool = parse_action('{"tool": {"name": "search", "arguments": {"query": "q"}}}')
    assert tool == {"tool": {"name": "search", "arguments": {"query": "q"}}}
    assert parse_action('<think>x</think>{"tool": {"name": "injury_status"}}')["tool"]["name"] == "injury_status"
    for bad in ("no json here", '{"neither": 1}', '{"tool": {"arguments": {}}}', "[1,2]"):
        with pytest.raises(ValueError):
            parse_action(bad)
