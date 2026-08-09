"""As-of gate poisoning, vacated-opportunity math, and news-window boundary
guards for the packet-v2 feature extractor (evals/packet_features.py)."""
import pytest

from evals.packet_features import (
    as_of_date,
    as_of_epoch_ms,
    asof_snapshot_depth,
    coaching_change,
    news_counts,
    pedigree,
    play_caller_for,
    s1_end_depth,
    season_usage,
    vacated_opportunity,
)

EVAL = 2023  # gate = 2023-09-01
GATE_MS = as_of_epoch_ms(EVAL)
DAY_MS = 86_400_000


def wk(season, week=1, targets=5, carries=2, team="LA"):
    return {"season": season, "week": week, "targets": targets,
            "carries": carries, "receiving_yards": 40, "target_share": "0.2",
            "air_yards_share": "0.3", "team": team}


# ---------------------------------------------------------------------------
# 1. as-of gate poisoning, channel by channel


def test_usage_rejects_eval_season_rows():
    with pytest.raises(AssertionError, match="leakage"):
        season_usage([wk(2022), wk(EVAL)], 2022, EVAL)
    with pytest.raises(AssertionError, match="leakage"):
        season_usage([wk(2024)], 2022, EVAL)


def test_usage_clean_rows_pass():
    u = season_usage([wk(2022), wk(2022, week=2)], 2022, EVAL)
    assert u["games"] == 2 and u["targets"] == 10


def dc_weekly(season, week=17, gsis="00-1", team="LA", rank=1):
    return {"season": season, "week": week, "game_type": "REG",
            "formation": "Offense", "gsis_id": gsis, "club_code": team,
            "depth_team": rank, "depth_position": "RB"}


def test_depth_s1_end_rejects_future_seasons():
    with pytest.raises(AssertionError, match="leakage"):
        s1_end_depth([dc_weekly(2022), dc_weekly(EVAL)], 2022, EVAL)


def test_depth_s1_end_uses_final_reg_week():
    rows = [dc_weekly(2022, week=5, rank=3), dc_weekly(2022, week=18, rank=1)]
    out = s1_end_depth(rows, 2022, EVAL)
    assert out["00-1"]["pos_rank"] == 1 and out["00-1"]["team"] == "LA"


def dc_snap(dt, gsis="00-1", team="LA", rank=2):
    return {"dt": dt, "gsis_id": gsis, "team": team, "pos_rank": rank,
            "pos_grp": "Offense"}


def test_depth_asof_ignores_snapshots_at_or_after_gate():
    clean = [dc_snap("2023-08-20T12:00:00Z", rank=2)]
    poisoned = clean + [dc_snap("2023-09-01T00:00:00Z", rank=1),
                        dc_snap("2023-11-05T12:00:00Z", rank=1)]
    assert asof_snapshot_depth(clean, EVAL) == asof_snapshot_depth(poisoned, EVAL)
    assert asof_snapshot_depth(poisoned, EVAL)["00-1"]["pos_rank"] == 2


def test_depth_asof_empty_when_only_future_snapshots():
    assert asof_snapshot_depth([dc_snap("2023-09-02T00:00:00Z")], EVAL) == {}


def test_coaching_rejects_future_season_query():
    rows = [{"coach": "A", "season": EVAL, "team": "MIA",
             "play_calling_weight": "1.0"}]
    with pytest.raises(AssertionError, match="leakage"):
        play_caller_for(rows, "MIA", EVAL + 1, EVAL)


def test_coaching_ignores_future_file_rows():
    base = [{"coach": "A", "season": 2022, "team": "MIA", "play_calling_weight": "1.0"},
            {"coach": "A", "season": 2023, "team": "MIA", "play_calling_weight": "1.0"}]
    poisoned = base + [{"coach": "Z", "season": 2024, "team": "MIA",
                        "play_calling_weight": "1.0"},
                       {"coach": "Z", "season": 2023, "team": "MIA",
                        "play_calling_weight": "0.0"}]
    assert coaching_change(base, "MIA", EVAL) == coaching_change(poisoned, "MIA", EVAL)
    assert coaching_change(poisoned, "MIA", EVAL)["team_changed_play_caller"] == 0


def test_coaching_change_derivable_and_not():
    rows = [{"coach": "A", "season": 2022, "team": "MIA", "play_calling_weight": "1.0"},
            {"coach": "B", "season": 2023, "team": "MIA", "play_calling_weight": "1.0"}]
    out = coaching_change(rows, "MIA", EVAL)
    assert out["team_changed_play_caller"] == 1 and out["coaching_status"] == "ok"
    out = coaching_change(rows, "SF", EVAL)
    assert out["team_changed_play_caller"] is None
    assert out["coaching_status"] == "not_derivable"


def test_pedigree_rejects_future_draft_class():
    with pytest.raises(AssertionError, match="leakage"):
        pedigree({"season": EVAL + 1, "round": "1", "pick": "1", "age": "21"},
                 None, EVAL)


def test_pedigree_age_and_draft_capital():
    out = pedigree({"season": 2023, "round": "5", "pick": "177", "age": "22"},
                   "2001-05-29", EVAL)  # Puka's birth date
    assert out["age_at_sept1"] == 22 and out["age_source"] == "sleeper_birth_date"
    assert out["draft_round"] == 5 and out["undrafted"] == 0
    assert out["years_since_draft"] == 0
    out = pedigree(None, None, EVAL)
    assert out["undrafted"] == 1 and out["age_at_sept1"] is None


def item(ts_ms, text="camp note"):
    return {"published": ts_ms, "text": text}


def test_news_poisoned_future_items_cannot_move_counts():
    clean = [item(GATE_MS - 10 * DAY_MS, "named the starter")]
    poisoned = clean + [item(GATE_MS, "breakout confirmed"),
                        item(GATE_MS + 40 * DAY_MS, "breakout star")]
    assert news_counts(clean, EVAL) == news_counts(poisoned, EVAL)
    assert news_counts(poisoned, EVAL)["n_news_90d"] == 1


# ---------------------------------------------------------------------------
# 2. vacated-opportunity math on a synthetic roster


def test_vacated_opportunity_synthetic_roster():
    # 2022 team LA: back A (kept), back B (left), WR C (left), WR D (kept),
    # plus a player on another team who must not count.
    s1_usage = [
        {"player_id": "A", "team": "LA", "targets": 30, "carries": 200},
        {"player_id": "B", "team": "LA", "targets": 20, "carries": 100},
        {"player_id": "C", "team": "LA", "targets": 120, "carries": 5},
        {"player_id": "D", "team": "LA", "targets": 80, "carries": 0},
        {"player_id": "E", "team": "SEA", "targets": 90, "carries": 50},
    ]
    roster_asof = {"A", "D", "NEWGUY"}
    out = vacated_opportunity("LA", roster_asof, s1_usage)
    assert out["vacated_targets"] == 140.0   # B 20 + C 120
    assert out["vacated_carries"] == 105.0   # B 100 + C 5
    assert out["vacated_opps"] == 245.0


def test_vacated_opportunity_full_retention_is_zero():
    s1_usage = [{"player_id": "A", "team": "LA", "targets": 10, "carries": 10}]
    assert vacated_opportunity("LA", {"A"}, s1_usage)["vacated_opps"] == 0.0


# ---------------------------------------------------------------------------
# 3. news-window boundary


def test_news_window_boundaries():
    window_start = GATE_MS - 90 * DAY_MS
    items = [
        item(window_start - 1, "too old"),          # out (older than 90d)
        item(window_start, "starting job"),          # in (inclusive start)
        item(GATE_MS - 1, "breakout impress"),       # in (last ms before gate)
        item(GATE_MS, "at the gate"),                # out (strictly before)
    ]
    out = news_counts(items, EVAL)
    assert out["n_news_90d"] == 2
    assert out["camp_promotion_flags"] == 1   # "starting job"
    assert out["hype_flags"] == 1             # "breakout impress"
    assert out["injury_mention_flags"] == 0


def test_news_keyword_flags():
    ts = GATE_MS - DAY_MS
    out = news_counts([
        item(ts, "Working with the first-team offense"),
        item(ts, "Suffers hamstring injury in camp"),
        item(ts, "Poised for a breakout season"),
        item(ts, "Traded to Denver"),
    ], EVAL)
    assert out == {"n_news_90d": 4, "camp_promotion_flags": 1,
                   "injury_mention_flags": 1, "hype_flags": 1}


def test_as_of_helpers():
    assert as_of_date(2023).isoformat() == "2023-09-01"
    # 2023-09-01T00:00:00Z
    assert as_of_epoch_ms(2023) == 1693526400000
