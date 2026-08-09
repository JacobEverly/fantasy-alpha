"""Tests for the BreakoutBench GBDT baseline (evals/gbdt_baseline.py).

Covers the three properties the baseline's credibility rests on:
1. no leakage — poisoning every row from the target season onward cannot
   change season-S predictions;
2. feature parity — the training/prediction features are byte-identical to
   the shipped anonymized slate packets;
3. determinism — same seed, same predictions.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.gbdt_baseline import (  # noqa: E402
    DEMO,
    build_candidates,
    load_data,
    score_season,
    train_predict,
    training_frame,
)

SEASON = 2015  # smallest training frame -> fastest fits


@pytest.fixture(scope="module")
def data():
    return load_data("ppr", "ppr")


def _poison_from(data, season: int):
    """Deep-copied Data with every row from `season` onward corrupted."""
    poisoned = copy.deepcopy(data)
    for pid, seasons in poisoned.stats.items():
        for y, row in seasons.items():
            if y >= season:
                for k in row:
                    if k.startswith(("total_", "ppg_", "pos_season_rank_")) or k == "games":
                        row[k] = "9999"
    for y, by_pid in poisoned.outcomes.items():
        if y >= season:
            for row in by_pid.values():
                row["breakout"] = "True" if row["breakout"] != "True" else "False"
    for y, players in poisoned.adp.items():
        if y > season:  # season-S ADP is legitimate pre-season input
            for p in players:
                p["adp"] = 1.0
                p["stdev"] = 0.0
    return poisoned


def test_no_leakage_poisoned_future_rows_do_not_change_predictions(data):
    clean = train_predict(data, SEASON, "ppr")
    dirty = train_predict(_poison_from(data, SEASON), SEASON, "ppr")
    assert [r["player_id"] for r in clean] == [r["player_id"] for r in dirty]
    assert [r["p_breakout"] for r in clean] == [r["p_breakout"] for r in dirty]


def test_training_frame_uses_only_past_seasons(data):
    x_before, _ = training_frame(data, SEASON)
    x_after, _ = training_frame(_poison_from(data, SEASON), SEASON)
    assert (x_before == x_after).all()


def test_feature_parity_with_shipped_slates(data):
    for season in (2015, 2020, 2024):
        slate = json.loads((DEMO / f"slate_{season}_ppr.json").read_text())
        rebuilt = build_candidates(data, season, "ppr")
        assert len(rebuilt) == slate["n_candidates"]
        shipped = sorted(
            json.dumps({k: v for k, v in c.items() if k != "anon_id"}, sort_keys=True)
            for c in slate["candidates"])
        ours = sorted(json.dumps(c["features"], sort_keys=True) for c in rebuilt)
        assert ours == shipped


def test_determinism_under_seed(data):
    a = train_predict(data, SEASON, "ppr", seed=123)
    b = train_predict(data, SEASON, "ppr", seed=123)
    assert [r["p_breakout"] for r in a] == [r["p_breakout"] for r in b]


def test_score_season_definitions(data):
    preds = train_predict(data, SEASON, "ppr")
    m = score_season(preds)
    assert m["n"] == len(preds)
    assert 0 <= m["hits"] <= 10
    assert m["expected_random"] == pytest.approx(m["base_rate"] * 10)
    assert 0.0 <= m["brier_full"] <= 1.0 and 0.0 <= m["brier_topk"] <= 1.0
