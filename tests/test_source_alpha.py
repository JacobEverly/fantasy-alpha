"""Tests for evals/source_alpha.py — lexicon determinism, as-of windowing
(the EvidenceStore gate, poison-tested), and stratification math."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from evals.source_alpha import (
    ADP_GATE,
    adp_band,
    bootstrap_lift_ci,
    classify,
    fetch_season_claims,
    pool_cells,
    position_base_rates,
    season_cell,
    window_for,
)


# ---------------------------------------------------------------------------
# lexicon

CASES = [
    ("Ross saw first-team reps during the Chiefs' first full practice",
     {"CAMP_PROMOTION"}),
    ("He was promoted to the lead role in the backfield",
     {"CAMP_PROMOTION"}),
    ("Listed as starter on the unofficial depth chart", {"CAMP_PROMOTION"}),
    ("Coach expects an expanded role and more targets this year",
     {"ROLE_EXPANSION"}),
    ("Should see an uptick in snaps after the trade", {"ROLE_EXPANSION"}),
    ("Cleared for full participation and looked fully healthy",
     {"INJURY_RECOVERY_POSITIVE"}),
    # recovery framing suppresses the injury-noun concern path
    ("Fully recovered from ACL surgery and removed from the PUP list",
     {"INJURY_RECOVERY_POSITIVE"}),
    # "cleared waivers" is not an injury recovery
    ("He cleared waivers and joined the practice squad", set()),
    ("Limited in practice with a hamstring setback", {"INJURY_CONCERN"}),
    ("Questionable after tweaking his high-ankle sprain",
     {"INJURY_CONCERN"}),
    ("Turning heads at camp, a popular breakout candidate", {"HYPE_SOFT"}),
    ("A standout through two weeks who has impressed coaches",
     {"HYPE_SOFT"}),
    ("Demoted to the second team and facing competition for snaps",
     {"NEGATIVE"}),
    ("The team waived him on Tuesday", {"NEGATIVE"}),
    # multi-type: concern + promotion in one item
    ("The starting job is his, though he remains questionable with a "
     "hamstring injury", {"CAMP_PROMOTION", "INJURY_CONCERN"}),
    ("", set()),
    # preseason-logistics phrasing must NOT count as promotion
    ("Starters are likely to play in Thursday's preseason game", set()),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_classify_hand_built(text, expected):
    assert classify(text) == frozenset(expected)


def test_classify_deterministic_and_case_insensitive():
    text = "PROMOTED to the FIRST-TEAM offense; more targets expected"
    first = classify(text)
    assert all(classify(text) == first for _ in range(5))
    assert first == classify(text.lower()) == classify(text.upper())
    assert first == {"CAMP_PROMOTION", "ROLE_EXPANSION"}


# ---------------------------------------------------------------------------
# as-of windowing (reuses EvidenceStore's gate; poisoned stream)

def _ms(*args) -> int:
    return int(datetime(*args, tzinfo=timezone.utc).timestamp() * 1000)


def _doc(pid: str, ms: int, title: str, source: str = "rotowire") -> dict:
    return {
        "doc_id": f"{source}:{ms}", "player_id": pid, "player_name": "Test Guy",
        "position": "WR", "team": "AA", "published_ms": ms,
        "published_utc": datetime.fromtimestamp(
            ms / 1000, tz=timezone.utc).isoformat(), "source": source,
        "source_key": None, "title": title, "text": "",
        "url": None, "internal_research_only": True,
    }


@pytest.fixture
def tiny_index(tmp_path: Path) -> Path:
    """One player; docs before the window, at its inclusive start, inside it,
    and POISON at/after the Sept-1 gate (published-ascending, as built)."""
    (tmp_path / "players").mkdir()
    (tmp_path / "docs").mkdir()
    players = {"999": {"player_id": "999", "name": "Test Guy",
                       "norm_name": "testguy", "position": "WR", "team": "AA"}}
    (tmp_path / "players.json").write_text(json.dumps(
        {"players": players, "by_norm_name": {"testguy": ["999"]}}))
    start, gate = window_for(2023)
    assert (start, gate) == (date(2023, 6, 3), date(2023, 9, 1))
    docs = [
        _doc("999", _ms(2023, 6, 2, 23, 59, 59),
             "waived by the team"),                       # pre-window: NEGATIVE
        _doc("999", _ms(2023, 6, 3, 0, 0, 0),
             "turning heads at camp"),                    # window start: HYPE
        _doc("999", _ms(2023, 8, 15, 12, 0, 0),
             "promoted to first-team reps"),              # in window: PROMOTION
        _doc("999", _ms(2023, 9, 1, 0, 0, 0),
             "expanded role and more targets"),           # AT gate: poison
        _doc("999", _ms(2023, 9, 20, 0, 0, 0),
             "limited with a hamstring setback"),         # after gate: poison
    ]
    with open(tmp_path / "players" / "999.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return tmp_path


def test_windowing_and_gate_poison(tiny_index):
    cand = {"player_id": "00-TEST", "player": "Test Guy", "position": "WR"}
    claims, stats = fetch_season_claims(2023, [cand], index_dir=tiny_index)
    got = claims["00-TEST"]["rotowire"]
    # in-window claims present (inclusive window start)
    assert got == {"CAMP_PROMOTION", "HYPE_SOFT"}
    assert claims["00-TEST"]["pooled"] == got
    # pre-window NEGATIVE excluded; at/after-gate poison never classified
    assert "NEGATIVE" not in got
    assert "ROLE_EXPANSION" not in got and "INJURY_CONCERN" not in got
    assert stats == {"season": 2023, "n_candidates": 1, "n_resolved": 1,
                     "n_items": 2, "n_unresolved": 0}


def test_unresolved_player_counts_as_unclaimed(tiny_index):
    cand = {"player_id": "00-NOPE", "player": "Nobody Home", "position": "RB"}
    claims, stats = fetch_season_claims(2023, [cand], index_dir=tiny_index)
    assert claims == {}
    assert stats["n_unresolved"] == 1 and stats["n_resolved"] == 0


# ---------------------------------------------------------------------------
# stratification / cell math

def _row(pid, pos, rank, breakout, alpha=None):
    return {"player_id": pid, "position": pos, "adp_pos_rank": rank,
            "breakout": breakout, "bust": False, "alpha": alpha,
            "player": pid, "season": 2020}


def test_adp_band_boundaries():
    assert adp_band("RB", ADP_GATE["RB"] + 1) == "near_gate"
    assert adp_band("RB", ADP_GATE["RB"] + 12) == "near_gate"
    assert adp_band("RB", ADP_GATE["RB"] + 13) == "mid"
    assert adp_band("QB", ADP_GATE["QB"] + 30) == "mid"
    assert adp_band("WR", ADP_GATE["WR"] + 31) == "deep"


def test_season_cell_and_pooling_math():
    # 6 players: 3 claimed (2 breakouts), 3 unclaimed (1 breakout)
    rows = [
        _row("a", "WR", 45, True, alpha=10.0),
        _row("b", "WR", 50, True, alpha=8.0),
        _row("c", "RB", 44, False, alpha=-4.0),
        _row("d", "WR", 60, True, alpha=6.0),
        _row("e", "RB", 46, False, alpha=-2.0),
        _row("f", "RB", 48, False, alpha=None),  # alpha missing, still counted
    ]
    claims = {p: {"rotowire": {"CAMP_PROMOTION"}} for p in ("a", "b", "c")}
    base = position_base_rates(rows, "breakout")
    assert base == {"WR": 3 / 3, "RB": 0.0}
    cell = season_cell(rows, claims, "rotowire", "CAMP_PROMOTION",
                       "breakout", base)
    assert (cell["k_c"], cell["n_c"], cell["k_u"], cell["n_u"]) == (2, 3, 1, 3)
    # expected hits for claimed group from position mix: WR + WR + RB = 2.0
    assert cell["exp_c"] == pytest.approx(2.0)
    assert cell["alpha_c_sum"] == pytest.approx(14.0) and cell["alpha_c_n"] == 3
    assert cell["alpha_u_sum"] == pytest.approx(4.0) and cell["alpha_u_n"] == 2

    pooled = pool_cells({2020: cell, 2021: cell})
    assert pooled["n_claimed"] == 6 and pooled["n_unclaimed"] == 6
    assert pooled["claimed_rate"] == pytest.approx(4 / 6)
    assert pooled["unclaimed_rate"] == pytest.approx(2 / 6)
    assert pooled["lift"] == pytest.approx(2.0)
    assert pooled["lift_vs_pos_base"] == pytest.approx(4 / 4.0)
    assert pooled["alpha_claimed_mean"] == pytest.approx(28 / 6)
    assert pooled["alpha_unclaimed_mean"] == pytest.approx(8 / 4)


def test_wrong_source_or_type_is_unclaimed():
    rows = [_row("a", "WR", 45, True), _row("b", "WR", 50, False)]
    claims = {"a": {"rotoballer": {"CAMP_PROMOTION"},
                    "rotowire": {"HYPE_SOFT"}}}
    base = position_base_rates(rows, "breakout")
    cell = season_cell(rows, claims, "rotowire", "CAMP_PROMOTION",
                       "breakout", base)
    assert cell["n_c"] == 0 and cell["n_u"] == 2


def test_bootstrap_ci_deterministic_and_brackets_constant_lift():
    cell = {"k_c": 4, "n_c": 10, "k_u": 4, "n_u": 20, "exp_c": 2.0,
            "alpha_c_sum": 0, "alpha_c_n": 0, "alpha_u_sum": 0, "alpha_u_n": 0}
    per_season = {s: dict(cell) for s in range(2016, 2025)}
    ci1 = bootstrap_lift_ci(per_season, reps=500, seed=7)
    ci2 = bootstrap_lift_ci(per_season, reps=500, seed=7)
    assert ci1 == ci2
    # identical seasons -> every resample gives exactly lift 2.0
    assert ci1 == (pytest.approx(2.0), pytest.approx(2.0))
    # degenerate: no unclaimed hits anywhere -> no defined draws
    empty = {2016: {**cell, "k_u": 0}}
    assert bootstrap_lift_ci(empty, reps=200, seed=1) == (None, None)
