"""T1 SFT datagen — anonymized-track leakage guards + season-split compliance.

Grounds: docs/training-risk-register.md risks 3-4 (T1 generation must be
ANONYMIZED-track; the named pilot only catches numeric memory leaks) and the
GAMEPLAN season-split amendment (train = 2008-2022 excluding 2013 and 2018;
2023+ never in SFT data; 2025 sealed).
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import pytest

from training import sft_datagen as dg

ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = ROOT / "data" / "processed" / "sft"
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


# ---------------------------------------------------------------------------
# season-split compliance (GAMEPLAN amendment 2026-08-09)


def test_t1_season_constants():
    seasons = set(dg.T1_SEASON_PRESET)
    assert seasons == set(range(2008, 2023)) - {2013, 2018}
    assert not seasons & {2013, 2018}, "mid-era diagnostics are sealed"
    assert not seasons & {2023, 2024, 2025}, "2023+ never in SFT data"
    assert dg.T1_SEASON_PRESET[2008] == "standard"
    assert dg.T1_SEASON_PRESET[2009] == "standard"
    assert all(dg.T1_SEASON_PRESET[s] == "ppr" for s in seasons if s >= 2010)


def test_t1_sample_deterministic_and_split_compliant():
    s1, s2 = dg.build_t1_sample(), dg.build_t1_sample()
    assert s1 == s2, "T1 sample must be deterministic"
    assert {it["season"] for it in s1} <= set(dg.T1_SEASON_PRESET)
    assert all(it["outcome"] in (0, 1) for it in s1)
    fams = {it["family"] for it in s1}
    assert fams == {"full_slate", "bust", "season_threshold", "weekly_h2h"}
    # 2008 has no strictly-prior labeled season: weekly_h2h only
    assert {it["family"] for it in s1 if it["season"] == 2008} == {"weekly_h2h"}
    assert 1200 <= len(s1) <= 1600


def test_h2h_grounding_uses_strictly_prior_seasons_only():
    packet = {"position": "WR", "week": 8,
              "a": {"games": 6, "total": 90.0, "ppg": 15.0},
              "b": {"games": 7, "total": 98.0, "ppg": 14.0}}
    g = dg.h2h_grounding(packet, "ppr", 2016)
    assert g["base_n"] == len(dg._h2h_history("ppr", 2016))
    # history for season s must be a strict prefix of history for s+1
    assert len(dg._h2h_history("ppr", 2010)) < len(dg._h2h_history("ppr", 2016))
    assert g["cohort_rate"] is not None and 0.0 <= g["cohort_rate"] <= 1.0


# ---------------------------------------------------------------------------
# anonymization: masked evidence surface (draftgym masked-mode pattern)


def test_masked_evidence_relativizes_dates_and_strips_identity():
    row = {"published_utc": "2016-08-14T12:00:00+00:00", "player_name": "Real Name",
           "team": "KC", "gsis_id": "00-001", "report_status": "Questionable",
           "practice_status": None, "primary_injury": "Hamstring", "week": "1",
           "season": 2016}
    out = dg._mask_injury(row, "B0042", "2016-09-01")
    assert out["player"] == "B0042"
    assert out["published"] == "18d_before_as_of"
    for leak in ("Real Name", "KC", "00-001", "2016"):
        assert leak not in json.dumps(out)

    depth = [{"published_utc": "2016-08-30T00:00:00+00:00", "player_name": "Real Name",
              "team": "KC", "gsis_id": "x", "depth_slot": "WR", "depth_rank": "1",
              "unit": "Offense", "snapshot": "2016-08-30"}]
    out = dg._mask_depth(depth, "B0042", "2016-09-01")
    assert out[0]["snapshot"] == "2d_before_as_of"
    assert "Real Name" not in json.dumps(out) and "2016" not in json.dumps(out)


def test_anon_evidence_disables_free_text_tools():
    class FakeStore:
        def injury_status(self, player):
            return None

        def depth_chart(self, player):
            return []

    log = dg.anon_evidence_lookups(FakeStore(), ["Real Name"], ["B0001"],
                                   "2016-09-01")
    assert log[0]["tool"] == "player_news" and log[0]["ok"] is False
    assert "disabled in anonymized mode" in log[0]["error"]
    called = {e["tool"] for e in log}
    assert called == {"player_news", "injury_status", "depth_chart"}
    assert "Real Name" not in json.dumps(log)


def test_anon_evidence_errors_are_sanitized():
    class BrokenStore:
        def injury_status(self, player):
            raise KeyError(f"unknown player: {player!r}")

        def depth_chart(self, player):
            raise ValueError(f"ambiguous player {player!r}")

    log = dg.anon_evidence_lookups(BrokenStore(), ["Real Name"], ["B0001"],
                                   "2016-09-01")
    assert "Real Name" not in json.dumps(log), \
        "error messages must never de-anonymize"
    assert all(e["error"] == "lookup failed" for e in log[1:])


# ---------------------------------------------------------------------------
# anonymization: the assembled prompt


@lru_cache(maxsize=1)
def _sample_by_family():
    sample = dg.build_t1_sample()
    picks = {}
    for it in sample:
        picks.setdefault(it["family"], it)
    return picks


def test_anon_user_message_masks_names_seasons_dates():
    ctx = dg._t1_ctx()
    for family, it in _sample_by_family().items():
        g = dg.t1_grounding_for_item(it, ctx)
        msg = dg.build_anon_user_message(it, g, tool_log=[])
        assert not DATE_RE.search(msg), f"{family}: absolute date in prompt"
        for name in it["_identity"]["players"]:
            assert name and name.lower() not in msg.lower(), \
                f"{family}: player name leaked"
        assert "year masked" in msg


def test_anon_user_message_asserts_on_date_leak():
    it = dict(_sample_by_family()["bust"])
    with pytest.raises(AssertionError):
        dg.build_anon_user_message(
            it, {"cohort_rate": 0.2},
            tool_log=[{"tool": "injury_status", "ok": True,
                       "result": {"published": "2016-08-14"}}])


# ---------------------------------------------------------------------------
# generated artifacts (skipped until they exist)


@lru_cache(maxsize=1)
def _key_names_by_season() -> dict[tuple[int, str], set]:
    """Every real player name in the KEY files for the T1 season map."""
    out = {}
    for season, preset in dg.T1_SEASON_PRESET.items():
        names = set()
        for stem in (f"breakoutbench_{season}_{preset}_KEY.json",
                     f"calibbench_{season}_{preset}_KEY.json"):
            path = dg.QUESTIONS_DIR / stem
            if not path.exists():
                continue
            for q in json.loads(path.read_text())["questions"].values():
                for k in ("player", "player_a", "player_b"):
                    if q.get(k):
                        names.add(q[k])
        out[(season, preset)] = names
    return out


def _iter_t1_records():
    for name in ("raw_traces_t1.jsonl", "t1_corpus_v1.jsonl"):
        path = SFT_DIR / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            yield name, json.loads(line)


def test_t1_outputs_have_no_player_names_on_anon_track():
    checked = 0
    for fname, rec in _iter_t1_records():
        meta = rec.get("meta", rec)
        if meta.get("track") == "named_pilot":
            continue  # deliberately-named flagged subset
        season, preset = meta["season"], meta["format"]
        if "messages" in rec:
            text = " ".join(m["content"] for m in rec["messages"])
        else:
            text = (rec.get("user") or "") + " " + (rec.get("assistant") or "")
        low = text.lower()
        for name in _key_names_by_season()[(season, preset)]:
            assert name.lower() not in low, \
                f"{fname}: player name {name!r} leaked into anon trace " \
                f"{meta.get('trace_id')}"
        checked += 1
    if checked == 0:
        pytest.skip("no T1 outputs generated yet")


def test_t1_outputs_season_split_compliance():
    checked = 0
    for fname, rec in _iter_t1_records():
        meta = rec.get("meta", rec)
        season = meta["season"]
        assert season in dg.T1_SEASON_PRESET, \
            f"{fname}: season {season} violates the T1 train split"
        assert season not in (2013, 2018, 2023, 2024, 2025)
        assert meta.get("track") in ("anonymized", "named_pilot")
        checked += 1
    if checked == 0:
        pytest.skip("no T1 outputs generated yet")
