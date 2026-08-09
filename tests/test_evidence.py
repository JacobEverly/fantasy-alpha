"""Tests for harness/evidence.py — the time-gated evidence store.

The core contract under test: EVERY query method hard-filters
``published < as_of`` inside the store. We poison a synthetic corpus with
future documents (including one at exactly the as_of boundary) and assert
none escape through any method.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from harness.evidence import (
    EvidenceStore,
    _as_of_to_cutoff_ms,
    _week_snapshot_date,
    build_index,
)

AS_OF = date(2023, 9, 1)
CUTOFF_MS = _as_of_to_cutoff_ms(AS_OF)

PAST_MS = CUTOFF_MS - 14 * 86400_000       # 2023-08-18
OLDER_MS = CUTOFF_MS - 40 * 86400_000      # 2023-07-23
FUTURE_MS = CUTOFF_MS + 30 * 86400_000     # 2023-10-01


def _sleeper_item(ms: int, key: str, title: str, desc: str, pid: str = "100") -> dict:
    return {
        "metadata": {"title": title, "description": desc, "analysis": ""},
        "player_id": pid,
        "published": ms,
        "source": "rotowire",
        "source_key": key,
    }


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("evidence")
    raw = root / "raw"
    (raw / "sleeper_backfill_internal").mkdir(parents=True)
    (raw / "sleeper_backfill_internal" / "100.json").write_text(json.dumps({
        "_source": "https://sleeper.com/graphql",
        "_captured_at": "2026-08-08T00:00:00+00:00",
        "_player": {
            "name": "Mike Evans", "norm_name": "mikeevans", "position": "WR",
            "team": "TB", "status": "Active", "sleeper_id": "100",
        },
        "news": [
            _sleeper_item(OLDER_MS, "1", "Evans steady in camp",
                          "Evans took first-team reps all week."),
            _sleeper_item(PAST_MS, "2", "Evans looks sharp",
                          "Evans caught everything in the preseason opener."),
            _sleeper_item(CUTOFF_MS, "3", "BOUNDARY DOC — must never appear",
                          "Published at exactly as_of; excluded by strict <."),
            _sleeper_item(FUTURE_MS, "4", "POISON FUTURE DOC zzyzxfuture",
                          "Evans erupts for 3 TDs in October. zzyzxfuture."),
        ],
    }))
    # A daily PFT capture whose items are ALL in the future.
    daily = raw / "2023-10-05"
    daily.mkdir()
    (daily / "pft_rss.json").write_text(json.dumps({
        "_source": "https://profootballtalk.nbcsports.com/feed/",
        "_captured_at": "2023-10-05T00:00:00+00:00",
        "items": [{
            "title": "POISON pft future item zzyzxfuture",
            "link": "https://example.com/poison",
            "pubDate": "Thu, 05 Oct 2023 12:00:00 -0400",
            "description": "future news that must never appear zzyzxfuture",
        }],
    }))

    nflverse = root / "nflverse"
    (nflverse / "depth_charts").mkdir(parents=True)
    (nflverse / "injuries").mkdir(parents=True)
    # Old (season, week) schema: week 1 ~ Aug 30 (pre-as_of), week 5 ~ Oct (post).
    (nflverse / "depth_charts" / "depth_charts_2023.csv").write_text(
        "season,club_code,week,game_type,depth_team,last_name,first_name,"
        "football_name,formation,gsis_id,jersey_number,position,elias_id,"
        "depth_position,full_name\n"
        "2023,TB,1,REG,1,Evans,Mike,Mike,Offense,00-0031408,13,WR,EVA001,WR,Mike Evans\n"
        "2023,TB,5,REG,1,Evans,Mike,Mike,Offense,00-0031408,13,WR,EVA001,LWR,Mike Evans\n"
    )
    # New dt schema (2025+): captured after as_of — must be invisible at 2023.
    (nflverse / "depth_charts" / "depth_charts_2025.csv").write_text(
        "dt,team,player_name,espn_id,gsis_id,pos_grp_id,pos_grp,pos_id,"
        "pos_name,pos_abb,pos_slot,pos_rank\n"
        "2025-06-01T07:00:00Z,TB,Mike Evans,1234,00-0031408,1,Offense,1,"
        "Wide Receiver,WR,1,1\n"
    )
    (nflverse / "injuries" / "injuries_2023.csv").write_text(
        "season,game_type,team,week,gsis_id,position,full_name,first_name,"
        "last_name,report_primary_injury,report_secondary_injury,report_status,"
        "practice_primary_injury,practice_secondary_injury,practice_status,"
        "date_modified\n"
        "2023,REG,TB,0,00-0031408,WR,Mike Evans,Mike,Evans,Hamstring,,"
        "Questionable,Hamstring,,Limited Participation in Practice,"
        "2023-08-20T18:00:00Z\n"
        "2023,REG,TB,3,00-0031408,WR,Mike Evans,Mike,Evans,Ankle,,Out,Ankle,,"
        "Did Not Participate In Practice,2023-09-20T18:00:00Z\n"
    )

    index = root / "index"
    manifest = build_index(raw, index, quiet=True)
    return {"raw": raw, "index": index, "nflverse": nflverse,
            "manifest": manifest}


@pytest.fixture()
def store(corpus):
    return EvidenceStore(AS_OF, index_dir=corpus["index"],
                         nflverse_dir=corpus["nflverse"])


def test_build_manifest(corpus):
    m = corpus["manifest"]
    assert m["n_docs"] == 5  # 4 sleeper + 1 pft
    assert m["n_players"] == 1
    assert m["docs_by_source"] == {"rotowire": 4, "pft_rss": 1}
    assert (corpus["index"] / "manifest.json").is_file()
    assert (corpus["index"] / "players" / "100.jsonl").is_file()


# --- player_news -----------------------------------------------------------

def test_player_news_excludes_future_and_boundary(store):
    items = store.player_news("Mike Evans", limit=10)
    assert [d["source_key"] for d in items] == ["2", "1"]  # newest first
    assert all(d["published_ms"] < CUTOFF_MS for d in items)
    titles = " ".join(d["title"] for d in items)
    assert "POISON" not in titles and "BOUNDARY" not in titles


def test_player_news_boundary_exactness(corpus):
    # A store as_of one ms AFTER the boundary doc includes it; at as_of it's out.
    at = EvidenceStore(AS_OF, index_dir=corpus["index"],
                       nflverse_dir=corpus["nflverse"])
    assert all(d["source_key"] != "3" for d in at.player_news("Mike Evans"))
    later = EvidenceStore(date(2023, 9, 2), index_dir=corpus["index"],
                          nflverse_dir=corpus["nflverse"])
    assert any(d["source_key"] == "3"
               for d in later.player_news("Mike Evans"))


def test_player_news_since(store):
    items = store.player_news("Mike Evans", limit=10, since=date(2023, 8, 1))
    assert [d["source_key"] for d in items] == ["2"]


def test_name_resolution_via_names_py(store):
    # evals.names canonicalizes Michael -> mike; suffix stripped.
    for query in ("Michael Evans", "mike evans jr.", "100"):
        items = store.player_news(query, limit=1)
        assert items and items[0]["player_id"] == "100"
    with pytest.raises(KeyError):
        store.player_news("Nonexistent Player")


def test_internal_only_flag(store):
    assert all(d["internal_research_only"] for d in store.player_news("100"))


# --- search ------------------------------------------------------------------

def test_search_excludes_future_docs(store):
    # The poison token exists ONLY in future docs (sleeper + pft).
    assert store.search("zzyzxfuture", limit=10) == []
    hits = store.search("Evans", limit=10)
    assert hits and all(d["published_ms"] < CUTOFF_MS for d in hits)


def test_search_terms_and_position(store):
    hits = store.search("first-team reps", limit=5)
    assert [d["source_key"] for d in hits] == ["1"]
    assert store.search("first-team reps", position="RB") == []
    assert store.search("first-team reps", position="WR")


# --- depth_chart ---------------------------------------------------------------

def test_depth_chart_latest_pre_cutoff(store):
    rows = store.depth_chart("Mike Evans")
    assert len(rows) == 1
    r = rows[0]
    assert r["week"] == 1 and r["depth_slot"] == "WR"
    assert r["snapshot"] == "2023-08-30" and r["date_basis"] == "approx_week"
    assert r["published_ms"] < CUTOFF_MS
    assert r["internal_research_only"] is False


def test_depth_chart_future_weeks_and_seasons_invisible(store):
    rows = store.depth_chart("Mike Evans", latest=False)
    assert all(r["published_ms"] < CUTOFF_MS for r in rows)
    assert all(r.get("week") != 5 for r in rows)
    assert all(r["season"] == 2023 for r in rows)  # 2025 dt file invisible


def test_depth_chart_dt_schema_visible_later(corpus):
    s = EvidenceStore(date(2025, 7, 1), index_dir=corpus["index"],
                      nflverse_dir=corpus["nflverse"])
    rows = s.depth_chart("Mike Evans")
    assert rows[0]["date_basis"] == "capture_dt"
    assert rows[0]["season"] == 2025


def test_depth_chart_by_team(store):
    rows = store.depth_chart("TB")
    assert rows and all(r["team"] == "TB" for r in rows)
    assert all(r["published_ms"] < CUTOFF_MS for r in rows)


# --- injury_status ------------------------------------------------------------

def test_injury_status_pre_cutoff_only(store):
    row = store.injury_status("Mike Evans")
    assert row is not None
    assert row["report_status"] == "Questionable"          # the Aug row
    assert row["published_ms"] < CUTOFF_MS                 # not the Sept "Out"


def test_injury_status_none_when_only_future(corpus):
    s = EvidenceStore(date(2023, 8, 1), index_dir=corpus["index"],
                      nflverse_dir=corpus["nflverse"])
    assert s.injury_status("Mike Evans") is None


# --- tool surface ---------------------------------------------------------------

def test_tool_definitions_shape(store):
    defs = store.as_tool_definitions()
    names = [d["function"]["name"] for d in defs]
    assert names == ["player_news", "search", "depth_chart", "injury_status"]
    json.dumps(defs)  # must be valid JSON-serializable schemas
    for d in defs:
        assert d["type"] == "function"
        assert d["function"]["parameters"]["type"] == "object"


def test_dispatch_round_trip(store):
    # OpenAI-style call with stringified arguments, as a chat model emits.
    call = {
        "function": {
            "name": "player_news",
            "arguments": json.dumps({"player": "Michael Evans", "limit": 2}),
        }
    }
    out = store.dispatch(json.dumps(call))
    assert out["ok"] is True
    assert [d["source_key"] for d in out["result"]] == ["2", "1"]
    json.dumps(out)  # result must round-trip to JSON

    for name, args in [
        ("search", {"query": "Evans"}),
        ("depth_chart", {"team_or_player": "TB"}),
        ("injury_status", {"player": "Mike Evans"}),
    ]:
        out = store.dispatch({"name": name, "arguments": args})
        assert out["ok"] is True, out
        payload = out["result"]
        rows = payload if isinstance(payload, list) else [payload]
        assert all(r["published_ms"] < CUTOFF_MS for r in rows if r)


def test_dispatch_errors(store):
    assert store.dispatch({"name": "nope", "arguments": {}})["ok"] is False
    out = store.dispatch({"name": "player_news",
                          "arguments": {"player": "Nonexistent Player"}})
    assert out["ok"] is False and "Nonexistent" in out["error"]


# --- helpers -----------------------------------------------------------------

def test_week_snapshot_dates():
    assert _week_snapshot_date(2023, 1) == date(2023, 8, 30)
    assert _week_snapshot_date(2023, 2) == date(2023, 9, 14)  # Thu of week 2
