"""Offline unit tests for scripts/backfill_sleeper_news.py (no network)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "backfill_sleeper_news", ROOT / "scripts" / "backfill_sleeper_news.py")
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

MS_2018_05 = 1525132800000   # 2018-05-01 UTC
MS_2017_06 = 1497312000000   # 2017-06-13 UTC
MS_2024_11 = 1730980800000   # 2024-11-07 UTC


def test_year_of_parses_ms_and_rejects_junk():
    assert bf.year_of(MS_2018_05) == 2018
    assert bf.year_of(MS_2024_11) == 2024
    assert bf.year_of(str(MS_2017_06)) == 2017  # numeric strings tolerated
    assert bf.year_of(None) is None
    assert bf.year_of("not-a-timestamp") is None
    assert bf.year_of(10**20) is None  # overflows datetime range


def test_bucket_years_histogram():
    items = [{"published": MS_2018_05}, {"published": MS_2018_05 + 1},
             {"published": MS_2024_11}, {"published": None}, {}]
    years = bf.bucket_years(items)
    assert years == {2018: 2, 2024: 1, "unknown": 2}


def test_split_resume_skips_only_nonempty_files(tmp_path):
    players = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    (tmp_path / "1.json").write_text('{"news": []}')  # captured -> skip
    (tmp_path / "2.json").write_text("")              # empty capture -> retry
    todo, resumed = bf.split_resume(players, tmp_path)
    assert [r["id"] for r in resumed] == ["1"]
    assert [r["id"] for r in todo] == ["2", "3"]


def test_select_players_inclusion_rule():
    dump = {
        "1": {"full_name": "Active NoNews", "position": "QB", "active": True},
        "2": {"full_name": "Retired Recent", "position": "WR", "active": False,
              "news_updated": MS_2024_11},
        "3": {"full_name": "Retired Ancient", "position": "RB", "active": False,
              "news_updated": MS_2017_06},
        "4": {"full_name": "Some Lineman", "position": "OT", "active": True},
        "5": {"first_name": "No", "last_name": "FullName", "position": "TE",
              "active": True, "team": "BUF"},
    }
    rows = bf.select_players(dump)
    assert [r["id"] for r in rows] == ["1", "2", "5"]  # 3 pre-window, 4 wrong pos
    by_id = {r["id"]: r for r in rows}
    assert by_id["5"]["name"] == "No FullName"  # first/last fallback
    assert by_id["5"]["team"] == "BUF"
