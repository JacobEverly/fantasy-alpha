"""Offline unit tests for scripts/archive_evidence.py (no network)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "archive_evidence", ROOT / "scripts" / "archive_evidence.py")
ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ae)

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>PFT</title>
<item><title>Player X signs</title><link>https://x/1</link>
<pubDate>Fri, 08 Aug 2026 12:00:00 +0000</pubDate>
<description><![CDATA[Big <b>news</b> here.]]></description></item>
<item><title>Second story</title><link>https://x/2</link>
<pubDate>Fri, 08 Aug 2026 13:00:00 +0000</pubDate>
<description>plain text</description></item>
</channel></rss>"""


def test_parse_rss_extracts_fields():
    items = ae.parse_rss(RSS)
    assert len(items) == 2
    assert items[0]["title"] == "Player X signs"
    assert items[0]["link"] == "https://x/1"
    assert items[0]["pubDate"].startswith("Fri, 08 Aug 2026")
    assert "news" in items[0]["description"]  # CDATA unwrapped
    assert items[1]["description"] == "plain text"


def test_match_id_nickname_team_and_position_gates():
    rows = [
        {"id": "1", "name": "Mike Williams", "pos": "WR", "team": "LAC", "active": True},
        {"id": "2", "name": "Mike Williams", "pos": "WR", "team": "NYJ", "active": True},
        {"id": "3", "name": "Josh Allen", "pos": "QB", "team": "BUF", "active": True},
    ]
    id_map = ae.build_id_map(rows)
    # nickname canonicalization (norm_name) + team tiebreak on duplicates
    assert ae.match_id(id_map, "Michael Williams", "WR", "NYJ") == "2"
    assert ae.match_id(id_map, "Josh Allen", "QB", "BUF") == "3"
    assert ae.match_id(id_map, "Josh Allen", "WR", "BUF") is None  # position gate
    assert ae.match_id(id_map, "Justin Tucker", "PK", "BAL") is None  # absent from map


def test_match_id_prefers_active_when_team_misses():
    rows = [
        {"id": "old", "name": "Zach Miller", "pos": "TE", "team": "", "active": False},
        {"id": "new", "name": "Zach Miller", "pos": "TE", "team": "CHI", "active": True},
    ]
    id_map = ae.build_id_map(rows)
    assert ae.match_id(id_map, "Zach Miller", "TE", "SEA") == "new"


def test_id_rows_from_dump_filters_to_fantasy_positions():
    dump = {
        "4984": {"full_name": "Josh Allen", "position": "QB", "team": "BUF", "active": True},
        "1166": {"first_name": "Justin", "last_name": "Tucker", "position": "K",
                 "team": "BAL", "active": True},
        "9999": {"full_name": "Some Lineman", "position": "OT", "team": "BUF", "active": True},
    }
    rows = ae.id_rows_from_dump(dump)
    assert {r["id"] for r in rows} == {"4984", "1166"}
    id_map = ae.build_id_map(rows)
    assert ae.match_id(id_map, "Justin Tucker", "PK", "BAL") == "1166"  # FFC PK -> sleeper K


def test_have_is_the_append_only_guard(tmp_path):
    p = tmp_path / "x.json"
    assert not ae.have(p)
    p.write_text("")
    assert not ae.have(p)  # empty capture may be retried
    p.write_text("{}")
    assert ae.have(p)  # non-empty evidence is immutable


def test_source_status_matrix():
    assert ae.source_status(0, 0, []) == "failed"
    assert ae.source_status(1, 2, [{"e": 1}]) == "partial"
    assert ae.source_status(2, 2, []) == "ok"
    assert ae.source_status(0, 2, []) == "cached"
