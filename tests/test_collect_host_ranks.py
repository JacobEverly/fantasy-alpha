"""Offline unit tests for scripts/collect_host_ranks.py (no network)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "collect_host_ranks", ROOT / "scripts" / "collect_host_ranks.py")
chr_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chr_)


# ---------------------------------------------------------------- fixtures

def espn_player(pid, name, pos_id, ppr, std, adp=None, auction=None):
    return {"player": {
        "id": pid, "fullName": name, "defaultPositionId": pos_id,
        "proTeamId": 8,
        "draftRanksByRankType": {
            "PPR": {"rank": ppr, "rankType": "PPR", "auctionValue": auction},
            "STANDARD": {"rank": std, "rankType": "STANDARD"},
        },
        "ownership": {"averageDraftPosition": adp, "percentOwned": 99.0},
    }}


def espn_payload():
    return {"players": [
        espn_player(2, "Bijan Robinson", 2, 2, 1, adp=1.8),
        espn_player(1, "Jahmyr Gibbs", 2, 1, 2, adp=1.6, auction=57),
        espn_player(3, "Ja'Marr Chase", 3, 3, 3, adp=3.1),
        {"player": {"id": 99, "fullName": "No Ranks Guy",
                    "defaultPositionId": 3, "draftRanksByRankType": {}}},
    ]}


def yahoo_player_fragments(pid, name, pos, team, avg_pick, status=None):
    meta = [{"player_id": pid}, {"name": {"full": name}},
            {"display_position": pos}, {"editorial_team_abbr": team},
            [],  # stray list, as the real API emits
            ]
    if status:
        meta.append({"status": status})
    return [meta, {"draft_analysis": [
        {"average_pick": avg_pick}, {"average_round": "1.0"},
        {"average_cost": "50.0"}, {"percent_drafted": "1.00"}]}]


def yahoo_page(players, count=None):
    coll = {"count": count if count is not None else len(players)}
    for i, p in enumerate(players):
        coll[str(i)] = {"player": p}
    return {"fantasy_content": {"game": [{"game_key": "470"},
                                         {"players": coll}]}}


# ---------------------------------------------------------------- ESPN parse

def test_parse_espn_orders_by_ppr_rank_and_drops_unranked():
    rows = chr_.parse_espn(espn_payload())
    assert [r["name"] for r in rows] == [
        "Jahmyr Gibbs", "Bijan Robinson", "Ja'Marr Chase"]
    gibbs = rows[0]
    assert gibbs["host_rank_ppr"] == 1
    assert gibbs["host_rank_standard"] == 2
    assert gibbs["pos"] == "RB"
    assert gibbs["espn_adp"] == 1.6
    assert gibbs["auction_value_ppr"] == 57
    assert gibbs["ranks"] == {"PPR": 1, "STANDARD": 2}


def test_parse_espn_empty_payload():
    assert chr_.parse_espn({}) == []
    assert chr_.parse_espn({"players": []}) == []


# ---------------------------------------------------------------- Yahoo parse

def test_parse_yahoo_rank_is_global_position_across_pages():
    p1 = yahoo_page([yahoo_player_fragments("40059", "Jahmyr Gibbs", "RB",
                                            "Det", "1.6", status="Q"),
                     yahoo_player_fragments("33393", "Bijan Robinson", "RB",
                                            "Atl", "1.8")])
    p2 = yahoo_page([yahoo_player_fragments("33399", "Ja'Marr Chase", "WR",
                                            "Cin", "3.3")])
    rows = chr_.parse_yahoo([p1, p2])
    assert [(r["name"], r["host_rank"]) for r in rows] == [
        ("Jahmyr Gibbs", 1), ("Bijan Robinson", 2), ("Ja'Marr Chase", 3)]
    assert rows[0]["yahoo_adp"] == 1.6
    assert rows[0]["status"] == "Q"
    assert rows[2]["team"] == "Cin"
    assert rows[2]["pct_drafted"] == 1.0


def test_parse_yahoo_tolerates_bad_numbers_and_empty():
    frag = yahoo_player_fragments("1", "X Y", "RB", "Det", "-")
    rows = chr_.parse_yahoo([yahoo_page([frag])])
    assert rows[0]["yahoo_adp"] is None
    assert chr_.parse_yahoo([]) == []


# ---------------------------------------------------------------- capture

def test_archive_host_ranks_writes_snapshot_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(chr_, "OUT", tmp_path)
    monkeypatch.setattr(chr_, "YAHOO_PAGES", 1)
    calls = []

    def fake_fetch(url, *, headers=None, timeout=60):
        calls.append(url)
        if "espn.com" in url:
            return json.dumps(espn_payload()).encode()
        return json.dumps(yahoo_page(
            [yahoo_player_fragments("40059", "Jahmyr Gibbs", "RB", "Det",
                                    "1.6")])).encode()

    monkeypatch.setattr(chr_, "fetch", fake_fetch)
    day_dir = tmp_path.parent / "evidence" / "2026-08-11"
    day_dir.mkdir(parents=True)
    entry = chr_.archive_host_ranks(day_dir)

    out = tmp_path / "2026-08-11"  # date from day_dir, output under OUT
    assert entry["status"] == "ok"
    assert entry["items"] == 4  # 3 espn + 1 yahoo
    assert (out / "espn_kona_player_info.json").exists()
    assert (out / "yahoo_players_or_p0.json").exists()
    parsed = json.loads((out / "host_ranks_parsed.json").read_text())
    assert parsed["espn"][0]["name"] == "Jahmyr Gibbs"
    assert parsed["yahoo"][0]["host_rank"] == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sources"]["espn"]["status"] == "ok"
    assert manifest["sources"]["yahoo"]["status"] == "ok"

    # idempotent re-run: nothing refetched, evidence untouched
    n_calls = len(calls)
    before = (out / "espn_kona_player_info.json").read_bytes()
    entry2 = chr_.archive_host_ranks(day_dir)
    assert len(calls) == n_calls
    assert entry2["status"] == "cached"
    assert (out / "espn_kona_player_info.json").read_bytes() == before


def test_one_host_failing_never_kills_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(chr_, "OUT", tmp_path)
    monkeypatch.setattr(chr_, "YAHOO_PAGES", 1)

    def fake_fetch(url, *, headers=None, timeout=60):
        if "espn.com" in url:
            raise OSError("espn down")
        return json.dumps(yahoo_page(
            [yahoo_player_fragments("40059", "Jahmyr Gibbs", "RB", "Det",
                                    "1.6")])).encode()

    monkeypatch.setattr(chr_, "fetch", fake_fetch)
    entry = chr_.archive_host_ranks()
    assert entry["status"] == "partial"
    assert entry["hosts"]["espn"]["status"] == "failed"
    assert "espn down" in entry["hosts"]["espn"]["error"]
    assert entry["hosts"]["yahoo"]["status"] == "ok"
    out = tmp_path / list(tmp_path.iterdir())[0].name
    parsed = json.loads((out / "host_ranks_parsed.json").read_text())
    assert parsed["espn"] == [] and len(parsed["yahoo"]) == 1


def test_yahoo_partial_page_failure_stops_rank_numbering(tmp_path, monkeypatch):
    monkeypatch.setattr(chr_, "YAHOO_PAGES", 3)
    monkeypatch.setattr(chr_, "SLEEP", 0)

    def fake_fetch(url, *, headers=None, timeout=60):
        if "start=0" in url:
            return json.dumps(yahoo_page(
                [yahoo_player_fragments("1", "A B", "RB", "Det", "1.0")])).encode()
        raise OSError("page down")

    monkeypatch.setattr(chr_, "fetch", fake_fetch)
    day = tmp_path / "2026-08-11"
    day.mkdir()
    entry, rows = chr_.capture_yahoo(day)
    assert entry["status"] == "partial"
    assert entry["pages"] == 1
    assert len(entry["failures"]) == 1
    assert [r["host_rank"] for r in rows] == [1]
