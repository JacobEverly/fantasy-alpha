"""Offline unit tests for scripts/collect_odds.py (no network)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "collect_odds", ROOT / "scripts" / "collect_odds.py")
co = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(co)


# ---------------------------------------------------------------- odds math

def test_american_to_prob():
    assert co.american_to_prob(-110) == pytest.approx(110 / 210)
    assert co.american_to_prob(+150) == pytest.approx(100 / 250)
    assert co.american_to_prob(-100) == pytest.approx(0.5)
    assert co.american_to_prob(+100) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        co.american_to_prob(0)


def test_devig_pair_normalizes_and_reports_overround():
    p_over, p_under, overround = co.devig_pair(-110, -110)
    assert p_over == pytest.approx(0.5)
    assert p_under == pytest.approx(0.5)
    assert overround == pytest.approx(2 * 110 / 210 - 1)  # ~4.76% margin
    # asymmetric market keeps ordering and sums to 1
    p_over, p_under, _ = co.devig_pair(-130, +105)
    assert p_over + p_under == pytest.approx(1.0)
    assert p_over > 0.5 > p_under


# ---------------------------------------------------------------- .env

def test_load_env_parses_and_environ_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nODDS_API_KEY=abc123\nOTHER='quoted'\nBROKEN LINE\n")
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    env = co.load_env(env_file)
    assert env["ODDS_API_KEY"] == "abc123"
    assert env["OTHER"] == "quoted"
    assert "BROKEN LINE" not in env
    monkeypatch.setenv("ODDS_API_KEY", "from-environ")
    assert co.load_env(env_file)["ODDS_API_KEY"] == "from-environ"
    # empty-value key in .env stays falsy -> archiver skips
    env_file.write_text("ODDS_API_KEY=\n")
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert not co.load_env(env_file).get("ODDS_API_KEY")


def test_load_env_missing_file(tmp_path):
    assert co.load_env(tmp_path / "nope.env").get("ODDS_API_KEY") in (None, "")


# ---------------------------------------------------------------- capture

ESPN_PAYLOAD = {"count": 2, "items": [
    {"id": 1, "name": "Regular Season MVP", "futures": []},
    {"id": 2, "name": "Super Bowl Winner", "futures": []},
]}


def test_espn_futures_capture_and_idempotency(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(url, **kw):
        calls.append(url)
        return json.dumps(ESPN_PAYLOAD).encode(), {}

    monkeypatch.setattr(co, "fetch", fake_fetch)
    entry = co.archive_espn_futures(tmp_path)
    assert entry["status"] == "ok" and entry["items"] == 2
    saved = json.loads((tmp_path / "espn_futures.json").read_text())
    assert saved["_source"].startswith("https://sports.core.api.espn.com/")
    assert saved["_captured_at"]

    # second run: cached, no refetch (immutable evidence)
    monkeypatch.setattr(co, "fetch", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not refetch")))
    entry2 = co.archive_espn_futures(tmp_path)
    assert entry2["status"] == "cached" and entry2["items"] == 2
    assert len(calls) == 1


def test_espn_futures_empty_payload_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(co, "fetch", lambda url, **kw: (b'{"items": []}', {}))
    with pytest.raises(ValueError):
        co.archive_espn_futures(tmp_path)
    assert not (tmp_path / "espn_futures.json").exists()


def test_odds_api_skips_without_key(tmp_path):
    entry = co.archive_odds_api(tmp_path, None)
    assert entry["status"] == "skipped"
    assert "the-odds-api.com" in entry["note"]
    assert list(tmp_path.iterdir()) == []


def test_odds_api_capture_discovers_futures_and_redacts_key(tmp_path, monkeypatch):
    sports = [
        {"key": "americanfootball_nfl", "active": True, "has_outrights": False},
        {"key": "americanfootball_nfl_super_bowl_winner", "active": True,
         "has_outrights": True},
        {"key": "americanfootball_nfl_old_future", "active": False,
         "has_outrights": True},
        {"key": "basketball_nba", "active": True, "has_outrights": False},
    ]
    events = [{"id": "e1"}, {"id": "e2"}]

    def fake_fetch(url, **kw):
        assert "sekrit" in url  # key goes out on the wire...
        payload = sports if "/sports?" in url else events
        return json.dumps(payload).encode(), {"x-requests-remaining": "497"}

    monkeypatch.setattr(co, "fetch", fake_fetch)
    monkeypatch.setattr(co, "SLEEP", 0)
    entry = co.archive_odds_api(tmp_path, "sekrit")
    assert entry["status"] == "ok"
    assert entry["futures_keys"] == ["americanfootball_nfl_super_bowl_winner"]
    assert entry["credits"] == {"x-requests-remaining": "497"}
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["oddsapi_futures_americanfootball_nfl_super_bowl_winner.json",
                     "oddsapi_nfl_odds.json", "oddsapi_sports.json"]
    for p in tmp_path.iterdir():  # ...but never into stored evidence
        text = p.read_text()
        assert "sekrit" not in text and "REDACTED" in text

    # idempotent: all cached, fetch never called again
    monkeypatch.setattr(co, "fetch", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not refetch")))
    entry2 = co.archive_odds_api(tmp_path, "sekrit")
    assert entry2["status"] == "cached"
    assert entry2["items"] == entry["items"]


def test_archive_odds_rollup_skip_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        co, "fetch", lambda url, **kw: (json.dumps(ESPN_PAYLOAD).encode(), {}))
    monkeypatch.setattr(co, "load_env", lambda path=None: {})
    entry = co.archive_odds(tmp_path)
    assert entry["status"] == "ok"  # espn ok + odds_api skipped -> ok, not failed
    assert entry["sources"]["odds_api"]["status"] == "skipped"
    manifest = json.loads((tmp_path / "odds" / "manifest.json").read_text())
    assert manifest["sources"]["espn_futures"]["status"] == "ok"


def test_archive_odds_all_failed_rolls_up_failed(tmp_path, monkeypatch):
    def boom(url, **kw):
        raise OSError("network down")

    monkeypatch.setattr(co, "fetch", boom)
    monkeypatch.setattr(co, "load_env", lambda path=None: {"ODDS_API_KEY": "k"})
    entry = co.archive_odds(tmp_path)
    assert entry["sources"]["espn_futures"]["status"] == "failed"
    assert entry["sources"]["odds_api"]["status"] == "failed"
    assert entry["status"] == "failed"
