"""Offline unit tests for scripts/collect_vegas_lines.py (no network)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "collect_vegas_lines", ROOT / "scripts" / "collect_vegas_lines.py")
cvl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cvl)


HEADER = ("game_id,season,game_type,week,gameday,home_team,away_team,"
          "home_score,away_score,spread_line,total_line")


def _fake_csv(n_rows: int = 6000, with_lines: float = 1.0) -> bytes:
    lines = [HEADER]
    for i in range(n_rows):
        season = 1999 + (i % 27)
        has = (i / n_rows) < with_lines
        spread, total = ("-3", "44.5") if has else ("", "")
        lines.append(
            f"{season}_01_KC_CHI_{i},{season},REG,1,{season}-09-12,CHI,KC,20,17,{spread},{total}")
    return "\n".join(lines).encode()


# ------------------------------------------------------------- validation

def test_validate_accepts_healthy_payload():
    cov = cvl.validate_games_csv(_fake_csv())
    assert cov["rows"] == 6000
    assert cov["rows_with_lines"] == 6000
    assert cov["season_min"] == 1999
    assert cov["season_max"] == 2025


def test_validate_rejects_missing_columns():
    bad = b"game_id,season\n1999_01_KC_CHI,1999\n"
    with pytest.raises(ValueError, match="missing expected columns"):
        cvl.validate_games_csv(bad)


def test_validate_rejects_tiny_and_lineless_payloads():
    with pytest.raises(ValueError, match="suspiciously small"):
        cvl.validate_games_csv(_fake_csv(n_rows=100))
    with pytest.raises(ValueError, match="line coverage too low"):
        cvl.validate_games_csv(_fake_csv(with_lines=0.5))


# ---------------------------------------------------------------- collect

def test_collect_writes_snapshot_and_manifest(tmp_path):
    payload = _fake_csv()
    entry = cvl.collect(out_dir=tmp_path, date="2026-08-08",
                        fetcher=lambda url: payload)
    assert entry["status"] == "captured"
    dest = tmp_path / "games_2026-08-08.csv"
    assert dest.read_bytes() == payload
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest) == 1
    m = manifest[0]
    assert m["source_url"] == cvl.GAMES_URL
    assert m["sha256"] == __import__("hashlib").sha256(payload).hexdigest()
    assert m["rows"] == 6000 and m["season_min"] == 1999


def test_collect_never_overwrites_existing_snapshot(tmp_path):
    dest = tmp_path / "games_2026-08-08.csv"
    dest.write_bytes(b"original immutable capture")

    def explode(url):  # network must not even be attempted
        raise AssertionError("fetch called despite existing snapshot")

    entry = cvl.collect(out_dir=tmp_path, date="2026-08-08", fetcher=explode)
    assert entry["status"] == "exists_immutable"
    assert dest.read_bytes() == b"original immutable capture"


def test_collect_bad_payload_leaves_no_file(tmp_path):
    with pytest.raises(ValueError):
        cvl.collect(out_dir=tmp_path, date="2026-08-08",
                    fetcher=lambda url: b"game_id,season\n")
    assert not (tmp_path / "games_2026-08-08.csv").exists()


def test_latest_snapshot_orders_by_date(tmp_path):
    assert cvl.latest_snapshot(tmp_path) is None
    (tmp_path / "games_2026-08-01.csv").write_bytes(b"a")
    (tmp_path / "games_2026-08-08.csv").write_bytes(b"b")
    assert cvl.latest_snapshot(tmp_path).name == "games_2026-08-08.csv"
