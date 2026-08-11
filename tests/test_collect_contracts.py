"""Offline unit tests for scripts/collect_contracts.py (no network)."""
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "collect_contracts", ROOT / "scripts" / "collect_contracts.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def _contract_row(**kw):
    row = {
        "player": "Test Player", "position": "RB", "team": "Giants",
        "is_active": True, "year_signed": 2026, "years": 4.0,
        "value": 31.2, "apy": 7.8, "guaranteed": 31.2, "apy_cap_pct": 0.04,
        "otc_id": 1, "gsis_id": "00-0000001", "draft_year": 2022.0,
        "draft_round": 1.0, "draft_overall": 2.0, "draft_team": "Giants",
        "contract_history": None, "season_history": None,
    }
    row.update(kw)
    return row


def _parquet_bytes(rows):
    import pandas as pd
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf)
    return buf.getvalue()


def _healthy_payload(n=20_001):
    rows = [_contract_row(otc_id=i, year_signed=2000 + (i % 27))
            for i in range(n - 1)]
    rows.append(_contract_row(otc_id=n, year_signed=2026))
    return _parquet_bytes(rows)


# ------------------------------------------------------- nested-block logic

def test_align_contract_type_unique_match():
    row = _contract_row(contract_history=[
        {"contract_type": "Drafted", "year_signed": 2026, "yrs": 4.0,
         "team": "Giants", "apy": 7.8},
        {"contract_type": "Franchise", "year_signed": 2030, "yrs": 1.0,
         "team": "Giants", "apy": 10.0},
    ])
    assert cc.align_contract_type(row) == "Drafted"


def test_align_contract_type_ambiguous_or_absent_is_blank():
    assert cc.align_contract_type(_contract_row()) == ""
    row = _contract_row(contract_history=[
        {"contract_type": "SFA", "year_signed": 2026, "yrs": 4.0,
         "team": "Giants", "apy": 7.8},
        {"contract_type": "UFA", "year_signed": 2026, "yrs": 4.0,
         "team": "Giants", "apy": 7.8},
    ])
    assert cc.align_contract_type(row) == ""


def test_detect_option_through_exercised():
    """Round-1 drafted deal + season_history year 5 with the drafted team,
    unexplained by any later contract -> option detected."""
    row = _contract_row(
        year_signed=2018, contract_type="Drafted",
        contract_history=[{"contract_type": "Franchise",
                           "year_signed": 2023.0}],
        season_history=[{"year": "2021", "team": "Giants"},
                        {"year": "2022", "team": "Giants"}])
    assert cc.detect_option_through(row) == "2022"


def test_detect_option_through_negative_cases():
    base = dict(year_signed=2018, contract_type="Drafted",
                season_history=[{"year": "2022", "team": "Giants"}],
                contract_history=None)
    # a contract signed within (year_signed, year_signed+4] explains year 5
    row = _contract_row(**{**base, "contract_history": [
        {"contract_type": "Extension", "year_signed": 2021.0}]})
    assert cc.detect_option_through(row) == ""
    # not round 1
    assert cc.detect_option_through(
        _contract_row(**base, draft_round=2.0)) == ""
    # pre-2011 CBA: no 5th-year options
    assert cc.detect_option_through(_contract_row(
        **{**base, "year_signed": 2009,
           "season_history": [{"year": "2013", "team": "Giants"}]})) == ""
    # year 5 played elsewhere
    assert cc.detect_option_through(_contract_row(
        **{**base, "season_history": [{"year": "2022", "team": "Jets"}]})) == ""


# ------------------------------------------------------------- validation

def test_validate_rejects_missing_columns():
    import pandas as pd
    buf = io.BytesIO()
    pd.DataFrame([{"player": "A", "team": "B"}]).to_parquet(buf)
    with pytest.raises(ValueError, match="missing columns"):
        cc.validate_and_flatten(buf.getvalue())


def test_validate_rejects_tiny_payload():
    with pytest.raises(ValueError, match="suspiciously small"):
        cc.validate_and_flatten(_parquet_bytes([_contract_row()] * 10))


def test_validate_rejects_stale_asset():
    """The frozen-at-2022 csv.gz failure mode must never slip through."""
    rows = [_contract_row(otc_id=i, year_signed=2015) for i in range(20_001)]
    with pytest.raises(ValueError, match="stale"):
        cc.validate_and_flatten(_parquet_bytes(rows))


# ---------------------------------------------------------------- collect

def test_collect_writes_snapshot_flat_csv_and_manifest(tmp_path):
    payload = _healthy_payload()
    entry = cc.collect(out_dir=tmp_path, date="2026-08-11",
                       fetcher=lambda url: payload)
    assert entry["status"] == "captured"
    assert (tmp_path / "historical_contracts_2026-08-11.parquet"
            ).read_bytes() == payload
    csv_path = tmp_path / "historical_contracts_2026-08-11.csv"
    header = csv_path.read_text().splitlines()[0]
    assert header == ",".join(cc.FLAT_COLUMNS)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest) == 1
    m = manifest[0]
    assert m["source_url"] == cc.PARQUET_URL
    assert m["sha256"] == __import__("hashlib").sha256(payload).hexdigest()
    assert m["rows"] == 20_001 and m["year_signed_max"] == 2026


def test_collect_never_overwrites_existing_snapshot(tmp_path):
    dest = tmp_path / "historical_contracts_2026-08-11.parquet"
    dest.write_bytes(b"original immutable capture")

    def explode(url):  # network must not even be attempted
        raise AssertionError("fetch called despite existing snapshot")

    entry = cc.collect(out_dir=tmp_path, date="2026-08-11", fetcher=explode)
    assert entry["status"] == "exists_immutable"
    assert dest.read_bytes() == b"original immutable capture"


def test_collect_bad_payload_leaves_no_files(tmp_path):
    with pytest.raises(ValueError):
        cc.collect(out_dir=tmp_path, date="2026-08-11",
                   fetcher=lambda url: _parquet_bytes([_contract_row()] * 5))
    assert not list(tmp_path.glob("historical_contracts_*"))
    assert not (tmp_path / "manifest.json").exists()


def test_latest_snapshot_orders_by_date(tmp_path):
    assert cc.latest_snapshot(tmp_path) is None
    (tmp_path / "historical_contracts_2026-08-01.csv").write_text("a")
    (tmp_path / "historical_contracts_2026-08-11.csv").write_text("b")
    assert cc.latest_snapshot(tmp_path).name == \
        "historical_contracts_2026-08-11.csv"
