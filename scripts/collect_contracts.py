#!/usr/bin/env python3
"""Download the nflverse contracts dataset (OverTheCap-sourced) into
data/raw/contracts/.

Source: the `contracts` release of github.com/nflverse/nflverse-data, which
mirrors OverTheCap.com player contracts (one row per contract: player,
position, team, year_signed, years, value/apy in $M, guarantees, draft
pedigree, gsis_id, plus nested per-year and contract-history blocks).
nflverse redistributes this openly for research (same channel as the rest of
our nflverse raw data), so no OTC scraping is needed or performed.

Format note (why this collector is NOT stdlib-only): the release's
stdlib-readable `historical_contracts.csv.gz` asset is frozen at 2022-05-29
(max year_signed 2022), while `historical_contracts.parquet` is refreshed
daily. We therefore snapshot the parquet (authoritative raw) and write a
flattened CSV conversion next to it via pandas/pyarrow — already project
dependencies for the GBDT evals. The flatten drops the nested
`season_history` block and reduces `contract_history` to one aligned
`contract_type` string per contract row (blank when alignment is ambiguous).

Snapshots are dated and immutable: an existing dated capture is never
overwritten (repo hard rule for market data). manifest.json records URL,
capture time, sha256, byte size, and coverage stats.

Usage:
    .venv/bin/python scripts/collect_contracts.py
"""
from __future__ import annotations

import hashlib
import io
import json
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PARQUET_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
               "contracts/historical_contracts.parquet")
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "contracts"
UA = {"User-Agent": "fantasy-alpha-collector/0.1"}


def _ssl_context() -> ssl.SSLContext:
    """Default context; fall back to the system bundle when the Python.org
    framework build ships without CA certs (house pattern from
    scripts/collect_odds.py)."""
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats().get("x509_ca") and Path("/etc/ssl/cert.pem").exists():
        ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    return ctx


SSL_CTX = _ssl_context()

REQUIRED_COLUMNS = {
    "player", "position", "team", "is_active", "year_signed", "years",
    "value", "apy", "guaranteed", "apy_cap_pct", "otc_id", "gsis_id",
    "draft_year", "draft_round", "draft_overall", "draft_team",
    "contract_history",
}

FLAT_COLUMNS = [
    "player", "position", "team", "is_active", "year_signed", "years",
    "value", "apy", "guaranteed", "apy_cap_pct", "otc_id", "gsis_id",
    "draft_year", "draft_round", "draft_overall", "draft_team",
    "contract_type",
]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300, context=SSL_CTX) as resp:
        return resp.read()


def align_contract_type(row) -> str:
    """One contract_type per top-level contract row, from the player-level
    contract_history list. A history entry matches on (year_signed, yrs,
    team, apy); blank when zero or multiple distinct types match (~10% of
    rows) — never guessed."""
    ch = row["contract_history"]
    if ch is None or len(ch) == 0:
        return ""
    types = {
        e["contract_type"]
        for e in ch
        if e.get("year_signed") == row["year_signed"]
        and e.get("yrs") == row["years"]
        and e.get("team") == row["team"]
        and abs((e.get("apy") or 0.0) - (row["apy"] or 0.0)) < 0.01
    }
    return types.pop() if len(types) == 1 else ""


def validate_and_flatten(data: bytes) -> tuple["object", dict]:
    """Parse the parquet payload, sanity-check it, and return
    (flat DataFrame, coverage stats). Raises ValueError on schema drift,
    suspiciously small payloads, or a stale asset (the frozen-CSV failure
    mode this collector exists to avoid)."""
    import pandas as pd  # deferred so tests can import module cheaply

    df = pd.read_parquet(io.BytesIO(data))
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"contracts parquet missing columns: {sorted(missing)}")
    if len(df) < 20_000:
        raise ValueError(f"contracts parquet suspiciously small: {len(df)} rows")
    year_max = int(df["year_signed"].max())
    if year_max < datetime.now(timezone.utc).year - 1:
        raise ValueError(
            f"contracts parquet looks stale: max year_signed {year_max}")

    flat = df.copy()
    flat["contract_type"] = flat.apply(align_contract_type, axis=1)
    flat = flat[FLAT_COLUMNS]
    stats = {
        "rows": int(len(df)),
        "year_signed_max": year_max,
        "gsis_id_coverage": round(float(df["gsis_id"].notna().mean()), 4),
        "contract_type_aligned": round(
            float((flat["contract_type"] != "").mean()), 4),
    }
    return flat, stats


def collect(out_dir: Path = OUT_DIR, date: str | None = None,
            fetcher=fetch) -> dict:
    """Capture a dated, immutable parquet snapshot + flat CSV + manifest."""
    date = date or datetime.now(timezone.utc).date().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_pq = out_dir / f"historical_contracts_{date}.parquet"
    dest_csv = out_dir / f"historical_contracts_{date}.csv"

    if dest_pq.exists() and dest_pq.stat().st_size > 0:
        print(f"skip {dest_pq.name} (immutable snapshot already captured)")
        return {"file": dest_pq.name, "status": "exists_immutable",
                "bytes": dest_pq.stat().st_size}

    data = fetcher(PARQUET_URL)
    flat, stats = validate_and_flatten(data)
    dest_pq.write_bytes(data)
    flat.to_csv(dest_csv, index=False)

    entry = {
        "file": dest_pq.name,
        "flat_csv": dest_csv.name,
        "status": "captured",
        "source_url": PARQUET_URL,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        **stats,
        "notes": ("OverTheCap contracts via nflverse-data `contracts` "
                  "release. One row per contract as signed; value/apy in $M. "
                  "year_signed is YEAR granularity only — no signing date, "
                  "so pre/post-Sept-1 timing of same-year signings is not "
                  "knowable from this source. The release's csv.gz asset is "
                  "frozen at 2022-05-29; the parquet is refreshed daily, "
                  "hence the parquet snapshot + local CSV conversion."),
    }
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest.append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"ok  {dest_pq.name}  {len(data)/1e6:.1f}MB  rows={stats['rows']}  "
          f"year_signed<= {stats['year_signed_max']}  "
          f"gsis coverage {stats['gsis_id_coverage']:.0%}")
    return entry


def latest_snapshot(out_dir: Path = OUT_DIR) -> Path | None:
    """Most recent dated flat-CSV snapshot, if any."""
    snaps = sorted(out_dir.glob("historical_contracts_*.csv"))
    return snaps[-1] if snaps else None


def main() -> int:
    collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
