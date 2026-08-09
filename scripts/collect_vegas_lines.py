#!/usr/bin/env python3
"""Download nflverse/nfldata historical game lines into data/raw/vegas/.

Source: Lee Sharpe's games.csv (github.com/nflverse/nfldata), which carries
per-game closing `spread_line` and `total_line` back to ~1999, plus scores.
Free, no key, redistributable-for-research.

Stdlib only. Snapshots are dated and immutable: an existing dated capture is
never overwritten (per the repo hard rule for market data). A provenance
manifest records URL, capture time, sha256, byte size, and row/season coverage.

Usage:
    python scripts/collect_vegas_lines.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "vegas"
UA = {"User-Agent": "fantasy-alpha-collector/0.1"}

REQUIRED_COLUMNS = {
    "game_id", "season", "game_type", "week", "gameday",
    "home_team", "away_team", "home_score", "away_score",
    "spread_line", "total_line",
}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def validate_games_csv(data: bytes) -> dict:
    """Sanity-check the payload and return coverage stats.

    Raises ValueError if the schema or line coverage looks broken, so a bad
    upstream change can never silently replace our understanding of the file.
    """
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    cols = set(reader.fieldnames or [])
    missing = REQUIRED_COLUMNS - cols
    if missing:
        raise ValueError(f"games.csv missing expected columns: {sorted(missing)}")

    rows = 0
    with_lines = 0
    seasons: set[int] = set()
    for row in reader:
        rows += 1
        seasons.add(int(row["season"]))
        if row.get("spread_line") and row.get("total_line"):
            with_lines += 1
    if rows < 5000:
        raise ValueError(f"games.csv suspiciously small: {rows} rows")
    if with_lines < 0.8 * rows:
        raise ValueError(
            f"games.csv line coverage too low: {with_lines}/{rows} rows have lines")
    return {
        "rows": rows,
        "rows_with_lines": with_lines,
        "season_min": min(seasons),
        "season_max": max(seasons),
    }


def collect(out_dir: Path = OUT_DIR, date: str | None = None,
            fetcher=fetch) -> dict:
    """Capture a dated, immutable snapshot of games.csv + manifest entry."""
    date = date or datetime.now(timezone.utc).date().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"games_{date}.csv"

    if dest.exists() and dest.stat().st_size > 0:
        entry = {"file": dest.name, "status": "exists_immutable",
                 "bytes": dest.stat().st_size}
        print(f"skip {dest.name} (immutable snapshot already captured)")
        return entry

    data = fetcher(GAMES_URL)
    coverage = validate_games_csv(data)
    dest.write_bytes(data)

    entry = {
        "file": dest.name,
        "status": "captured",
        "source_url": GAMES_URL,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        **coverage,
        "notes": ("nflverse/nfldata games.csv; spread_line is home-team "
                  "expected margin (positive = home favored); total_line is "
                  "the closing game total. Lines back to ~1999. Closing lines "
                  "are per-week (set days before each game), NOT preseason."),
    }

    manifest_path = out_dir / "manifest.json"
    manifest = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest.append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"ok  {dest.name}  {len(data)/1e6:.1f}MB  "
          f"seasons {coverage['season_min']}-{coverage['season_max']}  "
          f"{coverage['rows_with_lines']}/{coverage['rows']} rows with lines")
    return entry


def latest_snapshot(out_dir: Path = OUT_DIR) -> Path | None:
    """Return the most recent dated games snapshot, if any."""
    snaps = sorted(out_dir.glob("games_*.csv"))
    return snaps[-1] if snaps else None


def main() -> int:
    collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
