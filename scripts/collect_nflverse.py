#!/usr/bin/env python3
"""Download nflverse historical data releases into data/raw/nflverse/.

Stdlib only. Skips files already on disk, tolerates 404s (some asset/year
combos don't exist), and writes a dated manifest of what landed.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "nflverse"

# (release_tag, filename_template(s) tried in order, year range or None for single file)
TARGETS = [
    # "stats_player" is the current-generation release (one schema, 1999-2025);
    # the legacy "player_stats" tag is frozen at 2024 with a different schema.
    ("stats_player", ["stats_player_week_{y}.csv"], range(1999, 2026)),
    ("snap_counts", ["snap_counts_{y}.csv"], range(2012, 2026)),
    # 2026 included: the live preseason depth chart is a leading indicator NOW
    ("depth_charts", ["depth_charts_{y}.csv"], range(2001, 2027)),
    ("injuries", ["injuries_{y}.csv"], range(2009, 2026)),
    ("combine", ["combine.csv"], None),
    ("draft_picks", ["draft_picks.csv"], None),
]

UA = {"User-Agent": "fantasy-alpha-collector/0.1"}


def fetch(url: str, dest: Path) -> tuple[bool, int]:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False, 404
        raise
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True, len(data)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    failures: list[str] = []

    for tag, templates, years in TARGETS:
        year_list = list(years) if years is not None else [None]
        for y in year_list:
            names = [t.format(y=y) for t in templates]
            dest = OUT / tag / names[0]
            if dest.exists() and dest.stat().st_size > 0:
                manifest.append({"tag": tag, "file": dest.name, "bytes": dest.stat().st_size, "cached": True})
                continue
            got = False
            for name in names:
                ok, size = fetch(f"{BASE}/{tag}/{name}", OUT / tag / name)
                if ok:
                    manifest.append({"tag": tag, "file": name, "bytes": size, "cached": False})
                    print(f"ok  {tag}/{name}  {size/1e6:.1f}MB", flush=True)
                    got = True
                    break
            if not got:
                failures.append(f"{tag}/{names[0]}")
                print(f"404 {tag}/{names[0]}", flush=True)
            time.sleep(0.2)

    summary = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "files": len([m for m in manifest if not m.get("cached")]),
        "cached": len([m for m in manifest if m.get("cached")]),
        "total_bytes": sum(m["bytes"] for m in manifest),
        "failures": failures,
        "manifest": manifest,
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE files={summary['files']} cached={summary['cached']} "
          f"total={summary['total_bytes']/1e6:.0f}MB failures={len(failures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
