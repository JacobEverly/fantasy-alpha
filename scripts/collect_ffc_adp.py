#!/usr/bin/env python3
"""Collect historical ADP from the FantasyFootballCalculator public API.

Grid: format x team-count x year -> data/raw/ffc_adp/{format}_{teams}_{year}.json
The API silently coerces unsupported team counts (observed: half-ppr 10 -> 12),
so responses are deduped by the RETURNED meta, not the requested params.
Includes per-player ADP mean/stdev/high/low — needed to fit DraftGym opponents.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://fantasyfootballcalculator.com/api/v1/adp"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "ffc_adp"

FORMATS = ["standard", "ppr", "half-ppr", "2qb"]
TEAMS = [8, 10, 12, 14]
YEARS = range(2005, 2027)  # probe confirmed real pools back to 2007; older years skip gracefully
UA = {"User-Agent": "fantasy-alpha-collector/0.1"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seen_meta: set[tuple] = set()
    saved, skipped, failures = 0, 0, []

    for fmt in FORMATS:
        for teams in TEAMS:
            for year in YEARS:
                dest = OUT / f"{fmt}_{teams}t_{year}.json"
                if dest.exists():
                    skipped += 1
                    continue
                url = f"{BASE}/{fmt}?teams={teams}&year={year}"
                try:
                    req = urllib.request.Request(url, headers=UA)
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        payload = json.loads(resp.read())
                except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as e:
                    failures.append({"url": url, "error": str(e)})
                    time.sleep(0.4)
                    continue

                meta = payload.get("meta", {})
                players = payload.get("players") or []
                key = (fmt, meta.get("teams"), year)
                if payload.get("status") != "Success" or not players:
                    failures.append({"url": url, "error": f"empty/status={payload.get('status')}"})
                elif key in seen_meta:
                    skipped += 1  # coerced duplicate of an already-captured pool
                else:
                    seen_meta.add(key)
                    payload["_requested"] = {"format": fmt, "teams": teams, "year": year}
                    payload["_collected_at"] = datetime.now(timezone.utc).isoformat()
                    dest.write_text(json.dumps(payload))
                    saved += 1
                    print(f"ok  {fmt} {teams}t {year}: {len(players)} players "
                          f"(returned teams={meta.get('teams')}, drafts={meta.get('total_drafts')})", flush=True)
                time.sleep(0.4)

    (OUT / "manifest.json").write_text(json.dumps({
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "saved": saved, "skipped": skipped, "failures": failures,
    }, indent=2))
    print(f"DONE saved={saved} skipped={skipped} failures={len(failures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
