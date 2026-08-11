#!/usr/bin/env python3
"""Collect historical NFL preseason season-win-total lines into data/raw/vegas/.

Source: SportsOddsHistory (sportsoddshistory.com, now hosted at
covers.com/sportsoddshistory) — "NFL Regular Season Win Total Results by Team"
pages, which publish the preseason Vegas win-total O/U line, the team's actual
regular-season wins, and the over/under/push result, per team per season back
to 1989. Identified as the free team-level history path in
evals/results/odds_channel_scoping.md ("Season win totals: solved, free").

ToS / copyright posture (checked 2026-08-09):
- robots.txt at both sportsoddshistory.com and covers.com/sportsoddshistory
  is `User-agent: * / Disallow:` (empty) — crawling permitted.
- Capture is a polite single pass over exactly three pages (2000s decade,
  2010s decade, current page) with a delay between requests.
- We store only uncopyrightable facts (season, team, line, wins, result) plus
  the source URL — never page text or HTML.
- Juice (over/under prices) is NOT shown on these pages; the columns exist in
  the schema for forward compatibility and are left empty.

Snapshots are dated and immutable (repo hard rule for market data): an
existing dated capture is never overwritten. Provenance goes into the shared
data/raw/vegas/manifest.json alongside the games.csv entries.

Team codes are normalized to current franchise codes (the same convention as
evals/vegas_features.py: SD->LAC, STL->LA, OAK->LV, JAC->JAX), so the emitted
tables join directly against data/processed/labels/*.csv and
data/processed/vegas/team_season_vegas.csv on (season, team).

Emitted:
- data/raw/vegas/win_totals_<capture-date>.csv   (immutable snapshot, facts
  incl. actual wins — raw archive only)
- data/processed/vegas/team_win_totals.csv        (draft-day-legal join table:
  keys (season, team), the preseason line ONLY — no outcome columns, so it
  can never leak future information into a packet)

Usage:
    python scripts/collect_win_totals.py                 # capture + join table
    python scripts/collect_win_totals.py --study         # sanity correlations
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html as htmllib
import json
import math
import re
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "vegas"
PROCESSED_DIR = ROOT / "data" / "processed" / "vegas"
UA = {"User-Agent": "fantasy-alpha-collector/0.1 (research; single-pass archival)"}

BASE = "https://www.covers.com/sportsoddshistory"
# Season columns carry two-digit year headers ('05, '18, '24) that already
# encode the decade; '89-'99 resolve to the 1900s, everything else to 2000+.
PAGES = [
    f"{BASE}/nfl-regular-season-win-total-results-by-team-2000s/",
    f"{BASE}/nfl-regular-season-win-total-results-by-team-2010s/",
    f"{BASE}/nfl-regular-season-win-total-results-by-team/",
]

SEASON_MIN, SEASON_MAX = 2005, 2025

# Era-native code -> current franchise code (mirrors evals/vegas_features.py).
NORMALIZE_TEAM = {"SD": "LAC", "STL": "LA", "OAK": "LV", "JAC": "JAX"}

# Full franchise name (as printed by SportsOddsHistory, which files each
# franchise's whole history under its CURRENT name) -> current franchise code,
# matching nflverse stats_player / labels retroactive coding.
TEAM_NAME_TO_CODE = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA",
    "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    # Era-native names, in case a page ever reverts to them.
    "San Diego Chargers": "LAC",
    "St. Louis Rams": "LA",
    "Oakland Raiders": "LV",
    "Washington Redskins": "WAS",
    "Washington Football Team": "WAS",
}


def normalize_team(name_or_code: str) -> str | None:
    """Full franchise name or era-native code -> current franchise code."""
    s = " ".join(name_or_code.split())
    if s in TEAM_NAME_TO_CODE:
        return TEAM_NAME_TO_CODE[s]
    if s in NORMALIZE_TEAM:
        return NORMALIZE_TEAM[s]
    if s.isupper() and 2 <= len(s) <= 3 and s in set(TEAM_NAME_TO_CODE.values()):
        return s
    return None


# ------------------------------------------------------------------ fetching

def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


# ------------------------------------------------------------------- parsing

_TAG = re.compile(r"<[^>]+>")
_CELL = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.S)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_YY = re.compile(r"^'?(\d{2})$")
# cell payload: "<line> <wins> <O|U|P>", e.g. "7.5 9 O"; future seasons show "?"
_CELL_DATA = re.compile(r"^(\d+(?:\.5)?)\s+(\d+)\s+([OUP])$")


def _cell_text(cell_html: str) -> str:
    txt = htmllib.unescape(_TAG.sub(" ", cell_html))
    return " ".join(txt.replace("\xa0", " ").split())


def _resolve_season(yy: int) -> int:
    return 1900 + yy if yy >= 89 else 2000 + yy


def parse_page(page_html: str) -> list[dict]:
    """Extract (season, team, line, wins, result) facts from one by-team page.

    Only season columns (two-digit year headers) are read; record/streak
    summary columns and empty or future ("?") cells are skipped.
    """
    i = page_html.find("<table")
    j = page_html.find("</table>", i)
    if i < 0 or j < 0:
        raise ValueError("no <table> found in page")
    rows = _ROW.findall(page_html[i:j + 8])
    if not rows:
        raise ValueError("empty table")

    header = [_cell_text(c) for c in _CELL.findall(rows[0])]
    # column index -> season (header cells like "'05"); index 0 is Team
    season_cols: dict[int, int] = {}
    for idx, cell in enumerate(header):
        m = _YY.match(cell.replace("’", "'"))
        if m:
            season_cols[idx] = _resolve_season(int(m.group(1)))
    if not season_cols:
        raise ValueError("no season columns recognized in header")

    out: list[dict] = []
    for row_html in rows[1:]:
        cells = [_cell_text(c) for c in _CELL.findall(row_html)]
        if not cells:
            continue
        team = normalize_team(cells[0])
        if team is None:
            continue  # decade sub-header / summary rows
        for idx, season in season_cols.items():
            if idx >= len(cells):
                continue
            m = _CELL_DATA.match(cells[idx])
            if not m:
                continue  # "?" (future season) or blank (pre-franchise)
            line, repaired = _repair_line(float(m.group(1)))
            rec = {
                "season": season,
                "team": team,
                "team_name": " ".join(cells[0].split()),
                "win_total_line": line,
                "actual_wins": int(m.group(2)),
                "ou_result": m.group(3),
                "over_juice": "",   # not published on these pages
                "under_juice": "",
            }
            if repaired:
                rec["_repaired_from"] = m.group(1)
            out.append(rec)
    return out


def _repair_line(line: float) -> tuple[float, bool]:
    """Fix the source's occasional dropped decimal point.

    SportsOddsHistory sometimes prints a half-win line without its decimal
    (observed: Eagles 2023 shown as "115" for 11.5). Any 'line' >= 18 is
    impossible for an NFL win total; if dividing by 10 lands on a plausible
    x.5 value, that is unambiguously the intended line.
    """
    if line >= 18 and (line / 10) % 1 == 0.5 and 0 < line / 10 < 17:
        return line / 10, True
    return line, False


def validate_records(records: list[dict],
                     season_min: int = SEASON_MIN,
                     season_max: int = SEASON_MAX) -> dict:
    """Coverage + plausibility checks; raises ValueError on a broken scrape."""
    in_range = [r for r in records
                if season_min <= r["season"] <= season_max]
    if not in_range:
        raise ValueError("no records in requested season range")
    by_season: dict[int, set[str]] = {}
    for r in in_range:
        by_season.setdefault(r["season"], set()).add(r["team"])
        if not (0.0 < r["win_total_line"] < 17.0):
            raise ValueError(f"implausible line {r['win_total_line']} ({r})")
        if not (0 <= r["actual_wins"] <= 17):
            raise ValueError(f"implausible wins {r['actual_wins']} ({r})")
        if r["ou_result"] == "P" and r["win_total_line"] != r["actual_wins"]:
            raise ValueError(f"push with line != wins: {r}")
    for season in range(season_min, season_max + 1):
        teams = by_season.get(season, set())
        if len(teams) != 32:
            raise ValueError(
                f"season {season}: expected 32 teams, got {len(teams)}")
    dupes = len(in_range) - len({(r["season"], r["team"]) for r in in_range})
    if dupes:
        raise ValueError(f"{dupes} duplicate (season, team) records")
    return {
        "rows": len(in_range),
        "season_min": min(by_season),
        "season_max": max(by_season),
        "teams_per_season": 32,
    }


# ---------------------------------------------------------------- collection

RAW_FIELDS = ["season", "team", "team_name", "win_total_line", "actual_wins",
              "ou_result", "over_juice", "under_juice", "source_url"]


def collect(out_dir: Path = RAW_DIR, date: str | None = None,
            fetcher=fetch, pages=PAGES, delay: float = 2.0,
            season_min: int = SEASON_MIN, season_max: int = SEASON_MAX) -> dict:
    """Single-pass capture -> immutable dated CSV + manifest entry."""
    date = date or datetime.now(timezone.utc).date().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"win_totals_{date}.csv"

    if dest.exists() and dest.stat().st_size > 0:
        entry = {"file": dest.name, "status": "exists_immutable",
                 "bytes": dest.stat().st_size}
        print(f"skip {dest.name} (immutable snapshot already captured)")
        return entry

    records: list[dict] = []
    for k, url in enumerate(pages):
        if k and delay:
            time.sleep(delay)
        page = fetcher(url).decode("utf-8", errors="replace")
        for rec in parse_page(page):
            rec["source_url"] = url
            records.append(rec)

    records = [r for r in records if season_min <= r["season"] <= season_max]
    repairs = [
        {"season": r["season"], "team": r["team"],
         "printed": r.pop("_repaired_from"), "stored": r["win_total_line"]}
        for r in records if "_repaired_from" in r]
    coverage = validate_records(records, season_min, season_max)
    records.sort(key=lambda r: (r["season"], r["team"]))

    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(records)

    entry = {
        "file": dest.name,
        "status": "captured",
        "source_urls": list(pages),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "bytes": dest.stat().st_size,
        **coverage,
        "line_repairs": repairs,
        "notes": ("SportsOddsHistory (covers.com/sportsoddshistory) preseason "
                  "NFL season-win-total O/U lines + actual wins + O/U/P result "
                  "per (season, team). Facts + source URL only, no page text. "
                  "robots.txt permits crawling; polite single pass. Juice not "
                  "published on source -> over/under_juice empty. Team codes "
                  "normalized to current franchise codes (SD->LAC, STL->LA, "
                  "OAK->LV, JAC->JAX convention)."),
    }
    manifest_path = out_dir / "manifest.json"
    manifest = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest.append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"ok  {dest.name}  {coverage['rows']} rows  seasons "
          f"{coverage['season_min']}-{coverage['season_max']}")
    return entry


# --------------------------------------------------------------- join table

def latest_snapshot(raw_dir: Path = RAW_DIR) -> Path:
    snaps = sorted(raw_dir.glob("win_totals_*.csv"))
    if not snaps:
        raise FileNotFoundError(
            f"no win_totals_*.csv snapshot in {raw_dir}; run this collector")
    return snaps[-1]


JOIN_FIELDS = ["season", "team", "win_total_line"]


def write_join_table(snapshot: Path | None = None,
                     out_dir: Path = PROCESSED_DIR) -> dict:
    """Draft-day-legal join table keyed (season, team).

    The preseason win-total line for season S is posted months before season-S
    drafts, so it is legal in a season-S draft packet. Outcome columns
    (actual_wins, ou_result) are deliberately EXCLUDED — this table can never
    leak season-S results into a season-S feature row.
    """
    snapshot = snapshot or latest_snapshot()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "team_win_totals.csv"
    rows = []
    with open(snapshot, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"season": int(row["season"]), "team": row["team"],
                         "win_total_line": float(row["win_total_line"])})
    rows.sort(key=lambda r: (r["season"], r["team"]))
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOIN_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {dest}")
    return {"rows": len(rows), "path": str(dest)}


# -------------------------------------------------------------------- study

HOLDOUT_SEASON = 2025  # never correlate against holdout-season outcomes

ERAS = [(2005, 2011), (2012, 2018), (2019, 2024)]


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def run_study(snapshot: Path | None = None,
              season_vegas: Path | None = None) -> dict:
    """Sanity cross-checks on the captured lines.

    (a) preseason win-total line vs realized wins, per era — the market
        should predict wins decently and stably.
    (b) preseason win-total line vs the same season's week-1 implied team
        total (from data/processed/vegas/team_season_vegas.csv, nflverse
        game lines) — two independent sources measuring team strength should
        agree strongly; a weak correlation would mean one of them is broken.

    2025 (untouched eval holdout) is EXCLUDED from (a) — its realized wins
    are outcomes. It is included in (b), which compares two preseason-knowable
    quantities and touches no outcomes.
    """
    snapshot = snapshot or latest_snapshot()
    season_vegas = season_vegas or (PROCESSED_DIR / "team_season_vegas.csv")

    lines: dict[tuple[int, str], float] = {}
    wins: dict[tuple[int, str], int] = {}
    with open(snapshot, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["season"]), row["team"])
            lines[key] = float(row["win_total_line"])
            wins[key] = int(row["actual_wins"])

    week1: dict[tuple[int, str], float] = {}
    if season_vegas.exists():
        with open(season_vegas, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("week1_implied_total"):
                    week1[(int(row["season"]), row["team"])] = float(
                        row["week1_implied_total"])

    results: dict = {"eras": {}}
    for lo, hi in ERAS + [(ERAS[0][0], ERAS[-1][1])]:
        label = f"{lo}-{hi}"
        wpairs = [(lines[k], wins[k]) for k in lines
                  if lo <= k[0] <= hi and k[0] != HOLDOUT_SEASON]
        results["eras"][label] = {
            "n_wins": len(wpairs),
            "r_line_vs_wins": _pearson([p[0] for p in wpairs],
                                       [p[1] for p in wpairs]),
        }
    # (b) full range incl. 2025 (both quantities preseason-knowable)
    xpairs = [(lines[k], week1[k]) for k in lines if k in week1]
    results["n_week1"] = len(xpairs)
    results["r_line_vs_week1_implied"] = _pearson(
        [p[0] for p in xpairs], [p[1] for p in xpairs])
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", action="store_true",
                    help="run sanity correlations on the latest snapshot")
    ap.add_argument("--skip-collect", action="store_true",
                    help="don't fetch; use the latest existing snapshot")
    args = ap.parse_args(argv)

    if args.study:
        res = run_study()
        for era, block in res["eras"].items():
            print(f"{era}: r(line, wins) = {block['r_line_vs_wins']:+.3f} "
                  f"(n={block['n_wins']})")
        print(f"line vs week-1 implied total: "
              f"r = {res['r_line_vs_week1_implied']:+.3f} "
              f"(n={res['n_week1']})")
        return 0

    if not args.skip_collect:
        collect()
    write_join_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
