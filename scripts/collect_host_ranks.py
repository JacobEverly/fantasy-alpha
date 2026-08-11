#!/usr/bin/env python3
"""Daily snapshot of host-platform DEFAULT draft ranks (edge family #1:
host-rank anchoring).

Drafters in ESPN/Yahoo rooms pick from the host's visible default queue, so
(host_rank - sharp ADP) deltas predict in-room falls -- an execution edge,
not a priced signal. This collector archives the ranks those queues render:

  data/raw/host_ranks/<YYYY-MM-DD>/
    espn_kona_player_info.json   ESPN lm-api-reads leaguedefaults payload --
                                 the exact endpoint+view ESPN's own draft room
                                 uses. Per player: draftRanksByRankType
                                 (STANDARD/PPR/ELIMINATION/SUPERFLEX rank +
                                 auction value = the default queue order) plus
                                 ownership ADP/%owned and rankings-by-source.
                                 Public unauthenticated JSON, no key.
    yahoo_players_or_p<N>.json   Yahoo pub-api-ro players collection sorted by
                                 OR (Yahoo overall rank -- the draft-room
                                 default order; the rank IS the sort position)
                                 with /draft_analysis (Yahoo ADP: average_pick,
                                 average_cost, percent_drafted). Same
                                 unauthenticated read-only API Yahoo's public
                                 draft-analysis pages call. 3 pages x 100.
    host_ranks_parsed.json       Compact join-ready table per host:
                                 {name, pos, team, host_rank, host_adp, ...}
    manifest.json                per-source status/items/bytes/timestamps

ToS posture (conservative, same bar as collect_odds.py): both endpoints are
public unauthenticated JSON backing pages/apps the hosts serve to logged-out
users; volume is ~4 requests/host/day with a polite gap and an identifying UA.
Deliberately NOT captured: Yahoo HTML pages (scraping-unfriendly ToS -- the
pub-api JSON makes it unnecessary), ESPN/Yahoo authenticated league endpoints,
Sleeper (no public default-ranks endpoint exists as of 2026-08;
players-dump search_rank is a search-relevance proxy, not the draft queue --
see evals/results/host_ranks_scoping.md).

Stdlib only. Append-only and idempotent per day like archive_evidence.py:
non-empty files are never overwritten, re-runs only fill gaps, one host
failing never kills the run. Runs standalone (python
scripts/collect_host_ranks.py) or as the "host_ranks" source inside
archive_evidence (day date is taken from the passed day_dir so both archives
stay on the same calendar day; files still land under data/raw/host_ranks/).
"""
from __future__ import annotations

import json
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "host_ranks"

SEASON = 2026
ESPN_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/"
    "segments/0/leaguedefaults/3?view=kona_player_info"
)
ESPN_LIMIT = 500            # one call covers the draftable pool
ESPN_FILTER = {"players": {"limit": ESPN_LIMIT,
                           "sortDraftRanks": {"sortPriority": 100,
                                              "sortAsc": True, "value": "PPR"}}}
# ESPN defaultPositionId -> position (fantasy-relevant subset)
ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DEF"}

YAHOO_URL = (
    "https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2/game/nfl/"
    "players;start={start};count={count};sort=OR/draft_analysis?format=json"
)
YAHOO_PAGE = 100            # verified max-safe page size
YAHOO_PAGES = 3             # top 300 by Yahoo overall rank

SLEEP = 0.5
UA = {"User-Agent": "fantasy-alpha-archiver/0.1"}


def _ssl_context() -> ssl.SSLContext:
    """python.org macOS builds may ship an empty trust store; fall back."""
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats().get("x509_ca") and Path("/etc/ssl/cert.pem").exists():
        ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    return ctx


SSL_CTX = _ssl_context()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch(url: str, *, headers: dict | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
        return resp.read()


def have(path: Path) -> bool:
    """Append-only guard: a non-empty file is immutable evidence."""
    return path.exists() and path.stat().st_size > 0


def write_json(path: Path, obj, indent: int | None = None) -> int:
    data = json.dumps(obj, indent=indent).encode()
    path.write_bytes(data)
    return len(data)


# ---------------------------------------------------------------- ESPN

def parse_espn(payload: dict) -> list[dict]:
    """kona_player_info payload -> compact rank rows (queue order = PPR rank)."""
    rows = []
    for entry in payload.get("players") or []:
        p = entry.get("player") or {}
        ranks = {rt: d.get("rank") for rt, d in
                 (p.get("draftRanksByRankType") or {}).items()
                 if isinstance(d, dict) and d.get("rank") is not None}
        if not ranks:
            continue
        own = p.get("ownership") or {}
        rows.append({
            "espn_id": p.get("id"),
            "name": p.get("fullName"),
            "pos": ESPN_POS.get(p.get("defaultPositionId"),
                                str(p.get("defaultPositionId"))),
            "pro_team_id": p.get("proTeamId"),
            "host_rank_ppr": ranks.get("PPR"),
            "host_rank_standard": ranks.get("STANDARD"),
            "ranks": ranks,
            "auction_value_ppr": (p.get("draftRanksByRankType") or {})
                                 .get("PPR", {}).get("auctionValue"),
            "espn_adp": own.get("averageDraftPosition"),
            "pct_owned": own.get("percentOwned"),
        })
    rows.sort(key=lambda r: (r["host_rank_ppr"] is None,
                             r["host_rank_ppr"], r["name"] or ""))
    return rows


def capture_espn(day_dir: Path) -> tuple[dict, list[dict]]:
    dest = day_dir / "espn_kona_player_info.json"
    fetched = 0
    if have(dest):
        payload = json.loads(dest.read_text())
    else:
        url = ESPN_URL.format(season=SEASON)
        payload = json.loads(fetch(
            url, headers={"X-Fantasy-Filter": json.dumps(ESPN_FILTER)},
            timeout=90))
        if not payload.get("players"):
            raise ValueError("espn: empty players list")
        payload["_source"] = url
        payload["_filter"] = ESPN_FILTER
        payload["_captured_at"] = utcnow()
        write_json(dest, payload)
        fetched = 1
    rows = parse_espn(payload)
    return {"status": "ok" if fetched else "cached", "items": len(rows),
            "bytes": dest.stat().st_size, "captured_at": utcnow()}, rows


# ---------------------------------------------------------------- Yahoo

def _flatten(fragments: list) -> dict:
    """Yahoo API entities are lists of single-key dicts (and stray lists)."""
    flat: dict = {}
    for item in fragments:
        if isinstance(item, dict):
            flat.update(item)
    return flat


def parse_yahoo(pages: list[dict]) -> list[dict]:
    """Yahoo players pages (sort=OR) -> rank rows. Yahoo never returns the
    rank number itself: the OR sort position IS the host rank, so rank is
    assigned from global position across pages (pages must be in start order)."""
    rows = []
    rank = 0
    for page in pages:
        game = page.get("fantasy_content", {}).get("game", [])
        players = next((g["players"] for g in game
                        if isinstance(g, dict) and "players" in g), {})
        for i in range(int(players.get("count") or 0)):
            player = players.get(str(i), {}).get("player") or []
            meta = _flatten(player[0]) if player else {}
            da = _flatten(next((f.get("draft_analysis", []) for f in player[1:]
                                if isinstance(f, dict) and "draft_analysis" in f),
                               []))
            rank += 1
            name = (meta.get("name") or {}).get("full")
            if not name:
                continue

            def _num(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None

            rows.append({
                "yahoo_id": meta.get("player_id"),
                "name": name,
                "pos": meta.get("display_position"),
                "team": meta.get("editorial_team_abbr"),
                "host_rank": rank,
                "yahoo_adp": _num(da.get("average_pick")),
                "yahoo_avg_cost": _num(da.get("average_cost")),
                "pct_drafted": _num(da.get("percent_drafted")),
                "status": meta.get("status"),
            })
    return rows


def capture_yahoo(day_dir: Path) -> tuple[dict, list[dict]]:
    pages: list[dict] = []
    failures: list[dict] = []
    fetched = 0
    for n in range(YAHOO_PAGES):
        dest = day_dir / f"yahoo_players_or_p{n}.json"
        if have(dest):
            pages.append(json.loads(dest.read_text()))
            continue
        url = YAHOO_URL.format(start=n * YAHOO_PAGE, count=YAHOO_PAGE)
        try:
            payload = json.loads(fetch(url, timeout=60))
            game = payload.get("fantasy_content", {}).get("game", [])
            if not any(isinstance(g, dict) and "players" in g for g in game):
                raise ValueError("no players collection in response")
        except Exception as e:  # noqa: BLE001  tolerate per page
            failures.append({"page": n, "url": url, "error": str(e)})
            break  # later pages would corrupt rank numbering
        payload["_source"] = url
        payload["_captured_at"] = utcnow()
        write_json(dest, payload)
        pages.append(payload)
        fetched += 1
        time.sleep(SLEEP)
    rows = parse_yahoo(pages)
    status = ("failed" if not pages else
              "partial" if failures else ("ok" if fetched else "cached"))
    return {"status": status, "items": len(rows), "pages": len(pages),
            "bytes": sum((day_dir / f"yahoo_players_or_p{n}.json").stat().st_size
                         for n in range(len(pages))),
            "failures": failures, "captured_at": utcnow()}, rows


# ---------------------------------------------------------------- main

def archive_host_ranks(day_dir: Path | None = None) -> dict:
    """Entry point for archive_evidence.py (and standalone runs). day_dir, when
    passed by the evidence archiver, only supplies the calendar day -- output
    always lands in data/raw/host_ranks/<day>/."""
    day = day_dir.name if day_dir is not None else \
        datetime.now().astimezone().strftime("%Y-%m-%d")
    out_dir = OUT / day
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: dict[str, dict] = {}
    parsed: dict[str, list[dict]] = {}
    for host, fn in (("espn", capture_espn), ("yahoo", capture_yahoo)):
        try:
            sources[host], parsed[host] = fn(out_dir)
        except Exception as e:  # noqa: BLE001  one host never kills the run
            sources[host] = {"status": "failed", "items": 0, "bytes": 0,
                             "error": f"{type(e).__name__}: {e}"}
            parsed[host] = []

    parsed_dest = out_dir / "host_ranks_parsed.json"
    if not have(parsed_dest) and any(parsed.values()):
        write_json(parsed_dest, {"_captured_at": utcnow(), "date": day,
                                 "espn": parsed["espn"], "yahoo": parsed["yahoo"]})

    total = sum(s.get("bytes", 0) for s in sources.values())
    write_json(out_dir / "manifest.json", {
        "date": day, "captured_at": utcnow(), "total_bytes": total,
        "sources": sources,
    }, indent=2)

    failed = [h for h, s in sources.items() if s["status"] == "failed"]
    return {"status": ("failed" if len(failed) == 2 else
                       "partial" if failed else
                       ("cached" if all(s["status"] == "cached"
                                        for s in sources.values()) else "ok")),
            "items": sum(s.get("items", 0) for s in sources.values()),
            "bytes": total, "hosts": sources, "captured_at": utcnow()}


def main() -> int:
    entry = archive_host_ranks()
    for host, s in entry["hosts"].items():
        err = f"  error={s['error']}" if s.get("error") else ""
        print(f"{s['status']:<7} {host}  items={s['items']} bytes={s['bytes']}{err}",
              flush=True)
    print(f"DONE host_ranks status={entry['status']} items={entry['items']} "
          f"total={entry['bytes'] / 1e6:.1f}MB")
    return 0 if entry["status"] in ("ok", "cached", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
