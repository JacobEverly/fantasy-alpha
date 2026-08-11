#!/usr/bin/env python3
"""Append-only daily archive of fantasy-relevant NFL evidence.

Snapshots live sources into data/raw/evidence/<YYYY-MM-DD>/ so the 2026
season's information trail is timestamped by us instead of reconstructed
later (docs/evidence-archive-feasibility.md):

  pft_rss.xml + pft_rss.json         ProFootballTalk RSS, raw + parsed
  ffc_adp_<format>.json              FFC current ADP (standard/ppr/half-ppr/2qb)
  sleeper_trending_<add|drop>.json   Sleeper trending players (public API)
  sleeper_news_internal/<id>.json    Sleeper GraphQL news for the FFC PPR
                                     top-200 -- INTERNAL RESEARCH ONLY:
                                     unlicensed third-party blurbs, never
                                     republish (see feasibility doc)
  rotoworld_player_news.html         NBC Rotoworld player-news page
  odds/                              NFL betting odds (scripts/collect_odds.py:
                                     ESPN futures free; The Odds API sources
                                     skipped gracefully without ODDS_API_KEY)
  (host_ranks)                       ESPN/Yahoo default draft ranks -- runs as
                                     a source here but writes to
                                     data/raw/host_ranks/<date>/ (see
                                     scripts/collect_host_ranks.py)
  manifest.json                      per-source status/items/bytes/timestamps

Stdlib only. Idempotent per day: existing non-empty files are never
overwritten, so a re-run only fills gaps. One source failing never kills the
run. Player-id mapping uses the research repo's sleeper_players CSV when
present, falling back to a one-time cached pull of the full Sleeper players
dump (data/raw/evidence/sleeper_players_cache.json).
"""
from __future__ import annotations

import csv
import json
import ssl
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evals.build_breakout_labels import norm_name  # noqa: E402  canonical name matcher
from scripts.collect_host_ranks import archive_host_ranks  # noqa: E402  host default ranks
from scripts.collect_odds import archive_odds  # noqa: E402  betting-odds sub-archiver

OUT = ROOT / "data" / "raw" / "evidence"
PLAYERS_CACHE = OUT / "sleeper_players_cache.json"
ID_CSV_DIRS = [
    ROOT / "data" / "processed",
    ROOT / "fantasy-football-alpha-2026" / "data" / "processed",
]

PFT_RSS_URL = "https://profootballtalk.nbcsports.com/feed/"
FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?teams=12&year={year}"
FFC_FORMATS = ["standard", "ppr", "half-ppr", "2qb"]
FFC_YEAR = 2026
TRENDING_URL = "https://api.sleeper.app/v1/players/nfl/trending/{kind}"
PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
GRAPHQL_URL = "https://sleeper.com/graphql"
ROTOWORLD_URL = "https://www.nbcsports.com/fantasy/football/player-news"

NEWS_TOP_N = 200   # FFC PPR top-N players get per-player news pulls
NEWS_LIMIT = 30    # news items requested per player
NEWS_SLEEP = 0.3   # polite gap between Sleeper requests
FANTASY_POS = {"QB", "RB", "WR", "TE", "K"}
UA = {"User-Agent": "fantasy-alpha-archiver/0.1"}


def _ssl_context() -> ssl.SSLContext:
    """python.org macOS builds ship an empty default trust store unless the user
    ran 'Install Certificates.command'; fall back to the system bundle."""
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats().get("x509_ca") and Path("/etc/ssl/cert.pem").exists():
        ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    return ctx


SSL_CTX = _ssl_context()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch(url: str, *, data: bytes | None = None, headers: dict | None = None,
          timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
        return resp.read()


def have(path: Path) -> bool:
    """Append-only guard: a non-empty file is immutable evidence, never redone."""
    return path.exists() and path.stat().st_size > 0


def write_json(path: Path, obj, indent: int | None = None) -> int:
    data = json.dumps(obj, indent=indent).encode()
    path.write_bytes(data)
    return len(data)


def source_status(fetched: int, present: int, failures: list) -> str:
    if present == 0:
        return "failed"
    if failures:
        return "partial"
    return "ok" if fetched else "cached"


# ---------------------------------------------------------------- PFT RSS

def parse_rss(raw: bytes) -> list[dict]:
    channel = ET.fromstring(raw).find("channel")
    items = [] if channel is None else channel.findall("item")
    return [{
        "title": (it.findtext("title") or "").strip(),
        "link": (it.findtext("link") or "").strip(),
        "pubDate": (it.findtext("pubDate") or "").strip(),
        "description": (it.findtext("description") or "").strip(),
    } for it in items]


def archive_pft(day_dir: Path) -> dict:
    xml_dest = day_dir / "pft_rss.xml"
    json_dest = day_dir / "pft_rss.json"
    fetched = 0
    if have(xml_dest):
        raw = xml_dest.read_bytes()
    else:
        raw = fetch(PFT_RSS_URL)
        xml_dest.write_bytes(raw)
        fetched = 1

    entry: dict = {"captured_at": utcnow()}
    if have(json_dest):
        entry["items"] = len(json.loads(json_dest.read_text()).get("items", []))
    else:
        try:
            items = parse_rss(raw)
            write_json(json_dest, {"_source": PFT_RSS_URL, "_captured_at": utcnow(),
                                   "items": items})
            entry["items"] = len(items)
            fetched = fetched or 1  # reparsed from cached raw still counts as new output
        except ET.ParseError as e:
            entry["items"] = 0
            entry["error"] = f"raw xml saved; parse failed: {e}"
    entry["bytes"] = sum(p.stat().st_size for p in (xml_dest, json_dest) if p.exists())
    entry["status"] = "partial" if entry.get("error") else ("ok" if fetched else "cached")
    return entry


# ---------------------------------------------------------------- FFC ADP

def archive_ffc(day_dir: Path) -> dict:
    files: dict[str, int] = {}
    failures: list[dict] = []
    fetched = items = total = 0
    for fmt in FFC_FORMATS:
        dest = day_dir / f"ffc_adp_{fmt}.json"
        if have(dest):
            payload = json.loads(dest.read_text())
        else:
            url = FFC_URL.format(fmt=fmt, year=FFC_YEAR)
            try:
                payload = json.loads(fetch(url))
                if payload.get("status") != "Success" or not payload.get("players"):
                    raise ValueError(f"empty/status={payload.get('status')}")
            except Exception as e:  # noqa: BLE001  tolerate per format
                failures.append({"format": fmt, "error": str(e)})
                continue
            payload["_source"] = url
            payload["_captured_at"] = utcnow()
            write_json(dest, payload)
            fetched += 1
            time.sleep(NEWS_SLEEP)
        n = len(payload.get("players") or [])
        files[dest.name] = n
        items += n
        total += dest.stat().st_size
    return {"status": source_status(fetched, len(files), failures), "items": items,
            "bytes": total, "files": files, "failures": failures, "captured_at": utcnow()}


# ---------------------------------------------------------------- Sleeper trending

def archive_trending(day_dir: Path) -> dict:
    failures: list[dict] = []
    fetched = items = total = present = 0
    for kind in ("add", "drop"):
        dest = day_dir / f"sleeper_trending_{kind}.json"
        if have(dest):
            items += len(json.loads(dest.read_text()).get("items", []))
        else:
            url = TRENDING_URL.format(kind=kind)
            try:
                lst = json.loads(fetch(url, timeout=30))
            except Exception as e:  # noqa: BLE001  tolerate per kind
                failures.append({"kind": kind, "error": str(e)})
                continue
            write_json(dest, {"_source": url, "_captured_at": utcnow(), "items": lst})
            fetched += 1
            items += len(lst)
            time.sleep(NEWS_SLEEP)
        present += 1
        total += dest.stat().st_size
    return {"status": source_status(fetched, present, failures), "items": items,
            "bytes": total, "failures": failures, "captured_at": utcnow()}


# ---------------------------------------------------------------- Sleeper news (internal)

def latest_ids_csv() -> Path | None:
    hits = [p for d in ID_CSV_DIRS for p in d.glob("sleeper_players_*.csv")]
    return max(hits, default=None, key=lambda p: p.name)


def id_rows_from_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [{"id": r["sleeper_id"], "name": r["full_name"], "pos": r["position"],
                 "team": r.get("team") or "", "active": r.get("active", "True") == "True"}
                for r in csv.DictReader(f) if r.get("sleeper_id") and r.get("full_name")]


def id_rows_from_dump(dump: dict) -> list[dict]:
    rows = []
    for pid, p in dump.items():
        if not isinstance(p, dict) or p.get("position") not in FANTASY_POS:
            continue
        name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        if name:
            rows.append({"id": str(pid), "name": name, "pos": p["position"],
                         "team": p.get("team") or "", "active": bool(p.get("active"))})
    return rows


def build_id_map(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    id_map: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        id_map[(norm_name(r["name"]), r["pos"])].append(r)
    return id_map


def match_id(id_map: dict, name: str, pos: str, team: str) -> str | None:
    cands = id_map.get((norm_name(name), {"PK": "K"}.get(pos, pos)), [])
    if len(cands) > 1:
        team_hits = [c for c in cands if team and c.get("team") == team]
        cands = team_hits or [c for c in cands if c.get("active", True)] or cands
    return cands[0]["id"] if cands else None


def load_players_dump() -> dict:
    if have(PLAYERS_CACHE):
        return json.loads(PLAYERS_CACHE.read_text())
    raw = fetch(PLAYERS_URL, timeout=180)
    dump = json.loads(raw)  # validate before caching
    PLAYERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PLAYERS_CACHE.write_bytes(raw)
    print(f"ok  sleeper players dump cached  {len(raw) / 1e6:.1f}MB", flush=True)
    return dump


def fetch_player_news(pid: str) -> list[dict]:
    query = ('query { get_player_news(sport: "nfl", player_id: "%s", limit: %d) '
             '{ metadata published source source_key player_id } }' % (pid, NEWS_LIMIT))
    raw = fetch(GRAPHQL_URL, data=json.dumps({"query": query}).encode(),
                headers={"Content-Type": "application/json"}, timeout=30)
    payload = json.loads(raw)
    if payload.get("errors"):
        raise RuntimeError(f"graphql: {payload['errors'][:1]}")
    return payload["data"]["get_player_news"]


def load_top_players(day_dir: Path) -> list[dict]:
    src = day_dir / "ffc_adp_ppr.json"
    if have(src):
        payload = json.loads(src.read_text())
    else:  # ffc source failed earlier today; fetch just the pool we need
        payload = json.loads(fetch(FFC_URL.format(fmt="ppr", year=FFC_YEAR)))
    players = sorted(payload.get("players") or [], key=lambda p: float(p["adp"]))
    return players[:NEWS_TOP_N]


def archive_sleeper_news(day_dir: Path) -> dict:
    news_dir = day_dir / "sleeper_news_internal"
    news_dir.mkdir(parents=True, exist_ok=True)
    top = load_top_players(day_dir)

    csv_path = latest_ids_csv()
    id_map = build_id_map(id_rows_from_csv(csv_path)) if csv_path else {}

    # Resolve sleeper ids: CSV first; team DEFs use the team abbrev by Sleeper
    # convention; anything left (e.g. kickers) retried against the full dump.
    resolved: list[tuple[dict, str]] = []
    missing: list[dict] = []
    for p in top:
        if p.get("position") == "DEF":
            pid = p.get("team") or ""
            resolved.append((p, pid)) if pid else missing.append(p)
        else:
            pid = match_id(id_map, p["name"], p["position"], p.get("team") or "")
            resolved.append((p, pid)) if pid else missing.append(p)
    dump_used, dump_error = False, None
    if missing:
        try:
            dump_map = build_id_map(id_rows_from_dump(load_players_dump()))
            dump_used = True
            still = []
            for p in missing:
                pid = match_id(dump_map, p["name"], p["position"], p.get("team") or "")
                resolved.append((p, pid)) if pid else still.append(p)
            missing = still
        except Exception as e:  # noqa: BLE001  dump is best-effort gap fill
            dump_error = str(e)

    fetched = cached = items = total = 0
    failures: list[dict] = []
    for i, (p, pid) in enumerate(resolved, 1):
        dest = news_dir / f"{pid}.json"
        if have(dest):
            cached += 1
            items += len(json.loads(dest.read_text()).get("news") or [])
            total += dest.stat().st_size
            continue
        try:
            news = fetch_player_news(pid)
        except Exception as e:  # noqa: BLE001  tolerate per player
            failures.append({"player": p["name"], "sleeper_id": pid, "error": str(e)})
            time.sleep(NEWS_SLEEP)
            continue
        total += write_json(dest, {
            "_source": GRAPHQL_URL, "_captured_at": utcnow(),
            "_player": {"name": p["name"], "position": p["position"],
                        "team": p.get("team"), "ffc_adp": p.get("adp"), "sleeper_id": pid},
            "news": news})
        fetched += 1
        items += len(news)
        if fetched % 50 == 0:
            print(f"..  sleeper news {i}/{len(resolved)} players", flush=True)
        time.sleep(NEWS_SLEEP)

    unmatched = [{"name": p["name"], "position": p["position"], "team": p.get("team")}
                 for p in missing]
    write_json(news_dir / "manifest.json", {
        "captured_at": utcnow(), "top_n": NEWS_TOP_N, "news_limit": NEWS_LIMIT,
        "id_source": {"csv": str(csv_path) if csv_path else None,
                      "dump_used": dump_used, "dump_error": dump_error},
        "players_resolved": len(resolved), "fetched": fetched, "cached": cached,
        "news_items": items, "unmatched": unmatched, "failures": failures,
        "internal_research_only": True,
    }, indent=2)
    return {"status": source_status(fetched, fetched + cached, failures), "items": items,
            "bytes": total, "players": len(resolved), "fetched": fetched, "cached": cached,
            "unmatched": len(unmatched), "failures": failures, "captured_at": utcnow()}


# ---------------------------------------------------------------- Rotoworld page

def archive_rotoworld(day_dir: Path) -> dict:
    dest = day_dir / "rotoworld_player_news.html"
    if have(dest):
        return {"status": "cached", "items": 1, "bytes": dest.stat().st_size}
    raw = fetch(ROTOWORLD_URL, timeout=90)
    dest.write_bytes(raw)
    return {"status": "ok", "items": 1, "bytes": len(raw), "captured_at": utcnow()}


# ---------------------------------------------------------------- main

SOURCES = [
    ("pft_rss", archive_pft),
    ("ffc_adp", archive_ffc),           # before sleeper_news: it feeds the top-200
    ("sleeper_trending", archive_trending),
    ("sleeper_news_internal", archive_sleeper_news),
    ("rotoworld_html", archive_rotoworld),
    ("odds", archive_odds),
    ("host_ranks", archive_host_ranks),  # writes to data/raw/host_ranks/<day>/
]


def main() -> int:
    day = datetime.now().astimezone().strftime("%Y-%m-%d")
    day_dir = OUT / day
    day_dir.mkdir(parents=True, exist_ok=True)
    started = utcnow()

    sources: dict[str, dict] = {}
    for name, fn in SOURCES:
        try:
            sources[name] = fn(day_dir)
        except Exception as e:  # noqa: BLE001  one source never kills the run
            sources[name] = {"status": "failed", "items": 0, "bytes": 0,
                             "error": f"{type(e).__name__}: {e}"}
        s = sources[name]
        err = f"  error={s['error']}" if s.get("error") else ""
        print(f"{s['status']:<7} {name}  items={s['items']} bytes={s['bytes']}{err}", flush=True)

    total = sum(s["bytes"] for s in sources.values())
    write_json(day_dir / "manifest.json", {
        "date": day, "started_at": started, "finished_at": utcnow(),
        "total_bytes": total, "sources": sources,
    }, indent=2)
    failed = [n for n, s in sources.items() if s["status"] == "failed"]
    print(f"DONE {day} total={total / 1e6:.1f}MB failed={failed if failed else 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
