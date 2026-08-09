#!/usr/bin/env python3
"""One-time historical backfill of Sleeper GraphQL player news (2018->2025).

Pulls the full per-player news history (limit 1000) for every fantasy-relevant
QB/RB/WR/TE in the cached Sleeper players dump and stores it under
data/raw/evidence/sleeper_backfill_internal/<player_id>.json -- one file per
player that has at least one item.

INTERNAL RESEARCH ONLY: these blurbs are third-party licensed text (RotoWire /
RotoBaller / FantasyPros) obtained via Sleeper's undocumented endpoint
(docs/evidence-archive-feasibility.md). Never republish or ship in-product;
a direct vendor license replaces this corpus before any commercial launch.

Player selection is inclusive rather than clever: active players, plus anyone
whose latest Sleeper news timestamp (news_updated) falls on/after 2018-01-01 --
the union of "still in the league" and "Sleeper has news for them in the
backfill window". Inactive players with no 2018+ news would only return empty.

Stdlib only. Resumable: existing non-empty per-player files are immutable
evidence and are skipped, so an interrupted run just continues where it left
off. One player failing never kills the run (one retry, then recorded in
MANIFEST.json). Expected runtime 20-50 min at the 0.35s rate limit.
"""
from __future__ import annotations

import json
import ssl
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evals.names import norm_name  # noqa: E402  canonical cross-source join key

OUT = ROOT / "data" / "raw" / "evidence"
PLAYERS_CACHE = OUT / "sleeper_players_cache.json"
BACKFILL_DIR = OUT / "sleeper_backfill_internal"

GRAPHQL_URL = "https://sleeper.com/graphql"
NEWS_LIMIT = 1000   # full history per player (feasibility scan: Josh Allen = 933)
NEWS_SLEEP = 0.35   # polite gap between Sleeper requests
FANTASY_POS = {"QB", "RB", "WR", "TE"}
NEWS_CUTOFF_MS = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
HIST_YEARS = range(2017, 2027)  # coverage histogram buckets
PROGRESS_EVERY = 100
UA = {"User-Agent": "fantasy-alpha-archiver/0.1"}

README = """# sleeper_backfill_internal -- INTERNAL RESEARCH ONLY

Historical Sleeper GraphQL player-news backfill (target window 2018->2025),
produced by scripts/backfill_sleeper_news.py (resumable; re-run to fill gaps).

This corpus is third-party licensed text (RotoWire / RotoBaller / FantasyPros
blurbs) obtained sideways via Sleeper's undocumented GraphQL endpoint -- see
docs/evidence-archive-feasibility.md. The standing rule applies:

- internal training/eval research only
- never ship, republish, or surface this text in-product
- a RotoWire (or equivalent) license replaces it before any commercial launch
- every stored item keeps its source attribution and capture timestamp

One `<player_id>.json` per player with >=1 news item. `MANIFEST.json` carries
run totals, failures, and the per-year coverage histogram.
"""


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


# ---------------------------------------------------------------- player universe

def select_players(dump: dict) -> list[dict]:
    """QB/RB/WR/TE with a plausible 2018+ footprint: active, or Sleeper's own
    latest-news timestamp lands in the backfill window."""
    rows = []
    for pid, p in dump.items():
        if not isinstance(p, dict) or p.get("position") not in FANTASY_POS:
            continue
        if not (p.get("active") or (p.get("news_updated") or 0) >= NEWS_CUTOFF_MS):
            continue
        name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        rows.append({"id": str(pid), "name": name, "pos": p["position"],
                     "team": p.get("team") or "", "status": p.get("status"),
                     "active": bool(p.get("active"))})
    rows.sort(key=lambda r: (len(r["id"]), r["id"]))  # numeric-string order, stable resume
    return rows


def split_resume(players: list[dict], news_dir: Path) -> tuple[list[dict], list[dict]]:
    """Partition into (todo, resumed): players whose per-player file already
    exists non-empty were captured by a previous (interrupted) run."""
    todo: list[dict] = []
    resumed: list[dict] = []
    for r in players:
        (resumed if have(news_dir / f"{r['id']}.json") else todo).append(r)
    return todo, resumed


# ---------------------------------------------------------------- GraphQL fetch

def fetch_player_news(pid: str) -> list[dict]:
    query = ('query { get_player_news(sport: "nfl", player_id: "%s", limit: %d) '
             '{ metadata published source source_key player_id } }' % (pid, NEWS_LIMIT))
    raw = fetch(GRAPHQL_URL, data=json.dumps({"query": query}).encode(),
                headers={"Content-Type": "application/json"}, timeout=60)
    payload = json.loads(raw)
    if payload.get("errors"):
        raise RuntimeError(f"graphql: {payload['errors'][:1]}")
    return payload["data"]["get_player_news"]


def fetch_with_retry(pid: str) -> list[dict]:
    try:
        return fetch_player_news(pid)
    except Exception:  # noqa: BLE001  one retry per player, then give up
        time.sleep(NEWS_SLEEP * 4)
        return fetch_player_news(pid)


# ---------------------------------------------------------------- coverage

def year_of(published_ms) -> int | None:
    """Calendar year (UTC) of a Sleeper ms timestamp; None for missing/junk."""
    try:
        return datetime.fromtimestamp(int(published_ms) / 1000, tz=timezone.utc).year
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def bucket_years(items: list[dict]) -> Counter:
    """News items -> Counter keyed by int year ('unknown' for unparseable)."""
    years: Counter = Counter()
    for it in items:
        y = year_of(it.get("published"))
        years[y if y is not None else "unknown"] += 1
    return years


def coverage(news_dir: Path) -> dict:
    """Aggregate every saved per-player file (incl. prior runs) into the
    year histogram, source counts, and totals."""
    years: Counter = Counter()
    sources: Counter = Counter()
    items = size = files = 0
    for path in sorted(news_dir.glob("*.json")):
        if path.name == "MANIFEST.json":
            continue
        news = json.loads(path.read_text()).get("news") or []
        files += 1
        items += len(news)
        size += path.stat().st_size
        years.update(bucket_years(news))
        sources.update((it.get("source") or "unknown") for it in news)
    return {"files": files, "items": items, "bytes": size,
            "years": dict(years), "top_sources": sources.most_common(5)}


def print_coverage(cov: dict) -> None:
    years = cov["years"]
    peak = max(years.values(), default=1)
    print("news items by calendar year (published, UTC):")
    for y in HIST_YEARS:
        n = years.get(y, 0)
        print(f"  {y}  {n:>7,}  {'#' * round(n / peak * 40)}")
    for k, v in sorted((k, v) for k, v in years.items()
                       if not (isinstance(k, int) and k in HIST_YEARS)):
        print(f"  {k}  {v:>7,}")
    print("top sources: " + ", ".join(f"{s}={n:,}" for s, n in cov["top_sources"]))


# ---------------------------------------------------------------- main

def main() -> int:
    players = select_players(json.loads(PLAYERS_CACHE.read_text()))
    BACKFILL_DIR.mkdir(parents=True, exist_ok=True)
    if not have(BACKFILL_DIR / "README.md"):
        (BACKFILL_DIR / "README.md").write_text(README)
    todo, resumed = split_resume(players, BACKFILL_DIR)
    started = utcnow()
    print(f"backfill: {len(players)} players selected, {len(resumed)} already on disk, "
          f"{len(todo)} to fetch (~{len(todo) * NEWS_SLEEP / 60:.0f} min floor)", flush=True)

    saved = empty = 0
    failures: list[dict] = []
    for i, r in enumerate(todo, 1):
        try:
            news = fetch_with_retry(r["id"])
        except Exception as e:  # noqa: BLE001  tolerate per player
            failures.append({"player": r["name"], "sleeper_id": r["id"],
                             "error": f"{type(e).__name__}: {e}"})
            news = None
        if news:
            write_json(BACKFILL_DIR / f"{r['id']}.json", {
                "_source": GRAPHQL_URL, "_captured_at": utcnow(),
                "_player": {"name": r["name"], "norm_name": norm_name(r["name"]),
                            "position": r["pos"], "team": r["team"],
                            "status": r["status"], "sleeper_id": r["id"]},
                "news": news})
            saved += 1
        elif news is not None:
            empty += 1
        if i % PROGRESS_EVERY == 0:
            print(f"..  {i}/{len(todo)} saved={saved} empty={empty} "
                  f"failed={len(failures)}", flush=True)
        time.sleep(NEWS_SLEEP)

    cov = coverage(BACKFILL_DIR)
    write_json(BACKFILL_DIR / "MANIFEST.json", {
        "started_at": started, "finished_at": utcnow(),
        "query": {"endpoint": GRAPHQL_URL, "operation": "get_player_news",
                  "sport": "nfl", "limit": NEWS_LIMIT},
        "selection": {"positions": sorted(FANTASY_POS),
                      "rule": "active OR news_updated >= 2018-01-01 UTC"},
        "players": {"selected": len(players), "attempted": len(todo),
                    "resumed": len(resumed), "saved": saved, "empty": empty,
                    "failed": len(failures)},
        "files": cov["files"], "news_items": cov["items"], "bytes": cov["bytes"],
        "items_by_year": cov["years"], "top_sources": cov["top_sources"],
        "failures": failures, "internal_research_only": True,
    }, indent=2)

    print_coverage(cov)
    print(f"DONE players={len(players)} files={cov['files']} items={cov['items']:,} "
          f"total={cov['bytes'] / 1e6:.1f}MB failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
