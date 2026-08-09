#!/usr/bin/env python3
"""Daily snapshot of freely/legitimately reachable NFL betting odds.

Betting markets enter the system as a MEASURED source, not doctrine
(evals/results/odds_channel_scoping.md is the scoping + de-vig design doc).
This module captures the raw evidence leg into
data/raw/evidence/<YYYY-MM-DD>/odds/:

  espn_futures.json        ESPN core-API NFL season futures (DraftKings-priced
                           award/division/conference/Super Bowl markets).
                           Public unauthenticated JSON, no key.
  oddsapi_sports.json      The Odds API sports index (0-credit call) -- used to
                           discover active NFL futures sport keys.
  oddsapi_nfl_odds.json    The Odds API NFL game odds: h2h/spreads/totals,
                           us region, American prices (3 credits).
  oddsapi_futures_<key>.json  Outright prices for each active NFL futures key
                           (1 credit each, capped).
  manifest.json            per-source status/items/bytes/credits/timestamps

The Odds API requires ODDS_API_KEY in .env (free tier: sign up at
https://the-odds-api.com -> "Get API Key", 500 credits/month, no card).
Without a key those sources report status "skipped" and the run still
succeeds -- ESPN futures alone is a valid daily capture.

Deliberately NOT captured here (see scoping doc): StartWho (ToS bans bots),
sportsbook site endpoints such as DraftKings/FanDuel internal JSON (ToS),
BettingPros (FantasyPros property -- preview-only constraint).

Stdlib only. Append-only and idempotent per day like archive_evidence.py:
non-empty files are never overwritten, re-runs only fill gaps, and one
source failing never kills the run. Runs standalone (python
scripts/collect_odds.py) or as the "odds" source inside archive_evidence.
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "evidence"

ESPN_FUTURES_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
    "seasons/{season}/futures?limit=100"
)
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_REGION = "us"
ODDS_API_GAME_MARKETS = "h2h,spreads,totals"  # featured markets: 1 credit/market/region
ODDS_API_FUTURES_CAP = 8   # max outright sport keys per day (1 credit each)
SEASON = 2026
SLEEP = 0.3
UA = {"User-Agent": "fantasy-alpha-archiver/0.1"}


def _ssl_context() -> ssl.SSLContext:
    """Same fallback as archive_evidence: python.org macOS builds may ship an
    empty trust store; fall back to the system bundle."""
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats().get("x509_ca") and Path("/etc/ssl/cert.pem").exists():
        ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    return ctx


SSL_CTX = _ssl_context()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch(url: str, *, timeout: int = 60) -> tuple[bytes, dict[str, str]]:
    """GET url; return (body, interesting response headers)."""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
        wanted = ("x-requests-remaining", "x-requests-used", "x-requests-last")
        headers = {k: resp.headers[k] for k in wanted if resp.headers.get(k)}
        return resp.read(), headers


def have(path: Path) -> bool:
    """Append-only guard: a non-empty file is immutable evidence."""
    return path.exists() and path.stat().st_size > 0


def write_json(path: Path, obj, indent: int | None = None) -> int:
    data = json.dumps(obj, indent=indent).encode()
    path.write_bytes(data)
    return len(data)


# ---------------------------------------------------------------- .env

def load_env(path: Path | None = None) -> dict[str, str]:
    """Minimal .env parser (KEY=value, # comments). os.environ wins."""
    env: dict[str, str] = {}
    path = path if path is not None else ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip("'\"")
    env.update({k: v for k, v in os.environ.items() if k in env or k == "ODDS_API_KEY"})
    return env


# ---------------------------------------------------------------- odds math
# First-party de-vig primitives, unit-tested from day one; the full
# props -> implied stat lines -> league points pipeline is specified in
# evals/results/odds_channel_scoping.md and will live in harness/.

def american_to_prob(odds: float) -> float:
    """American odds -> raw implied probability (vig included)."""
    if odds == 0:
        raise ValueError("American odds of 0 are undefined")
    a = abs(odds)
    return a / (a + 100.0) if odds < 0 else 100.0 / (a + 100.0)


def devig_pair(over_odds: float, under_odds: float) -> tuple[float, float, float]:
    """De-vig a two-way market by proportional normalization.

    Returns (p_over_fair, p_under_fair, overround) where overround is the
    bookmaker margin (sum of raw implied probabilities minus 1).
    """
    p_over = american_to_prob(over_odds)
    p_under = american_to_prob(under_odds)
    total = p_over + p_under
    return p_over / total, p_under / total, total - 1.0


# ---------------------------------------------------------------- ESPN futures

def archive_espn_futures(odds_dir: Path) -> dict:
    dest = odds_dir / "espn_futures.json"
    if have(dest):
        payload = json.loads(dest.read_text())
        return {"status": "cached", "items": len(payload.get("items", [])),
                "bytes": dest.stat().st_size}
    url = ESPN_FUTURES_URL.format(season=SEASON)
    raw, _ = fetch(url)
    payload = json.loads(raw)  # validate before writing
    if not payload.get("items"):
        raise ValueError("espn futures: empty items")
    payload["_source"] = url
    payload["_captured_at"] = utcnow()
    n = write_json(dest, payload)
    return {"status": "ok", "items": len(payload["items"]), "bytes": n,
            "captured_at": utcnow()}


# ---------------------------------------------------------------- The Odds API

def _oddsapi_url(path: str, key: str, **params: str) -> str:
    query = "&".join([f"apiKey={key}"] + [f"{k}={v}" for k, v in params.items()])
    return f"{ODDS_API_BASE}/{path}?{query}"


def _redact(url: str, key: str) -> str:
    return url.replace(key, "REDACTED")


def archive_odds_api(odds_dir: Path, api_key: str | None) -> dict:
    if not api_key:
        return {"status": "skipped", "items": 0, "bytes": 0,
                "note": ("no ODDS_API_KEY in .env -- free key: "
                         "https://the-odds-api.com (500 credits/month)")}

    failures: list[dict] = []
    quota: dict[str, str] = {}
    fetched = items = total = present = 0

    def grab(dest: Path, path: str, count, **params) -> None:
        nonlocal fetched, items, total, present
        if have(dest):
            present += 1
            items += count(json.loads(dest.read_text())["data"])
            total += dest.stat().st_size
            return
        url = _oddsapi_url(path, api_key, **params)
        try:
            raw, headers = fetch(url, timeout=45)
            payload = json.loads(raw)
        except Exception as e:  # noqa: BLE001  tolerate per endpoint
            failures.append({"file": dest.name, "error": str(e)})
            return
        quota.update(headers)
        wrapped = {"_source": _redact(url, api_key), "_captured_at": utcnow(),
                   "data": payload}
        total += write_json(dest, wrapped)
        fetched += 1
        present += 1
        items += count(payload)
        time.sleep(SLEEP)

    # 0-credit sports index; also discovers active NFL futures sport keys.
    sports_dest = odds_dir / "oddsapi_sports.json"
    grab(sports_dest, "sports", len, all="true")

    futures_keys: list[str] = []
    if have(sports_dest):
        sports = json.loads(sports_dest.read_text())["data"]
        futures_keys = sorted(
            s["key"] for s in sports
            if s.get("key", "").startswith("americanfootball_nfl_")
            and s.get("active") and s.get("has_outrights")
        )[:ODDS_API_FUTURES_CAP]

    # Featured game markets: cost = 3 markets x 1 region = 3 credits.
    grab(odds_dir / "oddsapi_nfl_odds.json", "sports/americanfootball_nfl/odds",
         len, regions=ODDS_API_REGION, markets=ODDS_API_GAME_MARKETS,
         oddsFormat="american")

    # Active futures (outrights): 1 credit each, capped.
    for key in futures_keys:
        grab(odds_dir / f"oddsapi_futures_{key}.json", f"sports/{key}/odds",
             len, regions=ODDS_API_REGION, markets="outrights",
             oddsFormat="american")

    if present == 0:
        status = "failed"
    elif failures:
        status = "partial"
    else:
        status = "ok" if fetched else "cached"
    return {"status": status, "items": items, "bytes": total,
            "futures_keys": futures_keys, "failures": failures,
            "credits": quota, "captured_at": utcnow()}


# ---------------------------------------------------------------- entry points

def archive_odds(day_dir: Path) -> dict:
    """Archive-evidence source entry: snapshot odds into <day_dir>/odds/."""
    odds_dir = day_dir / "odds"
    odds_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict] = {}
    try:
        sources["espn_futures"] = archive_espn_futures(odds_dir)
    except Exception as e:  # noqa: BLE001  one source never kills the run
        sources["espn_futures"] = {"status": "failed", "items": 0, "bytes": 0,
                                   "error": f"{type(e).__name__}: {e}"}
    try:
        sources["odds_api"] = archive_odds_api(odds_dir, load_env().get("ODDS_API_KEY"))
    except Exception as e:  # noqa: BLE001
        sources["odds_api"] = {"status": "failed", "items": 0, "bytes": 0,
                               "error": f"{type(e).__name__}: {e}"}

    statuses = {s["status"] for s in sources.values()}
    real = statuses - {"skipped"}
    if real == {"failed"}:
        status = "failed"
    elif "failed" in real or "partial" in real:
        status = "partial"
    elif "ok" in real:
        status = "ok"
    else:
        status = "cached" if "cached" in real else "skipped"
    entry = {"status": status,
             "items": sum(s["items"] for s in sources.values()),
             "bytes": sum(s["bytes"] for s in sources.values()),
             "sources": sources, "captured_at": utcnow()}
    write_json(odds_dir / "manifest.json", entry, indent=2)
    return entry


def main() -> int:
    day_dir = OUT / datetime.now().astimezone().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    entry = archive_odds(day_dir)
    for name, s in entry["sources"].items():
        note = s.get("error") or s.get("note") or ""
        print(f"{s['status']:<7} odds/{name}  items={s['items']} "
              f"bytes={s['bytes']}{'  ' + note if note else ''}", flush=True)
    print(f"DONE odds {entry['status']} total={entry['bytes'] / 1e6:.2f}MB")
    return 1 if entry["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
