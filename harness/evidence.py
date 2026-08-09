"""Time-gated evidence retrieval — "search the world as of a past date".

The single chokepoint for point-in-time news/depth-chart/injury lookups used by
BreakoutBench open-tool mode (docs/breakoutbench-design.md §2b), DraftGym's
action space, and SFT trace generation. The time gate (``published < as_of``)
lives HERE, in the store — never in the caller's prompt. Every public query
method hard-filters and then re-asserts that nothing at-or-after the cutoff
escapes.

Corpus (see docs/evidence-archive-feasibility.md for provenance):
- ``data/raw/evidence/sleeper_backfill_internal/<sleeper_id>.json`` — dated
  player news blurbs (rotowire / rotoballer / fantasy_pros via Sleeper's
  GraphQL). **internal_research_only=True** — third-party licensed text,
  never republished in-product without a license.
- ``data/raw/evidence/<YYYY-MM-DD>/`` — nightly archiver captures: parsed PFT
  RSS (public capture, not internal-only) and ``sleeper_news_internal/``
  (same internal-only rule as the backfill; deduped against it by doc_id).
- ``data/raw/nflverse/depth_charts/depth_charts_<season>.csv`` — 2011-2024 are
  (season, week) snapshots without a capture timestamp; we assign an
  approximate snapshot date (week 1 -> Aug 30 of the season, the initial
  post-cutdown depth chart; week W>=2 -> the Thursday of that game week) and
  mark rows ``date_basis="approx_week"``. 2025+ files carry a real ``dt``
  capture timestamp (``date_basis="capture_dt"``).
- ``data/raw/nflverse/injuries/injuries_<season>.csv`` — weekly designations
  with a real ``date_modified`` timestamp.

Index build (``python -m harness.evidence build-index``) writes
``data/processed/evidence_index/``:
- ``players/<player_id>.jsonl`` — per-player, published-ascending doc records
- ``docs/docs_<YYYY>.jsonl``   — all docs sharded by publish year (search scan)
- ``players.json``             — player metadata + norm_name -> player_id map
- ``manifest.json``            — build provenance (inputs, counts, timings)

Search is a raw-line substring prefilter + JSON parse of hits over the year
shards, newest year first with early exit. Measured on the full corpus
(~310k docs): a full-history scan is well under a second, so no inverted
index is needed at this scale (decision recorded in the manifest).

Stdlib only. Python 3.11+.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from evals.names import norm_name
except ImportError:  # pragma: no cover - direct script use from repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evals.names import norm_name

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_EVIDENCE_DIR = REPO_ROOT / "data" / "raw" / "evidence"
NFLVERSE_DIR = REPO_ROOT / "data" / "raw" / "nflverse"
INDEX_DIR = REPO_ROOT / "data" / "processed" / "evidence_index"

# Text obtained via Sleeper's GraphQL endpoint is third-party licensed
# (rotowire/rotoballer/fantasy_pros) — internal research scaffolding only.
_SLEEPER_DIR_NAMES = ("sleeper_backfill_internal", "sleeper_news_internal")
_DAILY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TEAM_RE = re.compile(r"^[A-Za-z]{2,3}$")


def _utc_ms_to_iso(ms: int) -> str:
    return (
        datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _as_of_to_cutoff_ms(as_of: date | datetime | str) -> int:
    """Cutoff in epoch-ms. A bare date means midnight UTC at the START of that
    day, so an item published exactly at ``as_of`` is EXCLUDED (published <
    as_of)."""
    if isinstance(as_of, str):
        as_of = date.fromisoformat(as_of)
    if isinstance(as_of, datetime):
        dt = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    else:
        dt = datetime(as_of.year, as_of.month, as_of.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _week_snapshot_date(season: int, week: int) -> date:
    """Approximate snapshot date for (season, week)-keyed nflverse depth charts.

    Week 1 -> Aug 30 of the season year (initial post-cutdown depth chart);
    week W >= 2 -> Thursday of that game week (kickoff Thursday follows the
    first Monday of September). Weeks are absolute (playoffs continue the
    numbering), so no game_type handling is needed.
    """
    if week <= 1:
        return date(season, 8, 30)
    d = date(season, 9, 1)
    while d.weekday() != 0:  # first Monday of September
        d += timedelta(days=1)
    kickoff_thursday = d + timedelta(days=3)
    return kickoff_thursday + timedelta(days=7 * (week - 1))


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def _sleeper_item_to_doc(item: dict, player: dict) -> dict | None:
    published = item.get("published")
    if not isinstance(published, (int, float)) or published <= 0:
        return None
    meta = item.get("metadata") or {}
    source = item.get("source") or "unknown"
    source_key = str(item.get("source_key") or "")
    pid = str(item.get("player_id") or player.get("sleeper_id") or "")
    doc_id = f"{source}:{source_key}" if source_key else (
        "sleeper:" + hashlib.sha1(
            f"{pid}|{published}|{meta.get('title','')}".encode()
        ).hexdigest()[:16]
    )
    text = " ".join(
        s.strip() for s in (meta.get("description"), meta.get("analysis")) if s
    )
    return {
        "doc_id": doc_id,
        "player_id": pid or None,
        "player_name": player.get("name"),
        "position": player.get("position") or None,
        "team": player.get("team") or None,
        "published_ms": int(published),
        "published_utc": _utc_ms_to_iso(int(published)),
        "source": source,
        "source_key": source_key or None,
        "title": meta.get("title") or "",
        "text": text,
        "url": None,
        "internal_research_only": True,
    }


def _iter_sleeper_files(raw_evidence_dir: Path) -> Iterator[Path]:
    backfill = raw_evidence_dir / "sleeper_backfill_internal"
    if backfill.is_dir():
        yield from sorted(backfill.glob("*.json"))
    for daily in sorted(raw_evidence_dir.iterdir()) if raw_evidence_dir.is_dir() else []:
        if daily.is_dir() and _DAILY_DIR_RE.match(daily.name):
            news_dir = daily / "sleeper_news_internal"
            if news_dir.is_dir():
                yield from sorted(news_dir.glob("*.json"))


def _iter_pft_docs(raw_evidence_dir: Path) -> Iterator[dict]:
    if not raw_evidence_dir.is_dir():
        return
    for daily in sorted(raw_evidence_dir.iterdir()):
        if not (daily.is_dir() and _DAILY_DIR_RE.match(daily.name)):
            continue
        pft = daily / "pft_rss.json"
        if not pft.is_file():
            continue
        try:
            payload = json.loads(pft.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for item in payload.get("items", []):
            try:
                dt = parsedate_to_datetime(item.get("pubDate", ""))
            except (TypeError, ValueError):
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ms = int(dt.timestamp() * 1000)
            link = item.get("link") or ""
            doc_id = "pft_rss:" + hashlib.sha1(
                (link or f"{ms}|{item.get('title','')}").encode()
            ).hexdigest()[:16]
            yield {
                "doc_id": doc_id,
                "player_id": None,
                "player_name": None,
                "position": None,
                "team": None,
                "published_ms": ms,
                "published_utc": _utc_ms_to_iso(ms),
                "source": "pft_rss",
                "source_key": None,
                "title": item.get("title") or "",
                "text": (item.get("description") or "").strip(),
                "url": link or None,
                "internal_research_only": False,
            }


def build_index(
    raw_evidence_dir: Path | str = RAW_EVIDENCE_DIR,
    out_dir: Path | str = INDEX_DIR,
    quiet: bool = False,
) -> dict:
    """One pass over the corpus into ``out_dir``. Returns the manifest."""
    t0 = time.monotonic()
    raw_evidence_dir = Path(raw_evidence_dir)
    out_dir = Path(out_dir)
    (out_dir / "players").mkdir(parents=True, exist_ok=True)
    (out_dir / "docs").mkdir(parents=True, exist_ok=True)

    docs_by_id: dict[str, dict] = {}
    players: dict[str, dict] = {}
    n_input_files = 0

    for path in _iter_sleeper_files(raw_evidence_dir):
        n_input_files += 1
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        player = payload.get("_player") or {}
        pid = str(player.get("sleeper_id") or path.stem)
        pmeta = players.setdefault(
            pid,
            {
                "player_id": pid,
                "name": player.get("name"),
                "norm_name": player.get("norm_name")
                or (norm_name(player["name"]) if player.get("name") else None),
                "position": player.get("position") or None,
                "team": player.get("team") or None,
            },
        )
        # Later files (daily captures) may carry fresher team info.
        if player.get("team"):
            pmeta["team"] = player["team"]
        for item in payload.get("news", []):
            doc = _sleeper_item_to_doc(item, {**player, "sleeper_id": pid})
            if doc is not None:
                docs_by_id.setdefault(doc["doc_id"], doc)

    for doc in _iter_pft_docs(raw_evidence_dir):
        docs_by_id.setdefault(doc["doc_id"], doc)

    docs = sorted(docs_by_id.values(), key=lambda d: (d["published_ms"], d["doc_id"]))

    # Per-player streams (published-ascending).
    per_player: dict[str, list[dict]] = {}
    for doc in docs:
        if doc["player_id"]:
            per_player.setdefault(doc["player_id"], []).append(doc)
    for old in (out_dir / "players").glob("*.jsonl"):
        old.unlink()
    for pid, items in per_player.items():
        with open(out_dir / "players" / f"{pid}.jsonl", "w") as f:
            for doc in items:
                f.write(json.dumps(doc, separators=(",", ":")) + "\n")

    # Year shards (published-ascending within shard).
    for old in (out_dir / "docs").glob("*.jsonl"):
        old.unlink()
    shard_counts: dict[str, int] = {}
    shard_files: dict[int, Any] = {}
    try:
        for doc in docs:
            year = datetime.fromtimestamp(
                doc["published_ms"] / 1000, tz=timezone.utc
            ).year
            f = shard_files.get(year)
            if f is None:
                f = open(out_dir / "docs" / f"docs_{year}.jsonl", "w")
                shard_files[year] = f
            f.write(json.dumps(doc, separators=(",", ":")) + "\n")
            shard_counts[str(year)] = shard_counts.get(str(year), 0) + 1
    finally:
        for f in shard_files.values():
            f.close()

    by_norm: dict[str, list[str]] = {}
    for pid, pmeta in players.items():
        if pmeta.get("norm_name"):
            by_norm.setdefault(pmeta["norm_name"], []).append(pid)
    (out_dir / "players.json").write_text(
        json.dumps({"players": players, "by_norm_name": by_norm}, indent=0)
    )

    src_counts: dict[str, int] = {}
    for doc in docs:
        src_counts[doc["source"]] = src_counts.get(doc["source"], 0) + 1
    manifest = {
        "built_at_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "raw_evidence_dir": str(raw_evidence_dir),
        "n_input_files": n_input_files,
        "n_docs": len(docs),
        "n_players": len(per_player),
        "docs_by_source": src_counts,
        "docs_by_year": shard_counts,
        "published_range_utc": (
            [docs[0]["published_utc"], docs[-1]["published_utc"]] if docs else None
        ),
        "index_bytes": sum(
            p.stat().st_size for p in out_dir.rglob("*") if p.is_file()
        ),
        "build_seconds": round(time.monotonic() - t0, 2),
        "search_strategy": (
            "raw-line substring prefilter over year shards, newest-first with "
            "early exit; no inverted index (full-history scan <1s at ~310k docs)"
        ),
        "time_gate": "published < as_of, enforced in EvidenceStore, never in prompts",
        "internal_research_only_rule": (
            "Sleeper-derived blurb text (rotowire/rotoballer/fantasy_pros) is "
            "internal research scaffolding only — never republish in-product "
            "without a license (docs/evidence-archive-feasibility.md)"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not quiet:
        print(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

class EvidenceStore:
    """Point-in-time evidence store. EVERY method hard-filters
    ``published < as_of`` — the gate lives here, never in the caller's prompt.
    """

    def __init__(
        self,
        as_of: date | datetime | str,
        index_dir: Path | str = INDEX_DIR,
        nflverse_dir: Path | str = NFLVERSE_DIR,
    ):
        self.index_dir = Path(index_dir)
        self.nflverse_dir = Path(nflverse_dir)
        self._cutoff_ms = _as_of_to_cutoff_ms(as_of)
        self.as_of_utc = _utc_ms_to_iso(self._cutoff_ms)
        players_path = self.index_dir / "players.json"
        if not players_path.is_file():
            raise FileNotFoundError(
                f"evidence index not found at {self.index_dir} — run "
                "`python -m harness.evidence build-index` first"
            )
        blob = json.loads(players_path.read_text())
        self._players: dict[str, dict] = blob["players"]
        self._by_norm: dict[str, list[str]] = blob["by_norm_name"]
        self._depth_cache: dict[int, list[dict]] = {}
        self._injury_cache: dict[int, list[dict]] = {}

    # -- gate -------------------------------------------------------------
    def _assert_gate(self, items: Iterable[dict]) -> list[dict]:
        out = list(items)
        for item in out:
            assert item["published_ms"] < self._cutoff_ms, (
                f"time-gate violation: doc published {item['published_utc']} "
                f">= as_of {self.as_of_utc}: {item}"
            )
        return out

    # -- player resolution --------------------------------------------------
    def resolve_player(self, player: str) -> dict:
        """Resolve a sleeper id or a name (via evals.names.norm_name) to
        player metadata. Raises KeyError if unknown, ValueError if ambiguous.
        """
        player = str(player).strip()
        if player in self._players:
            return self._players[player]
        key = norm_name(player)
        pids = self._by_norm.get(key, [])
        if not pids:
            raise KeyError(f"unknown player: {player!r}")
        if len(pids) > 1:
            cands = [
                f"{p}={self._players[p].get('name')} "
                f"({self._players[p].get('position')})"
                for p in pids
            ]
            raise ValueError(
                f"ambiguous player {player!r}: {', '.join(cands)} — pass a "
                "sleeper id"
            )
        return self._players[pids[0]]

    # -- news ---------------------------------------------------------------
    def player_news(
        self,
        player: str,
        limit: int = 10,
        since: date | str | None = None,
    ) -> list[dict]:
        """Most-recent-first pre-``as_of`` news items for a player (id or name)."""
        meta = self.resolve_player(player)
        path = self.index_dir / "players" / f"{meta['player_id']}.jsonl"
        since_ms = _as_of_to_cutoff_ms(since) if since is not None else None
        items: list[dict] = []
        if path.is_file():
            with open(path) as f:
                for line in f:
                    doc = json.loads(line)
                    if doc["published_ms"] >= self._cutoff_ms:
                        break  # ascending file: everything after is future
                    if since_ms is not None and doc["published_ms"] < since_ms:
                        continue
                    items.append(doc)
        items = items[-max(0, int(limit)):][::-1]
        return self._assert_gate(items)

    # -- search ---------------------------------------------------------------
    def search(
        self,
        query_terms: str | list[str],
        limit: int = 10,
        position: str | None = None,
    ) -> list[dict]:
        """Keyword search (all terms must appear, case-insensitive substring)
        across the pre-cutoff corpus, most recent first."""
        if isinstance(query_terms, str):
            terms = [t for t in re.split(r"\s+", query_terms.lower()) if t]
        else:
            terms = [str(t).lower() for t in query_terms if str(t).strip()]
        if not terms:
            return []
        limit = max(0, int(limit))
        pos = position.upper() if position else None
        cutoff_year = datetime.fromtimestamp(
            (self._cutoff_ms - 1) / 1000, tz=timezone.utc
        ).year
        shards = sorted(
            (
                p
                for p in (self.index_dir / "docs").glob("docs_*.jsonl")
                if int(p.stem.split("_")[1]) <= cutoff_year
            ),
            key=lambda p: int(p.stem.split("_")[1]),
            reverse=True,
        )
        results: list[dict] = []
        for shard in shards:
            hits: list[dict] = []
            with open(shard) as f:
                for line in f:
                    low = line.lower()
                    if not all(t in low for t in terms):
                        continue
                    doc = json.loads(line)
                    if doc["published_ms"] >= self._cutoff_ms:
                        break  # ascending within shard
                    if pos and (doc.get("position") or "").upper() != pos:
                        continue
                    hay = f"{doc.get('title','')} {doc.get('text','')} " \
                          f"{doc.get('player_name') or ''}".lower()
                    if all(t in hay for t in terms):
                        hits.append(doc)
            hits.sort(key=lambda d: -d["published_ms"])
            results.extend(hits)
            if len(results) >= limit:
                break
        return self._assert_gate(results[:limit])

    # -- depth charts -----------------------------------------------------
    def _load_depth_season(self, season: int) -> list[dict]:
        if season in self._depth_cache:
            return self._depth_cache[season]
        path = self.nflverse_dir / "depth_charts" / f"depth_charts_{season}.csv"
        rows: list[dict] = []
        if path.is_file():
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    if "dt" in r and r.get("dt"):  # 2025+ capture-timestamp schema
                        try:
                            dt = datetime.fromisoformat(r["dt"].replace("Z", "+00:00"))
                        except ValueError:
                            continue
                        ms = int(dt.timestamp() * 1000)
                        rows.append(
                            {
                                "season": season,
                                "snapshot": r["dt"],
                                "date_basis": "capture_dt",
                                "published_ms": ms,
                                "published_utc": _utc_ms_to_iso(ms),
                                "team": r.get("team"),
                                "player_name": r.get("player_name"),
                                "position": r.get("pos_abb"),
                                "depth_slot": r.get("pos_abb"),
                                "depth_rank": r.get("pos_rank"),
                                "unit": r.get("pos_grp"),
                                "gsis_id": r.get("gsis_id"),
                                "source": "nflverse_depth_charts",
                                "internal_research_only": False,
                            }
                        )
                    else:  # 2011-2024 (season, week) schema; approximate date
                        try:
                            week = int(r.get("week") or 0)
                        except ValueError:
                            continue
                        d = _week_snapshot_date(season, week)
                        ms = _as_of_to_cutoff_ms(d)
                        rows.append(
                            {
                                "season": season,
                                "week": week,
                                "snapshot": d.isoformat(),
                                "date_basis": "approx_week",
                                "published_ms": ms,
                                "published_utc": _utc_ms_to_iso(ms),
                                "team": r.get("club_code"),
                                "player_name": r.get("full_name"),
                                "position": r.get("position"),
                                "depth_slot": r.get("depth_position"),
                                "depth_rank": r.get("depth_team"),
                                "unit": r.get("formation"),
                                "gsis_id": r.get("gsis_id"),
                                "source": "nflverse_depth_charts",
                                "internal_research_only": False,
                            }
                        )
        self._depth_cache[season] = rows
        return rows

    def _depth_seasons(self) -> list[int]:
        seasons = []
        for p in (self.nflverse_dir / "depth_charts").glob("depth_charts_*.csv"):
            try:
                seasons.append(int(p.stem.split("_")[-1]))
            except ValueError:
                continue
        cutoff_year = datetime.fromtimestamp(
            (self._cutoff_ms - 1) / 1000, tz=timezone.utc
        ).year
        return sorted((s for s in seasons if s <= cutoff_year), reverse=True)

    def depth_chart(self, team_or_player: str, latest: bool = True) -> list[dict]:
        """Depth-chart rows strictly before ``as_of``. ``latest=True`` returns
        rows from the most recent pre-cutoff snapshot only; ``latest=False``
        returns the full pre-cutoff history (newest first)."""
        query = str(team_or_player).strip()
        is_team = bool(_TEAM_RE.match(query)) and query.upper() == query
        key = query.upper() if is_team else norm_name(query)
        matches: list[dict] = []
        for season in self._depth_seasons():
            for row in self._load_depth_season(season):
                if row["published_ms"] >= self._cutoff_ms:
                    continue
                if is_team:
                    if (row.get("team") or "").upper() == key:
                        matches.append(row)
                elif row.get("player_name") and norm_name(row["player_name"]) == key:
                    matches.append(row)
            if matches and latest:
                break  # newest season with data is enough for `latest`
        matches.sort(key=lambda r: -r["published_ms"])
        if latest and matches:
            newest = matches[0]["snapshot"]
            matches = [r for r in matches if r["snapshot"] == newest]
        return self._assert_gate(matches)

    # -- injuries -----------------------------------------------------------
    def _load_injury_season(self, season: int) -> list[dict]:
        if season in self._injury_cache:
            return self._injury_cache[season]
        path = self.nflverse_dir / "injuries" / f"injuries_{season}.csv"
        rows: list[dict] = []
        if path.is_file():
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    stamp = r.get("date_modified") or ""
                    try:
                        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ms = int(dt.timestamp() * 1000)
                    rows.append(
                        {
                            "season": season,
                            "week": r.get("week"),
                            "published_ms": ms,
                            "published_utc": _utc_ms_to_iso(ms),
                            "team": r.get("team"),
                            "player_name": r.get("full_name"),
                            "position": r.get("position"),
                            "report_status": r.get("report_status") or None,
                            "practice_status": r.get("practice_status") or None,
                            "primary_injury": r.get("report_primary_injury")
                            or r.get("practice_primary_injury")
                            or None,
                            "secondary_injury": r.get("report_secondary_injury")
                            or r.get("practice_secondary_injury")
                            or None,
                            "gsis_id": r.get("gsis_id"),
                            "source": "nflverse_injuries",
                            "internal_research_only": False,
                        }
                    )
        self._injury_cache[season] = rows
        return rows

    def injury_status(self, player: str) -> dict | None:
        """Most recent pre-``as_of`` injury-report designation for a player,
        or None if the player has no pre-cutoff designation on file."""
        key = norm_name(str(player))
        cutoff_year = datetime.fromtimestamp(
            (self._cutoff_ms - 1) / 1000, tz=timezone.utc
        ).year
        seasons = []
        for p in (self.nflverse_dir / "injuries").glob("injuries_*.csv"):
            try:
                s = int(p.stem.split("_")[-1])
            except ValueError:
                continue
            if s <= cutoff_year:
                seasons.append(s)
        best: dict | None = None
        for season in sorted(seasons, reverse=True):
            for row in self._load_injury_season(season):
                if row["published_ms"] >= self._cutoff_ms:
                    continue
                if not row.get("player_name"):
                    continue
                if norm_name(row["player_name"]) != key:
                    continue
                if best is None or row["published_ms"] > best["published_ms"]:
                    best = row
            if best is not None:
                break  # newest season containing this player wins
        if best is not None:
            self._assert_gate([best])
        return best

    # -- tool surface ------------------------------------------------------
    def as_tool_definitions(self) -> list[dict]:
        """OpenAI-style function-calling schemas — the exact surface DraftGym
        and open-tool evals hand a model."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "player_news",
                    "description": (
                        "Dated news items for one NFL player, most recent "
                        "first. Only items published strictly before the "
                        "session's as-of date exist; the archive contains "
                        "nothing after it."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "player": {
                                "type": "string",
                                "description": "Player full name or sleeper id",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max items to return (default 10)",
                            },
                            "since": {
                                "type": "string",
                                "description": "Optional ISO date lower bound (YYYY-MM-DD)",
                            },
                        },
                        "required": ["player"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": (
                        "Keyword search over the dated NFL news archive "
                        "(all terms must appear). Most recent first, "
                        "pre-as-of only."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Space-separated keywords, e.g. 'first team reps'",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max items to return (default 10)",
                            },
                            "position": {
                                "type": "string",
                                "description": "Optional position filter (QB/RB/WR/TE)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "depth_chart",
                    "description": (
                        "Latest pre-as-of team depth-chart snapshot rows for a "
                        "team code (e.g. 'LA', 'KC') or a single player's "
                        "row(s). Dates may be approximate for 2011-2024 "
                        "(week-keyed snapshots)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "team_or_player": {
                                "type": "string",
                                "description": "Team abbreviation (uppercase) or player name",
                            },
                            "latest": {
                                "type": "boolean",
                                "description": "Latest snapshot only (default true); false returns pre-as-of history",
                            },
                        },
                        "required": ["team_or_player"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "injury_status",
                    "description": (
                        "Most recent pre-as-of official injury-report "
                        "designation for a player (nflverse), or null if none "
                        "on file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "player": {
                                "type": "string",
                                "description": "Player full name",
                            },
                        },
                        "required": ["player"],
                    },
                },
            },
        ]

    def dispatch(self, tool_call: dict | str) -> dict:
        """Execute an OpenAI-style tool call and return a JSON-serializable
        result envelope: {"ok": true, "result": ...} or {"ok": false,
        "error": ...}. Accepts either the full call dict, its "function"
        sub-object, or a JSON string of either."""
        try:
            if isinstance(tool_call, str):
                tool_call = json.loads(tool_call)
            call = tool_call.get("function", tool_call)
            name = call["name"]
            args = call.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            if name == "player_news":
                result: Any = self.player_news(
                    args["player"],
                    limit=int(args.get("limit", 10)),
                    since=args.get("since"),
                )
            elif name == "search":
                result = self.search(
                    args["query"],
                    limit=int(args.get("limit", 10)),
                    position=args.get("position"),
                )
            elif name == "depth_chart":
                result = self.depth_chart(
                    args["team_or_player"], latest=bool(args.get("latest", True))
                )
            elif name == "injury_status":
                result = self.injury_status(args["player"])
            else:
                return {"ok": False, "error": f"unknown tool: {name}"}
            return {"ok": True, "result": result}
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def as_tool_definitions() -> list[dict]:
    """Module-level convenience: tool schemas are store-independent."""
    return EvidenceStore.as_tool_definitions(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_doc(doc: dict) -> str:
    flag = " [internal-only]" if doc.get("internal_research_only") else ""
    who = f" ({doc['player_name']})" if doc.get("player_name") else ""
    return (
        f"  {doc['published_utc']}  {doc['source']:<12}{flag}{who}\n"
        f"    {doc['title']}\n"
        f"    {doc['text'][:220]}{'…' if len(doc.get('text','')) > 220 else ''}"
    )


def _demo() -> None:
    as_of = date(2023, 9, 1)
    store = EvidenceStore(as_of=as_of)
    print(f"EvidenceStore(as_of={as_of}) — everything below is published < {as_of}\n")

    print("== Puka Nacua: last 5 pre-cutoff news items ==")
    t0 = time.monotonic()
    news = store.player_news("Puka Nacua", limit=5)
    dt_news = time.monotonic() - t0
    for doc in news:
        print(_fmt_doc(doc))
    print(f"  ({len(news)} items, {dt_news*1000:.1f} ms)\n")

    print("== Puka Nacua: depth chart (latest pre-cutoff snapshot) ==")
    t0 = time.monotonic()
    rows = store.depth_chart("Puka Nacua")
    dt_dc = time.monotonic() - t0
    for r in rows:
        print(
            f"  {r['snapshot']} ({r['date_basis']})  {r['team']}  "
            f"{r['unit']}/{r['depth_slot']} depth {r['depth_rank']}  "
            f"season {r['season']} week {r.get('week', '-')}"
        )
    print(f"  ({len(rows)} rows, {dt_dc*1000:.1f} ms)\n")

    print("== search('first team reps'), Aug 2023 window ==")
    t0 = time.monotonic()
    hits = store.search("first team reps", limit=5)
    dt_s = time.monotonic() - t0
    for doc in hits:
        print(_fmt_doc(doc))
    print(f"  ({len(hits)} hits, {dt_s*1000:.1f} ms)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harness.evidence",
        description="Time-gated evidence retrieval (build + demo).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-index", help="build data/processed/evidence_index/")
    b.add_argument("--raw", default=str(RAW_EVIDENCE_DIR))
    b.add_argument("--out", default=str(INDEX_DIR))
    sub.add_parser("demo", help="face-validity demo (Puka Nacua @ 2023-09-01)")
    args = parser.parse_args(argv)
    if args.cmd == "build-index":
        build_index(args.raw, args.out)
    elif args.cmd == "demo":
        _demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
