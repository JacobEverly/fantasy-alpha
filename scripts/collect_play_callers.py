"""Historical NFL head-coach / offensive play-caller table, 2008-2025.

Covers edge family #6 (coordinator / play-caller changes), which the packet
coverage table shows at 0-8% historically: the research repo's manual
play_caller_history.csv only reaches back to ~2022.

Sources, in ToS order:
  1. Wikipedia "<year> <Team> season" pages (CC BY-SA; fetched via the
     MediaWiki API with a descriptive User-Agent). The
     {{Infobox NFL team season}} carries |coach= and |off_coach=.
  2. The research repo's manual play_caller_history.csv as seed/validation
     (notably head_coach_play_caller facts).
  3. Pro-Football-Reference is deliberately NOT used (ToS prohibits scraping).

As-of discipline: the *first-listed* coach/coordinator in the infobox is the
season opener — offseason hires are announced Jan-Feb, so the opener HC/OC
for season S is knowable before the Sept-1 gate. Midseason changes (fired /
interim / multiple names) are flagged separately and never leak into the
opener columns.

Play-caller ambiguity: whether the HC or the OC actually calls plays is often
not determinable from structured sources. We record the OC (or the HC when no
OC is listed) and set play_caller_ambiguous=1 unless the research-repo seed
carries an explicit fact. changed_play_caller (opener vs prior-season opener,
franchise-normalized) is the load-bearing flag.

Join keys match evals/vegas_features.py normalization (NORMALIZE_TEAM):
Rams=LA, Chargers=LAC, Raiders=LV, Jaguars=JAX, franchise-normalized across
relocations. Output joins packet_features / vegas tables on (season, team).

Outputs:
  data/raw/coaching/wikipedia_team_seasons_<date>.json   (immutable snapshot;
      per-page fetch is checkpointed into a cache dir so interrupted runs
      resume without refetching)
  data/processed/coaching/play_callers.csv

Usage:
  python scripts/collect_play_callers.py            # fetch + build table
  python scripts/collect_play_callers.py --study    # + breakout study ->
                                                    #   evals/results/play_caller_history.md
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "coaching"
CACHE_DIR = RAW_DIR / "_fetch_cache"  # checkpointed per-batch fetches
OUT_DIR = ROOT / "data" / "processed" / "coaching"
OUT_CSV = OUT_DIR / "play_callers.csv"
SEED_CSV = (ROOT / "fantasy-football-alpha-2026" / "data" / "manual"
            / "play_caller_history.csv")
RESULTS_MD = ROOT / "evals" / "results" / "play_caller_history.md"
LABELS_CSV = ROOT / "data" / "processed" / "labels" / "breakouts.csv"

SEASONS = range(2008, 2026)
API = "https://en.wikipedia.org/w/api.php"
UA = "FantasyAlpha/0.1 (research data collection; jacob@plantmangroup.com)"

# Same convention as evals/vegas_features.py NORMALIZE_TEAM
# ({"SD": "LAC", "STL": "LA", "OAK": "LV", "JAC": "JAX"}): franchise codes.
NORMALIZE_TEAM = {"SD": "LAC", "STL": "LA", "OAK": "LV", "JAC": "JAX",
                  "LAR": "LA"}

STUDY_SEASONS = range(2010, 2025)  # 2025 is the untouched eval holdout
ADP_GATE = {"QB": 18, "TE": 18, "RB": 40, "WR": 40}  # build_breakout_labels v0.2
BOOTSTRAP_REPS = 2000
SEED = 20260811


def franchise_name(team: str, season: int) -> str:
    """Wikipedia franchise name for a normalized team code in a season."""
    if team == "LA":
        return "St. Louis Rams" if season <= 2015 else "Los Angeles Rams"
    if team == "LAC":
        return "San Diego Chargers" if season <= 2016 else "Los Angeles Chargers"
    if team == "LV":
        return "Oakland Raiders" if season <= 2019 else "Las Vegas Raiders"
    if team == "WAS":
        if season <= 2019:
            return "Washington Redskins"
        return ("Washington Football Team" if season <= 2021
                else "Washington Commanders")
    return _STATIC_NAMES[team]


_STATIC_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos",
    "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints",
    "NYG": "New York Giants", "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans",
}
TEAMS = sorted(list(_STATIC_NAMES) + ["LA", "LAC", "LV", "WAS"])


def page_title(team: str, season: int) -> str:
    return f"{season} {franchise_name(team, season)} season"


# ---------------------------------------------------------------------------
# Fetch (checkpointed)

def _api_get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def fetch_all(titles: list[str], cache_dir: Path = CACHE_DIR,
              sleep_s: float = 0.5) -> dict[str, str]:
    """title -> wikitext. Batches of 50; each batch is checkpointed to
    cache_dir so an interrupted run resumes without refetching."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for f in cache_dir.glob("batch_*.json"):
        out.update(json.loads(f.read_text()))
    missing = [t for t in titles if t not in out]
    for i in range(0, len(missing), 50):
        batch = missing[i:i + 50]
        data = _api_get({
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "format": "json", "formatversion": "2",
            "redirects": "1", "titles": "|".join(batch)})
        got: dict[str, str] = {}
        redirect_back = {r["to"]: r["from"]
                         for r in data.get("query", {}).get("redirects", [])}
        for p in data.get("query", {}).get("pages", []):
            if "revisions" not in p:
                continue
            title = redirect_back.get(p["title"], p["title"])
            got[title] = p["revisions"][0]["slots"]["main"]["content"]
        stamp = int(time.time() * 1000)
        (cache_dir / f"batch_{stamp}.json").write_text(json.dumps(got))
        out.update(got)
        print(f"fetched batch {i // 50 + 1}: {len(got)}/{len(batch)} pages",
              file=sys.stderr)
        time.sleep(sleep_s)
    return out


def write_snapshot(pages: dict[str, str]) -> Path:
    """Immutable dated snapshot of the raw wikitext (never overwrite)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"wikipedia_team_seasons_{date.today().isoformat()}.json"
    if path.exists():
        print(f"snapshot {path.name} already exists; leaving it untouched",
              file=sys.stderr)
        return path
    path.write_text(json.dumps(
        {"source": "en.wikipedia.org (CC BY-SA 4.0)", "api": API,
         "fetched": date.today().isoformat(), "pages": pages},
        indent=None))
    return path


# ---------------------------------------------------------------------------
# Parse

_FIELD_RE = {
    "coach": re.compile(r"^\s*\|\s*coach\s*=\s*(.*)$", re.MULTILINE),
    "off_coach": re.compile(r"^\s*\|\s*off_coach\s*=\s*(.*)$", re.MULTILINE),
}
_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")
_MIDSEASON_RE = re.compile(r"fired|interim|resigned|stepped down|released"
                           r"|mutually|relieved|placed on leave|until",
                           re.IGNORECASE)
_PAREN_RE = re.compile(r"\([^)]*\)")
_TAG_RE = re.compile(r"<[^>]+>")
_REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>/]*/>", re.DOTALL)
_TMPL_RE = re.compile(r"\{\{[^{}]*\}\}")


def _field(text: str, name: str) -> str | None:
    m = _FIELD_RE[name].search(text)
    return m.group(1).strip() if m else None


def parse_names(raw: str) -> list[str]:
    """Ordered coach names from an infobox field value. First = opener."""
    if raw is None:
        return []
    s = _REF_RE.sub(" ", raw)
    s = _TMPL_RE.sub(" ", s)
    names: list[str] = []
    for m in _LINK_RE.finditer(s):
        name = (m.group(2) or m.group(1)).strip()
        if name and name not in names:
            names.append(name)
    if names:
        return names
    # plain-text fallback: strip markup/parentheticals, split on separators
    s = _TAG_RE.sub("<br>", s)
    s = _PAREN_RE.sub(" ", s)
    for part in re.split(r"<br>|,|;|/| and ", s):
        p = part.strip().strip("'\"*")
        if p and p.lower() not in {"none", "vacant", "n/a", "tba"}:
            if p not in names:
                names.append(p)
    return names


def norm_coach(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace(".", "").strip()).casefold()


_STAFF_OC_RE = re.compile(
    r"^\*+\s*(?P<pre>[^\n–—]*?)offensive coordinators?\b[^\n–—]*?"
    r"(?:–|—|\s-\s?)\s*(?P<val>.+)$", re.IGNORECASE | re.MULTILINE)


_FIRED_OC_RE = re.compile(
    r"(?:fired|dismissed|relieved)\s+(?:its\s+|their\s+)?offensive "
    r"coordinator\s*,?\s*(?P<who>\[\[[^\]]+\]\])"
    r"|offensive coordinator\s+(?P<who2>\[\[[^\]]+\]\])\s+was\s+"
    r"(?:fired|dismissed|relieved)", re.IGNORECASE)


def staff_section_ocs(text: str) -> tuple[list[str], bool]:
    """Fallback: 'Offensive coordinator – [[Name]]' lines from the page-body
    staff list. Returns (opener-first names, midseason_flag). Interim rows
    never become the opener; assistant/pass-game titles are skipped."""
    names: list[str] = []
    interim: list[str] = []
    midseason = False
    for m in _STAFF_OC_RE.finditer(text):
        pre = m.group("pre").lower()
        # exclude only when assistant/associate directly modifies the OC
        # title; "Assistant head coach/offensive coordinator" IS the OC.
        if re.search(r"(assistant|associate)\s*$", pre):
            continue
        val = m.group("val")
        if _MIDSEASON_RE.search(val):
            midseason = True
        bucket = interim if "interim" in pre else names
        for n in parse_names(val):
            if n not in bucket:
                bucket.append(n)
    if interim:
        midseason = True
    # staff lists reflect END-of-season staff; when prose records a midseason
    # OC firing, the fired coach is the season OPENER (as-of discipline).
    for m in _FIRED_OC_RE.finditer(text):
        fired = parse_names(m.group("who") or m.group("who2") or "")
        if fired:
            midseason = True
            if fired[0] not in names:
                names.insert(0, fired[0])
    return names + interim, midseason


def parse_page(text: str) -> dict:
    coach_raw = _field(text, "coach")
    oc_raw = _field(text, "off_coach")
    hcs = parse_names(coach_raw) if coach_raw else []
    ocs = parse_names(oc_raw) if oc_raw else []
    oc_midseason_extra = False
    if not ocs:
        ocs, oc_midseason_extra = staff_section_ocs(text)
    return {
        "head_coach": hcs[0] if hcs else "",
        "offensive_coordinator": ocs[0] if ocs else "",
        "midseason_hc_change": int(len(hcs) > 1 or bool(
            coach_raw and _MIDSEASON_RE.search(coach_raw))),
        "midseason_oc_change": int(len(ocs) > 1 or oc_midseason_extra or bool(
            oc_raw and _MIDSEASON_RE.search(oc_raw))),
    }


# ---------------------------------------------------------------------------
# Seed (research repo manual facts)

def load_seed(path: Path = SEED_CSV) -> dict[tuple[int, str], dict]:
    """(season, team) -> best-weight seed row (facts only)."""
    if not path.exists():
        return {}
    out: dict[tuple[int, str], dict] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("is_fact", "")).strip().lower() != "true":
                continue
            key = (int(r["season"]), NORMALIZE_TEAM.get(r["team"], r["team"]))
            w = float(r.get("play_calling_weight") or 0)
            if key not in out or w > float(
                    out[key].get("play_calling_weight") or 0):
                out[key] = r
    return out


# ---------------------------------------------------------------------------
# Assemble

def assemble(pages: dict[str, str],
             seed: dict[tuple[int, str], dict]) -> list[dict]:
    rows = []
    for team in TEAMS:
        for season in SEASONS:
            title = page_title(team, season)
            parsed = parse_page(pages[title]) if title in pages else {
                "head_coach": "", "offensive_coordinator": "",
                "midseason_hc_change": "", "midseason_oc_change": ""}
            hc, oc = parsed["head_coach"], parsed["offensive_coordinator"]
            # play caller: OC when listed, else HC; ambiguous unless a seed
            # fact pins who called plays.
            if oc:
                play_caller, pc_source, ambiguous = oc, "wikipedia_oc", 1
            elif hc:
                play_caller, pc_source, ambiguous = hc, "wikipedia_hc_no_oc", 1
            else:
                play_caller, pc_source, ambiguous = "", "missing", ""
            srow = seed.get((season, team))
            if srow and float(srow.get("play_calling_weight") or 0) >= 1.0:
                play_caller, pc_source, ambiguous = (
                    srow["coach"], "seed_fact", 0)
            rows.append({
                "season": season, "team": team,
                "head_coach": hc, "offensive_coordinator": oc,
                "play_caller": play_caller,
                "play_caller_source": pc_source,
                "play_caller_ambiguous": ambiguous,
                "midseason_hc_change": parsed["midseason_hc_change"],
                "midseason_oc_change": parsed["midseason_oc_change"],
                "source_url": ("https://en.wikipedia.org/wiki/"
                               + urllib.parse.quote(title.replace(" ", "_"))
                               if title in pages else ""),
            })
    # changed flags: opener vs prior-season opener, franchise-normalized.
    by_key = {(r["season"], r["team"]): r for r in rows}
    for r in rows:
        prev = by_key.get((r["season"] - 1, r["team"]))
        for col, field in (("changed_play_caller", "play_caller"),
                           ("changed_head_coach", "head_coach")):
            if prev and r[field] and prev[field]:
                r[col] = int(norm_coach(r[field]) != norm_coach(prev[field]))
            else:
                r[col] = ""
    return rows


COLUMNS = ["season", "team", "head_coach", "offensive_coordinator",
           "play_caller", "play_caller_source", "play_caller_ambiguous",
           "changed_play_caller", "changed_head_coach",
           "midseason_hc_change", "midseason_oc_change", "source_url"]


def write_table(rows: list[dict], path: Path = OUT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["season"], r["team"])):
            w.writerow(r)


def coverage(rows: list[dict], since: int = 2010) -> float:
    pop = [r for r in rows if r["season"] >= since]
    ok = [r for r in pop if r["play_caller"] and r["changed_play_caller"]
          != "" or (r["season"] == 2008 and r["play_caller"])]
    # 2008 has no prior season in-table; count play_caller presence there.
    return len(ok) / len(pop) if pop else 0.0


def validate_against_seed(rows: list[dict],
                          seed: dict[tuple[int, str], dict]) -> list[str]:
    """Report seed rows whose coach matches neither our OC nor HC."""
    by_key = {(r["season"], r["team"]): r for r in rows}
    issues = []
    for (season, team), s in seed.items():
        if float(s.get("play_calling_weight") or 0) < 1.0:
            continue
        r = by_key.get((season, team))
        if not r:
            continue
        ours = {norm_coach(r["head_coach"]),
                norm_coach(r["offensive_coordinator"]),
                norm_coach(r["play_caller"])}
        if norm_coach(s["coach"]) not in ours:
            issues.append(f"{season} {team}: seed says {s['coach']}, "
                          f"table has HC={r['head_coach']} "
                          f"OC={r['offensive_coordinator']}")
    return issues


# ---------------------------------------------------------------------------
# Descriptive study: breakout rate vs changed play caller, by ADP band
# (mirrors evals/source_alpha.py: pooled multiplicative lift, season-
#  resampled bootstrap CI)

def adp_band(position: str, adp_pos_rank: int) -> str:
    off = adp_pos_rank - ADP_GATE[position]
    if off <= 12:
        return "near_gate"
    if off <= 30:
        return "mid"
    return "deep"


def _lift(k_c, n_c, k_u, n_u) -> float | None:
    if n_c == 0 or n_u == 0 or k_u == 0:
        return None
    return (k_c / n_c) / (k_u / n_u)


def bootstrap_lift_ci(per_season: dict[int, dict], reps: int = BOOTSTRAP_REPS,
                      seed: int = SEED) -> tuple[float | None, float | None]:
    seasons = sorted(per_season)
    if not seasons:
        return None, None
    rng = random.Random(seed)
    lifts = []
    for _ in range(reps):
        k_c = n_c = k_u = n_u = 0
        for _ in seasons:
            c = per_season[seasons[rng.randrange(len(seasons))]]
            k_c += c["k_c"]; n_c += c["n_c"]
            k_u += c["k_u"]; n_u += c["n_u"]
        lift = _lift(k_c, n_c, k_u, n_u)
        if lift is not None:
            lifts.append(lift)
    if len(lifts) < max(100, reps // 100):
        return None, None
    lifts.sort()
    return (lifts[int(0.025 * (len(lifts) - 1))],
            lifts[int(0.975 * (len(lifts) - 1))])


def load_study_candidates(labels_csv: Path = LABELS_CSV) -> list[dict]:
    """Gate-eligible ppr breakout candidates 2010-2024 with a joinable team."""
    out = []
    with open(labels_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["format"] != "ppr":
                continue
            season = int(r["season"])
            if season not in STUDY_SEASONS:
                continue
            pos = r["position"]
            if pos not in ADP_GATE:
                continue
            rank = int(r["adp_pos_rank"]) if r["adp_pos_rank"] else None
            if rank is None or rank <= ADP_GATE[pos]:
                continue  # breakout population is beyond the ADP gate
            team = NORMALIZE_TEAM.get(r["adp_team"], r["adp_team"])
            if not team or "/" in team or team == "FA":
                continue
            out.append({"season": season, "team": team, "position": pos,
                        "band": adp_band(pos, rank),
                        "breakout": r["breakout"] == "True"})
    return out


def run_study(rows: list[dict], candidates: list[dict]) -> dict:
    changed = {(r["season"], r["team"]): r["changed_play_caller"]
               for r in rows}
    cells: dict[str, dict[int, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"k_c": 0, "n_c": 0, "k_u": 0, "n_u": 0}))
    dropped = 0
    for c in candidates:
        flag = changed.get((c["season"], c["team"]))
        if flag == "" or flag is None:
            dropped += 1
            continue
        suffix = "c" if int(flag) else "u"
        for band in ("all", c["band"]):
            cell = cells[band][c["season"]]
            cell[f"n_{suffix}"] += 1
            cell[f"k_{suffix}"] += int(c["breakout"])
    out = {"dropped_no_flag": dropped, "n_candidates": len(candidates),
           "bands": {}}
    for band, per_season in cells.items():
        tot = defaultdict(int)
        for cell in per_season.values():
            for k, v in cell.items():
                tot[k] += v
        lo, hi = bootstrap_lift_ci(per_season)
        out["bands"][band] = {
            "n_changed": tot["n_c"], "k_changed": tot["k_c"],
            "n_unchanged": tot["n_u"], "k_unchanged": tot["k_u"],
            "rate_changed": tot["k_c"] / tot["n_c"] if tot["n_c"] else None,
            "rate_unchanged": tot["k_u"] / tot["n_u"] if tot["n_u"] else None,
            "lift": _lift(tot["k_c"], tot["n_c"], tot["k_u"], tot["n_u"]),
            "ci_lo": lo, "ci_hi": hi,
        }
    return out


def _fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def write_report(rows: list[dict], study: dict, seed_issues: list[str],
                 path: Path = RESULTS_MD) -> None:
    cov_2010 = coverage(rows, 2010)
    cov_all = coverage(rows, 2008)
    lines = [
        "# Play-caller history: coverage + breakout study",
        "",
        f"Generated {date.today().isoformat()} by "
        "`scripts/collect_play_callers.py`. Source: Wikipedia team-season "
        "infoboxes (CC BY-SA) + research-repo seed facts. "
        "Pro-Football-Reference deliberately not used (ToS).",
        "",
        "## Coverage",
        "",
        f"- (season, team) rows 2008-2025: {len(rows)}",
        f"- play_caller + changed flag coverage 2010+: {cov_2010:.1%}",
        f"- coverage 2008+ (2008 lacks an in-table prior season): "
        f"{cov_all:.1%}",
        f"- seed validation issues: {len(seed_issues)}",
    ]
    lines += [f"  - {s}" for s in seed_issues]
    lines += [
        "",
        "Join key: (season, team) with vegas_features NORMALIZE_TEAM codes "
        "(Rams=LA, Chargers=LAC, Raiders=LV, Jaguars=JAX). "
        "`team_changed_play_caller` for packet_features = "
        "`changed_play_caller` from `data/processed/coaching/"
        "play_callers.csv`.",
        "",
        "## Breakout rate vs changed play caller (ppr, 2010-2024)",
        "",
        f"Gate-eligible candidates: {study['n_candidates']} "
        f"(dropped for missing flag/team: {study['dropped_no_flag']}). "
        f"Lift = rate_changed / rate_unchanged; season-resampled bootstrap "
        f"95% CI, {BOOTSTRAP_REPS} reps.",
        "",
        "| band | n changed | rate changed | n unchanged | rate unchanged "
        "| lift | 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for band in ("all", "near_gate", "mid", "deep"):
        b = study["bands"].get(band)
        if not b:
            continue
        lines.append(
            f"| {band} | {b['n_changed']} | {_fmt(b['rate_changed'])} "
            f"| {b['n_unchanged']} | {_fmt(b['rate_unchanged'])} "
            f"| {_fmt(b['lift'], 2)} "
            f"| [{_fmt(b['ci_lo'], 2)}, {_fmt(b['ci_hi'], 2)}] |")
    all_b = study["bands"].get("all", {})
    lift, lo, hi = all_b.get("lift"), all_b.get("ci_lo"), all_b.get("ci_hi")
    if lift is not None and lo is not None:
        if lo > 1.0:
            verdict = ("Changed-play-caller teams show a breakout lift whose "
                       "bootstrap CI excludes 1.0 - carries post-ADP signal.")
        elif hi < 1.0:
            verdict = ("Changed-play-caller teams break out LESS often; CI "
                       "excludes 1.0.")
        else:
            verdict = ("CI includes 1.0 - no clear pooled post-ADP signal; "
                       "check bands before using as a standalone feature.")
    else:
        verdict = "Insufficient data for a verdict."
    lines += [
        "", f"**Verdict:** {verdict}", "",
        "## Limitations",
        "",
        "- Who *actually* called plays (HC vs OC) is ambiguous from "
        "structured sources; the table records the OC (HC when no OC is "
        "listed) with `play_caller_ambiguous=1` unless a research-repo seed "
        "fact pins it. `changed_play_caller` is the load-bearing flag.",
        "- Wikipedia staff lists reflect end-of-season staff; prose-recorded "
        "midseason OC firings are folded back to the opener, but an "
        "unrecorded midseason change could mislabel an opener (noise "
        "attenuates the lift rather than inflating it).",
        "- Study is descriptive (no controls beyond ADP bands); the `deep` "
        "band is tiny. 2025 is the untouched eval holdout and is excluded.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    titles = [page_title(t, s) for t in TEAMS for s in SEASONS]
    pages = fetch_all(titles)
    write_snapshot(pages)
    seed = load_seed()
    rows = assemble(pages, seed)
    write_table(rows)
    cov = coverage(rows)
    issues = validate_against_seed(rows, seed)
    print(f"wrote {OUT_CSV} ({len(rows)} rows; coverage 2010+ {cov:.1%}; "
          f"{len(issues)} seed issues)")
    for s in issues:
        print("  seed mismatch:", s)
    if "--study" in argv:
        study = run_study(rows, load_study_candidates())
        write_report(rows, study, issues)
        print(f"wrote {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
