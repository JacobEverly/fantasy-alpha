#!/usr/bin/env python3
"""Historical source-credibility table: whose preseason claims predicted outcomes.

For each season S in 2016-2024 (ppr labels), the study window is the 90 days
strictly before Sept 1 of S (the BreakoutBench as-of gate). Every news item in
the window is pulled through ``harness.evidence.EvidenceStore`` — the time
gate (published < as_of) lives in the store, never here — and classified into
claim types with a versioned keyword lexicon that extends the flag regexes in
``evals/packet_features.py``.

For each (source x claim_type) cell and pooled across sources, among
*gate-eligible breakout candidates* (breakouts.csv, ppr, adp_pos_rank beyond
the v0.2 ADP gates): breakout rate of players who received >=1 such claim
in-window vs those who received none, lift over the no-claim group, lift over
the positional base rate (expected hits given the claimed group's position
mix), and a season-resampled bootstrap 95% CI on the lift (the established
pattern: seasons are the independent unit, 10k resamples, seeded).

Bust analysis: among top-12 positional ADP picks, do August INJURY_CONCERN /
NEGATIVE claims predict busts? Reported twice: all outcomes, and
freak-excluded — candidates whose season ended with an acute in-season injury
(injury_context.csv season_ending_acute) are dropped from the population
entirely, because an August blurb cannot get credit for "predicting" a week-2
ACL (nor be penalized for missing one).

Volume control: August coverage correlates with fame. Pooled per-claim-type
lifts are re-reported within positional-ADP-rank bands (distance beyond the
gate) so "gets written about" is not conflated with "gets written about
POSITIVELY".

Outputs:
  data/processed/labels/source_alpha.csv
  evals/results/source_alpha.md

Usage:
  .venv/bin/python -m evals.source_alpha run [--reps 10000] [--seed 20260808]

Stdlib only. Python 3.11+.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from evals.names import norm_name
from evals.build_breakout_labels import ADP_GATE, BUST_ADP_GATE
from harness.evidence import INDEX_DIR, EvidenceStore

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "processed" / "labels"
OUT_CSV = LABELS / "source_alpha.csv"
OUT_MD = ROOT / "evals" / "results" / "source_alpha.md"

SEASONS = tuple(range(2016, 2025))  # news archive starts ~2016-01
FORMAT = "ppr"
WINDOW_DAYS = 90
SOURCES = ("rotowire", "rotoballer", "fantasy_pros")
POOLED = "pooled"
BOOTSTRAP_REPS = 10_000
SEED = 20260808

# ---------------------------------------------------------------------------
# Claim lexicon (versioned). Extends evals/packet_features.py flag regexes:
#   CAMP_PROMOTION_RE: first[- ]team|\bstarter\b|\bstarting\b|\bWR1\b|\bRB1\b
#   INJURY_RE:         injur|ACL|achilles|hamstring|concussion|sprain|fractur|
#                      surgery|MCL|high[- ]ankle|IR
#   HYPE_RE:           breakout|impress
# One item can carry multiple claim types. Classification is over
# title + " " + text, case-insensitive, pure regex — deterministic by
# construction (same string -> same set, no state).

LEXICON_VERSION = "source-alpha-lexicon-v1"

CLAIM_TYPES = (
    "CAMP_PROMOTION",
    "ROLE_EXPANSION",
    "INJURY_RECOVERY_POSITIVE",
    "INJURY_CONCERN",
    "HYPE_SOFT",
    "NEGATIVE",
)

# packet_features.CAMP_PROMOTION_RE uses bare \bstarter\b|\bstarting\b; in
# full blurb prose that matches preseason-game logistics ("starters will play
# Thursday") on ~80% of candidates and carries no discrimination, so the
# claim-level flag requires promotion-specific phrasing around the word.
CAMP_PROMOTION_RE = re.compile(
    r"first[- ]team|with the (?:1s|ones|starters)\b"
    r"|named (?:the |him )?(?:\w+\s)?starter|listed as (?:the |a )?starter"
    r"|won the (?:starting )?job|starting job is his"
    r"|\bWR1\b|\bRB1\b|\bTE1\b|\bQB1\b"
    r"|promot(?:ed|ion)|lead\s+(?:role|back|receiver)"
    r"|no\.\s*1\s+(?:receiver|wideout|back|running back|tight end|option)"
    r"|top of the depth chart|atop the depth chart",
    re.IGNORECASE)

ROLE_EXPANSION_RE = re.compile(
    r"(?:expanded|increased|bigger|larger|growing|expanding)\s+"
    r"(?:role|workload|share|usage)"
    r"|more\s+(?:targets|touches|snaps|carries|work|involvement|opportunit)"
    r"|uptick in\s+(?:targets|touches|snaps|usage|work)"
    r"|increase[d]?\s+(?:targets|touches|snaps|workload|usage)",
    re.IGNORECASE)

INJURY_RECOVERY_POSITIVE_RE = re.compile(
    r"cleared(?!\s+waivers)|full participant"
    r"|fully\s+(?:healthy|recovered|cleared|practicing)"
    r"|return(?:ed|s)? to (?:practice|the field|team drills)"
    r"|good to go|removed from (?:the )?(?:PUP|physically unable)"
    r"|activated (?:off|from)|back at practice|100 percent|100%"
    r"|no (?:setbacks|limitations|restrictions)",
    re.IGNORECASE)

# packet_features.INJURY_RE, reused verbatim as the injury-noun layer
INJURY_NOUN_RE = re.compile(
    r"injur|\bACL\b|achilles|hamstring|concussion|sprain|fractur|surgery"
    r"|\bMCL\b|high[- ]ankle|\bIR\b", re.IGNORECASE)

INJURY_CONCERN_TERM_RE = re.compile(
    r"limited (?:participant|in practice|practice|at practice)|setback"
    r"|questionable|doubtful|will miss|expected to miss|sidelined"
    r"|day[- ]to[- ]day|carted|placed on (?:the )?(?:IR|injured reserve|PUP)"
    r"|underwent surgery|week[- ]to[- ]week|re[- ]?aggravat",
    re.IGNORECASE)

HYPE_SOFT_RE = re.compile(
    r"breakout|impress|stand[- ]?out|standout|turn(?:ing|ed) heads|stood out"
    r"|camp buzz|\bhype\b|\bsleeper\b|eye[- ](?:opening|popping)",
    re.IGNORECASE)

NEGATIVE_RE = re.compile(
    r"demot(?:ed|ion)|buried|\bwaived\b|\breleased\b|competition"
    r"|battl(?:e|ing) for (?:the |a )?(?:starting )?job|lost the job"
    r"|passed on the depth chart|f(?:a|e)ll(?:s|ing)? behind|benched",
    re.IGNORECASE)


def classify(text: str) -> frozenset[str]:
    """Claim types carried by one news item's text. Deterministic regex-only.

    INJURY_CONCERN fires on an explicit concern term, OR on an injury noun
    that is not framed as recovery (so "fully recovered from ACL surgery" is
    recovery-only, "hamstring injury, limited" is concern).
    """
    out: set[str] = set()
    if CAMP_PROMOTION_RE.search(text):
        out.add("CAMP_PROMOTION")
    if ROLE_EXPANSION_RE.search(text):
        out.add("ROLE_EXPANSION")
    recovery = bool(INJURY_RECOVERY_POSITIVE_RE.search(text))
    if recovery:
        out.add("INJURY_RECOVERY_POSITIVE")
    if INJURY_CONCERN_TERM_RE.search(text) or (
            INJURY_NOUN_RE.search(text) and not recovery):
        out.add("INJURY_CONCERN")
    if HYPE_SOFT_RE.search(text):
        out.add("HYPE_SOFT")
    if NEGATIVE_RE.search(text):
        out.add("NEGATIVE")
    return frozenset(out)


# ---------------------------------------------------------------------------
# Windows and ADP bands

def window_for(season: int) -> tuple[date, date]:
    """[gate - 90d, gate) — gate is Sept 1 of the season (exclusive)."""
    gate = date(season, 9, 1)
    return gate - timedelta(days=WINDOW_DAYS), gate


def adp_band(position: str, adp_pos_rank: int) -> str:
    """Positional-ADP-rank band for gate-eligible breakout candidates,
    measured as distance beyond the eligibility gate."""
    off = adp_pos_rank - ADP_GATE[position]
    if off <= 12:
        return "near_gate"
    if off <= 30:
        return "mid"
    return "deep"


# ---------------------------------------------------------------------------
# Candidate populations (breakouts.csv, ppr)

def load_candidates(fmt: str = FORMAT) -> dict[int, dict[str, list[dict]]]:
    """season -> {"breakout": [...], "bust": [...]} candidate rows.

    breakout population: gate-eligible (adp_pos_rank > v0.2 ADP gate),
    matching evals/harness_breakout.load_breakout_rows.
    bust population: drafted top-12 positional (adp_pos_rank <= 12).
    """
    out: dict[int, dict[str, list[dict]]] = {
        s: {"breakout": [], "bust": []} for s in SEASONS}
    with open(LABELS / "breakouts.csv", newline="") as f:
        for r in csv.DictReader(f):
            season = int(r["season"])
            if r["format"] != fmt or season not in out or not r["adp_pos_rank"]:
                continue
            pos, rank = r["position"], int(r["adp_pos_rank"])
            if pos not in ADP_GATE:
                continue
            row = {
                "season": season,
                "player": r["player"],
                "player_id": r["player_id"],
                "position": pos,
                "adp_pos_rank": rank,
                "breakout": r["breakout"] == "True",
                "bust": r["bust"] == "True",
                "alpha": float(r["alpha"]) if r["alpha"] else None,
            }
            if rank > ADP_GATE[pos]:
                row["band"] = adp_band(pos, rank)
                out[season]["breakout"].append(row)
            if rank <= BUST_ADP_GATE:
                out[season]["bust"].append(row)
    return out


def load_freak_exclusions() -> set[tuple[int, str]]:
    """(season, player_id) whose season ended with an acute in-season injury —
    excluded entirely from the freak-excluded bust population."""
    out: set[tuple[int, str]] = set()
    with open(LABELS / "injury_context.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["season_ending_acute"] == "True":
                out.add((int(r["season"]), r["player_id"]))
    return out


# ---------------------------------------------------------------------------
# Claim fetch (per season, batched over the per-player index streams)

def _player_resolver(index_dir: Path) -> dict:
    """(norm_name, position) and norm_name -> sleeper pid maps from the
    evidence index's players.json (read-only)."""
    blob = json.loads((Path(index_dir) / "players.json").read_text())
    players = blob["players"]
    by_norm = blob["by_norm_name"]
    return {"players": players, "by_norm": by_norm}


def resolve_pid(resolver: dict, player: str, position: str) -> str | None:
    pids = resolver["by_norm"].get(norm_name(player), [])
    if len(pids) > 1:
        pids = [p for p in pids
                if resolver["players"][p].get("position") == position]
    return pids[0] if len(pids) == 1 else None


def fetch_season_claims(
    season: int,
    candidates: list[dict],
    index_dir: Path | str = INDEX_DIR,
) -> tuple[dict[str, dict[str, set[str]]], dict]:
    """gsis player_id -> source -> set(claim types) for all in-window items.

    All retrieval goes through EvidenceStore(as_of=Sept 1) so the time gate is
    the store's, and the `since` bound trims to the 90-day window (inclusive
    start, exclusive gate — EvidenceStore keeps published >= since and
    hard-excludes published >= as_of).
    """
    start, gate = window_for(season)
    store = EvidenceStore(as_of=gate, index_dir=index_dir)
    resolver = _player_resolver(Path(index_dir))
    claims: dict[str, dict[str, set[str]]] = {}
    stats = {"season": season, "n_candidates": len(candidates),
             "n_resolved": 0, "n_unresolved": 0, "n_items": 0}
    seen: set[str] = set()
    for c in sorted(candidates, key=lambda c: (c["player_id"], c["player"])):
        gsis = c["player_id"]
        if gsis in seen:
            continue
        seen.add(gsis)
        pid = resolve_pid(resolver, c["player"], c["position"])
        if pid is None:
            stats["n_unresolved"] += 1
            continue
        stats["n_resolved"] += 1
        items = store.player_news(pid, limit=10_000, since=start)
        stats["n_items"] += len(items)
        for it in items:
            types = classify(f"{it.get('title') or ''} {it.get('text') or ''}")
            if not types:
                continue
            src = it["source"]
            by_src = claims.setdefault(gsis, {})
            by_src.setdefault(src, set()).update(types)
            by_src.setdefault(POOLED, set()).update(types)
    return claims, stats


# ---------------------------------------------------------------------------
# Cell math

def season_cell(rows: list[dict], claims: dict[str, dict[str, set[str]]],
                source: str, claim_type: str, outcome_key: str,
                base_rates: dict[str, float]) -> dict:
    """One season's counts for a (source, claim_type) cell.

    base_rates: position -> outcome base rate among this season's population
    (used for the expected-hits-by-position-mix denominator).
    """
    k_c = n_c = k_u = n_u = 0
    exp_c = 0.0
    a_c: list[float] = []
    a_u: list[float] = []
    for r in rows:
        claimed = claim_type in claims.get(r["player_id"], {}).get(source, set())
        hit = int(bool(r[outcome_key]))
        if claimed:
            n_c += 1
            k_c += hit
            exp_c += base_rates.get(r["position"], 0.0)
            if r["alpha"] is not None:
                a_c.append(r["alpha"])
        else:
            n_u += 1
            k_u += hit
            if r["alpha"] is not None:
                a_u.append(r["alpha"])
    return {"k_c": k_c, "n_c": n_c, "k_u": k_u, "n_u": n_u, "exp_c": exp_c,
            "alpha_c_sum": sum(a_c), "alpha_c_n": len(a_c),
            "alpha_u_sum": sum(a_u), "alpha_u_n": len(a_u)}


def position_base_rates(rows: list[dict], outcome_key: str) -> dict[str, float]:
    by_pos: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_pos[r["position"]].append(int(bool(r[outcome_key])))
    return {p: (sum(v) / len(v) if v else 0.0) for p, v in by_pos.items()}


def _lift(k_c: int, n_c: int, k_u: int, n_u: int) -> float | None:
    if n_c == 0 or n_u == 0 or k_u == 0:
        return None
    return (k_c / n_c) / (k_u / n_u)


def pool_cells(per_season: dict[int, dict]) -> dict:
    """Point estimates pooled over seasons (sum counts, then ratio)."""
    tot = defaultdict(float)
    for c in per_season.values():
        for k, v in c.items():
            tot[k] += v
    claimed_rate = tot["k_c"] / tot["n_c"] if tot["n_c"] else None
    unclaimed_rate = tot["k_u"] / tot["n_u"] if tot["n_u"] else None
    return {
        "n_claimed": int(tot["n_c"]), "n_unclaimed": int(tot["n_u"]),
        "claimed_rate": claimed_rate, "unclaimed_rate": unclaimed_rate,
        "lift": _lift(int(tot["k_c"]), int(tot["n_c"]),
                      int(tot["k_u"]), int(tot["n_u"])),
        "lift_vs_pos_base": (tot["k_c"] / tot["exp_c"]
                             if tot["exp_c"] > 0 else None),
        "alpha_claimed_mean": (tot["alpha_c_sum"] / tot["alpha_c_n"]
                               if tot["alpha_c_n"] else None),
        "alpha_unclaimed_mean": (tot["alpha_u_sum"] / tot["alpha_u_n"]
                                 if tot["alpha_u_n"] else None),
    }


def bootstrap_lift_ci(per_season: dict[int, dict], reps: int = BOOTSTRAP_REPS,
                      seed: int = SEED) -> tuple[float | None, float | None]:
    """Season-resampled bootstrap 95% CI on the pooled lift (the established
    pattern: seasons are the independent unit; draws where the lift is
    undefined — empty claimed group or zero unclaimed hits — are skipped)."""
    seasons = sorted(per_season)
    if not seasons:
        return None, None
    rng = random.Random(seed)
    lifts: list[float] = []
    for _ in range(reps):
        k_c = n_c = k_u = n_u = 0
        for _ in seasons:
            c = per_season[seasons[rng.randrange(len(seasons))]]
            k_c += c["k_c"]; n_c += c["n_c"]
            k_u += c["k_u"]; n_u += c["n_u"]
        lift = _lift(k_c, n_c, k_u, n_u)
        if lift is not None:
            lifts.append(lift)
    if len(lifts) < max(100, reps // 100):  # too degenerate to interpret
        return None, None
    lifts.sort()
    lo = lifts[int(0.025 * (len(lifts) - 1))]
    hi = lifts[int(0.975 * (len(lifts) - 1))]
    return lo, hi


# ---------------------------------------------------------------------------
# Study assembly

def _fmt(v, nd=3):
    return "" if v is None else round(v, nd)


def run_study(index_dir: Path | str = INDEX_DIR, reps: int = BOOTSTRAP_REPS,
              seed: int = SEED, quiet: bool = False) -> dict:
    cands = load_candidates()
    freak = load_freak_exclusions()

    # one claim fetch per season, over the union of both populations
    season_claims: dict[int, dict] = {}
    fetch_stats: list[dict] = []
    for s in SEASONS:
        union = {r["player_id"]: r
                 for r in cands[s]["breakout"] + cands[s]["bust"]}
        claims, stats = fetch_season_claims(s, list(union.values()), index_dir)
        season_claims[s] = claims
        fetch_stats.append(stats)
        if not quiet:
            print(f"  {s}: {stats['n_resolved']}/{stats['n_candidates']} "
                  f"resolved, {stats['n_items']} in-window items")

    rows_out: list[dict] = []
    sources = list(SOURCES) + [POOLED]

    def emit(outcome: str, source: str, claim_type: str,
             pops: dict[int, list[dict]], band: str,
             outcome_key: str) -> None:
        per_season: dict[int, dict] = {}
        for s in SEASONS:
            rows = pops[s]
            if not rows:
                continue
            base = position_base_rates(rows, outcome_key)
            per_season[s] = season_cell(rows, season_claims[s], source,
                                        claim_type, outcome_key, base)
        pooled = pool_cells(per_season)
        lo, hi = bootstrap_lift_ci(per_season, reps, seed)
        rows_out.append({
            "source": source, "claim_type": claim_type, "season": POOLED,
            "n_claimed": pooled["n_claimed"],
            "n_unclaimed": pooled["n_unclaimed"],
            "claimed_rate": _fmt(pooled["claimed_rate"]),
            "unclaimed_rate": _fmt(pooled["unclaimed_rate"]),
            "lift": _fmt(pooled["lift"]),
            "ci_lo": _fmt(lo), "ci_hi": _fmt(hi),
            "adp_band": band, "outcome": outcome,
            "lift_vs_pos_base": _fmt(pooled["lift_vs_pos_base"]),
            "alpha_claimed_mean": _fmt(pooled["alpha_claimed_mean"], 2),
            "alpha_unclaimed_mean": _fmt(pooled["alpha_unclaimed_mean"], 2),
        })
        if band != "all":
            return  # strata rows: pooled only
        for s, c in sorted(per_season.items()):
            rows_out.append({
                "source": source, "claim_type": claim_type, "season": s,
                "n_claimed": c["n_c"], "n_unclaimed": c["n_u"],
                "claimed_rate": _fmt(c["k_c"] / c["n_c"] if c["n_c"] else None),
                "unclaimed_rate": _fmt(c["k_u"] / c["n_u"] if c["n_u"] else None),
                "lift": _fmt(_lift(c["k_c"], c["n_c"], c["k_u"], c["n_u"])),
                "ci_lo": "", "ci_hi": "",  # one season cannot be season-resampled
                "adp_band": band, "outcome": outcome,
                "lift_vs_pos_base": _fmt(c["k_c"] / c["exp_c"]
                                         if c["exp_c"] > 0 else None),
                "alpha_claimed_mean": _fmt(c["alpha_c_sum"] / c["alpha_c_n"]
                                           if c["alpha_c_n"] else None, 2),
                "alpha_unclaimed_mean": _fmt(c["alpha_u_sum"] / c["alpha_u_n"]
                                             if c["alpha_u_n"] else None, 2),
            })

    # 1) breakout cells: source x claim_type, seasons + pooled
    bo_pop = {s: cands[s]["breakout"] for s in SEASONS}
    for source in sources:
        for ct in CLAIM_TYPES:
            emit("breakout", source, ct, bo_pop, "all", "breakout")

    # 2) ADP-band strata (volume control): pooled source, pooled seasons
    for band in ("near_gate", "mid", "deep"):
        pop = {s: [r for r in cands[s]["breakout"] if r["band"] == band]
               for s in SEASONS}
        for ct in CLAIM_TYPES:
            emit("breakout", POOLED, ct, pop, band, "breakout")

    # 3) bust cells (top-12 positional ADP): all outcomes + freak-excluded
    bust_pop = {s: cands[s]["bust"] for s in SEASONS}
    bust_pop_fx = {s: [r for r in cands[s]["bust"]
                       if (s, r["player_id"]) not in freak]
                   for s in SEASONS}
    for source in sources:
        for ct in ("INJURY_CONCERN", "NEGATIVE"):
            emit("bust_all", source, ct, bust_pop, "all", "bust")
            emit("bust_freak_excluded", source, ct, bust_pop_fx, "all",
                 "bust")

    return {"rows": rows_out, "fetch_stats": fetch_stats,
            "n_breakout_candidates": sum(len(v) for v in bo_pop.values()),
            "n_bust_candidates": sum(len(v) for v in bust_pop.values()),
            "n_bust_candidates_fx": sum(len(v) for v in bust_pop_fx.values())}


# ---------------------------------------------------------------------------
# Outputs

CSV_COLUMNS = ["source", "claim_type", "season", "n_claimed", "n_unclaimed",
               "claimed_rate", "unclaimed_rate", "lift", "ci_lo", "ci_hi",
               "adp_band", "outcome", "lift_vs_pos_base",
               "alpha_claimed_mean", "alpha_unclaimed_mean"]


def write_csv(rows: list[dict], path: Path = OUT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _cell_str(r: dict) -> str:
    if r["lift"] == "":
        return "—"
    ci = (f" [{r['ci_lo']}, {r['ci_hi']}]"
          if r["ci_lo"] != "" else "")
    return f"{r['lift']}{ci} (n={r['n_claimed']})"


def write_md(study: dict, path: Path = OUT_MD, reps: int = BOOTSTRAP_REPS,
             seed: int = SEED) -> None:
    rows = study["rows"]
    idx = {(r["outcome"], r["source"], r["claim_type"], str(r["season"]),
            r["adp_band"]): r for r in rows}
    sources = list(SOURCES) + [POOLED]

    lines: list[str] = []
    a = lines.append
    a("# Source credibility: whose August claims predicted outcomes")
    a("")
    a(f"Generated by `evals/source_alpha.py` (lexicon `{LEXICON_VERSION}`, "
      f"seed {seed}, {reps:,} season-resampled bootstrap draws). "
      f"Seasons {SEASONS[0]}-{SEASONS[-1]}, ppr labels; window = the 90 days "
      "strictly before Sept 1, retrieved via `EvidenceStore` (the time gate "
      "lives in the store).")
    a("")
    a("## Populations")
    a("")
    a(f"- Breakout candidates (gate-eligible, v0.2 ADP gates): "
      f"{study['n_breakout_candidates']} player-seasons")
    a(f"- Bust candidates (top-12 positional ADP): "
      f"{study['n_bust_candidates']} player-seasons "
      f"({study['n_bust_candidates_fx']} after freak-injury exclusion)")
    for s in study["fetch_stats"]:
        a(f"  - {s['season']}: {s['n_resolved']}/{s['n_candidates']} players "
          f"resolved to the news index, {s['n_items']} in-window items")
    a("")
    a("## Headline: breakout lift by (claim type x source), pooled seasons")
    a("")
    a("Lift = breakout rate of candidates with >=1 in-window claim of the "
      "type from the source, over the rate of candidates with none. "
      "`[lo, hi]` = season-resampled bootstrap 95% CI; n = claimed group "
      "size. Cells without a CI were too sparse to bootstrap.")
    a("")
    a("| claim_type | " + " | ".join(sources) + " |")
    a("|---|" + "---|" * len(sources))
    for ct in CLAIM_TYPES:
        cells = [_cell_str(idx[("breakout", src, ct, POOLED, "all")])
                 for src in sources]
        a(f"| {ct} | " + " | ".join(cells) + " |")
    a("")
    a("Pooled-source rows also carry `lift_vs_pos_base` (observed breakouts "
      "over those expected from the claimed group's positional base rates) "
      "and realized-alpha means in the CSV.")
    a("")
    a("## Volume control: pooled lift within positional-ADP bands")
    a("")
    a("Players with more August coverage are more famous. Within-band lifts "
      "(distance of positional ADP rank beyond the eligibility gate: "
      "near_gate = +1..+12, mid = +13..+30, deep = +31+) separate \"gets "
      "written about\" from \"gets written about positively\".")
    a("")
    a("| claim_type | near_gate | mid | deep |")
    a("|---|---|---|---|")
    for ct in CLAIM_TYPES:
        cells = [_cell_str(idx[("breakout", POOLED, ct, POOLED, b)])
                 for b in ("near_gate", "mid", "deep")]
        a(f"| {ct} | " + " | ".join(cells) + " |")
    a("")
    a("## Busts: do August concerns predict them?")
    a("")
    a("Population: top-12 positional ADP picks. `freak_excluded` drops "
      "players whose season ended on an acute in-season injury "
      "(`injury_context.csv` season_ending_acute) from the population "
      "entirely — an August blurb cannot predict a week-2 ACL, and is not "
      "penalized for missing one.")
    a("")
    a("| claim_type | outcome | " + " | ".join(sources) + " |")
    a("|---|---|" + "---|" * len(sources))
    for ct in ("INJURY_CONCERN", "NEGATIVE"):
        for oc in ("bust_all", "bust_freak_excluded"):
            cells = [_cell_str(idx[(oc, src, ct, POOLED, "all")])
                     for src in sources]
            a(f"| {ct} | {oc} | " + " | ".join(cells) + " |")
    a("")
    a("## Caveats (honest ones)")
    a("")
    a("- **Only three sources**, and all three arrive via Sleeper's blurb "
      "feed. Rotowire/RotoBaller/FantasyPros blurbs are themselves "
      "aggregations of primary reporting (beat writers, coach quotes), so "
      "these lifts measure the *pipeline*, not original sourcing, and the "
      "three columns are heavily correlated — a real camp story is usually "
      "blurbed by all of them within a day.")
    a("- **Lexicon is keyword regex** (`" + LEXICON_VERSION + "`, extending "
      "`evals/packet_features.py`). No negation handling beyond the "
      "recovery/concern split; sarcasm, hedges, and \"competition\" used "
      "positively will misclassify. Rerun the table when the lexicon "
      "version changes; do not compare lifts across lexicon versions.")
    a("- **Small cells.** Gate-eligible pools are shallow (~40-130 "
      "candidates/season), so many per-season lifts are noise; trust only "
      "pooled rows whose bootstrap CI excludes 1.0, and even those are "
      f"based on {len(SEASONS)} seasons.")
    a("- **Coverage != claims.** The volume-control bands mitigate but do "
      "not remove fame confounding; deeper strata have fewer claimed "
      "players and wider CIs.")
    a("- **Population is conditioned on the FFC ADP pool and the v0.2 "
      "gates.** Camp-promotion breakouts drafted inside the gate (e.g. 2023 "
      "LaPorta at TE) or absent from the ADP pool entirely never enter the "
      "denominator, so these lifts answer \"among draftable longshots, does "
      "the claim separate hitters from missers\", not \"do promotions ever "
      "matter\".")
    a("- **Name resolution** joins labels to the news index on normalized "
      "name + position; a handful of candidates per season don't resolve "
      "and are treated as unclaimed (counts above).")
    a("- Sleeper-derived blurb text is internal-research-only; this table "
      "publishes derived statistics, never the text.")
    a("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="compute the table + report")
    r.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    r.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        print(f"source_alpha: seasons {SEASONS[0]}-{SEASONS[-1]}, "
              f"lexicon {LEXICON_VERSION}")
        study = run_study(reps=args.reps, seed=args.seed)
        write_csv(study["rows"])
        write_md(study, reps=args.reps, seed=args.seed)
        print(f"wrote {OUT_CSV.relative_to(ROOT)} "
              f"({len(study['rows'])} rows) and {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
