#!/usr/bin/env python3
"""Classify injury-report causes and attribute missed games -> injury_context.csv.

BreakoutBench v0.3 (docs/breakoutbench-design.md amendment): a player's own
acute traumatic event (ACL/Achilles/fracture class) is unforecastable noise,
while chronic/soft-tissue durability risk is predictable skill. This builder
labels every (player, season) with games missed by injury cause class so
benchmark metrics can be reported in two panels (all-outcomes vs
freak-excluded).

Outputs:
- data/processed/labels/injury_context.csv — per (player, season): games
  played, scheduled-team games missed while carrying an injury designation,
  attributed by class, plus a `season_ending_acute` flag.
- data/processed/labels/injury_context_unmatched.txt — raw injury strings the
  lexicon could not classify (fuel for lexicon iteration).
- evals/results/injury_context_report.md — cross-tab vs breakouts.csv (ppr):
  bust class split, base-rate shift when acute-injury busts are excluded,
  face-validity reclassification list.

Schema notes (what data/raw/nflverse/injuries/*.csv actually contains):
- Columns: season, game_type, team, week, gsis_id, position, full_name,
  report_primary_injury, report_secondary_injury, report_status
  (Out/Doubtful/Questionable/Probable/blank — no IR status), practice_*.
- Injury text is body-part granularity ("Knee", "Achilles", "Hamstring") —
  the words "torn"/"ACL"/"fracture" essentially never appear in-week; the
  bone/tendon-specific terms (achilles, fibula, tibia, collarbone) are the
  observable acute markers.
- Players on injured reserve DROP OFF the report entirely unless a practice
  window opens (Aaron Rodgers 2023 appears only weeks 13-18). Attribution
  therefore carries a designation forward across report-less missed weeks
  until the player next plays.
- Preseason injuries (July/August ACLs) never appear: a top pick with zero
  games and zero in-season rows is unattributable from this source.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from evals.names import norm_name

ROOT = Path(__file__).resolve().parent.parent
INJ_DIR = ROOT / "data" / "raw" / "nflverse" / "injuries"
STATS_DIR = ROOT / "data" / "raw" / "nflverse" / "stats_player"
LABELS = ROOT / "data" / "processed" / "labels"
RESULTS = ROOT / "evals" / "results"

SEASONS = range(2011, 2026)
POSITIONS = {"QB", "RB", "WR", "TE"}

ACUTE, SOFT, OTHER = "ACUTE_TRAUMATIC", "SOFT_TISSUE_CHRONIC", "OTHER"

# ---------------------------------------------------------------------------
# Classification lexicon — versioned; changes require a version bump.
# Precedence: ACUTE > SOFT > tagged OTHER > EXCLUDED > body-part OTHER >
# unmatched OTHER (raw string logged). Multi-injury strings ("Knee, Achilles")
# resolve to the most severe class present.

LEXICON_VERSION = "2026-08-08.v1"

ACUTE_PATTERNS = [
    r"\bacl\b", r"\bmcl\b", r"\bpcl\b", r"\blcl\b", r"achil", r"\btorn\b",
    r"\btear\b", r"ruptur", r"fractur", r"\bbroken?\b", r"dislocat",
    r"lisfranc", r"patellar tendon", r"\bfibula", r"\btibia", r"collarbone",
    r"clavic",  # clavicle/sternoclavicular — joint trauma, acute
]
SOFT_PATTERNS = [
    r"hamstring", r"groin", r"\bcal(f|ves)\b", r"\bquad", r"thigh",
    r"oblique", r"adductor", r"hip flexor", r"glute", r"soft tissue",
    r"tight", r"\bstrain", r"\bcore\b", r"pector", r"\bbicep", r"\btricep",
    r"cramp", r"hernia",
]
# OTHER with explicit subtags (concussion kept separately queryable).
CONCUSSION_PATTERNS = [r"concussion"]
ILLNESS_PATTERNS = [
    r"illness", r"migraine", r"infection", r"\bcovid", r"\bflu\b",
    r"appendix", r"\bsick", r"nir-medical", r"\bmedical\b",
]
PERSONAL_PATTERNS = [r"personal"]
# Not injury designations at all — rest days, coach/roster decisions.
EXCLUDED_PATTERNS = [
    r"not injury related", r"not football related", r"\brest", r"veteran",
    r"coach", r"travel", r"suspension", r"load management", r"^--$",
    r"^other$", r"ramp up",
]
# Recognized body parts / generic injuries -> OTHER (subtag "body").
BODY_PATTERNS = [
    r"knee", r"ankle", r"shoulder", r"\bf(oo|ee)t\b", r"\bback\b", r"\bhip",
    r"neck", r"\btoe", r"elbow", r"\bhand", r"wrist", r"\brib", r"thumb",
    r"chest", r"abdom", r"finger", r"heel", r"\bshin\b", r"\bhead\b",
    r"forearm", r"\beye", r"\bjaw\b", r"\bt(oo|ee)th\b", r"nose", r"mouth",
    r"\bear\b", r"chin", r"face", r"throat", r"kidney", r"liver", r"\blung",
    r"stomach", r"pelvis", r"tailbone", r"stinger", r"\barm\b", r"\bleg\b",
    r"buttock", r"\btrap\b", r"lumbar", r"spine", r"upper arm", r"lower leg",
    r"heat", r"concussion protocol",
]

# Structural escalation: weekly reports never say "torn ACL" — a season tear
# is reported as "Knee" and the player then vanishes to IR. A designation on a
# structural (bone/joint) body part that proves SEASON-ENDING (player never
# plays again despite >=3 remaining team games) is escalated OTHER -> ACUTE
# with subtag "escalated_structural". Soft-tissue season-enders (hamstring IR)
# stay SOFT — that is the predictable durability class v0.3 wants kept in.
ESCALATION_PATTERNS = [
    r"knee", r"ankle", r"\bf(oo|ee)t\b", r"\bleg\b", r"\btoe", r"heel",
    r"\bshin\b", r"\bhip", r"pelvis", r"shoulder", r"elbow", r"wrist",
    r"\bhand", r"thumb", r"forearm", r"\barm\b", r"\brib", r"chest",
    r"neck", r"\bjaw\b",
]
ESCALATION_RX = re.compile("|".join(ESCALATION_PATTERNS))

_TIERS = [
    (ACUTE, "acute", ACUTE_PATTERNS),
    (SOFT, "soft", SOFT_PATTERNS),
    (OTHER, "concussion", CONCUSSION_PATTERNS),
    (OTHER, "illness", ILLNESS_PATTERNS),
    (OTHER, "personal", PERSONAL_PATTERNS),
    (None, "excluded", EXCLUDED_PATTERNS),
    (OTHER, "body", BODY_PATTERNS),
]
_COMPILED = [(cls, tag, re.compile("|".join(pats))) for cls, tag, pats in _TIERS]


def classify(text: str) -> tuple[str, str] | None:
    """Injury-report text -> (class, subtag), or None when not an injury
    designation (empty text / rest / coach's decision). Unmatched non-empty
    text -> (OTHER, "unmatched") — caller logs the raw string."""
    t = text.strip().lower()
    if not t:
        return None
    for cls, tag, rx in _COMPILED:
        if rx.search(t):
            return None if cls is None else (cls, tag)
    return (OTHER, "unmatched")


# ---------------------------------------------------------------------------
# Attribution: a missed scheduled-team game attributes to the most recent
# designation not cleared by an intervening played game (IR carry-forward).


def attribute_season(played: set[int], team_weeks: set[int],
                     designations: list[dict]) -> dict:
    """Pure attribution over one (player, season).

    played: weeks the player appeared in; team_weeks: weeks the player's team
    played; designations: [{week, cls, subtag, text}] sorted by week.
    A designation at week W stays active for missed week m >= W unless the
    player played some week in (W, m]. season_ending_acute = an ACUTE
    designation with no played week after it and >=3 team games after it
    (structural body-part designations meeting that condition are first
    escalated to ACUTE — see ESCALATION_PATTERNS). vanished_unattributed
    (not written to the CSV; feeds the cross-tab report) = player played,
    then missed >=3 trailing team games with no designation at all — the
    straight-to-IR pattern this data source cannot attribute.
    """
    designations = sorted((dict(d) for d in designations), key=lambda d: d["week"])

    def season_ending(d: dict) -> bool:
        return (not any(w > d["week"] for w in played)
                and sum(1 for w in team_weeks if w > d["week"]) >= 3)

    for d in designations:
        if (d["cls"] == OTHER and d["subtag"] == "body" and season_ending(d)
                and ESCALATION_RX.search(d["text"].lower())):
            d["cls"], d["subtag"] = ACUTE, "escalated_structural"

    missed = {ACUTE: 0, SOFT: 0, OTHER: 0}
    texts: Counter = Counter()
    unattributed_missed: set[int] = set()
    for m in sorted(team_weeks):
        if m in played:
            continue
        active = None
        for d in designations:
            if d["week"] > m:
                break
            if not any(d["week"] < w <= m for w in played):
                active = d
        if active:
            missed[active["cls"]] += 1
            texts[(active["cls"], active["text"])] += 1
        else:
            unattributed_missed.add(m)

    season_ending_acute = any(d["cls"] == ACUTE and season_ending(d)
                              for d in designations)
    last_played = max(played) if played else None
    trailing = ({w for w in team_weeks if w > last_played}
                if last_played is not None else set())
    vanished_unattributed = (len(trailing) >= 3
                             and trailing <= unattributed_missed)

    if texts:
        sample = texts.most_common(1)[0][0][1]
    elif designations:
        sample = designations[-1]["text"]
    else:
        sample = ""
    return {
        "games_played": len(played),
        "missed_acute": missed[ACUTE],
        "missed_soft": missed[SOFT],
        "missed_other": missed[OTHER],
        "season_ending_acute": season_ending_acute,
        "primary_injury_text_sample": sample,
        "vanished_unattributed": vanished_unattributed,  # report-only field
    }


# ---------------------------------------------------------------------------
# Season assembly

_SEVERITY = {"Out": 4, "Doubtful": 3, "Questionable": 2, "Probable": 1, "": 0}


def injury_text(r: dict) -> str:
    parts = [r.get("report_primary_injury", ""), r.get("report_secondary_injury", "")]
    if not any(p.strip() for p in parts):
        parts = [r.get("practice_primary_injury", ""), r.get("practice_secondary_injury", "")]
    return ", ".join(p.strip() for p in parts if p.strip())


def build_season(season: int, unmatched_log: list[str]) -> list[dict]:
    played: dict[str, set[int]] = defaultdict(set)
    events: dict[str, dict[int, str]] = defaultdict(dict)  # pid -> week -> team
    team_weeks: dict[str, set[int]] = defaultdict(set)
    meta: dict[str, dict] = {}

    with open(STATS_DIR / f"stats_player_week_{season}.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("season_type") != "REG":
                continue
            week, team = int(r["week"]), r.get("team", "")
            team_weeks[team].add(week)
            if r.get("position") not in POSITIONS:
                continue
            pid = r["player_id"]
            played[pid].add(week)
            events[pid][week] = team
            meta[pid] = {"player": r.get("player_display_name") or r.get("player_name"),
                         "position": r["position"]}

    # best designation row per (pid, week): most severe status wins
    desig_rows: dict[tuple[str, int], dict] = {}
    with open(INJ_DIR / f"injuries_{season}.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("game_type") != "REG":
                continue
            pid, pos = r["gsis_id"], r.get("position", "")
            if not pid or (pos not in POSITIONS and pid not in meta):
                continue
            week = int(r["week"])
            events[pid].setdefault(week, r.get("team", ""))
            meta.setdefault(pid, {"player": r.get("full_name", ""), "position": pos})
            prev = desig_rows.get((pid, week))
            if prev is None or _SEVERITY.get(r.get("report_status", ""), 0) > \
                    _SEVERITY.get(prev.get("report_status", ""), 0):
                desig_rows[(pid, week)] = r

    designations: dict[str, list[dict]] = defaultdict(list)
    for (pid, week), r in sorted(desig_rows.items()):
        text = injury_text(r)
        hit = classify(text)
        if hit is None:
            continue
        cls, subtag = hit
        if subtag == "unmatched":
            unmatched_log.append(f"{season} wk{week} {r.get('full_name', pid)}: {text}")
        designations[pid].append({"week": week, "cls": cls, "subtag": subtag, "text": text})

    rows = []
    for pid, m in sorted(meta.items()):
        if m["position"] not in POSITIONS:
            continue
        ev = events[pid]
        weeks_all = sorted(set().union(*team_weeks.values())) if team_weeks else []
        ev_weeks = sorted(ev)
        my_team_weeks = set()
        for w in weeks_all:
            past = [x for x in ev_weeks if x <= w]
            team = ev[past[-1]] if past else ev[ev_weeks[0]]
            if w in team_weeks.get(team, ()):
                my_team_weeks.add(w)
        rows.append({
            "season": season, "player_id": pid, "player": m["player"],
            "position": m["position"],
            **attribute_season(played[pid], my_team_weeks, designations[pid]),
        })
    return rows


# ---------------------------------------------------------------------------
# Cross-tab report vs breakouts.csv (ppr, 2015-2024)

REPORT_SEASONS = range(2015, 2025)
ACUTE_MISS_GATE = 5     # >=5 games lost to an acute class -> acute-injury bust
INJURY_MISS_GATE = 5    # >=5 games lost to any injury class -> injury bust


def _true(v) -> bool:
    return v is True or v == "True"


def bust_class(b: dict, ctx: dict | None) -> str:
    if ctx and (_true(ctx["season_ending_acute"])
                or int(ctx["missed_acute"]) >= ACUTE_MISS_GATE):
        return "acute_injury_bust"
    if ctx and _true(ctx.get("vanished_unattributed", False)):
        return "vanish_bust"  # straight-to-IR, cause invisible to this source
    games = int(b["games"] or 0)
    inj_missed = int(ctx["missed_soft"]) + int(ctx["missed_other"]) if ctx else 0
    if games < 8 or inj_missed >= INJURY_MISS_GATE:
        return "other_injury_bust"
    return "performance_bust"


def write_report(context_rows: list[dict]) -> None:
    ctx_by_id = {(int(r["season"]), r["player_id"]): r for r in context_rows}
    ctx_by_name = {(int(r["season"]), norm_name(r["player"]), r["position"]): r
                   for r in context_rows}

    with open(LABELS / "breakouts.csv", newline="") as f:
        lab = [r for r in csv.DictReader(f)
               if r["format"] == "ppr" and int(r["season"]) in REPORT_SEASONS]

    def ctx_for(b: dict) -> dict | None:
        season = int(b["season"])
        return (ctx_by_id.get((season, b["player_id"]))
                or ctx_by_name.get((season, norm_name(b["player"]), b["position"])))

    busts = [b for b in lab if b["bust"] == "True"]
    pool = [b for b in lab if int(b["adp_pos_rank"]) <= 12
            and b["match_status"] != "undrafted_in_pool"]
    for b in busts:
        b["_class"] = bust_class(b, ctx_for(b))
    split = Counter(b["_class"] for b in busts)
    no_data = sum(1 for b in busts if ctx_for(b) is None)

    breakouts = [b for b in lab if b["breakout"] == "True"]
    bo_acute = [b for b in breakouts
                if (c := ctx_for(b)) and _true(c["season_ending_acute"])]

    per_season = []
    for season in REPORT_SEASONS:
        p = sum(1 for b in pool if int(b["season"]) == season)
        bs = [b for b in busts if int(b["season"]) == season]
        a = sum(1 for b in bs if b["_class"] == "acute_injury_bust")
        v = sum(1 for b in bs if b["_class"] == "vanish_bust")
        per_season.append((
            season, p, len(bs), a, v,
            len(bs) / p if p else 0.0,
            (len(bs) - a) / (p - a) if p - a else 0.0,
            (len(bs) - a - v) / (p - a - v) if p - a - v else 0.0))
    P, B = len(pool), len(busts)
    A, V = split["acute_injury_bust"], split["vanish_bust"]
    rate_all = B / P
    rate_excl = (B - A) / (P - A)
    rate_excl_v = (B - A - V) / (P - A - V)

    reclass = sorted((b for b in busts
                      if b["_class"] in ("acute_injury_bust", "vanish_bust")),
                     key=lambda b: float(b["adp"] or 999))[:15]

    lines = [
        "# Injury-context cross-tab — BreakoutBench v0.3 freak-injury split",
        "",
        f"Built by `evals/build_injury_context.py` (lexicon {LEXICON_VERSION}) from",
        "`data/processed/labels/injury_context.csv` x `breakouts.csv` (ppr, 2015-2024).",
        "",
        "Definitions: bust pool = drafted top-12 positional (ppr ADP). A bust is an",
        f"`acute_injury_bust` if season_ending_acute or >={ACUTE_MISS_GATE} games missed in the",
        "ACUTE_TRAUMATIC class (lexical acute terms, or a structural body-part designation",
        "that proved season-ending); a `vanish_bust` if the player went straight to IR with",
        f"no report row at all (cause invisible to this source); else `other_injury_bust` if",
        f"games < 8 or >={INJURY_MISS_GATE} games missed to soft-tissue/other designations; else",
        "`performance_bust`.",
        "",
        "## Bust class split (pooled 2015-2024, ppr)",
        "",
        "| class | n | share |",
        "|---|---|---|",
    ]
    for k in ("performance_bust", "acute_injury_bust", "vanish_bust", "other_injury_bust"):
        lines.append(f"| {k} | {split[k]} | {split[k] / B:.1%} |")
    lines += [
        f"| total busts | {B} | 100% |",
        "",
        f"Busts with no joinable injury-context row (zero games, no in-season report —",
        f"typically preseason injuries invisible to this source): {no_data} (bucketed as other_injury).",
        "",
        "## Breakouts that broke out despite their own season-ending acute injury",
        "",
        f"{len(bo_acute)} of {len(breakouts)} labeled breakouts:",
        "",
    ]
    for b in sorted(bo_acute, key=lambda b: (b["season"], b["player"])):
        c = ctx_for(b)
        lines.append(f"- {b['season']} {b['player']} ({b['position']}): played "
                     f"{c['games_played']}, sample `{c['primary_injury_text_sample']}`")
    lines += [
        "",
        "## THE NUMBER: bust base rate, all-outcomes vs freak-excluded",
        "",
        "Freak-excluded removes acute-injury busts from numerator and pool.",
        "`excl acute` is the confirmed-acute panel (headline, lower bound);",
        "`excl acute+vanish` also drops straight-to-IR unknowns (upper bound).",
        "",
        "| season | pool | busts | acute | vanish | rate (all) | excl acute | excl acute+vanish |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for season, p, b, a, v, r_all, r_ex, r_exv in per_season:
        lines.append(f"| {season} | {p} | {b} | {a} | {v} | {r_all:.1%} | {r_ex:.1%} "
                     f"| {r_exv:.1%} |")
    lines += [
        f"| **pooled** | {P} | {B} | {A} | {V} | **{rate_all:.1%}** | **{rate_excl:.1%}** "
        f"| **{rate_excl_v:.1%}** |",
        "",
        f"Pooled shift: {rate_all:.1%} -> {rate_excl:.1%} "
        f"({(rate_excl - rate_all) * 100:+.1f} pp confirmed-acute; "
        f"{(rate_excl_v - rate_all) * 100:+.1f} pp including vanish busts).",
        "",
        "## Face validity: 15 most consequential acute-injury reclassifications",
        "",
        "Sorted by overall ADP (earliest picks whose bust label is excused as freak",
        "injury); includes vanish busts, marked as such.",
        "",
        "| season | player | pos | ADP | pos rank | finish | games | class | injury sample |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for b in reclass:
        c = ctx_for(b)
        sample = (c["primary_injury_text_sample"] or "straight to IR, no report row") \
            if c else "-"
        lines.append(f"| {b['season']} | {b['player']} | {b['position']} | {b['adp']} "
                     f"| {b['adp_pos_rank']} | {b['finish_pos_rank'] or '-'} | {b['games']} "
                     f"| {b['_class'].replace('_bust', '')} | {sample} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- nflverse weekly reports carry body-part text only; IR players drop off the",
        "  report, so attribution carries the last designation forward until a played game,",
        "  and season-ending structural body-part designations are escalated to ACUTE.",
        "- Players who go straight to IR between games leave no report row at all — those",
        "  busts are the `vanish` class (cause unknown, historically mostly ACL-grade).",
        "- Preseason (July/August) injuries never appear; zero-game busts without any",
        "  in-season row cannot be attributed and are counted as other_injury, not acute —",
        "  the confirmed-acute shift is therefore a lower bound.",
        "- `season_ending_acute` in injury_context.csv is computed for every rostered",
        "  player; for fringe players a season-ending designation can also mean a roster",
        "  cut, so interpret the flag jointly with draft capital (as this report does).",
        "",
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "injury_context_report.md").write_text("\n".join(lines))
    print(f"report: busts={B} split={dict(split)} pooled rate {rate_all:.1%} -> {rate_excl:.1%}")


# ---------------------------------------------------------------------------

COLUMNS = ["season", "player_id", "player", "position", "games_played",
           "missed_acute", "missed_soft", "missed_other", "season_ending_acute",
           "primary_injury_text_sample"]


def main() -> int:
    unmatched_log: list[str] = []
    rows: list[dict] = []
    for season in SEASONS:
        srows = build_season(season, unmatched_log)
        rows.extend(srows)
        flagged = sum(1 for r in srows if r["season_ending_acute"])
        print(f"{season}: {len(srows)} player-seasons, {flagged} season_ending_acute", flush=True)

    LABELS.mkdir(parents=True, exist_ok=True)
    with open(LABELS / "injury_context.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (LABELS / "injury_context_unmatched.txt").write_text("\n".join(unmatched_log))
    print(f"rows={len(rows)} unmatched_texts={len(unmatched_log)}")

    write_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
