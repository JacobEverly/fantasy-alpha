#!/usr/bin/env python3
"""Contract-year channel: enter it as data, measure it walk-forward.

Jacob's coverage flag: "was the player in a contract year?" This module
(1) derives the join-ready table data/processed/contracts/contract_status.csv
from the raw OverTheCap-via-nflverse snapshot (scripts/collect_contracts.py),
and (2) measures whether the flag carries selection signal after ADP
conditioning, local compute only:

  (a) descriptive: breakout/bust rates for contract-year vs non-contract-year
      players within positional-ADP bands (the fame-control pattern from
      evals/source_alpha.py), pooled 2011-2024 ppr, season-resampled
      bootstrap 95% CIs;
  (b) GBDT delta: the four contract features added to the v2 packet feature
      set, strict walk-forward, paired season-bootstrap deltas vs v2 (the
      evals/gbdt_baseline.py + evals/vegas_packet_test.py pattern), plus
      permutation importances for the contract columns.

As-of discipline: a season-S row in contract_status.csv is derived ONLY from
contracts with year_signed <= S, so nothing signed after season S can leak
backwards. The unavoidable residual ambiguity — OTC/nflverse carries
year_signed at YEAR granularity, no signing date — is that a deal signed in
calendar year S is treated as known before Sept 1 of S even when it was
actually an in-season (Sept+) extension. Rows where the selected contract has
year_signed == S carry signed_recently=True as the explicit ambiguity marker;
the report quantifies how many candidates that touches.

Outputs:
  data/processed/contracts/contract_status.csv   (join-ready, harness-facing)
  evals/results/contract_year.md

Usage:
  .venv/bin/python -m evals.contract_year_test build     # derive the table
  .venv/bin/python -m evals.contract_year_test run       # measurement + report
  .venv/bin/python -m evals.contract_year_test all
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.names import norm_name  # noqa: E402
from evals.build_breakout_labels import ADP_GATE, BUST_ADP_GATE  # noqa: E402
from evals.source_alpha import adp_band, bootstrap_lift_ci, _lift  # noqa: E402

LABELS = ROOT / "data" / "processed" / "labels"
OUT_STATUS = ROOT / "data" / "processed" / "contracts" / "contract_status.csv"
OUT_MD = ROOT / "evals" / "results" / "contract_year.md"

SEASONS = tuple(range(2011, 2025))       # descriptive study, ppr labels
DERIVE_SEASONS = tuple(range(2011, 2027))
GBDT_SEASONS = tuple(range(2016, 2025))  # v2-packet frame (train from 2015)
GBDT_TRAIN_START = 2015
FORMAT = "ppr"
SEED = 20260808
REPS = 10_000

SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}
CONTRACT_FIELDS = ("in_contract_year", "years_remaining",
                   "signed_recently", "rookie_deal")

STATUS_COLUMNS = [
    "season", "gsis_id", "otc_id", "player", "position", "pos_fantasy",
    "team", "in_contract_year", "years_remaining", "signed_recently",
    "rookie_deal", "fifth_year_option", "contract_type", "year_signed",
    "contract_years", "end_year", "apy", "value", "apy_cap_pct",
]


# ---------------------------------------------------------------------------
# Derivation: raw contract rows -> per (player, season) status

def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_contracts", ROOT / "scripts" / "collect_contracts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_raw_contracts(path: Path | None = None) -> list[dict]:
    """Skill-position contract rows from the latest dated raw snapshot,
    filtered to placeable contracts (real year_signed, years >= 1)."""
    if path is None:
        path = _load_collector().latest_snapshot()
        if path is None:
            raise FileNotFoundError(
                "no contracts snapshot — run scripts/collect_contracts.py")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["position"] not in SKILL_POSITIONS:
                continue
            try:
                year_signed = int(float(r["year_signed"]))
                years = float(r["years"])
            except (ValueError, TypeError):
                continue
            if year_signed < 1990 or years < 1:
                continue
            rows.append({
                "player": r["player"],
                "position": r["position"],
                "team": r["team"],
                "otc_id": r["otc_id"],
                "gsis_id": r["gsis_id"] or "",
                "year_signed": year_signed,
                "years": int(years),
                "value": float(r["value"] or 0),
                "apy": float(r["apy"] or 0),
                "apy_cap_pct": r["apy_cap_pct"] or "",
                "contract_type": r.get("contract_type") or "",
                "option_through": (int(float(r["option_through"]))
                                   if r.get("option_through") else None),
                "draft_year": (int(float(r["draft_year"]))
                               if r.get("draft_year") not in (None, "", "NA")
                               else None),
            })
    return rows


def _end_year(c: dict) -> int:
    """Scheduled final season of a contract: signed term, extended by an
    exercised round-1 5th-year option when the raw snapshot detected one
    (as-of legal — option decisions land in May of contract-year 4)."""
    end = c["year_signed"] + c["years"] - 1
    if c.get("option_through"):
        end = max(end, c["option_through"])
    return end


def select_contract(contracts: list[dict], season: int) -> dict | None:
    """The current deal entering season S: the most recently signed contract
    with year_signed <= S (ties: largest value, then longest), provided its
    scheduled term still covers S. None = no contract on file for S.

    Year-granularity caveat: a deal with year_signed == S is assumed known
    before Sept 1 of S. That is right for the typical March-August signing
    and wrong for a September+ in-season extension; callers see
    signed_recently=True on exactly these rows.
    """
    elig = [c for c in contracts if c["year_signed"] <= season]
    if not elig:
        return None
    ymax = max(c["year_signed"] for c in elig)
    sel = max((c for c in elig if c["year_signed"] == ymax),
              key=lambda c: (c["value"], c["years"]))
    return sel if _end_year(sel) >= season else None


def status_row(sel: dict, season: int) -> dict:
    end = _end_year(sel)
    ctype = sel["contract_type"]
    rookie = (ctype in ("Drafted", "UDFA") if ctype else
              (sel["draft_year"] is not None
               and sel["year_signed"] == sel["draft_year"]))
    return {
        "season": season,
        "gsis_id": sel["gsis_id"],
        "otc_id": sel["otc_id"],
        "player": sel["player"],
        "position": sel["position"],
        "pos_fantasy": "RB" if sel["position"] == "FB" else sel["position"],
        "team": sel["team"],
        "in_contract_year": end == season,
        "years_remaining": end - season + 1,
        "signed_recently": sel["year_signed"] == season,
        "rookie_deal": rookie,
        "fifth_year_option": (sel.get("option_through") is not None
                              and season == sel["option_through"]),
        "contract_type": ctype,
        "year_signed": sel["year_signed"],
        "contract_years": sel["years"],
        "end_year": end,
        "apy": sel["apy"],
        "value": sel["value"],
        "apy_cap_pct": sel["apy_cap_pct"],
    }


def derive_status(raw: list[dict],
                  seasons: tuple[int, ...] = DERIVE_SEASONS) -> list[dict]:
    """One row per (player, season) under a known contract. Uses only
    contracts with year_signed <= season (as-of by construction)."""
    by_player: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_player[r["otc_id"]].append(r)
    out = []
    for otc_id in sorted(by_player):
        contracts = by_player[otc_id]
        for season in seasons:
            sel = select_contract(contracts, season)
            if sel is not None:
                out.append(status_row(sel, season))
    return out


def write_status(rows: list[dict], path: Path = OUT_STATUS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATUS_COLUMNS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Status lookup (join-ready: gsis_id first, (norm_name, pos_fantasy) fallback)

def load_status_index(path: Path = OUT_STATUS) -> dict:
    by_gsis: dict[tuple[int, str], dict] = {}
    by_name: dict[tuple[int, str, str], dict] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            season = int(r["season"])
            row = {
                "in_contract_year": r["in_contract_year"] == "True",
                "years_remaining": int(r["years_remaining"]),
                "signed_recently": r["signed_recently"] == "True",
                "rookie_deal": r["rookie_deal"] == "True",
            }
            if r["gsis_id"]:
                by_gsis[(season, r["gsis_id"])] = row
            key = (season, norm_name(r["player"]), r["pos_fantasy"])
            # name collisions (rare): first writer wins, deterministic by
            # the sorted otc_id derivation order
            by_name.setdefault(key, row)
    return {"by_gsis": by_gsis, "by_name": by_name}


def status_for(index: dict, season: int, player: str, position: str,
               player_id: str | None) -> dict | None:
    if player_id:
        hit = index["by_gsis"].get((season, player_id))
        if hit is not None:
            return hit
    return index["by_name"].get((season, norm_name(player), position))


# ---------------------------------------------------------------------------
# (a) Descriptive: flag vs outcome within ADP bands, pooled seasons

def load_candidates() -> dict[int, dict[str, list[dict]]]:
    """season -> {"breakout": [...], "bust": [...]} from breakouts.csv (ppr).
    Same populations as evals/source_alpha.load_candidates, seasons 2011-2024."""
    out: dict[int, dict[str, list[dict]]] = {
        s: {"breakout": [], "bust": []} for s in SEASONS}
    with open(LABELS / "breakouts.csv", newline="") as f:
        for r in csv.DictReader(f):
            season = int(r["season"])
            if (r["format"] != FORMAT or season not in out
                    or not r["adp_pos_rank"]):
                continue
            pos, rank = r["position"], int(r["adp_pos_rank"])
            if pos not in ADP_GATE:
                continue
            row = {
                "season": season, "player": r["player"],
                "player_id": r["player_id"], "position": pos,
                "adp_pos_rank": rank,
                "breakout": r["breakout"] == "True",
                "bust": r["bust"] == "True",
            }
            if rank > ADP_GATE[pos]:
                row["band"] = adp_band(pos, rank)
                out[season]["breakout"].append(row)
            if rank <= BUST_ADP_GATE:
                out[season]["bust"].append(row)
    return out


def flag_cells(pop: dict[int, list[dict]], index: dict, flag: str,
               outcome_key: str) -> tuple[dict[int, dict], dict]:
    """Per-season k/n counts for flag-holders vs non-holders among MATCHED
    candidates (unmatched are excluded from denominators, not assumed
    un-flagged), plus coverage stats."""
    per_season: dict[int, dict] = {}
    cov = {"n": 0, "matched": 0, "flag_true": 0, "ambiguous": 0}
    for season, rows in pop.items():
        cell = {"k_c": 0, "n_c": 0, "k_u": 0, "n_u": 0}
        for r in rows:
            cov["n"] += 1
            st = status_for(index, season, r["player"], r["position"],
                            r["player_id"])
            if st is None:
                continue
            cov["matched"] += 1
            cov["ambiguous"] += st["signed_recently"]
            hit = int(bool(r[outcome_key]))
            if st[flag]:
                cov["flag_true"] += 1
                cell["n_c"] += 1
                cell["k_c"] += hit
            else:
                cell["n_u"] += 1
                cell["k_u"] += hit
        if rows:
            per_season[season] = cell
    return per_season, cov


def pooled_row(per_season: dict[int, dict], reps: int, seed: int) -> dict:
    tot = defaultdict(int)
    for c in per_season.values():
        for k, v in c.items():
            tot[k] += v
    lo, hi = bootstrap_lift_ci(per_season, reps, seed)
    return {
        "n_flag": tot["n_c"], "n_noflag": tot["n_u"],
        "rate_flag": tot["k_c"] / tot["n_c"] if tot["n_c"] else None,
        "rate_noflag": tot["k_u"] / tot["n_u"] if tot["n_u"] else None,
        "lift": _lift(tot["k_c"], tot["n_c"], tot["k_u"], tot["n_u"]),
        "ci": (lo, hi),
    }


def descriptive_study(index: dict, reps: int = REPS,
                      seed: int = SEED) -> dict:
    cands = load_candidates()
    bo_pop = {s: cands[s]["breakout"] for s in SEASONS}
    bust_pop = {s: cands[s]["bust"] for s in SEASONS}
    out: dict = {"rows": {}, "coverage": {}}

    # pooled rows: each flag x each population
    for flag in CONTRACT_FIELDS:
        if flag == "years_remaining":
            continue  # not boolean; carried by the GBDT arm instead
        for name, pop, key in (("breakout", bo_pop, "breakout"),
                               ("bust", bust_pop, "bust")):
            per_season, cov = flag_cells(pop, index, flag, key)
            out["rows"][(flag, name, "all")] = pooled_row(per_season, reps, seed)
            out["coverage"][(flag, name)] = cov

    # ADP-band strata for the headline flag (fame-control pattern)
    for band in ("near_gate", "mid", "deep"):
        pop = {s: [r for r in bo_pop[s] if r["band"] == band] for s in SEASONS}
        per_season, _ = flag_cells(pop, index, "in_contract_year", "breakout")
        out["rows"][("in_contract_year", "breakout", band)] = pooled_row(
            per_season, reps, seed)
    return out


# ---------------------------------------------------------------------------
# (b) GBDT delta: v2 + contract features vs v2, walk-forward, paired

def gbdt_study(index: dict, reps: int = REPS, seed: int = SEED) -> dict:
    import numpy as np
    from evals.gbdt_baseline import (
        V2_FEATURE_NAMES, _fit, build_candidates_v2, encode_v2, label_of,
        load_data, permutation_importance_brier, score_season,
        train_predict_v2)
    from evals.vegas_packet_test import paired_delta, pool_slate, slate_ci

    feature_names = list(V2_FEATURE_NAMES) + [
        *CONTRACT_FIELDS, "contract_missing"]

    def encode_v2c(features: dict, enrichment: dict, st: dict | None) -> list:
        row = encode_v2(features, enrichment)
        if st is None:
            row += [float("nan")] * len(CONTRACT_FIELDS) + [1.0]
        else:
            row += [float(st[f]) for f in CONTRACT_FIELDS] + [0.0]
        return row

    def candidates_v2c(data, season: int) -> list[dict]:
        cands = build_candidates_v2(data, season, FORMAT)
        for c in cands:
            c["contract"] = status_for(
                index, season, c["player"], c["features"]["position"],
                c["player_id"])
        return cands

    def arm_v2c() -> dict:
        per_season: dict[int, dict] = {}
        importances = np.zeros(len(feature_names))
        n_total = 0
        n_cand = n_matched = 0
        for season in GBDT_SEASONS:
            xs, ys = [], []
            for s in range(GBDT_TRAIN_START, season):
                assert s < season, "leakage: training on target/future season"
                for c in candidates_v2c(data, s):
                    xs.append(encode_v2c(c["features"], c["enrichment"],
                                         c["contract"]))
                    ys.append(label_of(data, s, c["player_id"]))
            model = _fit(np.asarray(xs, dtype=float),
                         np.asarray(ys, dtype=float), seed)
            slate = candidates_v2c(data, season)
            n_cand += len(slate)
            n_matched += sum(c["contract"] is not None for c in slate)
            x_eval = np.asarray([encode_v2c(c["features"], c["enrichment"],
                                            c["contract"]) for c in slate])
            y_eval = np.asarray(
                [label_of(data, season, c["player_id"]) for c in slate],
                dtype=float)
            probs = model.predict_proba(x_eval)[:, 1]
            preds = [{"player": c["player"], "player_id": c["player_id"],
                      "p_breakout": float(p), "label": int(y)}
                     for c, p, y in zip(slate, probs, y_eval)]
            importances += permutation_importance_brier(
                model, x_eval, y_eval, seed) * len(y_eval)
            n_total += len(y_eval)
            per_season[season] = score_season(preds)
            m = per_season[season]
            print(f"  gbdt-v2c {season}: n={m['n']} hits@10={m['hits']} "
                  f"lift={m['lift']:.1f}x brier={m['brier_full']:.3f}",
                  flush=True)
        return {"per_season": per_season, "pooled": pool_slate(per_season),
                "ci": slate_ci(per_season, reps, seed),
                "importances": importances / n_total,
                "n_cand": n_cand, "n_matched": n_matched}

    def arm_v2() -> dict:
        per_season: dict[int, dict] = {}
        for season in GBDT_SEASONS:
            preds, *_ = train_predict_v2(data, season, FORMAT, seed,
                                         GBDT_TRAIN_START)
            per_season[season] = score_season(preds)
            m = per_season[season]
            print(f"  gbdt-v2  {season}: n={m['n']} hits@10={m['hits']} "
                  f"lift={m['lift']:.1f}x brier={m['brier_full']:.3f}",
                  flush=True)
        return {"per_season": per_season, "pooled": pool_slate(per_season),
                "ci": slate_ci(per_season, reps, seed)}

    data = load_data(FORMAT, FORMAT,
                     seasons=tuple(range(GBDT_TRAIN_START,
                                         max(GBDT_SEASONS) + 1)))
    print("== GBDT v2 (baseline) ==", flush=True)
    v2 = arm_v2()
    print("== GBDT v2 + contract features ==", flush=True)
    v2c = arm_v2c()
    delta = paired_delta(v2c, v2, reps, seed)
    imp = dict(zip(feature_names, v2c["importances"]))
    ranked = sorted(imp.items(), key=lambda t: -t[1])
    rank = {name: i + 1 for i, (name, _) in enumerate(ranked)}
    return {"v2": v2, "v2c": v2c, "delta": delta, "imp": imp, "rank": rank,
            "n_features": len(feature_names)}


# ---------------------------------------------------------------------------
# Report

def _fmt_ci(ci, digits=2):
    if ci is None or ci[0] is None:
        return "[—]"
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _row_str(r: dict) -> str:
    if r["lift"] is None:
        return "—"
    return (f"{r['rate_flag']:.1%} ({r['n_flag']}) vs "
            f"{r['rate_noflag']:.1%} ({r['n_noflag']}) | "
            f"{r['lift']:.2f} {_fmt_ci(r['ci'])}")


def write_report(desc: dict, gbdt: dict, status_stats: dict,
                 path: Path = OUT_MD, reps: int = REPS,
                 seed: int = SEED) -> None:
    d = gbdt["delta"]
    band_rows = {b: desc["rows"][("in_contract_year", "breakout", b)]
                 for b in ("near_gate", "mid", "deep")}
    any_band_clear = any(
        r["ci"][0] is not None and (r["ci"][0] > 1.0 or r["ci"][1] < 1.0)
        for r in band_rows.values())
    pooled_bo = desc["rows"][("in_contract_year", "breakout", "all")]
    pooled_clear = (pooled_bo["ci"][0] is not None
                    and (pooled_bo["ci"][0] > 1.0 or pooled_bo["ci"][1] < 1.0))
    brier_sig = d["brier_delta_ci"][1] < 0
    lift_sig = d["lift_delta_ci"][0] > 0
    additive = brier_sig or lift_sig or any_band_clear

    if additive:
        verdict = ("**Contract year carries selection signal after ADP "
                   "conditioning** — at least one pre-registered read (paired "
                   "GBDT Brier/lift delta vs v2, or within-band breakout "
                   "stratification) is clear of the null. Weight it in the "
                   "packet; keep the walk-forward delta as the regression "
                   "gate.")
    else:
        verdict = ("**Contract year is PRICED — no detectable selection "
                   "signal after ADP conditioning.** Adding the contract "
                   "features to the v2 packet moves neither full-slate Brier "
                   "nor top-10 lift by a bootstrap-detectable amount, and the "
                   "contract-year flag does not stratify breakout rate within "
                   "positional-ADP bands. This matches the prior: the market "
                   "has known about contract-year incentives for decades. "
                   "Ship it as CONVERSATIONAL COVERAGE (the table and tool "
                   "shape below), with zero weight in selection.")

    cy_cov = desc["coverage"][("in_contract_year", "breakout")]
    cy_cov_bust = desc["coverage"][("in_contract_year", "bust")]
    imp = gbdt["imp"]

    lines = [
        "# Contract-year channel — data entry + walk-forward measurement",
        "",
        f"Generated {__import__('datetime').date.today().isoformat()} by "
        f"`evals/contract_year_test.py` (seed {seed}, {reps:,} "
        "season-resampled bootstrap draws; GBDT machinery only, no API "
        "calls). Raw source: OverTheCap contracts via the nflverse-data "
        "`contracts` release (`scripts/collect_contracts.py`, dated "
        "immutable snapshot).",
        "",
        "## Data entered",
        "",
        f"- `data/processed/contracts/contract_status.csv`: "
        f"{status_stats['rows']} (player, season) rows, seasons "
        f"{status_stats['season_min']}-{status_stats['season_max']}, "
        f"{status_stats['players']} distinct players (QB/RB/WR/TE/FB).",
        "- Per row: `in_contract_year` (final scheduled year of the current "
        "deal entering season S), `years_remaining`, `signed_recently` (deal "
        "signed in calendar year S), `rookie_deal`, plus contract terms "
        "(type, years, end_year, apy/value in $M).",
        "- **As-of discipline**: season-S rows use only contracts with "
        "`year_signed <= S`. Residual ambiguity: the source carries year "
        "granularity only (no signing dates), so a September-of-S extension "
        "is indistinguishable from a March-of-S signing; every such row is "
        "flagged `signed_recently=True` "
        f"({cy_cov['ambiguous']}/{cy_cov['matched']} matched breakout "
        "candidates, pooled).",
        "",
        "## Join coverage (labels x contract table)",
        "",
        "| population | candidate rows | matched | coverage | in contract year |",
        "|---|---|---|---|---|",
        f"| breakout (gate-eligible, {SEASONS[0]}-{SEASONS[-1]} ppr) "
        f"| {cy_cov['n']} | {cy_cov['matched']} "
        f"| {cy_cov['matched'] / cy_cov['n']:.1%} "
        f"| {cy_cov['flag_true']} ({cy_cov['flag_true'] / cy_cov['matched']:.1%}) |",
        f"| bust (top-12 positional ADP) | {cy_cov_bust['n']} "
        f"| {cy_cov_bust['matched']} "
        f"| {cy_cov_bust['matched'] / cy_cov_bust['n']:.1%} "
        f"| {cy_cov_bust['flag_true']} "
        f"({cy_cov_bust['flag_true'] / cy_cov_bust['matched']:.1%}) |",
        "",
        "Unmatched candidates are EXCLUDED from descriptive denominators "
        "(missing contract data is not evidence of no contract year); the "
        "GBDT arm encodes them as missing.",
        "",
        "## (a) Descriptive: outcome rate, flag vs no-flag, pooled "
        f"{SEASONS[0]}-{SEASONS[-1]}",
        "",
        "Lift = rate(flag) / rate(no flag); [lo, hi] = season-resampled "
        "bootstrap 95% CI (seasons are the independent unit).",
        "",
        "| flag | population | rate flag (n) vs rate no-flag (n) | lift [95% CI] |",
        "|---|---|---|---|",
    ]
    for flag in ("in_contract_year", "signed_recently", "rookie_deal"):
        for popn in ("breakout", "bust"):
            r = desc["rows"][(flag, popn, "all")]
            lines.append(f"| {flag} | {popn} | {_row_str(r)} |")
    lines += [
        "",
        "### Fame control: in_contract_year x breakout within "
        "positional-ADP bands",
        "",
        "Bands = distance of positional ADP rank beyond the eligibility gate "
        "(near_gate +1..+12, mid +13..+30, deep +31+; the "
        "`evals/source_alpha.py` pattern).",
        "",
        "| band | rate flag (n) vs rate no-flag (n) | lift [95% CI] |",
        "|---|---|---|",
    ]
    for b in ("near_gate", "mid", "deep"):
        lines.append(f"| {b} | {_row_str(band_rows[b])} |")
    v2p, v2cp = gbdt["v2"]["pooled"], gbdt["v2c"]["pooled"]
    v2ci, v2cci = gbdt["v2"]["ci"], gbdt["v2c"]["ci"]
    lines += [
        "",
        "## (b) GBDT delta: v2 packet + contract features vs v2 "
        "(walk-forward, paired)",
        "",
        f"Frame: seasons {GBDT_SEASONS[0]}-{GBDT_SEASONS[-1]}, ppr, train "
        f"{GBDT_TRAIN_START}..S-1 for both arms (identical folds; the "
        "`evals/vegas_packet_test.py` machinery). Contract features: "
        + ", ".join(CONTRACT_FIELDS) + " + a shared missing indicator "
        f"(join coverage {gbdt['v2c']['n_matched']}/{gbdt['v2c']['n_cand']} "
        f"= {gbdt['v2c']['n_matched'] / gbdt['v2c']['n_cand']:.0%} of "
        "gate-eligible candidates).",
        "",
        "| arm | Brier (full slate) [95% CI] | top-10 hits "
        f"(of {10 * len(GBDT_SEASONS)}) | top-10 lift [95% CI] |",
        "|---|---|---|---|",
        f"| GBDT v2 | {v2p['brier']:.4f} "
        f"[{v2ci['brier'][0]:.4f}, {v2ci['brier'][1]:.4f}] | {v2p['hits']} "
        f"| {v2p['lift']:.2f}x [{v2ci['lift'][0]:.2f}, {v2ci['lift'][1]:.2f}] |",
        f"| **GBDT v2 + contracts** | {v2cp['brier']:.4f} "
        f"[{v2cci['brier'][0]:.4f}, {v2cci['brier'][1]:.4f}] | {v2cp['hits']} "
        f"| {v2cp['lift']:.2f}x [{v2cci['lift'][0]:.2f}, "
        f"{v2cci['lift'][1]:.2f}] |",
        "",
        "Paired season-bootstrap deltas (negative Brier delta = contracts "
        "helped):",
        "",
        "| Δ Brier [95% CI] | Δ top-10 lift [95% CI] | Δ hits [95% CI] |",
        "|---|---|---|",
        f"| {d['brier_delta']:+.4f} [{d['brier_delta_ci'][0]:+.4f}, "
        f"{d['brier_delta_ci'][1]:+.4f}] | {d['lift_delta']:+.2f} "
        f"[{d['lift_delta_ci'][0]:+.2f}, {d['lift_delta_ci'][1]:+.2f}] "
        f"| {d['hits_delta']:+.0f} [{d['hits_delta_ci'][0]:+.0f}, "
        f"{d['hits_delta_ci'][1]:+.0f}] |",
        "",
        "Permutation importances (mean held-out Brier increase when "
        "shuffled, n-weighted across folds) for the contract columns:",
        "",
        f"| column | Δ Brier when shuffled | rank of {gbdt['n_features']} |",
        "|---|---|---|",
    ]
    for f in (*CONTRACT_FIELDS, "contract_missing"):
        lines.append(f"| {f} | {imp[f]:+.5f} | {gbdt['rank'][f]} |")
    lines += [
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"- Headline paired delta (v2+contracts − v2): Brier "
        f"{d['brier_delta']:+.4f} [{d['brier_delta_ci'][0]:+.4f}, "
        f"{d['brier_delta_ci'][1]:+.4f}], top-10 lift {d['lift_delta']:+.2f} "
        f"[{d['lift_delta_ci'][0]:+.2f}, {d['lift_delta_ci'][1]:+.2f}].",
        f"- Pooled in-contract-year breakout lift: "
        f"{pooled_bo['lift']:.2f} {_fmt_ci(pooled_bo['ci'])}"
        f"{' (CI excludes 1.0)' if pooled_clear else ' (CI includes 1.0)'}; "
        "within-band lifts: "
        + "; ".join(f"{b} {band_rows[b]['lift']:.2f} "
                    f"{_fmt_ci(band_rows[b]['ci'])}"
                    if band_rows[b]["lift"] is not None else f"{b} —"
                    for b in ("near_gate", "mid", "deep")) + ".",
        "",
        "## Conversational coverage (wired regardless of verdict)",
        "",
        "The harness answers contract-year questions from "
        "`data/processed/contracts/contract_status.csv` — a "
        "`contract_status(player, season=current)` tool resolves via "
        "gsis_id, else (normalized name, fantasy position), and returns:",
        "",
        "```json",
        '{"player": "...", "season": 2026, "in_contract_year": true,',
        ' "years_remaining": 1, "signed_recently": false,',
        ' "rookie_deal": false, "contract_type": "UFA",',
        ' "year_signed": 2023, "contract_years": 4, "end_year": 2026,',
        ' "apy_musd": 12.5, "value_musd": 50.0,',
        ' "caveat": "OverTheCap via nflverse; year-level signing dates;',
        '  scheduled term as signed (extensions supersede when signed)"}',
        "```",
        "",
        "`null` (no row) means no contract on file for that season — say so, "
        "never guess. The measured verdict above is what the model should "
        "SAY when asked whether contract year matters for drafting.",
        "",
        "## Caveats (honest ones)",
        "",
        "- **Year-granularity signing dates.** A deal signed in calendar "
        "year S is treated as known before Sept 1 of S. True September+ "
        "extensions (rare for the draft-relevant pool, but real) wrongly "
        "unset the contract-year flag for season S; every affected row is "
        "identifiable via `signed_recently`.",
        "- **Scheduled term as signed.** `years` is the contract's scheduled "
        "length; void years and later renegotiations can shift the real "
        "expiry. Extensions appear as new contracts and supersede from "
        "their signing year.",
        "- **OTC coverage thins pre-2013** (2011: ~40 unmatched candidates "
        "vs <10/season after). Pooled 2011-2024 estimates lean on the "
        "well-covered years.",
        "- **is_active/terms are a current snapshot** — contract terms as "
        "signed don't change retroactively, but data-entry corrections at "
        "OTC do land silently; snapshots are dated for exactly this reason.",
        "- The bust read is all-outcomes (no freak-injury exclusion here); "
        "a contract-year flag cannot predict a week-2 ACL either way.",
        "- 2025 remains the untouched holdout; nothing here touches it.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI

def build() -> dict:
    raw = load_raw_contracts()
    rows = derive_status(raw)
    write_status(rows)
    stats = {
        "rows": len(rows),
        "players": len({r["otc_id"] for r in rows}),
        "season_min": min(r["season"] for r in rows),
        "season_max": max(r["season"] for r in rows),
    }
    print(f"wrote {OUT_STATUS.relative_to(ROOT)}: {stats['rows']} rows, "
          f"{stats['players']} players, seasons "
          f"{stats['season_min']}-{stats['season_max']}")
    return stats


def status_table_stats() -> dict:
    rows = 0
    players = set()
    smin, smax = 9999, 0
    with open(OUT_STATUS, newline="") as f:
        for r in csv.DictReader(f):
            rows += 1
            players.add(r["otc_id"])
            s = int(r["season"])
            smin, smax = min(smin, s), max(smax, s)
    return {"rows": rows, "players": len(players),
            "season_min": smin, "season_max": smax}


def run(reps: int = REPS, seed: int = SEED) -> Path:
    index = load_status_index()
    print("== descriptive study ==", flush=True)
    desc = descriptive_study(index, reps, seed)
    gbdt = gbdt_study(index, reps, seed)
    write_report(desc, gbdt, status_table_stats(), reps=reps, seed=seed)
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    return OUT_MD


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="derive contract_status.csv from raw snapshot")
    r = sub.add_parser("run", help="measurement + report")
    r.add_argument("--reps", type=int, default=REPS)
    r.add_argument("--seed", type=int, default=SEED)
    al = sub.add_parser("all", help="build then run")
    al.add_argument("--reps", type=int, default=REPS)
    al.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if args.cmd in ("build", "all"):
        build()
    if args.cmd in ("run", "all"):
        run(args.reps, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
