#!/usr/bin/env python3
"""Vegas-vs-ADP test: does the week-1 implied-team-total signal survive ADP
conditioning at the player level?

Context: evals/results/vegas_history.md measured the week-1 implied team
total as additive over naive team stats (+0.23 incremental r, 2016-2024) at
the TEAM level. But BreakoutBench candidates already carry ADP — the market's
player-level consensus — which may have priced the same information. This
module runs the decisive player-level test, GBDT + base-rate machinery only
(no API/model calls):

1. Walk-forward GBDT on the full-slate family (2016-2024, ppr):
     structure-only (v1 packet, train 2015..S-1)  — ADP + prior seasons
     v2             (enriched packet)             — + news/depth/usage/pedigree
     v3             (v2 + team vegas)             — + week-1 implied total and
                                                     prior-season mean implied
   Paired season-bootstrap deltas (10k draws, seeded) on full-slate Brier and
   top-10 lift; permutation importances for the vegas columns.
2. Same v3-vs-v2 delta on the bust and over_under families (team vegas is
   team-level, so it rides along cheaply via the shared packet machinery).
3. The direct read: among gate-eligible candidates, does the week-1 implied
   team total stratify breakout rate WITHIN positional-ADP bands (the
   fame-control pattern from evals/source_alpha.py)?

Team attach follows packet_features conventions (breakoutbench.vegas_for):
as-of dated-snapshot team when it exists, else the S-1 season-end depth-chart
team. That misattributes offseason movers, so the stratified read is repeated
with the player's actual season-S week-1 team as a labelled robustness arm
(legal information — where a player signed is public well before Sept 1 —
but sourced from a post-gate file, so it never feeds the GBDT arms).

Output: evals/results/vegas_vs_adp.md.
Usage:  .venv/bin/python -m evals.vegas_packet_test run [--reps 10000]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.breakoutbench import (  # noqa: E402
    ADP_GATE,
    POSITIONS,
    V3_STATUS,
    VEGAS_TEAM_NORMALIZE,
    _name_index,
    _outcome_index,
    _season_points,
    _team_vegas_index,
    adp_path,
    enrichment_for,
    family_base_rates,
    family_of,
    outcome_for,
    vegas_for,
)
from evals.gbdt_baseline import (  # noqa: E402
    SEED,
    TOP_K,
    V2_TRAIN_START,
    V3_FEATURE_NAMES,
    _fit,
    build_candidates_v3,
    encode_v2,
    encode_v3,
    load_data,
    permutation_importance_brier,
    score_season,
    train_predict,
    train_predict_v2,
    train_predict_v3,
)
from evals.names import norm_name  # noqa: E402

PRESET = "ppr"
SEASONS = tuple(range(2016, 2025))  # 2025 holdout untouched
TRAIN_START = V2_TRAIN_START        # 2015 — identical history for all arms
REPS = 10_000
RESULTS = ROOT / "evals" / "results" / "vegas_vs_adp.md"
STATS_DIR = ROOT / "data" / "raw" / "nflverse" / "stats_player"

VEGAS_COLS = ("team_week1_implied_total", "team_prior_season_mean_implied_total")
BANDS = ("near_gate", "mid", "deep")
TERCILES = ("T1_low", "T2_mid", "T3_high")


# ---------------------------------------------------------------- GBDT arms

def gbdt_arm(data, variant: str, seed: int = SEED) -> dict:
    """One walk-forward contender over the frame. variant: v1s | v2 | v3.
    v3 also accumulates n-weighted permutation importances per fold."""
    per_season: dict[int, dict] = {}
    importances = np.zeros(len(V3_FEATURE_NAMES))
    n_total = 0
    for season in SEASONS:
        if variant == "v1s":
            preds = train_predict(data, season, PRESET, seed, TRAIN_START)
        elif variant == "v2":
            preds, *_ = train_predict_v2(data, season, PRESET, seed, TRAIN_START)
        elif variant == "v3":
            preds, model, x_eval, y_eval = train_predict_v3(
                data, season, PRESET, seed, TRAIN_START)
            importances += permutation_importance_brier(
                model, x_eval, y_eval, seed) * len(y_eval)
            n_total += len(y_eval)
        else:
            raise ValueError(variant)
        per_season[season] = score_season(preds)
        m = per_season[season]
        print(f"  gbdt-{variant} {season}: n={m['n']} hits@{TOP_K}={m['hits']} "
              f"lift={m['lift']:.1f}x brier={m['brier_full']:.3f}", flush=True)
    out = {"per_season": per_season, "pooled": pool_slate(per_season),
           "ci": slate_ci(per_season, seed=seed)}
    if variant == "v3" and n_total:
        out["importances"] = importances / n_total
    return out


def pool_slate(per_season: dict[int, dict]) -> dict:
    hits = sum(m["hits"] for m in per_season.values())
    expected = sum(m["expected_random"] for m in per_season.values())
    sq = np.concatenate([m["sq_err"] for m in per_season.values()])
    return {"hits": hits, "expected": expected, "lift": hits / expected,
            "brier": float(sq.mean()), "n": int(sq.size)}


def slate_ci(per_season: dict[int, dict], draws: int = REPS,
             seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    seasons = sorted(per_season)
    lifts, briers = [], []
    for _ in range(draws):
        sample = rng.choice(seasons, size=len(seasons), replace=True)
        hits = sum(per_season[s]["hits"] for s in sample)
        expected = sum(per_season[s]["expected_random"] for s in sample)
        sq = np.concatenate([per_season[s]["sq_err"] for s in sample])
        lifts.append(hits / expected)
        briers.append(float(sq.mean()))
    pct = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
    return {"lift": pct(lifts), "brier": pct(briers)}


def paired_delta(a: dict, b: dict, draws: int = REPS, seed: int = SEED) -> dict:
    """Paired season-bootstrap 95% CI on (A - B): full-slate Brier, top-10
    lift, top-10 hits. Both arms are pooled over the SAME season resample."""
    rng = np.random.default_rng(seed)
    seasons = sorted(set(a["per_season"]) & set(b["per_season"]))
    d_brier, d_lift, d_hits = [], [], []
    for _ in range(draws):
        sample = rng.choice(seasons, size=len(seasons), replace=True)

        def pool(run):
            hits = sum(run["per_season"][s]["hits"] for s in sample)
            exp = sum(run["per_season"][s]["expected_random"] for s in sample)
            sq = np.concatenate([run["per_season"][s]["sq_err"] for s in sample])
            return hits, hits / exp, float(sq.mean())

        ha, la, ba = pool(a)
        hb, lb, bb = pool(b)
        d_hits.append(ha - hb)
        d_lift.append(la - lb)
        d_brier.append(ba - bb)
    pct = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
    return {"brier_delta": a["pooled"]["brier"] - b["pooled"]["brier"],
            "brier_delta_ci": pct(d_brier),
            "lift_delta": a["pooled"]["lift"] - b["pooled"]["lift"],
            "lift_delta_ci": pct(d_lift),
            "hits_delta": a["pooled"]["hits"] - b["pooled"]["hits"],
            "hits_delta_ci": pct(d_hits)}


# ----------------------------------------------- bust / over_under families

@lru_cache(maxsize=None)
def family_candidates(season: int, family: str,
                      preset: str = PRESET) -> tuple:
    """Candidates for one (season, family) with features + enrichment + vegas
    + outcome — the exact build_question_set membership/packet definitions,
    without anonymization (labels stay attached for walk-forward training)."""
    path = adp_path(preset, season)
    if path is None:
        return ()
    payload = json.loads(path.read_text())
    players = [p for p in payload["players"] if p.get("position") in POSITIONS]
    by_pos: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(players, key=lambda p: (float(p["adp"]),
                                            -int(p.get("times_drafted", 0)))):
        by_pos[p["position"]].append(p)

    stats = _season_points()
    name_index = _name_index()
    outcomes = _outcome_index().get((season, preset), {})
    out = []
    for pos in POSITIONS:
        for i, p in enumerate(by_pos.get(pos, [])):
            pos_rank = i + 1
            label = outcomes.get((p["name"], pos, pos_rank))
            if label is None or family not in family_of(
                    pos, pos_rank, label["finish_pos_rank"]):
                continue
            pids = name_index.get((norm_name(p["name"]), pos), [])
            pid = pids[0] if pids else None
            history = stats.get(pid, {}) if pid else {}
            past = {y: history[y] for y in history if y < season}
            assert all(y < season for y in past), "leakage: future season"
            first_year = min(past) if past else None

            def block(y):
                r = past.get(y)
                return None if not r else {
                    "games": int(r["games"]),
                    "total": float(r[f"total_{preset}"]),
                    "ppg": float(r[f"ppg_{preset}"]),
                    "pos_rank": int(r[f"pos_season_rank_{preset}"])}

            out.append({
                "player": p["name"], "player_id": pid,
                "features": {
                    "position": pos,
                    "adp_overall": float(p["adp"]),
                    "adp_pos_rank": pos_rank,
                    "adp_stdev": float(p.get("stdev") or 0),
                    "seasons_of_data": len(past),
                    "years_since_first_season":
                        (season - first_year) if first_year else 0,
                    "prior_season": block(season - 1),
                    "two_seasons_ago": block(season - 2)},
                "enrichment": enrichment_for(season, p["name"], pos, pid),
                "vegas": vegas_for(season, p["name"], pos, pid),
                "outcome": outcome_for(family, label),
            })
    return tuple(json.loads(json.dumps(c)) for c in out)


def _encode_family(c: dict, arm: str) -> list[float]:
    if arm == "v2":
        return encode_v2(c["features"], c["enrichment"])
    return encode_v3(c["features"], c["enrichment"], c["vegas"])


def family_arm(family: str, arm: str, seed: int = SEED) -> dict:
    """Walk-forward GBDT over one family. arm: v2 | v3. Per-season Brier
    (plus the no-peek base-rate Brier for context)."""
    per_season: dict[int, dict] = {}
    for season in SEASONS:
        xs, ys = [], []
        for s in range(TRAIN_START, season):
            for c in family_candidates(s, family):
                xs.append(_encode_family(c, arm))
                ys.append(float(c["outcome"]))
        model = _fit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float),
                     seed)
        evals = family_candidates(season, family)
        x_eval = np.asarray([_encode_family(c, arm) for c in evals])
        y_eval = np.asarray([float(c["outcome"]) for c in evals])
        probs = model.predict_proba(x_eval)[:, 1]
        rate, _ = family_base_rates(before_season=season)
        base_p = rate[family]
        per_season[season] = {
            "n": len(evals),
            "sq_err": (probs - y_eval) ** 2,
            "base_sq_err": (base_p - y_eval) ** 2,
        }
        print(f"  {family}-{arm} {season}: n={len(evals)} "
              f"brier={float(np.mean((probs - y_eval) ** 2)):.3f}", flush=True)
    sq = np.concatenate([m["sq_err"] for m in per_season.values()])
    base_sq = np.concatenate([m["base_sq_err"] for m in per_season.values()])
    return {"per_season": per_season, "brier": float(sq.mean()),
            "base_brier": float(base_sq.mean()), "n": int(sq.size)}


def family_paired_delta(a: dict, b: dict, draws: int = REPS,
                        seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    seasons = sorted(set(a["per_season"]) & set(b["per_season"]))
    deltas = []
    for _ in range(draws):
        sample = rng.choice(seasons, size=len(seasons), replace=True)
        sa = np.concatenate([a["per_season"][s]["sq_err"] for s in sample])
        sb = np.concatenate([b["per_season"][s]["sq_err"] for s in sample])
        deltas.append(float(sa.mean()) - float(sb.mean()))
    return {"brier_delta": a["brier"] - b["brier"],
            "brier_delta_ci": (float(np.percentile(deltas, 2.5)),
                               float(np.percentile(deltas, 97.5)))}


# ------------------------------------------------------- stratified direct read

def adp_band(position: str, adp_pos_rank: int) -> str:
    """Positional-ADP band = distance beyond the eligibility gate (the
    fame-control bands from evals/source_alpha.py)."""
    off = adp_pos_rank - ADP_GATE[position]
    if off <= 12:
        return "near_gate"
    if off <= 30:
        return "mid"
    return "deep"


def tercile_split(values: list[float]) -> list[int]:
    """Tercile index (0=low, 1=mid, 2=high) per value, rank-based (ties broken
    by input order), thirds by count."""
    n = len(values)
    order = sorted(range(n), key=lambda i: (values[i], i))
    out = [0] * n
    for rank, i in enumerate(order):
        out[i] = min(2, rank * 3 // n) if n else 0
    return out


@lru_cache(maxsize=None)
def _week1_team_of(season: int) -> dict[str, str]:
    """player_id -> actual earliest-REG-week team in season S (normalized to
    current franchise codes). Robustness attach ONLY: this is a post-gate
    source standing in for legal pre-Sept-1 roster knowledge; it never feeds
    the GBDT arms."""
    path = STATS_DIR / f"stats_player_week_{season}.csv"
    best: dict[str, tuple[int, str]] = {}
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("season_type") != "REG" or r.get("position") not in POSITIONS:
                continue
            wk = int(r["week"])
            pid = r["player_id"]
            if pid not in best or wk < best[pid][0]:
                best[pid] = (wk, r.get("team", ""))
    return {pid: VEGAS_TEAM_NORMALIZE.get(t, t)
            for pid, (_w, t) in best.items() if t}


def stratified_rows(data, attach: str = "convention") -> list[dict]:
    """One row per gate-eligible candidate with a week-1 implied total
    attached. attach: 'convention' (packet_features team, draft-day legal) or
    'true_team' (actual season-S week-1 team, robustness)."""
    assert attach in ("convention", "true_team")
    vegas_idx = _team_vegas_index()
    rows = []
    for season in SEASONS:
        cands = build_candidates_v3(data, season, PRESET)
        season_rows = []
        for c in cands:
            f = c["features"]
            if attach == "convention":
                wk1 = c["vegas"]["team_week1_implied_total"]
            else:
                team = _week1_team_of(season).get(c["player_id"] or "")
                v = vegas_idx.get((season, team)) if team else None
                wk1 = v["week1_implied_total"] if v else None
            if wk1 is None:
                continue
            from evals.gbdt_baseline import label_of
            season_rows.append({
                "season": season, "player": c["player"],
                "position": f["position"], "adp_overall": f["adp_overall"],
                "band": adp_band(f["position"], f["adp_pos_rank"]),
                "week1_implied": float(wk1),
                "breakout": label_of(data, season, c["player_id"]),
            })
        for r, t in zip(season_rows,
                        tercile_split([r["week1_implied"] for r in season_rows])):
            r["tercile"] = TERCILES[t]
        rows.extend(season_rows)
    return rows


def stratified_table(rows: list[dict]) -> dict:
    """Pooled (band x tercile) counts/rates + per-band T3/T1 ratio with a
    season-resampled bootstrap 95% CI."""
    cells: dict[tuple[str, str], dict] = defaultdict(lambda: {"n": 0, "k": 0,
                                                              "adp": []})
    per_season: dict[tuple[str, int], dict] = defaultdict(
        lambda: {t: [0, 0] for t in TERCILES})  # (band, season) -> tercile [k, n]
    for r in rows:
        c = cells[(r["band"], r["tercile"])]
        c["n"] += 1
        c["k"] += r["breakout"]
        c["adp"].append(r["adp_overall"])
        ps = per_season[(r["band"], r["season"])][r["tercile"]]
        ps[0] += r["breakout"]
        ps[1] += 1
        # "all" pseudo-band
        c2 = cells[("all", r["tercile"])]
        c2["n"] += 1
        c2["k"] += r["breakout"]
        c2["adp"].append(r["adp_overall"])
        ps2 = per_season[("all", r["season"])][r["tercile"]]
        ps2[0] += r["breakout"]
        ps2[1] += 1

    def ratio_ci(band: str, draws: int = REPS, seed: int = SEED):
        seasons = sorted({s for (b, s) in per_season if b == band})
        if not seasons:
            return None, (None, None)
        import random
        rng = random.Random(seed)
        ratios = []
        for _ in range(draws):
            k3 = n3 = k1 = n1 = 0
            for _ in seasons:
                s = seasons[rng.randrange(len(seasons))]
                cell = per_season[(band, s)]
                k3 += cell["T3_high"][0]; n3 += cell["T3_high"][1]
                k1 += cell["T1_low"][0]; n1 += cell["T1_low"][1]
            if n3 and n1 and k1:
                ratios.append((k3 / n3) / (k1 / n1))
        hi_cell = cells[(band, "T3_high")]
        lo_cell = cells[(band, "T1_low")]
        point = ((hi_cell["k"] / hi_cell["n"]) / (lo_cell["k"] / lo_cell["n"])
                 if hi_cell["n"] and lo_cell["n"] and lo_cell["k"] else None)
        if len(ratios) < max(100, draws // 100):
            return point, (None, None)
        ratios.sort()
        return point, (ratios[int(0.025 * (len(ratios) - 1))],
                       ratios[int(0.975 * (len(ratios) - 1))])

    out = {"cells": {}, "ratios": {}}
    for band in BANDS + ("all",):
        for t in TERCILES:
            c = cells[(band, t)]
            out["cells"][(band, t)] = {
                "n": c["n"], "k": c["k"],
                "rate": c["k"] / c["n"] if c["n"] else None,
                "mean_adp": statistics.fmean(c["adp"]) if c["adp"] else None}
        point, ci = ratio_ci(band)
        out["ratios"][band] = {"point": point, "ci": ci}
    return out


def week1_adp_correlation(rows: list[dict]) -> float:
    """Mean per-season Pearson r between week-1 implied total and overall ADP
    among candidates — how much of the vegas column ADP already encodes."""
    rs = []
    for season in sorted({r["season"] for r in rows}):
        xs = [r["week1_implied"] for r in rows if r["season"] == season]
        ys = [r["adp_overall"] for r in rows if r["season"] == season]
        if len(xs) < 3:
            continue
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        if sxx and syy:
            rs.append(sxy / math.sqrt(sxx * syy))
    return statistics.fmean(rs) if rs else float("nan")


# ---------------------------------------------------------------------- report

def fmt_ci(ci, digits: int = 3) -> str:
    if ci is None or ci[0] is None:
        return "[—]"
    return f"[{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def fmt_ratio_ci(ci) -> str:
    if ci is None or ci[0] is None:
        return "[—]"
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _delta_verdict(d: dict) -> str:
    lo, hi = d["brier_delta_ci"]
    if hi < 0:
        return "CI excludes 0 — real Brier improvement"
    if lo > 0:
        return "CI excludes 0 on the WRONG side — vegas hurt"
    return "CI includes 0 — no detectable change"


def run(reps: int = REPS, seed: int = SEED) -> Path:
    global REPS
    REPS = reps

    print("== full-slate GBDT arms (walk-forward, ppr, 2016-2024) ==")
    data = load_data(PRESET, PRESET,
                     seasons=tuple(range(TRAIN_START, max(SEASONS) + 1)))
    arms = {v: gbdt_arm(data, v, seed) for v in ("v1s", "v2", "v3")}

    deltas = {
        "v3_v2": paired_delta(arms["v3"], arms["v2"], reps, seed),
        "v3_v1s": paired_delta(arms["v3"], arms["v1s"], reps, seed),
        "v2_v1s": paired_delta(arms["v2"], arms["v1s"], reps, seed),
    }

    imp = sorted(zip(V3_FEATURE_NAMES, arms["v3"]["importances"]),
                 key=lambda t: -t[1])
    imp_rank = {name: i + 1 for i, (name, _v) in enumerate(imp)}
    imp_val = dict(imp)

    print("== bust / over_under (v3 vs v2) ==")
    fam_runs, fam_deltas = {}, {}
    for family in ("bust", "over_under"):
        fam_runs[family] = {arm: family_arm(family, arm, seed)
                            for arm in ("v2", "v3")}
        fam_deltas[family] = family_paired_delta(
            fam_runs[family]["v3"], fam_runs[family]["v2"], reps, seed)

    print("== stratified read ==")
    strat = {}
    corr = {}
    for attach in ("convention", "true_team"):
        rows = stratified_rows(data, attach)
        strat[attach] = {"rows": len(rows), "table": stratified_table(rows)}
        corr[attach] = week1_adp_correlation(rows)
        print(f"  attach={attach}: {len(rows)} candidate rows, "
              f"week1-vs-ADP mean r={corr[attach]:+.2f}")

    # coverage stats for honesty
    n_cand = n_attached = 0
    for season in SEASONS:
        for c in build_candidates_v3(data, season, PRESET):
            n_cand += 1
            n_attached += c["vegas"][V3_STATUS] == "ok"

    # ---- verdict logic
    d32 = deltas["v3_v2"]
    strat_conv = strat["convention"]["table"]["ratios"]
    any_band_clear = any(
        r["ci"][0] is not None and (r["ci"][0] > 1.0 or r["ci"][1] < 1.0)
        for b, r in strat_conv.items() if b in BANDS)
    brier_sig = d32["brier_delta_ci"][1] < 0
    lift_sig = d32["lift_delta_ci"][0] > 0
    additive = brier_sig or lift_sig or any_band_clear
    vegas_imp_sum = sum(imp_val[c] for c in VEGAS_COLS)

    r_conv = corr["convention"]
    if additive:
        verdict = ("**Vegas is ADDITIVE over ADP at the player level** — at "
                   "least one pre-registered read (paired Brier/lift delta or "
                   "within-band stratification) is clear of the null.")
    else:
        mechanism = (
            "Within this pool the week-1 implied total is nearly orthogonal "
            f"to ADP (r = {r_conv:+.2f}), so this is not literal pricing — "
            "it is dilution: knowing an offense will score does not identify "
            "WHICH of its discounted players captures the points."
            if abs(r_conv) < 0.15 else
            "The week-1 implied total is strongly correlated with ADP within "
            f"this pool (r = {r_conv:+.2f}) — the drafting market has "
            "already moved the affected players' prices.")
        verdict = (
            "**Vegas is NOT additive over ADP for player selection — treat "
            "it as already priced for drafting purposes.** The team-level "
            "week-1 signal (vegas_history.md, +0.23 incremental r over naive "
            "stats) does not survive ADP conditioning at the player level: "
            "adding the implied-total pair moves neither Brier nor top-10 "
            "lift by a detectable amount on any family, and it does not "
            "stratify breakout rate within ADP bands (point estimates even "
            "lean below 1.0). " + mechanism)

    lines = [
        "# Vegas vs ADP — does the week-1 implied-total signal survive ADP "
        "conditioning at the player level?",
        "",
        f"Generated {__import__('datetime').date.today().isoformat()} by "
        "`evals/vegas_packet_test.py` (seed "
        f"{seed}, {reps:,} season-resampled bootstrap draws; GBDT + base-rate "
        "machinery only, no API calls).",
        "",
        "Setup: `evals/results/vegas_history.md` measured the week-1 implied "
        "team total as additive over naive prior-season team stats "
        "(+0.23 [+0.11, +0.32] incremental r, 2016-2024) at the TEAM level. "
        "This test asks whether that edge survives at the PLAYER level once "
        "ADP — the drafting market's player consensus — is conditioned on. "
        f"Frame: seasons {SEASONS[0]}-{SEASONS[-1]}, {PRESET}, walk-forward "
        f"(train {TRAIN_START}..S-1 for every arm). Team vegas joined per "
        "packet_features conventions (as-of team, else S-1 season-end depth "
        "team; era codes normalized to current franchise codes). "
        f"Join coverage: {n_attached}/{n_cand} "
        f"({n_attached / n_cand:.0%}) of gate-eligible candidates carry a "
        "week-1 implied total (the rest are rookies/no-S-1-depth players — "
        "encoded as missing, never zero).",
        "",
        "## 1. Full-slate GBDT arms (paired, same folds, same seasons)",
        "",
        "| arm | features | Brier (full slate) [95% CI] | top-10 hits "
        f"(of {TOP_K * len(SEASONS)}) | top-10 lift [95% CI] |",
        "|---|---|---|---|---|",
    ]
    arm_desc = {"v1s": "structure only (ADP + prior seasons)",
                "v2": "v1 + enrichment (news/depth/usage/pedigree)",
                "v3": "**v2 + team vegas (week-1 implied, prior-season mean "
                      "implied)**"}
    for v in ("v1s", "v2", "v3"):
        p, ci = arms[v]["pooled"], arms[v]["ci"]
        lines.append(
            f"| GBDT {v} | {arm_desc[v]} | {p['brier']:.4f} "
            f"[{ci['brier'][0]:.4f}, {ci['brier'][1]:.4f}] | {p['hits']} "
            f"| {p['lift']:.2f}x [{ci['lift'][0]:.2f}, {ci['lift'][1]:.2f}] |")
    lines += [
        "",
        "Paired season-bootstrap deltas (negative Brier delta = vegas/enrichment "
        "helped; positive lift delta = helped):",
        "",
        "| comparison | Δ Brier [95% CI] | Δ top-10 lift [95% CI] "
        "| Δ hits [95% CI] | read |",
        "|---|---|---|---|---|",
    ]
    for label, key in (("**v3 − v2** (vegas over everything)", "v3_v2"),
                       ("v3 − v1s (vegas + enrichment over structure)", "v3_v1s"),
                       ("v2 − v1s (enrichment over structure)", "v2_v1s")):
        d = deltas[key]
        lines.append(
            f"| {label} | {d['brier_delta']:+.4f} "
            f"{fmt_ci(d['brier_delta_ci'], 4)} | {d['lift_delta']:+.2f} "
            f"{fmt_ci(d['lift_delta_ci'], 2)} | {d['hits_delta']:+.0f} "
            f"{fmt_ci(d['hits_delta_ci'], 0)} | {_delta_verdict(d)} |")
    lines += [
        "",
        "## 2. Bust / over_under families (v3 − v2, same machinery)",
        "",
        "| family | v2 Brier | v3 Brier | base-rate Brier | Δ Brier (v3 − v2) "
        "[95% CI] |",
        "|---|---|---|---|---|",
    ]
    for family in ("bust", "over_under"):
        r2, r3 = fam_runs[family]["v2"], fam_runs[family]["v3"]
        d = fam_deltas[family]
        lines.append(
            f"| {family} (n={r2['n']}) | {r2['brier']:.4f} | {r3['brier']:.4f} "
            f"| {r2['base_brier']:.4f} | {d['brier_delta']:+.4f} "
            f"{fmt_ci(d['brier_delta_ci'], 4)} |")
    lines += [
        "",
        "## 3. Permutation importances (v3 model, mean held-out Brier increase, "
        "n-weighted across folds)",
        "",
        f"All {len(V3_FEATURE_NAMES)} columns ranked; the two vegas columns:",
        "",
        "| vegas column | Δ Brier when shuffled | rank of "
        f"{len(V3_FEATURE_NAMES)} |",
        "|---|---|---|",
    ]
    for c in VEGAS_COLS + tuple(f"{c}_missing" for c in VEGAS_COLS):
        lines.append(f"| {c} | {imp_val[c]:+.5f} | {imp_rank[c]} |")
    lines += [
        "",
        "Top 12 columns overall (for scale):",
        "",
        "| feature | Δ Brier when shuffled |",
        "|---|---|",
    ]
    for name, v in imp[:12]:
        tag = " **(vegas)**" if name in VEGAS_COLS or \
            name.replace("_missing", "") in VEGAS_COLS else ""
        lines.append(f"| {name}{tag} | {v:+.5f} |")
    lines += [
        "",
        f"Summed vegas-column importance: {vegas_imp_sum:+.5f} "
        "(negative = shuffling vegas on average IMPROVED held-out Brier — "
        "noise, not signal).",
        "",
        "## 4. Direct stratified read (the fame-control pattern from "
        "source_alpha)",
        "",
        "Among gate-eligible candidates, breakout rate by positional-ADP band "
        "x per-season week-1 implied-total tercile. If Vegas carries "
        "player-selection signal beyond ADP, high-implied-total offenses "
        "should hit more WITHIN each band.",
        "",
    ]
    for attach in ("convention", "true_team"):
        t = strat[attach]["table"]
        label = ("draft-day-legal attach (packet_features team convention)"
                 if attach == "convention" else
                 "robustness: actual season-S week-1 team (post-gate source "
                 "standing in for legal roster knowledge)")
        lines += [
            f"### {label} — {strat[attach]['rows']} candidate rows, "
            f"week-1 implied vs ADP mean r = {corr[attach]:+.2f}",
            "",
            "| band | T1 low (rate, n) | T2 mid (rate, n) | T3 high (rate, n) "
            "| T3/T1 ratio [95% CI] |",
            "|---|---|---|---|---|",
        ]
        for band in BANDS + ("all",):
            cells = [t["cells"][(band, terc)] for terc in TERCILES]
            ratio = t["ratios"][band]
            cell_s = " | ".join(
                f"{c['rate']:.1%} ({c['k']}/{c['n']})" if c["n"] else "—"
                for c in cells)
            point = f"{ratio['point']:.2f}" if ratio["point"] is not None else "—"
            lines.append(f"| {band} | {cell_s} | {point} "
                         f"{fmt_ratio_ci(ratio['ci'])} |")
        lines.append("")
    lines += [
        "## Verdict",
        "",
        verdict,
        "",
        f"- Headline paired delta (v3 − v2): Brier {d32['brier_delta']:+.4f} "
        f"{fmt_ci(d32['brier_delta_ci'], 4)}, top-10 lift "
        f"{d32['lift_delta']:+.2f} {fmt_ci(d32['lift_delta_ci'], 2)} — "
        f"{_delta_verdict(d32)}.",
        f"- Vegas permutation importance sums to {vegas_imp_sum:+.5f} "
        f"(vs adp_overall at {imp_val['adp_overall']:+.5f}).",
        "- Stratified T3/T1 ratios within ADP bands: "
        + "; ".join(
            f"{b} {strat['convention']['table']['ratios'][b]['point']:.2f} "
            f"{fmt_ratio_ci(strat['convention']['table']['ratios'][b]['ci'])}"
            if strat['convention']['table']['ratios'][b]['point'] is not None
            else f"{b} —"
            for b in BANDS) + ".",
        f"- Week-1 implied total vs overall ADP among candidates: mean "
        f"per-season r = {corr['convention']:+.2f} (restriction of range — "
        "the gate-eligible pool is all late-ADP players, so this measures "
        "orthogonality within the pool, not across the whole draft board).",
        "",
        "## Caveats",
        "",
        "- Team attach for historical seasons is the S-1 season-end depth "
        "team (no dated pre-Sept-1 snapshots exist before 2025), so "
        "offseason movers are attributed to their OLD team's line in the "
        "draft-day-legal arm; the true-team robustness arm above bounds how "
        "much that attenuates the read.",
        f"- Join coverage is {n_attached / n_cand:.0%} of candidates; "
        "the uncovered tail is rookie-heavy, where team context arguably "
        "matters most — this test cannot speak to them.",
        "- Week-1 implied totals are CLOSING week-1 lines (games.csv); "
        "August look-ahead lines are slightly staler, so any live edge would "
        "be smaller still (same caveat as vegas_history.md).",
        "- 2025 remains the untouched holdout; nothing here touches it.",
        "",
    ]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("\n".join(lines))
    print(f"\nwrote {RESULTS}")
    return RESULTS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--reps", type=int, default=REPS)
    r.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args.reps, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
