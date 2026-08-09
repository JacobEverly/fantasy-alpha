#!/usr/bin/env python3
"""Enriched-packet experiment — does the evidence channel add rankable signal?

The decisive test the structure-only powered null (evals/results/
expanded_baseline.md) sets up: same BreakoutBench v0.3 machinery, but the
candidate dossiers now carry the anonymization-safe, historically-covered
packet-v2 enrichment (news window counts, S-1 season-end depth rank, age,
draft capital, opportunities/game + target share + YoY). Frame: seasons
2016-2024 (news coverage), ppr, all three families; full_slate is the
primary question.

Contenders on the SAME frame:
  - GBDT v1  (structure features, train 2011..S-1)   — the boring-ML bar
  - GBDT v1s (structure features, train 2015..S-1)   — history-length control
  - GBDT v2  (enriched features,  train 2015..S-1)   — does enrichment help ML?
  - Qwen naked / harness on v1 packets (existing answers, re-scored on frame)
  - Qwen naked_v2 / harness_v2 on v2 packets (evals/run_expanded_qwen.py run_v2)
  - base rate (no-peek) + coin — feature-blind anchors

Output: evals/results/enriched_experiment.md (side-by-side, era split, paired
season-bootstrap deltas, GBDT-v2 permutation importances, verdicts).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.breakoutbench import (  # noqa: E402
    ANSWERS_DIR,
    BOOTSTRAP_SEED,
    FAMILIES,
    QUESTIONS_DIR,
    TOP_K,
    build,
    cell_aggregates,
    load_answer_files,
    score_run,
    season_formats,
    write_baseline_answers,
    _pool,
)
from evals.gbdt_baseline import (  # noqa: E402
    SEED,
    TRAIN_START,
    V2_EVAL_SEASONS,
    V2_FEATURE_NAMES,
    V2_TRAIN_START,
    load_data,
    permutation_importance_brier,
    score_season,
    train_predict,
    train_predict_v2,
)
from evals.run_expanded_qwen import fmt_ci, overlap, run_v2  # noqa: E402

PRESET = "ppr"
SEASONS = tuple(V2_EVAL_SEASONS)  # 2016-2024
ERAS = (("2016-2019", range(2016, 2020)), ("2020-2024", range(2020, 2025)))
RESULTS = ROOT / "evals" / "results" / "enriched_experiment.md"
REPS = 10_000


# --------------------------------------------------------------- Qwen scoring

def answer_paths(run_name: str) -> list[Path]:
    paths = []
    for s in SEASONS:
        for fam in FAMILIES:
            p = ANSWERS_DIR / f"{run_name}_{s}_{PRESET}_{fam}.json"
            if p.exists():
                paths.append(p)
    return paths


def baseline_paths(name: str) -> list[Path]:
    return [ANSWERS_DIR / f"{name}_{s}_{PRESET}.json" for s in SEASONS]


def paired_delta_ci(paths_a: list[Path], paths_b: list[Path],
                    reps: int = REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired season-bootstrap 95% CI on (A - B) for pooled Brier and top-10
    lift, resampling the SAME seasons for both runs."""
    import random
    agg_a = cell_aggregates(load_answer_files(paths_a)["cells"])
    agg_b = cell_aggregates(load_answer_files(paths_b)["cells"])
    seasons = sorted({s for s, _ in agg_a} & {s for s, _ in agg_b})
    rng = random.Random(seed)
    d_brier, d_lift = [], []
    for _ in range(reps):
        draw = [seasons[rng.randrange(len(seasons))] for _ in seasons]
        pa, pb = _pool(agg_a, draw), _pool(agg_b, draw)
        if pa["pooled_brier"] is not None and pb["pooled_brier"] is not None:
            d_brier.append(pa["pooled_brier"] - pb["pooled_brier"])
        if pa["lift"] is not None and pb["lift"] is not None:
            d_lift.append(pa["lift"] - pb["lift"])

    def ci(vals):
        if not vals:
            return None
        vs = sorted(vals)
        return (vs[int(0.025 * (len(vs) - 1))], vs[int(0.975 * (len(vs) - 1))])

    pa = _pool(agg_a, seasons)
    pb = _pool(agg_b, seasons)
    return {"brier_delta": pa["pooled_brier"] - pb["pooled_brier"],
            "brier_delta_ci": ci(d_brier),
            "lift_delta": pa["lift"] - pb["lift"],
            "lift_delta_ci": ci(d_lift)}


# --------------------------------------------------------------- GBDT battery

def gbdt_run(data, variant: str, seed: int = SEED,
             importance: bool = False) -> dict:
    """One GBDT contender over the frame. variant: v1 | v1s | v2."""
    per_season: dict[int, dict] = {}
    importances = np.zeros(len(V2_FEATURE_NAMES))
    n_total = 0
    for season in SEASONS:
        if variant == "v1":
            preds = train_predict(data, season, PRESET, seed, TRAIN_START)
        elif variant == "v1s":
            preds = train_predict(data, season, PRESET, seed, V2_TRAIN_START)
        elif variant == "v2":
            preds, model, x_eval, y_eval = train_predict_v2(
                data, season, PRESET, seed, V2_TRAIN_START)
            if importance:
                importances += permutation_importance_brier(
                    model, x_eval, y_eval, seed) * len(y_eval)
                n_total += len(y_eval)
        else:
            raise ValueError(variant)
        per_season[season] = score_season(preds)
        m = per_season[season]
        print(f"  gbdt-{variant} {season}: n={m['n']} hits@{TOP_K}={m['hits']} "
              f"lift={m['lift']:.1f}x brier={m['brier_full']:.3f}", flush=True)
    out = {"per_season": per_season, "pooled": _gbdt_pool(per_season),
           "ci": _gbdt_ci(per_season, seed=seed), "eras": {}}
    for era, era_seasons in ERAS:
        sub = {s: per_season[s] for s in per_season if s in era_seasons}
        out["eras"][era] = {"pooled": _gbdt_pool(sub), "ci": _gbdt_ci(sub, seed=seed)}
    if importance and n_total:
        out["importances"] = importances / n_total
    return out


def _gbdt_pool(per_season: dict[int, dict]) -> dict:
    hits = sum(m["hits"] for m in per_season.values())
    expected = sum(m["expected_random"] for m in per_season.values())
    sq = np.concatenate([m["sq_err"] for m in per_season.values()])
    return {"hits": hits, "expected": expected, "lift": hits / expected,
            "brier": float(sq.mean()), "n": int(sq.size)}


def _gbdt_ci(per_season: dict[int, dict], draws: int = REPS,
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


def gbdt_paired_delta(a: dict, b: dict, draws: int = REPS,
                      seed: int = SEED) -> dict:
    """Paired season-bootstrap CI on (A - B) full-slate Brier and lift."""
    rng = np.random.default_rng(seed)
    seasons = sorted(set(a["per_season"]) & set(b["per_season"]))
    d_brier, d_lift = [], []
    for _ in range(draws):
        sample = rng.choice(seasons, size=len(seasons), replace=True)

        def pool(run):
            hits = sum(run["per_season"][s]["hits"] for s in sample)
            exp = sum(run["per_season"][s]["expected_random"] for s in sample)
            sq = np.concatenate([run["per_season"][s]["sq_err"] for s in sample])
            return hits / exp, float(sq.mean())

        la, ba = pool(a)
        lb, bb = pool(b)
        d_lift.append(la - lb)
        d_brier.append(ba - bb)
    pct = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
    return {"brier_delta": a["pooled"]["brier"] - b["pooled"]["brier"],
            "brier_delta_ci": pct(d_brier),
            "lift_delta": a["pooled"]["lift"] - b["pooled"]["lift"],
            "lift_delta_ci": pct(d_lift)}


# --------------------------------------------------------------------- report

def _clear_of_one(ci) -> str:
    if ci is None:
        return "no data"
    if ci[0] > 1.0:
        return "YES — CI clear of 1.0 above"
    if ci[1] < 1.0:
        return "below 1.0 (anti-signal)"
    return "NO — CI brackets 1.0"


def qwen_row(label: str, r: dict) -> str:
    p, ci = r["pooled"], r["ci"]
    fs = p["families"]["full_slate"]
    return (f"| {label} | {fs['brier']:.4f} {fmt_ci(ci.get('brier_full_slate'))} "
            f"| {p['pooled_brier']:.4f} {fmt_ci(ci.get('pooled_brier'))} "
            f"| {p['lift']:.2f}x {fmt_ci(ci.get('lift'), 2)} | {r['answered']} |")


def gbdt_row(label: str, r: dict) -> str:
    p, ci = r["pooled"], r["ci"]
    return (f"| {label} | {p['brier']:.4f} {fmt_ci(ci['brier'])} | — "
            f"| {p['lift']:.2f}x {fmt_ci(ci['lift'], 2)} | {p['n']} |")


def main() -> int:
    # 1. enriched slates (idempotent; asserts anon-id/key identity)
    print("building v2 slates ...")
    build(list(SEASONS), (PRESET,), packet_version="v2")

    # 2. Qwen enriched batteries (resume-safe, budget-capped)
    spend = run_v2(presets=(PRESET,))

    # 3. baselines regenerated for exact parity, then score everything on the frame
    write_baseline_answers(SEASONS, (PRESET,))
    runs = {
        "qwen_naked (structure)": score_run(answer_paths("qwen_naked")),
        "qwen_harness (structure)": score_run(answer_paths("qwen_harness")),
        "qwen_naked_v2 (enriched)": score_run(answer_paths("qwen_naked_v2")),
        "qwen_harness_v2 (enriched)": score_run(answer_paths("qwen_harness_v2")),
        "base_rate": score_run(baseline_paths("base_rate")),
        "coin": score_run(baseline_paths("coin")),
    }

    # 4. GBDT battery
    data = load_data(PRESET, PRESET, seasons=tuple(range(TRAIN_START, max(SEASONS) + 1)))
    gbdt = {
        "v1": gbdt_run(data, "v1"),
        "v1s": gbdt_run(data, "v1s"),
        "v2": gbdt_run(data, "v2", importance=True),
    }

    # 5. paired deltas (enriched minus structure, same seasons)
    q_delta = {
        "naked": paired_delta_ci(answer_paths("qwen_naked_v2"),
                                 answer_paths("qwen_naked")),
        "harness": paired_delta_ci(answer_paths("qwen_harness_v2"),
                                   answer_paths("qwen_harness")),
    }
    g_delta = gbdt_paired_delta(gbdt["v2"], gbdt["v1s"])

    # 6. importances, grouped
    imp = sorted(zip(V2_FEATURE_NAMES, gbdt["v2"]["importances"]),
                 key=lambda t: -t[1])

    # ---- report
    n_fs = runs["base_rate"]["pooled"]["families"]["full_slate"]["n"]
    lines = [
        "# Enriched-packet experiment — does the evidence channel add rankable signal?",
        "",
        f"Frame: seasons {SEASONS[0]}-{SEASONS[-1]} (news coverage), format {PRESET}, "
        f"all three families; **full_slate is the primary question** ({n_fs} judgments). "
        "Enrichment = packet-v2 fields with honest historical coverage (news 90-day window "
        "counts + keyword flags, S-1 season-end depth rank, age, draft capital, "
        "opportunities/game + target share + YoY), joined from "
        "`data/processed/packet_features/`; the as-of-null channels (depth movement, "
        "vacated opportunity, coaching) are excluded. Structure-only rows are the SAME "
        "contenders re-scored on this identical frame (subset of the expanded battery), so "
        "every comparison is like-for-like. 95% CIs: season-resampled bootstrap "
        f"({REPS:,} draws, seeded). GBDT rows are full_slate-only by construction.",
        "",
        "## Side-by-side: structure-only vs enriched",
        "",
        "| contender | full_slate Brier [95% CI] | pooled Brier [95% CI] "
        "| top-10 lift [95% CI] | n answered |",
        "|---|---|---|---|---|",
        gbdt_row("GBDT structure (train 2011..S-1)", gbdt["v1"]),
        gbdt_row("GBDT structure, short history (train 2015..S-1)", gbdt["v1s"]),
        gbdt_row("**GBDT enriched v2** (train 2015..S-1)", gbdt["v2"]),
        qwen_row("Qwen naked, structure", runs["qwen_naked (structure)"]),
        qwen_row("**Qwen naked, enriched v2**", runs["qwen_naked_v2 (enriched)"]),
        qwen_row("Qwen harness, structure", runs["qwen_harness (structure)"]),
        qwen_row("**Qwen harness, enriched v2**", runs["qwen_harness_v2 (enriched)"]),
        qwen_row("base rate (no-peek, feature-blind)", runs["base_rate"]),
        qwen_row("coin (p=0.5)", runs["coin"]),
        "",
        "Paired season-bootstrap deltas (enriched − structure, same seasons; negative "
        "Brier delta = enrichment helped, positive lift delta = enrichment helped):",
        "",
        "| contender | Δ full-frame Brier [95% CI] | Δ top-10 lift [95% CI] |",
        "|---|---|---|",
        f"| GBDT (v2 − v1-short) | {g_delta['brier_delta']:+.4f} "
        f"{fmt_ci(g_delta['brier_delta_ci'], 4)} | {g_delta['lift_delta']:+.2f} "
        f"{fmt_ci(g_delta['lift_delta_ci'], 2)} |",
        f"| Qwen naked (v2 − v1) | {q_delta['naked']['brier_delta']:+.4f} "
        f"{fmt_ci(q_delta['naked']['brier_delta_ci'], 4)} | "
        f"{q_delta['naked']['lift_delta']:+.2f} "
        f"{fmt_ci(q_delta['naked']['lift_delta_ci'], 2)} |",
        f"| Qwen harness (v2 − v1) | {q_delta['harness']['brier_delta']:+.4f} "
        f"{fmt_ci(q_delta['harness']['brier_delta_ci'], 4)} | "
        f"{q_delta['harness']['lift_delta']:+.2f} "
        f"{fmt_ci(q_delta['harness']['lift_delta_ci'], 2)} |",
        "",
        "(GBDT delta pools full_slate only; Qwen deltas pool all three families for "
        "Brier and full_slate for lift — same convention as the headline table.)",
        "",
        "## Per-era split",
        "",
        "| contender | era | Brier [95% CI] | top-10 lift [95% CI] | n |",
        "|---|---|---|---|---|",
    ]
    for label, key in (("GBDT structure", "v1"), ("GBDT structure-short", "v1s"),
                       ("GBDT enriched v2", "v2")):
        for era, er in gbdt[key]["eras"].items():
            ep, eci = er["pooled"], er["ci"]
            lines.append(f"| {label} | {era} | {ep['brier']:.4f} {fmt_ci(eci['brier'])} "
                         f"| {ep['lift']:.2f}x {fmt_ci(eci['lift'], 2)} | {ep['n']} |")
    for label in ("qwen_naked (structure)", "qwen_naked_v2 (enriched)",
                  "qwen_harness (structure)", "qwen_harness_v2 (enriched)",
                  "base_rate"):
        r = runs[label]
        for era, er in r["eras"].items():
            ep = er["pooled"]
            era_label = era.replace("2015-2019", "2016-2019")
            lines.append(f"| {label} | {era_label} | {ep['pooled_brier']:.4f} "
                         f"{fmt_ci(er['ci'].get('pooled_brier'))} | {ep['lift']:.2f}x "
                         f"{fmt_ci(er['ci'].get('lift'), 2)} | {ep['n']} |")

    # verdicts
    g2, g1s = gbdt["v2"], gbdt["v1s"]
    qn2 = runs["qwen_naked_v2 (enriched)"]
    qh2 = runs["qwen_harness_v2 (enriched)"]
    lines += [
        "",
        "## Verdicts",
        "",
        f"**(i) Does enrichment give GBDT rankable signal?** GBDT-v2 top-10 lift "
        f"{g2['pooled']['lift']:.2f}x {fmt_ci(g2['ci']['lift'], 2)} — "
        f"{_clear_of_one(g2['ci']['lift'])}. Paired Brier delta vs the structure-only "
        f"control (same training history): {g_delta['brier_delta']:+.4f} "
        f"{fmt_ci(g_delta['brier_delta_ci'], 4)}"
        f"{' — CI excludes 0, a real change' if g_delta['brier_delta_ci'][1] < 0 or g_delta['brier_delta_ci'][0] > 0 else ' — CI includes 0, no detectable change'}.",
        "",
        f"**(ii) Does enrichment give Qwen any?** naked_v2 lift "
        f"{qn2['pooled']['lift']:.2f}x {fmt_ci(qn2['ci'].get('lift'), 2)} — "
        f"{_clear_of_one(qn2['ci'].get('lift'))}; harness_v2 lift "
        f"{qh2['pooled']['lift']:.2f}x {fmt_ci(qh2['ci'].get('lift'), 2)} — "
        f"{_clear_of_one(qh2['ci'].get('lift'))}. Paired deltas vs their structure-only "
        f"runs: naked Δlift {q_delta['naked']['lift_delta']:+.2f} "
        f"{fmt_ci(q_delta['naked']['lift_delta_ci'], 2)}, harness Δlift "
        f"{q_delta['harness']['lift_delta']:+.2f} "
        f"{fmt_ci(q_delta['harness']['lift_delta_ci'], 2)}."
        + (" Note: naked Qwen's pooled Brier delta "
           f"({q_delta['naked']['brier_delta']:+.4f} "
           f"{fmt_ci(q_delta['naked']['brier_delta_ci'], 4)}) excludes 0 on the "
           "WRONG side — enrichment measurably worsened its calibration "
           "(it overreacts to the added evidence)."
           if q_delta['naked']['brier_delta_ci'][0] > 0 else "")
        + (" Note: harness Qwen's pooled Brier delta "
           f"({q_delta['harness']['brier_delta']:+.4f} "
           f"{fmt_ci(q_delta['harness']['brier_delta_ci'], 4)}) excludes 0 on the "
           "WRONG side — enrichment measurably worsened its calibration."
           if q_delta['harness']['brier_delta_ci'][0] > 0 else ""),
        "",
        f"**(iii) Does the LLM gain more from enrichment than GBDT (co-design)?** "
        f"GBDT Δlift {g_delta['lift_delta']:+.2f} {fmt_ci(g_delta['lift_delta_ci'], 2)} vs "
        f"Qwen naked Δlift {q_delta['naked']['lift_delta']:+.2f} "
        f"{fmt_ci(q_delta['naked']['lift_delta_ci'], 2)} / harness Δlift "
        f"{q_delta['harness']['lift_delta']:+.2f} "
        f"{fmt_ci(q_delta['harness']['lift_delta_ci'], 2)}"
        f"{' — the delta CIs overlap heavily; no evidence either learner extracts more from the channel' if overlap(g_delta['lift_delta_ci'], q_delta['naked']['lift_delta_ci']) else ' — the delta CIs separate'}.",
        "",
        "## GBDT-v2 permutation importances (mean held-out Brier increase, "
        "n-weighted across folds)",
        "",
        "| feature | Δ Brier when shuffled |",
        "|---|---|",
    ]
    enrich_names = set(V2_FEATURE_NAMES[19:])  # 19 structure cols, then enrichment
    for name, v in imp[:18]:
        tag = " (enrichment)" if name in enrich_names else ""
        lines.append(f"| {name}{tag} | {v:+.5f} |")
    enrich_total = sum(v for n, v in imp if n in enrich_names)
    struct_total = sum(v for n, v in imp if n not in enrich_names)
    lines += [
        "",
        f"Summed importance — structure columns {struct_total:+.5f}, enrichment columns "
        f"{enrich_total:+.5f}. (Negative sums mean shuffling those columns on average "
        "IMPROVED held-out Brier — noise, not signal.)",
        "",
        "## Cost",
        "",
        f"Qwen v2 batteries this run: {spend['requests']} requests, "
        f"{spend['in']}in/{spend['out']}out tokens ≈ ${spend['cost']:.3f} "
        "(cap $3.00; logged in docs/budget-ledger.md). GBDT/baselines: local CPU, $0.",
        "",
    ]
    RESULTS.write_text("\n".join(lines))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
