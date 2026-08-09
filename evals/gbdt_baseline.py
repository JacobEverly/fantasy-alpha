#!/usr/bin/env python3
"""BreakoutBench "boring ML" baseline: gradient-boosted trees, walk-forward.

The bar any LLM must beat to justify itself (docs/breakoutbench-design.md §5.4).
For each target season S in 2015-2024 (ppr), a HistGradientBoostingClassifier
is trained ONLY on gate-eligible candidates from seasons 2011..S-1, built with
the exact slate feature definitions from evals/anon_demo.py (ADP snapshot +
season_points.csv history via evals/names.py joins). It then predicts
p(breakout) for every candidate in the season-S slate.

Feature set is exactly the anonymized-slate packet — position (one-hot),
adp_overall, adp_pos_rank, adp_stdev, seasons_of_data,
years_since_first_season, prior_season{games,total,ppg,pos_rank},
two_seasons_ago{...} — with sentinel + missing-indicator encoding for absent
prior seasons. No extra features: this measures the same-information bar.

Calibration: CalibratedClassifierCV (isotonic when the training frame is large
enough, sigmoid otherwise) fit inside the walk-forward fold, so calibration
data is also strictly <= S-1.

CLI: `python evals/gbdt_baseline.py run` evaluates 2015-2024 and writes
evals/results/gbdt_baseline.md.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evals.names import norm_name  # noqa: E402

LABELS = ROOT / "data" / "processed" / "labels"
DEMO = ROOT / "evals" / "demo"
RESULTS = ROOT / "evals" / "results"

ADP_GATE = {"QB": 18, "TE": 18, "RB": 40, "WR": 40}
POSITIONS = ("QB", "RB", "TE", "WR")  # fixed order for one-hot encoding

SEED = 20260808
TRAIN_START = 2011
EVAL_SEASONS = tuple(range(2015, 2025))
TOP_K = 10
MISSING = -1.0  # sentinel for absent prior-season blocks

SEASON_FIELDS = ("games", "total", "ppg", "pos_rank")
FEATURE_NAMES = (
    [f"pos_{p}" for p in POSITIONS]
    + ["adp_overall", "adp_pos_rank", "adp_stdev",
       "seasons_of_data", "years_since_first_season"]
    + [f"prior_{f}" for f in SEASON_FIELDS] + ["prior_missing"]
    + [f"two_ago_{f}" for f in SEASON_FIELDS] + ["two_ago_missing"]
)

# --- packet v2 (enriched) --------------------------------------------------
# Same enrichment join as the shipped v2 slates (evals/breakoutbench.py), so
# GBDT-v2 and Qwen-v2 see byte-identical added information. Missing values are
# NaN (native HGB missing support) + an explicit coverage indicator per field,
# so the model can learn missingness (e.g. news pre-coverage seasons).
from evals.breakoutbench import V2_FIELDS, V3_FIELDS, enrichment_for, vegas_for  # noqa: E402

V2_ORDER = tuple(V2_FIELDS)
V2_FEATURE_NAMES = list(FEATURE_NAMES) + [
    name for f in V2_ORDER for name in (f, f"{f}_missing")]
V2_TRAIN_START = 2015  # packet_features tables start 2015 (news pre-coverage there)
V2_EVAL_SEASONS = tuple(range(2016, 2025))  # news coverage starts ~2016-05

# --- packet v3 (v2 + draft-day-legal team Vegas) ---------------------------
# Same join as breakoutbench.vegas_for: team attached per packet_features
# conventions (as-of dated snapshot when it exists, else S-1 season-end depth
# team), then (season, current-franchise team) -> week-1 implied total +
# prior-season mean implied total. Never the retrospective season-S mean.
V3_ORDER = tuple(V3_FIELDS)
V3_FEATURE_NAMES = list(V2_FEATURE_NAMES) + [
    name for f in V3_ORDER for name in (f, f"{f}_missing")]
V3_TRAIN_START = V2_TRAIN_START
V3_EVAL_SEASONS = V2_EVAL_SEASONS


# ---------------------------------------------------------------- data access

def load_season_points() -> dict[str, dict[int, dict]]:
    by_player: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(LABELS / "season_points.csv", newline="") as f:
        for r in csv.DictReader(f):
            by_player[r["player_id"]][int(r["season"])] = r
    return by_player


def load_outcomes(preset: str) -> dict[int, dict[str, dict]]:
    """season -> player_id -> breakouts.csv row (for one scoring format)."""
    by_season: dict[int, dict[str, dict]] = defaultdict(dict)
    with open(LABELS / "breakouts.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["format"] == preset and r["player_id"]:
                by_season[int(r["season"])][r["player_id"]] = r
    return by_season


def load_adp(season: int, fmt: str) -> list[dict]:
    adp_dir = ROOT / "data" / "raw" / "ffc_adp"
    path = next(p for t in (12, 10, 14, 8)
                if (p := adp_dir / f"{fmt}_{t}t_{season}.json").exists())
    payload = json.loads(path.read_text())
    return [p for p in payload["players"] if p.get("position") in POSITIONS]


@dataclass(frozen=True)
class Data:
    stats: dict[str, dict[int, dict]]           # player_id -> season -> row
    outcomes: dict[int, dict[str, dict]]        # season -> player_id -> row
    adp: dict[int, list[dict]]                  # season -> ADP player list
    name_index: dict[tuple, list[str]]          # (norm_name, pos) -> [pid]


def load_data(fmt: str = "ppr", preset: str = "ppr",
              seasons: tuple[int, ...] | None = None) -> Data:
    stats = load_season_points()
    name_index: dict[tuple, list[str]] = defaultdict(list)
    for pid, seasons_rows in stats.items():
        any_row = next(iter(seasons_rows.values()))
        name_index[(norm_name(any_row["player"]), any_row["position"])].append(pid)
    wanted = seasons or tuple(range(TRAIN_START, max(EVAL_SEASONS) + 1))
    adp = {s: load_adp(s, fmt) for s in wanted}
    return Data(stats=stats, outcomes=load_outcomes(preset), adp=adp,
                name_index=name_index)


# ------------------------------------------------------------------- features

def build_candidates(data: Data, season: int, preset: str = "ppr") -> list[dict]:
    """Gate-eligible candidates for `season`, replicating anon_demo.build().

    Features use ONLY seasons strictly before `season` (plus the
    target-season ADP snapshot, which is legitimately pre-season info).
    Returns [{"player", "player_id", "features": {...}}] in ADP order.
    """
    by_pos: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(data.adp[season],
                    key=lambda p: (float(p["adp"]), -int(p.get("times_drafted", 0)))):
        by_pos[p["position"]].append(p)

    candidates = []
    for pos, group in by_pos.items():
        for i, p in enumerate(group):
            pos_rank = i + 1
            if pos_rank <= ADP_GATE[pos]:
                continue
            pids = data.name_index.get((norm_name(p["name"]), pos), [])
            pid = pids[0] if pids else None
            history = data.stats.get(pid, {}) if pid else {}
            past = {y: history[y] for y in history if y < season}
            assert all(y < season for y in past), "leakage: future season in features"
            first_year = min(past) if past else None

            def season_block(y: int) -> dict | None:
                r = past.get(y)
                if not r:
                    return None
                return {
                    "games": int(r["games"]),
                    "total": float(r[f"total_{preset}"]),
                    "ppg": float(r[f"ppg_{preset}"]),
                    "pos_rank": int(r[f"pos_season_rank_{preset}"]),
                }

            candidates.append({
                "player": p["name"],
                "player_id": pid,
                "features": {
                    "position": pos,
                    "adp_overall": float(p["adp"]),
                    "adp_pos_rank": pos_rank,
                    "adp_stdev": float(p.get("stdev") or 0),
                    "seasons_of_data": len(past),
                    "years_since_first_season": (season - first_year) if first_year else 0,
                    "prior_season": season_block(season - 1),
                    "two_seasons_ago": season_block(season - 2),
                },
            })
    return candidates


def encode(features: dict) -> list[float]:
    """Numeric vector for one candidate's slate-packet features."""
    row = [1.0 if features["position"] == p else 0.0 for p in POSITIONS]
    row += [float(features["adp_overall"]), float(features["adp_pos_rank"]),
            float(features["adp_stdev"]), float(features["seasons_of_data"]),
            float(features["years_since_first_season"])]
    for block_key in ("prior_season", "two_seasons_ago"):
        block = features.get(block_key)
        if block is None:
            row += [MISSING] * len(SEASON_FIELDS) + [1.0]
        else:
            row += [float(block[f]) for f in SEASON_FIELDS] + [0.0]
    return row


def label_of(data: Data, season: int, pid: str | None) -> int:
    """Same hit definition as anon_demo.score: unmatched/no-data -> 0."""
    out = data.outcomes.get(season, {}).get(pid) if pid else None
    return 1 if out and out["breakout"] == "True" else 0


def training_frame(data: Data, target_season: int, preset: str = "ppr",
                   train_start: int = TRAIN_START) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix + labels from seasons train_start..target_season-1 only."""
    xs, ys = [], []
    for s in range(train_start, target_season):
        assert s < target_season, "leakage: training on target/future season"
        for c in build_candidates(data, s, preset):
            xs.append(encode(c["features"]))
            ys.append(label_of(data, s, c["player_id"]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


# --------------------------------------------------------------- v2 features

def build_candidates_v2(data: Data, season: int, preset: str = "ppr") -> list[dict]:
    """build_candidates + the v2 enrichment block (identical join to the
    shipped v2 slates: player_id first, (norm_name, position) fallback)."""
    cands = build_candidates(data, season, preset)
    for c in cands:
        c["enrichment"] = enrichment_for(
            season, c["player"], c["features"]["position"], c["player_id"])
    return cands


def encode_v2(features: dict, enrichment: dict) -> list[float]:
    row = encode(features)
    for f in V2_ORDER:
        v = enrichment.get(f)
        missing = v is None
        row += [float("nan") if missing else float(v), 1.0 if missing else 0.0]
    return row


def training_frame_v2(data: Data, target_season: int, preset: str = "ppr",
                      train_start: int = V2_TRAIN_START) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for s in range(train_start, target_season):
        assert s < target_season, "leakage: training on target/future season"
        for c in build_candidates_v2(data, s, preset):
            xs.append(encode_v2(c["features"], c["enrichment"]))
            ys.append(label_of(data, s, c["player_id"]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


# --------------------------------------------------------------- v3 features

def build_candidates_v3(data: Data, season: int, preset: str = "ppr") -> list[dict]:
    """build_candidates_v2 + the draft-day-legal team-vegas block."""
    cands = build_candidates_v2(data, season, preset)
    for c in cands:
        c["vegas"] = vegas_for(
            season, c["player"], c["features"]["position"], c["player_id"])
    return cands


def encode_v3(features: dict, enrichment: dict, vegas: dict) -> list[float]:
    row = encode_v2(features, enrichment)
    for f in V3_ORDER:
        v = vegas.get(f)
        missing = v is None
        row += [float("nan") if missing else float(v), 1.0 if missing else 0.0]
    return row


def training_frame_v3(data: Data, target_season: int, preset: str = "ppr",
                      train_start: int = V3_TRAIN_START) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for s in range(train_start, target_season):
        assert s < target_season, "leakage: training on target/future season"
        for c in build_candidates_v3(data, s, preset):
            xs.append(encode_v3(c["features"], c["enrichment"], c["vegas"]))
            ys.append(label_of(data, s, c["player_id"]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def train_predict_v3(data: Data, season: int, preset: str = "ppr",
                     seed: int = SEED, train_start: int = V3_TRAIN_START
                     ) -> tuple[list[dict], object, np.ndarray, np.ndarray]:
    """v3 walk-forward fold (v2 + team vegas). Same return shape as v2 so the
    caller can run permutation importance on the held-out slate."""
    x_train, y_train = training_frame_v3(data, season, preset, train_start)
    model = _fit(x_train, y_train, seed)
    slate = build_candidates_v3(data, season, preset)
    x_eval = np.asarray([encode_v3(c["features"], c["enrichment"], c["vegas"])
                         for c in slate])
    y_eval = np.asarray([label_of(data, season, c["player_id"]) for c in slate],
                        dtype=float)
    probs = model.predict_proba(x_eval)[:, 1]
    preds = [{"player": c["player"], "player_id": c["player_id"],
              "p_breakout": float(p), "label": int(y)}
             for c, p, y in zip(slate, probs, y_eval)]
    return preds, model, x_eval, y_eval


# ---------------------------------------------------------------------- model

def _fit(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier

    base = HistGradientBoostingClassifier(
        random_state=seed, max_iter=300, learning_rate=0.05,
        max_leaf_nodes=15, min_samples_leaf=20, early_stopping=False)
    method = "isotonic" if len(y_train) >= 500 else "sigmoid"
    model = CalibratedClassifierCV(base, method=method, cv=5)
    model.fit(x_train, y_train)
    return model


def train_predict(data: Data, season: int, preset: str = "ppr",
                  seed: int = SEED, train_start: int = TRAIN_START) -> list[dict]:
    """Walk-forward fold: fit on <= season-1, predict the season slate.

    Returns [{"player", "player_id", "p_breakout", "label"}] in slate order.
    """
    x_train, y_train = training_frame(data, season, preset, train_start)
    model = _fit(x_train, y_train, seed)
    slate = build_candidates(data, season, preset)
    probs = model.predict_proba(np.asarray([encode(c["features"]) for c in slate]))[:, 1]
    return [{"player": c["player"], "player_id": c["player_id"],
             "p_breakout": float(p), "label": label_of(data, season, c["player_id"])}
            for c, p in zip(slate, probs)]


def train_predict_v2(data: Data, season: int, preset: str = "ppr",
                     seed: int = SEED, train_start: int = V2_TRAIN_START
                     ) -> tuple[list[dict], object, np.ndarray, np.ndarray]:
    """v2 walk-forward fold. Returns (preds, fitted model, X_eval, y_eval) so
    the caller can run permutation importance on the held-out slate."""
    x_train, y_train = training_frame_v2(data, season, preset, train_start)
    model = _fit(x_train, y_train, seed)
    slate = build_candidates_v2(data, season, preset)
    x_eval = np.asarray([encode_v2(c["features"], c["enrichment"]) for c in slate])
    y_eval = np.asarray([label_of(data, season, c["player_id"]) for c in slate],
                        dtype=float)
    probs = model.predict_proba(x_eval)[:, 1]
    preds = [{"player": c["player"], "player_id": c["player_id"],
              "p_breakout": float(p), "label": int(y)}
             for c, p, y in zip(slate, probs, y_eval)]
    return preds, model, x_eval, y_eval


def permutation_importance_brier(model, x_eval: np.ndarray, y_eval: np.ndarray,
                                 seed: int = SEED, reps: int = 10) -> np.ndarray:
    """Mean Brier increase on the held-out slate when each column is shuffled.
    Positive = the column carried signal for this fold."""
    rng = np.random.default_rng(seed)
    base = float(np.mean((model.predict_proba(x_eval)[:, 1] - y_eval) ** 2))
    out = np.zeros(x_eval.shape[1])
    for j in range(x_eval.shape[1]):
        deltas = []
        for _ in range(reps):
            xp = x_eval.copy()
            rng.shuffle(xp[:, j])
            deltas.append(float(np.mean(
                (model.predict_proba(xp)[:, 1] - y_eval) ** 2)) - base)
        out[j] = float(np.mean(deltas))
    return out


# -------------------------------------------------------------------- scoring

def score_season(preds: list[dict], k: int = TOP_K) -> dict:
    labels = np.array([r["label"] for r in preds], dtype=float)
    probs = np.array([r["p_breakout"] for r in preds], dtype=float)
    order = np.argsort(-probs, kind="stable")
    top = order[:k]
    base_rate = labels.mean()
    hits = int(labels[top].sum())
    return {
        "n": len(preds),
        "n_breakouts": int(labels.sum()),
        "base_rate": float(base_rate),
        "hits": hits,
        "expected_random": float(base_rate * k),
        "lift": hits / (base_rate * k) if base_rate else float("nan"),
        "brier_full": float(np.mean((probs - labels) ** 2)),
        "brier_topk": float(np.mean((probs[top] - labels[top]) ** 2)),
        "sq_err": (probs - labels) ** 2,
        "sq_err_topk": (probs[top] - labels[top]) ** 2,
        "probs": probs,
        "labels": labels,
        "top_players": [(preds[i]["player"], float(probs[i]), int(labels[i]))
                        for i in top],
    }


def pooled_metrics(per_season: dict[int, dict]) -> dict:
    hits = sum(m["hits"] for m in per_season.values())
    expected = sum(m["expected_random"] for m in per_season.values())
    sq = np.concatenate([m["sq_err"] for m in per_season.values()])
    sq_top = np.concatenate([m["sq_err_topk"] for m in per_season.values()])
    return {"hits": hits, "picks": TOP_K * len(per_season),
            "expected_random": expected, "lift": hits / expected,
            "brier_full": float(sq.mean()), "brier_topk": float(sq_top.mean())}


def bootstrap_ci(per_season: dict[int, dict], draws: int = 10_000,
                 seed: int = SEED) -> dict:
    """Season-resampled bootstrap 95% CIs for pooled metrics."""
    rng = np.random.default_rng(seed)
    seasons = sorted(per_season)
    stats = {"hits": [], "lift": [], "brier_full": [], "brier_topk": []}
    for _ in range(draws):
        sample = rng.choice(seasons, size=len(seasons), replace=True)
        hits = sum(per_season[s]["hits"] for s in sample)
        expected = sum(per_season[s]["expected_random"] for s in sample)
        sq = np.concatenate([per_season[s]["sq_err"] for s in sample])
        sq_top = np.concatenate([per_season[s]["sq_err_topk"] for s in sample])
        stats["hits"].append(hits)
        stats["lift"].append(hits / expected)
        stats["brier_full"].append(sq.mean())
        stats["brier_topk"].append(sq_top.mean())
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in stats.items()}


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> tuple[float, list]:
    """10-bin expected calibration error + per-bin reliability rows."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    total, rows = 0.0, []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append((edges[b], edges[b + 1], 0, None, None))
            continue
        conf, acc = float(probs[mask].mean()), float(labels[mask].mean())
        total += (n / len(probs)) * abs(acc - conf)
        rows.append((edges[b], edges[b + 1], n, conf, acc))
    return total, rows


# ------------------------------------------------------------------ reporting

QWEN_ROWS = [
    ("Qwen naked (top-10-only Brier)", 19, "1.3x", "n/a", 0.493),
    ("Qwen + harness (top-10-only Brier)", 18, "1.2x", "n/a", 0.241),
]


def run(preset: str = "ppr", seed: int = SEED) -> Path:
    data = load_data(preset, preset)
    per_season: dict[int, dict] = {}
    for season in EVAL_SEASONS:
        preds = train_predict(data, season, preset, seed)
        per_season[season] = score_season(preds)
        m = per_season[season]
        print(f"{season}: n={m['n']} breakouts={m['n_breakouts']} "
              f"hits@{TOP_K}={m['hits']} lift={m['lift']:.1f}x "
              f"brier_full={m['brier_full']:.3f} brier_top{TOP_K}={m['brier_topk']:.3f}")

    pooled = pooled_metrics(per_season)
    ci = bootstrap_ci(per_season, seed=seed)
    all_probs = np.concatenate([m["probs"] for m in per_season.values()])
    all_labels = np.concatenate([m["labels"] for m in per_season.values()])
    ece_val, ece_rows = ece(all_probs, all_labels)

    lines = [
        "# BreakoutBench — GBDT baseline (boring ML bar)",
        "",
        f"Generated {__import__('datetime').date.today().isoformat()} by "
        "`evals/gbdt_baseline.py` (seed 20260808, scikit-learn "
        f"HistGradientBoostingClassifier + CalibratedClassifierCV).",
        "",
        "Protocol: strict walk-forward. To predict season S, the model trains only",
        f"on gate-eligible candidates from seasons {TRAIN_START}..S-1 (features rebuilt",
        "with the exact `evals/anon_demo.py` slate definitions; labels from",
        "`data/processed/labels/breakouts.csv`). Feature set is exactly the anonymized",
        "slate packet — no extra information. Calibration (sigmoid <500 rows, isotonic",
        "otherwise) is fit inside each fold, so it also sees only seasons <= S-1.",
        "Format: ppr. Eval seasons: 2015-2024. Top-10 picks per slate.",
        "",
        "## Per-season results",
        "",
        "| Season | Slate n | Breakouts | Base rate | Hits@10 | Lift | Brier (full slate) | Brier (top-10) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in EVAL_SEASONS:
        m = per_season[s]
        lines.append(f"| {s} | {m['n']} | {m['n_breakouts']} | {m['base_rate']:.1%} "
                     f"| {m['hits']} | {m['lift']:.1f}x | {m['brier_full']:.3f} "
                     f"| {m['brier_topk']:.3f} |")
    lines += [
        "",
        "## Pooled (2015-2024) with season-resampled bootstrap 95% CI (10,000 draws)",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
        f"| Hits@10 (of {pooled['picks']}) | {pooled['hits']} "
        f"| [{ci['hits'][0]:.0f}, {ci['hits'][1]:.0f}] |",
        f"| Lift vs random | {pooled['lift']:.2f}x "
        f"| [{ci['lift'][0]:.2f}x, {ci['lift'][1]:.2f}x] |",
        f"| Brier, full slate | {pooled['brier_full']:.3f} "
        f"| [{ci['brier_full'][0]:.3f}, {ci['brier_full'][1]:.3f}] |",
        f"| Brier, top-10 subset | {pooled['brier_topk']:.3f} "
        f"| [{ci['brier_topk'][0]:.3f}, {ci['brier_topk'][1]:.3f}] |",
        "",
        "## GBDT vs measured Qwen (same slates, 2015-2024)",
        "",
        "Note: the Qwen runs reported Brier over their 10 picks only, so the",
        "apples-to-apples GBDT column is the top-10-subset Brier, not full-slate.",
        "",
        "| Model | Hits (of 100) | Lift | Brier full slate | Brier top-10 |",
        "|---|---|---|---|---|",
        f"| **GBDT baseline** | **{pooled['hits']}** | **{pooled['lift']:.1f}x** "
        f"| {pooled['brier_full']:.3f} | **{pooled['brier_topk']:.3f}** |",
    ]
    for name, hits, lift, bf, bt in QWEN_ROWS:
        lines.append(f"| {name} | {hits} | {lift} | {bf} | {bt} |")
    lines += [
        "| Random | ~{:.0f} | 1.0x | n/a | n/a |".format(pooled["expected_random"]),
        "",
        "## Reliability (10-bin, pooled full-slate predictions)",
        "",
        f"**ECE = {ece_val:.3f}** over {len(all_probs)} candidate predictions.",
        "",
        "| Bin | n | Mean p | Hit rate |",
        "|---|---|---|---|",
    ]
    for lo, hi, n, conf, acc in ece_rows:
        conf_s = f"{conf:.3f}" if conf is not None else "—"
        acc_s = f"{acc:.3f}" if acc is not None else "—"
        lines.append(f"| [{lo:.1f}, {hi:.1f}) | {n} | {conf_s} | {acc_s} |")
    lines.append("")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "gbdt_baseline.md"
    out.write_text("\n".join(lines))
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run()
    else:
        print("usage: python evals/gbdt_baseline.py run")
