#!/usr/bin/env python3
"""Standard-presentation benchmark artifacts: model comparison + dashboard.

Generates the two industry-standard benchmark surfaces from the measured data:

  evals/results/model_comparison.md   quality x cost x latency table + reading
  evals/results/dashboard.html        (a) quality-vs-cost scatter + leaderboard
                                      (b) training-progress chart (our
                                          checkpoints vs flat baselines)

Data sources (read live at build time):
  - evals/results/answers/frontier_spend.json  exact per-model OpenRouter spend
  - data/processed/telemetry/eval_requests.jsonl  per-request latency/cost
    (evals/telemetry.py; empty for pre-instrumentation runs -> latency "n/a")
  - training/checkpoints_log.jsonl  our-model checkpoint metrics over training

Quality metrics (pooled Brier / lift / canary verdicts) are pinned constants
below, transcribed from evals/results/frontier_battery.md — after any new
battery run, update LEADERBOARD from that report and re-run this script.

training/checkpoints_log.jsonl schema (one JSON object per line):

    run_id           str   training run identifier (e.g. "b3-sft-v0")
    phase            str   "B2" | "B3" | "B4" (gameplan phase)
    step             int   optimizer step (0 = untrained / proxy point)
    tokens_trained   int   cumulative training tokens at this checkpoint
    cost_cumulative  float cumulative training spend (USD) at this checkpoint
    metrics          dict  {"pooled_brier": float|null,
                            "draftgym_masked_reward": float|null,
                            "league_index": float|null}
                           (extra metric keys allowed; null = not evaluated)
    timestamp        str   ISO-8601 checkpoint eval time
    note             str   optional provenance note

T1/T2 append one row per evaluated checkpoint; this script re-renders the
training-progress chart from whatever rows exist.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.telemetry import TELEMETRY_PATH, load as load_telemetry, summarize  # noqa: E402

SPEND_PATH = ROOT / "evals" / "results" / "answers" / "frontier_spend.json"
CHECKPOINTS_PATH = ROOT / "training" / "checkpoints_log.jsonl"
COMPARISON_MD = ROOT / "evals" / "results" / "model_comparison.md"
DASHBOARD_HTML = ROOT / "evals" / "results" / "dashboard.html"

# ---------------------------------------------------------------------------
# Pinned quality rows — transcribed from evals/results/frontier_battery.md
# (run date 2026-08-08, anonymized ppr track 2015-2024). Update after each
# battery run. Only COMPLETE runs (full 10-season grid) are listed; partial
# pre-restriction rows are excluded exactly as the battery's verdicts exclude
# them.

LEADERBOARD = [
    # model label, config, slug (cost key), OpenRouter/provider id,
    # pooled Brier, CI, top-10 lift, CI, canary verdict
    dict(label="Claude Fable 5", config="naked", slug="fable5",
         model_id="anthropic/claude-fable-5",
         brier=0.2000, brier_ci=(0.190, 0.210), lift=1.27, lift_ci=(0.84, 1.73),
         canary="PASS (swap idx -0.194 [-0.333, -0.045])"),
    dict(label="Claude Fable 5", config="+ harness", slug="fable5",
         model_id="anthropic/claude-fable-5",
         brier=0.1918, brier_ci=(0.184, 0.200), lift=1.40, lift_ci=(0.88, 1.74),
         canary="PASS (swap idx -0.194 [-0.333, -0.045])"),
    dict(label="DeepSeek V4 Flash", config="naked", slug="dsv4flash",
         model_id="deepseek/deepseek-v4-flash",
         brier=0.2052, brier_ci=(0.195, 0.216), lift=1.20, lift_ci=(0.63, 1.83),
         canary="PASS (swap idx +0.033 [-0.223, 0.218])"),
    dict(label="DeepSeek V4 Flash", config="+ harness", slug="dsv4flash",
         model_id="deepseek/deepseek-v4-flash",
         brier=0.1930, brier_ci=(0.185, 0.201), lift=1.33, lift_ci=(0.68, 1.85),
         canary="PASS (swap idx +0.033 [-0.223, 0.218])"),
    dict(label="Qwen3.5-35B-A3B", config="naked", slug="qwen35b",
         model_id="qwen/qwen3.5-35b-a3b",
         brier=0.2122, brier_ci=(0.203, 0.222), lift=0.87, lift_ci=(0.58, 1.30),
         canary="PASS (swap idx -0.039 [-0.168, 0.127])"),
    dict(label="Qwen3.5-35B-A3B", config="+ harness", slug="qwen35b",
         model_id="qwen/qwen3.5-35b-a3b",
         brier=0.1931, brier_ci=(0.186, 0.200), lift=1.27, lift_ci=(0.64, 1.69),
         canary="PASS (swap idx -0.039 [-0.168, 0.127])"),
    dict(label="Qwen3.5-9B (ours, untrained)", config="naked", slug="qwen9b",
         model_id="Qwen/Qwen3.5-9B",
         brier=0.1992, brier_ci=(0.191, 0.210), lift=1.21, lift_ci=(0.93, 1.48),
         canary="PASS (swap idx -0.145 [-0.390, 0.081])", ours=True),
    dict(label="Qwen3.5-9B (ours, untrained)", config="+ harness", slug="qwen9b",
         model_id="Qwen/Qwen3.5-9B",
         brier=0.1949, brier_ci=(0.186, 0.205), lift=1.01, lift_ci=(0.67, 1.29),
         canary="PASS (swap idx -0.145 [-0.390, 0.081])", ours=True),
    dict(label="GBDT walk-forward", config="boring-ML bar", slug="gbdt",
         model_id=None,
         brier=None, brier_ci=None, lift=1.02, lift_ci=(0.77, 1.30),
         canary="n/a (code)",
         note="full_slate Brier 0.1270 [0.103, 0.151]; no bust/over_under "
              "judgments, so no pooled Brier"),
    dict(label="base rate (no-peek)", config="flat baseline", slug="base_rate",
         model_id=None,
         brier=0.1922, brier_ci=(0.184, 0.203), lift=1.13, lift_ci=(0.96, 1.26),
         canary="n/a (code)"),
    dict(label="coin p=0.5", config="flat baseline", slug="coin",
         model_id=None,
         brier=0.2500, brier_ci=(0.250, 0.250), lift=1.13, lift_ci=(0.96, 1.26),
         canary="n/a (code)"),
]

# Scored judgments per model across the runs its recorded cost covers.
# Frontier models: naked + harness answered counts (frontier_battery.md);
# their recorded cost additionally covers the canary tier (+ DraftGym for
# Fable) — footnoted, not silently dropped. Qwen9B: the all-format expanded
# battery (5,030 judgments x naked+harness at $0.420, docs/budget-ledger.md).
JUDGMENTS = {"fable5": 1848 + 1846, "dsv4flash": 1849 + 1849,
             "qwen35b": 1849 + 1849, "qwen9b": 5030 + 5030}

# Spend not covered by frontier_spend.json (docs/budget-ledger.md).
EXTRA_SPEND = {"qwen9b": {"cost": 0.420, "requests": 162,
                          "source": "docs/budget-ledger.md (Prime token counts)"},
               "gbdt": {"cost": 0.0, "requests": 0, "source": "local compute"},
               "base_rate": {"cost": 0.0, "requests": 0, "source": "local compute"},
               "coin": {"cost": 0.0, "requests": 0, "source": "local compute"}}

COST_FOOTNOTE = {
    "fable5": "cost basis includes canary tier + 4 DraftGym episodes",
    "dsv4flash": "cost basis includes canary tier",
    "qwen35b": "cost basis includes canary tier",
    "qwen9b": "all-format expanded battery, 3 scoring formats",
}

# Flat reference lines for the training-progress charts (metric -> label -> y).
# Only metrics a reference was actually measured on appear; GBDT has no pooled
# Brier / DraftGym / League Index reading and market has no Brier reading.
TRAINING_REFERENCES = {
    "pooled_brier": {
        "Fable 5 + harness": 0.1918,
        "base rate (no-peek)": 0.1922,
    },
    "draftgym_masked_reward": {
        "Fable 5 masked (n=2)": 135.7,       # (-108.4 + 379.8) / 2
        "market (autopick ADP)": 0.0,        # control is 0 pts vs itself
    },
    "league_index": {
        "market (autopick ADP)": 87.0,   # control grid, normalized_rescore.md
        "Fable 5 masked (n=2)": 79.0,    # (71.5 + 86.4) / 2
    },
}

METRIC_META = {
    "pooled_brier": dict(title="Pooled Brier", better="lower", digits=4),
    "draftgym_masked_reward": dict(title="DraftGym masked reward (pts)",
                                   better="higher", digits=1),
    "league_index": dict(title="League Index", better="higher", digits=1),
}

CHECKPOINT_REQUIRED = ("run_id", "phase", "step", "tokens_trained",
                       "cost_cumulative", "metrics", "timestamp")
PHASES = ("B2", "B3", "B4")


# ---------------------------------------------------------------------------
# data loading


def load_spend() -> dict[str, dict]:
    spend = {}
    if SPEND_PATH.exists():
        snap = json.loads(SPEND_PATH.read_text())
        for slug, m in snap.get("by_model", {}).items():
            spend[slug] = {"cost": m["cost"], "requests": m["requests"],
                           "source": "OpenRouter usage.cost (frontier_spend.json)"}
    for slug, m in EXTRA_SPEND.items():
        spend.setdefault(slug, m)
    return spend


def cost_per_1k(slug: str, spend: dict[str, dict]) -> float | None:
    s = spend.get(slug)
    n = JUDGMENTS.get(slug)
    if s is None or not n:
        return 0.0 if s and s["cost"] == 0.0 else None
    return s["cost"] / n * 1000


def cost_per_request(slug: str, spend: dict[str, dict]) -> float | None:
    s = spend.get(slug)
    if s is None or not s["requests"]:
        return None
    return s["cost"] / s["requests"]


def latency_cell(model_id: str | None, tele_summary: dict[str, dict]) -> str:
    if model_id is None:
        return "n/a (local compute)"
    m = tele_summary.get(model_id)
    if m is None:
        return "n/a (pre-instrumentation)"
    return f"{m['latency_mean_s']:.1f} s mean ({m['n']} req)"


def validate_checkpoint_row(row: dict) -> list[str]:
    """Schema check for one training/checkpoints_log.jsonl row ([] = valid)."""
    problems = [f"missing field: {f}" for f in CHECKPOINT_REQUIRED if f not in row]
    if problems:
        return problems
    if not isinstance(row["run_id"], str) or not row["run_id"]:
        problems.append("run_id must be a non-empty string")
    if row["phase"] not in PHASES:
        problems.append(f"phase must be one of {PHASES}")
    if not isinstance(row["step"], int) or row["step"] < 0:
        problems.append("step must be a non-negative int")
    if not isinstance(row["tokens_trained"], int) or row["tokens_trained"] < 0:
        problems.append("tokens_trained must be a non-negative int")
    if not isinstance(row["cost_cumulative"], (int, float)) or row["cost_cumulative"] < 0:
        problems.append("cost_cumulative must be a non-negative number")
    if not isinstance(row["metrics"], dict):
        problems.append("metrics must be a dict")
    else:
        for k, v in row["metrics"].items():
            if v is not None and not isinstance(v, (int, float)):
                problems.append(f"metrics.{k} must be a number or null")
    return problems


def load_checkpoints(path: Path | None = None) -> list[dict]:
    target = path or CHECKPOINTS_PATH
    if not target.exists():
        return []
    rows = []
    for i, line in enumerate(target.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        problems = validate_checkpoint_row(row)
        if problems:
            raise ValueError(f"checkpoints_log.jsonl line {i + 1}: {problems}")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# model_comparison.md


def fmt_ci(ci, digits=3):
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else "—"


def fmt_cost(c: float | None) -> str:
    if c is None:
        return "—"
    if c == 0.0:
        return "~$0"
    return f"${c:,.3f}" if c < 10 else f"${c:,.2f}"


def _overlap(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def quality_vs_cost_reading(spend: dict[str, dict]) -> str:
    """Data-checked reading; only claims what the CIs and costs support."""
    fable = next(r for r in LEADERBOARD
                 if r["slug"] == "fable5" and r["config"] == "+ harness")
    flash = next(r for r in LEADERBOARD
                 if r["slug"] == "dsv4flash" and r["config"] == "+ harness")
    c_fable = cost_per_1k("fable5", spend)
    c_flash = cost_per_1k("dsv4flash", spend)
    lines = []
    if _overlap(fable["brier_ci"], flash["brier_ci"]) and c_fable and c_flash:
        ratio = c_fable / c_flash
        lines.append(
            f"**DeepSeek V4 Flash delivers frontier-tied calibration at roughly "
            f"1/{ratio:.0f} of Fable's cost per judgment** — harness pooled "
            f"Brier {flash['brier']:.4f} {fmt_ci(flash['brier_ci'])} vs Fable's "
            f"{fable['brier']:.4f} {fmt_ci(fable['brier_ci'])} (CIs overlap: "
            f"statistically tied) at {fmt_cost(c_flash)} vs {fmt_cost(c_fable)} "
            "per 1,000 judgments. (Fable's cost basis also carries its DraftGym "
            "episodes, so the true battery-only ratio is somewhat smaller but "
            "stays around two orders of magnitude.)")
    base = next(r for r in LEADERBOARD if r["slug"] == "base_rate")
    best = min((r for r in LEADERBOARD if r["brier"] is not None
                and r["slug"] not in ("base_rate", "coin")),
               key=lambda r: r["brier"])
    if _overlap(best["brier_ci"], base["brier_ci"]):
        lines.append(
            f"The quality axis is compressed: the best model row "
            f"({best['label']} {best['config']}, {best['brier']:.4f}) is "
            f"statistically tied with the free no-peek base rate "
            f"({base['brier']:.4f}) — on this anonymized track, paying more "
            "buys latency and cost, not calibration. Cost is currently the "
            "only axis that separates the field.")
    return "\n\n".join(lines)


def build_markdown(spend: dict[str, dict], tele_summary: dict[str, dict],
                   now: str) -> str:
    lines = [
        "# Model comparison — quality x cost x latency",
        "",
        f"Generated {now} by evals/report_dashboard.py. Quality rows: "
        "anonymized ppr track 2015-2024, 1,849 judgments/run "
        "(evals/results/frontier_battery.md; Qwen3.5-9B cost basis is its "
        "all-format 5,030-judgment battery). Costs: exact API-reported spend "
        "(evals/results/answers/frontier_spend.json + docs/budget-ledger.md). "
        "Latency: data/processed/telemetry/eval_requests.jsonl "
        "(evals/telemetry.py) — pre-instrumentation runs have no recoverable "
        "per-request timing and are marked n/a, not estimated. Partial "
        "pre-restriction rows (GPT-5.6-Luna-Pro, Gemini 3.1 Pro, DeepSeek V4 "
        "Pro, Qwen3.5-397B, GLM-5.2) are excluded, as in the battery verdicts.",
        "",
        "| model | config | pooled Brier [95% CI] | top-10 lift [95% CI] | "
        "canary verdict | cost / 1,000 judgments | cost / request | "
        "latency / request |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in LEADERBOARD:
        c1k = cost_per_1k(r["slug"], spend)
        creq = cost_per_request(r["slug"], spend)
        note = COST_FOOTNOTE.get(r["slug"])
        cost_cell = fmt_cost(c1k) + (f" ({note})" if note and c1k else "")
        brier_cell = (f"{r['brier']:.4f} {fmt_ci(r['brier_ci'])}"
                      if r["brier"] is not None else
                      "— (" + r.get("note", "n/a") + ")")
        lines.append(
            f"| {r['label']} | {r['config']} | {brier_cell} | "
            f"{r['lift']:.2f}x {fmt_ci(r['lift_ci'], 2)} | {r['canary']} | "
            f"{cost_cell} | {fmt_cost(creq)} | "
            f"{latency_cell(r['model_id'], tele_summary)} |")
    lines += [
        "",
        "Cost / 1,000 judgments = the model's total recorded API spend divided "
        "by its scored naked+harness judgments; per-model spend also covers "
        "the canary tier (and Fable's 4 DraftGym episodes), so LLM rows are "
        "slight overestimates of pure battery cost. A model's naked and "
        "harness rows share one cost basis. Cost / request = spend / API "
        "requests (a request answers a whole question batch — one "
        "(season, family) slate of ~40-90 questions, or one DraftGym pick).",
        "",
        "## Quality-vs-cost reading",
        "",
        quality_vs_cost_reading(spend),
        "",
        "## Keeping this current",
        "",
        "- New battery runs record latency/cost per request automatically via "
        "evals/telemetry.py (wired into run_frontier_battery.py and "
        "run_expanded_qwen.py).",
        "- After a battery run, transcribe any new/changed leaderboard rows "
        "from frontier_battery.md into LEADERBOARD in "
        "evals/report_dashboard.py, then `python evals/report_dashboard.py` "
        "to regenerate this file and dashboard.html.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# dashboard.html — inline SVG, DESIGN.md tokens, no external assets


def _log_x(cost: float, x0: float, x1: float, lo=0.01, hi=10.0) -> float:
    c = min(max(cost, lo), hi)
    return x0 + (math.log10(c) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)


def scatter_svg(spend: dict[str, dict]) -> str:
    """Quality-vs-cost scatter: log-x cost per 1k judgments, y pooled Brier
    (axis inverted: better calibration sits higher). Base-rate par band."""
    W, H = 760, 430
    x0, x1, y0, y1 = 70, 700, 30, 360  # plot box (y0 top)
    b_lo, b_hi = 0.186, 0.216          # brier domain (low = top)

    def ybrier(b: float) -> float:
        return y0 + (b - b_lo) / (b_hi - b_lo) * (y1 - y0)

    base = next(r for r in LEADERBOARD if r["slug"] == "base_rate")
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             'aria-label="Quality versus cost scatter">']
    # base-rate par band (CI) + center line (clamped to the plot box)
    band_top = max(ybrier(base["brier_ci"][0]), y0)
    band_bot = min(ybrier(base["brier_ci"][1]), y1)
    parts.append(f'<rect x="{x0}" y="{band_top:.1f}" width="{x1 - x0}" '
                 f'height="{band_bot - band_top:.1f}" class="parband"/>')
    parts.append(f'<line x1="{x0}" x2="{x1}" y1="{ybrier(base["brier"]):.1f}" '
                 f'y2="{ybrier(base["brier"]):.1f}" class="refline"/>')
    parts.append(f'<text x="{x1 - 8}" y="{band_top + 14:.1f}" class="reflabel" '
                 'text-anchor="end">base-rate par (no-peek, free) '
                 f'{base["brier"]:.4f} [CI band]</text>')
    # gridlines + axes
    for b in (0.19, 0.20, 0.21):
        y = ybrier(b)
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="tick" '
                     f'text-anchor="end">{b:.2f}</text>')
    for c in (0.01, 0.1, 1, 10):
        x = _log_x(c, x0, x1)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}" class="grid"/>')
        lbl = f"${c:g}"
        parts.append(f'<text x="{x:.1f}" y="{y1 + 18}" class="tick" '
                     f'text-anchor="middle">{lbl}</text>')
    parts.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 14}" class="axis" '
                 'text-anchor="middle">cost per 1,000 judgments (log scale)</text>')
    parts.append(f'<text x="18" y="{(y0 + y1) / 2:.0f}" class="axis" text-anchor="middle" '
                 f'transform="rotate(-90 18 {(y0 + y1) / 2:.0f})">pooled Brier '
                 '(lower = better, plotted upward)</text>')
    # points, with estimated-width label collision avoidance (labels shift up
    # in 13px rows until they overlap nothing already placed); right-half
    # points get end-anchored labels so text stays inside the plot
    placed: list[tuple[float, float, float]] = []  # (x_start, x_end, baseline_y)

    def place_label(x: float, y: float, text: str, anchor: str) -> float:
        w = 6.3 * len(text)
        xs_, xe = (x - w, x) if anchor == "end" else (x, x + w)
        yy = y
        for _ in range(12):
            if any(abs(yy - py) < 12 and xs_ < pe and pxs < xe
                   for pxs, pe, py in placed):
                yy -= 13
            else:
                break
        placed.append((xs_, xe, yy))
        return yy

    for r in LEADERBOARD:
        if r["brier"] is None or r["slug"] in ("base_rate", "coin", "gbdt"):
            continue
        c1k = cost_per_1k(r["slug"], spend)
        if c1k is None:
            continue
        x, y = _log_x(c1k, x0, x1), ybrier(r["brier"])
        cls = "pt ours" if r.get("ours") else "pt"
        fill = 'class="%s"' % cls if r["config"] == "+ harness" else \
               'class="%s hollow"' % cls
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" {fill}/>')
        text = f'{r["label"]} {r["config"]}'
        if x > (x0 + x1) / 2:
            ly = place_label(x - 11, y + 4, text, "end")
            parts.append(f'<text x="{x - 11:.1f}" y="{ly:.1f}" '
                         f'class="ptlabel" text-anchor="end">{text}</text>')
        else:
            ly = place_label(x + 11, y + 4, text, "start")
            parts.append(f'<text x="{x + 11:.1f}" y="{ly:.1f}" class="ptlabel">'
                         f'{text}</text>')
    # legend
    parts.append(f'<circle cx="{x0 + 8}" cy="{y0 - 12}" r="5" class="pt"/>'
                 f'<text x="{x0 + 18}" y="{y0 - 8}" class="ptlabel">+ harness</text>'
                 f'<circle cx="{x0 + 108}" cy="{y0 - 12}" r="5" class="pt hollow"/>'
                 f'<text x="{x0 + 118}" y="{y0 - 8}" class="ptlabel">naked</text>'
                 f'<circle cx="{x0 + 188}" cy="{y0 - 12}" r="5" class="pt ours"/>'
                 f'<text x="{x0 + 198}" y="{y0 - 8}" class="ptlabel">our base model</text>')
    parts.append("</svg>")
    return "".join(parts)


def training_svg(rows: list[dict], metric: str) -> str:
    """One training-progress small multiple: metric vs step, one line per
    run_id, flat dashed reference lines."""
    meta = METRIC_META[metric]
    refs = TRAINING_REFERENCES.get(metric, {})
    W, H = 760, 300
    x0, x1, y0, y1 = 70, 560, 26, 240
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        if r["metrics"].get(metric) is not None:
            by_run.setdefault(r["run_id"], []).append(r)
    values = [r["metrics"][metric] for rs in by_run.values() for r in rs]
    values += list(refs.values())
    if not values:
        return ""
    vlo, vhi = min(values), max(values)
    pad = (vhi - vlo) * 0.25 or abs(vhi) * 0.1 or 0.01
    vlo, vhi = vlo - pad, vhi + pad
    max_step = max([r["step"] for rs in by_run.values() for r in rs] + [0])
    # before any trained checkpoint exists, show a real step scale so the
    # empty run-space ahead is visible rather than a degenerate 0..1 axis
    x_hi = max(max_step * 1.15, 1000)

    invert = meta["better"] == "lower"

    def yv(v: float) -> float:
        frac = (v - vlo) / (vhi - vlo)
        if not invert:
            frac = 1 - frac
        return y0 + frac * (y1 - y0)

    def xs(step: int) -> float:
        return x0 + step / x_hi * (x1 - x0)

    d = meta["digits"]
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{meta["title"]} '
             'over training">']
    parts.append(f'<text x="{x0}" y="16" class="charttitle">{meta["title"]} '
                 f'<tspan class="tick">({meta["better"]} is better)</tspan></text>')
    for frac in (0.0, 0.5, 1.0):
        v = vlo + frac * (vhi - vlo)
        y = yv(v)
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="tick" '
                     f'text-anchor="end">{v:.{d}f}</text>')
    for frac in (0, 0.5, 1.0):
        step = int(x_hi * frac)
        x = xs(step)
        parts.append(f'<text x="{x:.1f}" y="{y1 + 18}" class="tick" '
                     f'text-anchor="middle">{step:,}</text>')
    parts.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 10}" class="axis" '
                 'text-anchor="middle">training step</text>')
    # reference lines; labels stack downward when two references nearly
    # coincide (e.g. Fable+harness 0.1918 vs base rate 0.1922)
    ref_pts = sorted(((yv(v), label, v) for label, v in refs.items()))
    prev_ly = -1e9
    for y, label, v in ref_pts:
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" class="refline"/>')
        ly = max(y + 4, prev_ly + 12)
        prev_ly = ly
        parts.append(f'<text x="{x1 + 8}" y="{ly:.1f}" class="reflabel">'
                     f'{label} {v:.{d}f}</text>')
    for run_id, rs in sorted(by_run.items()):
        rs = sorted(rs, key=lambda r: r["step"])
        pts = [(xs(r["step"]), yv(r["metrics"][metric])) for r in rs]
        if len(pts) > 1:
            path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<path d="{path}" class="runline"/>')
        for (x, y), r in zip(pts, rs):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="pt ours"/>')
        lx, ly = pts[-1]
        parts.append(f'<text x="{lx + 10:.1f}" y="{ly - 8:.1f}" class="ptlabel">'
                     f'{run_id} (step {rs[-1]["step"]:,})</text>')
    parts.append("</svg>")
    return "".join(parts)


def leaderboard_table_html(spend: dict[str, dict],
                           tele_summary: dict[str, dict]) -> str:
    rows = []
    for r in LEADERBOARD:
        c1k = cost_per_1k(r["slug"], spend)
        brier = (f"{r['brier']:.4f} <span class='ci'>{fmt_ci(r['brier_ci'])}</span>"
                 if r["brier"] is not None else "—")
        ours = " class='oursrow'" if r.get("ours") else ""
        canary = r["canary"].replace(
            " (swap idx ", " <span class='ci'>").replace(")", "</span>") \
            if r["canary"].startswith("PASS") else r["canary"]
        rows.append(
            f"<tr{ours}><td>{r['label']}</td><td>{r['config']}</td>"
            f"<td class='num'>{brier}</td>"
            f"<td class='num'>{r['lift']:.2f}x <span class='ci'>{fmt_ci(r['lift_ci'], 2)}</span></td>"
            f"<td class='num'>{canary}</td>"
            f"<td class='num'>{fmt_cost(c1k)}</td>"
            f"<td class='num'>{latency_cell(r['model_id'], tele_summary)}</td></tr>")
    return ("<div class='tablewrap'><table><thead><tr>"
            "<th>model</th><th>config</th><th>pooled Brier [95% CI]</th>"
            "<th>top-10 lift [95% CI]</th><th>canary</th>"
            "<th>cost / 1k judgments</th><th>latency / request</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


CSS = """
:root {
  --bg: #0D0D0D; --fg: #FAFAFA; --fg-faint: #8A8A8A;
  --surface: #1A1A1A; --border: #2A2A2A;
  --accent: #10B981; --gain: #34D399; --loss: #F87171; --warn: #FBBF24;
  --pt: #A3A3A3;
}
@media (prefers-color-scheme: light) {
  :root { --bg: #FAFAFA; --fg: #111111; --fg-faint: #6E6E6E;
          --surface: #FFFFFF; --border: #E2E2E2; --pt: #5A5A5A; }
}
:root[data-theme="dark"] { --bg: #0D0D0D; --fg: #FAFAFA; --fg-faint: #8A8A8A;
  --surface: #1A1A1A; --border: #2A2A2A; --pt: #A3A3A3; }
:root[data-theme="light"] { --bg: #FAFAFA; --fg: #111111; --fg-faint: #6E6E6E;
  --surface: #FFFFFF; --border: #E2E2E2; --pt: #5A5A5A; }
* { box-sizing: border-box; margin: 0; }
body { background: var(--bg); color: var(--fg);
  font: 15px/1.55 "Inter", system-ui, -apple-system, sans-serif;
  padding: 32px 24px 64px; max-width: 980px; margin: 0 auto; }
h1, h2 { font-family: "Space Grotesk", "Inter", system-ui, sans-serif;
  font-weight: 700; letter-spacing: -0.02em; }
h1 { font-size: 26px; margin-bottom: 4px; }
h2 { font-size: 19px; margin: 40px 0 8px; }
.stamp { font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12px; color: var(--fg-faint); margin-bottom: 28px; }
.sub { color: var(--fg-faint); font-size: 13.5px; margin-bottom: 14px;
  max-width: 72ch; }
.card { background: var(--surface); border-radius: 8px;
  padding: 18px 16px; margin-bottom: 18px; }
svg { width: 100%; height: auto; display: block; }
.grid { stroke: var(--border); stroke-width: 1; }
.tick { font: 11px "IBM Plex Mono", ui-monospace, monospace;
  fill: var(--fg-faint); font-variant-numeric: tabular-nums; }
.axis { font: 12px "Inter", system-ui, sans-serif; fill: var(--fg-faint); }
.charttitle { font: 700 13px "Space Grotesk", "Inter", sans-serif;
  fill: var(--fg); }
.pt { fill: var(--pt); }
.pt.hollow { fill: none; stroke: var(--pt); stroke-width: 2; }
.pt.ours { fill: var(--accent); }
.pt.ours.hollow { fill: none; stroke: var(--accent); stroke-width: 2; }
.ptlabel { font: 11px "Inter", system-ui, sans-serif; fill: var(--fg); }
.parband { fill: var(--accent); opacity: 0.08; }
.refline { stroke: var(--fg-faint); stroke-width: 1;
  stroke-dasharray: 5 4; }
.reflabel { font: 10.5px "IBM Plex Mono", ui-monospace, monospace;
  fill: var(--fg-faint); }
.runline { stroke: var(--accent); stroke-width: 2; fill: none; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; min-width: 860px;
  font-size: 13.5px; }
th { text-align: left; font-weight: 600; color: var(--fg-faint);
  padding: 8px 10px; border-bottom: 1px solid var(--border);
  white-space: nowrap; }
td { padding: 7px 10px; border-bottom: 1px solid var(--border);
  vertical-align: top; }
td.num { font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12.5px; font-variant-numeric: tabular-nums;
  white-space: nowrap; }
.ci { color: var(--fg-faint); }
.oursrow td { background: rgba(16, 185, 129, 0.06); }
.foot { font-size: 12.5px; color: var(--fg-faint); margin-top: 10px;
  max-width: 78ch; }
"""


def build_dashboard(spend: dict[str, dict], tele_summary: dict[str, dict],
                    checkpoints: list[dict], now: str) -> str:
    training_charts = "".join(
        f"<div class='card'>{svg}</div>"
        for metric in METRIC_META
        if (svg := training_svg(checkpoints, metric)))
    n_ckpt = len(checkpoints)
    runs = sorted({r["run_id"] for r in checkpoints})
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fantasy Alpha — benchmark dashboard</title>
<style>{CSS}</style>
</head>
<body>
<h1>Fantasy Alpha — benchmark dashboard</h1>
<p class="stamp">data as of {now} · anonymized ppr track 2015-2024 · 2025 holdout untouched</p>

<h2>Quality vs cost</h2>
<p class="sub">Every complete run from the frontier battery, on the two axes that
matter for shipping: calibration (pooled Brier, 95% CI bootstrap over seasons)
against exact API cost per 1,000 judgments. The shaded band is the no-peek
base-rate par — free, and statistically tied with everything above it.</p>
<div class="card">{scatter_svg(spend)}</div>
{leaderboard_table_html(spend, tele_summary)}
<p class="foot">Cost basis = each model's total recorded battery spend
(naked + canary + harness tiers; Fable additionally includes 4 DraftGym
episodes) over its scored naked+harness judgments. GBDT (full_slate Brier
0.1270 [0.103, 0.151], ~$0) has no pooled Brier and is omitted from the
scatter. Latency for pre-instrumentation runs is not recoverable and is
never estimated. Sources: frontier_battery.md, frontier_spend.json,
budget-ledger.md.</p>

<h2>Training progress</h2>
<p class="sub">Our checkpoints (training/checkpoints_log.jsonl — {n_ckpt} row{"s" if n_ckpt != 1 else ""},
run{"s" if len(runs) != 1 else ""}: {", ".join(runs) if runs else "none"}) against the flat
reference bars they must beat. Step 0 = the untrained Qwen3.5-9B proxy points;
T1/T2 checkpoint evals append rows and this chart re-renders.</p>
{training_charts}
<p class="foot">References are measured values, not aspirations: Fable 5 +
harness and the no-peek base rate from frontier_battery.md; DraftGym masked
means (n=2 paired episodes) and League Index / market (autopick ADP) rows from
normalized_rescore.md. Metrics a reference was never measured on show no line
(GBDT has no pooled Brier, DraftGym, or League Index reading; the market has
no Brier reading).</p>
</body>
</html>
"""


# ---------------------------------------------------------------------------


def generate(out_md: Path | None = None, out_html: Path | None = None,
             checkpoints_path: Path | None = None,
             telemetry_path: Path | None = None) -> tuple[Path, Path]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    spend = load_spend()
    tele_summary = summarize(load_telemetry(telemetry_path or TELEMETRY_PATH))
    checkpoints = load_checkpoints(checkpoints_path)
    md_path = out_md or COMPARISON_MD
    html_path = out_html or DASHBOARD_HTML
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_markdown(spend, tele_summary, now))
    html_path.write_text(build_dashboard(spend, tele_summary, checkpoints, now))
    return md_path, html_path


def main() -> int:
    md, html = generate()
    print(f"wrote {md}\nwrote {html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
