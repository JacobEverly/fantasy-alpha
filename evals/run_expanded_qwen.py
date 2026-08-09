#!/usr/bin/env python3
"""BreakoutBench v0.3 expanded battery — the two Qwen baselines, re-run.

Naked and harness-grounded `Qwen/Qwen3.5-9B` (Prime Intellect serverless) over
every (season 2015-2024, format, family) question set built by
evals/breakoutbench.py. One request per (season, format, family) carrying all
of that family's questions; strict-JSON parse with one repair retry;
skip-and-record on failure (evals/results/answers/failures_<run>.json).

Harness grounding ports evals/harness_breakout.py's pure-code stage to all
three families: per candidate, historical rates computed STRICTLY from label
seasons before the eval season (asserted):
  - full_slate: cohort (±8 adp_pos_rank), prior-rank cohort (±10 prior finish),
    position base rate — identical math to harness_breakout.
  - bust: cohort (±3 adp_pos_rank — the band is only 1-12) + position base rate.
  - over_under: cohort (±8 adp_pos_rank) + position base rate.

BUDGET: hard cap $3.00 at Prime serverless rates ($0.18/M in, $0.54/M out);
the run aborts before any request that could plausibly cross it. Actuals are
appended to docs/budget-ledger.md.

Resume-safe: an existing answers file skips the request. Final comparison
table (naked / harness / base-rate / coin, Brier ± 95% CI, top-10 lift ± CI,
era rows, CI-overlap verdicts) -> evals/results/expanded_baseline.md.
"""
from __future__ import annotations

import json
import re
import ssl
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.telemetry import record as telemetry_record  # noqa: E402

from evals.breakoutbench import (  # noqa: E402
    ANSWERS_DIR,
    FAMILIES,
    QUESTIONS_DIR,
    SEASONS,
    TOP_K,
    _breakout_rows,
    _season_points,
    family_of,
    outcome_for,
    score_run,
    season_formats,
    write_baseline_answers,
)

API = "https://api.pinference.ai/api/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-9B"
PRICE_IN, PRICE_OUT = 0.18 / 1e6, 0.54 / 1e6  # $/token, Prime serverless
BUDGET_CAP = 3.00
MAX_REQUEST_COST = 0.02  # conservative per-request ceiling used for the pre-flight check
COHORT_WINDOW = {"full_slate": 8, "over_under": 8, "bust": 3}

_ctx = ssl.create_default_context()
if not _ctx.cert_store_stats().get("x509_ca"):
    _ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")


def api_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("PRIME_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no PRIME_API_KEY")


def chat(messages: list[dict], max_tokens: int = 4000, task: str = "") -> tuple[str, dict]:
    t0 = time.monotonic()
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
                 # bill the team account, not the (empty) personal balance
                 "X-Prime-Team-ID": "cmskuc3x7018j7g10tr01gukl",
                 # the API's WAF rejects urllib's default Python-urllib UA
                 "User-Agent": "fantasy-alpha-evals/0.1"},
    )
    with urllib.request.urlopen(req, timeout=180, context=_ctx) as resp:
        out = json.loads(resp.read())
    usage = out.get("usage", {})
    telemetry_record(
        runner="run_expanded_qwen", model=MODEL, task_family=task,
        latency_s=time.monotonic() - t0,
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("completion_tokens", 0),
        cost_usd=(usage.get("prompt_tokens", 0) * PRICE_IN
                  + usage.get("completion_tokens", 0) * PRICE_OUT),
        cost_source="computed")
    return out["choices"][0]["message"]["content"], usage


def parse_answers(text: str, valid_ids: set[str]) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, flags=re.S)
    raw = json.loads(m.group(0) if m else text)
    assert isinstance(raw, list) and raw, f"bad answers shape: {type(raw)}"
    answers, seen = [], set()
    for a in raw:
        qid = str(a["question_id"])
        if qid not in valid_ids or qid in seen:
            continue
        p = float(a["p"])
        answers.append({"question_id": qid, "p": min(max(p, 0.0), 1.0)})
        seen.add(qid)
    coverage = len(answers) / len(valid_ids)
    assert coverage >= 0.5, f"only {coverage:.0%} of questions answered"
    return answers


# ---------------- grounding stage (pure code, all three families) ----------


def _family_rows(preset: str) -> dict[str, list[dict]]:
    """family -> label rows [{season, position, adp_pos_rank, player_id, y}]."""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in _breakout_rows():
        if r["format"] != preset:
            continue
        for family in family_of(r["position"], r["adp_pos_rank"], r["finish_pos_rank"]):
            out[family].append({
                "season": r["season"], "position": r["position"],
                "adp_pos_rank": r["adp_pos_rank"], "player_id": r["player_id"],
                "y": outcome_for(family, r),
            })
    return dict(out)


def _prior_ranks(preset: str) -> dict[tuple[str, int], int]:
    return {(pid, y): int(row[f"pos_season_rank_{preset}"])
            for pid, seasons in _season_points().items()
            for y, row in seasons.items()}


def _rate(subset: list[dict]) -> tuple[float | None, int]:
    n = len(subset)
    return (round(sum(r["y"] for r in subset) / n, 3) if n else None), n


def grounding_for(family: str, packet: dict, rows: list[dict],
                  prior_ranks: dict[tuple[str, int], int], eval_season: int) -> dict:
    assert all(r["season"] < eval_season for r in rows), "leakage: future season in grounding rows"
    at_pos = [r for r in rows if r["position"] == packet["position"]]
    window = COHORT_WINDOW[family]
    cohort = [r for r in at_pos if abs(r["adp_pos_rank"] - packet["adp_pos_rank"]) <= window]
    cohort_rate, cohort_n = _rate(cohort)
    base_rate, base_n = _rate(at_pos)
    g = {"cohort_rate": cohort_rate, "cohort_n": cohort_n,
         "position_base_rate": base_rate, "position_base_n": base_n}
    if family == "full_slate":
        prior = packet.get("prior_season")
        if prior:
            prior_cohort = [r for r in at_pos
                            if (pr := prior_ranks.get((r["player_id"], r["season"] - 1))) is not None
                            and abs(pr - prior["pos_rank"]) <= 10]
            g["prior_rank_cohort_rate"], g["prior_rank_cohort_n"] = _rate(prior_cohort)
        else:
            g["prior_rank_cohort_rate"], g["prior_rank_cohort_n"] = None, 0
    return g


# ---------------- prompts ----------------

SYSTEM = "You are a fantasy football analyst. Answer with strict JSON only."

HARNESS_EXTRA = (
    "The historical_grounding fields are real base rates computed from past seasons — "
    "anchor your probabilities to them and deviate only with a stated structural reason. "
    "Well-calibrated probabilities beat confident ones."
)

ENRICHMENT_NOTE = (
    "Each candidate's `enrichment` block adds real pre-season evidence: prior-season "
    "usage (opportunities per game, target share, year-over-year change), season-end "
    "depth-chart position rank, age, draft capital, and counts of news items about the "
    "player in the 90 days before the season (n_news_90d, with camp-promotion / injury / "
    "hype keyword-flag counts). A null value means UNAVAILABLE, not zero — the *_status "
    "fields say why."
)


def build_prompt(qpayload: dict, family: str, grounded: bool) -> tuple[list[dict], set[str]]:
    questions = [q for q in qpayload["questions"] if q["family"] == family]
    items = [{"question_id": q["question_id"], **q["packet"]} for q in questions]
    task = questions[0]["question"]
    enriched = any("enrichment" in q["packet"] for q in questions)
    content = (
        f"Season and player identities are masked. Format: {qpayload['format']}.\n"
        f"For EVERY candidate below, answer: {task}\n"
        + (ENRICHMENT_NOTE + "\n" if enriched else "")
        + (HARNESS_EXTRA + "\n" if grounded else "")
        + "\nCandidates:\n" + json.dumps(items)
        + '\n\nReturn a JSON array with EXACTLY one object per candidate: '
          '{"question_id": str, "p": float 0-1}. Cover every question_id once. No prose.'
    )
    return ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            {q["question_id"] for q in questions})


# ---------------- run loop ----------------


def run_battery(run_name: str, grounded: bool, spend: dict,
                suffix: str = "", cells: list[tuple[int, str]] | None = None) -> None:
    """`suffix` selects the question-file variant (e.g. "_v2" for enriched
    packets); `cells` restricts the (season, preset) grid (default: all)."""
    failures_path = ANSWERS_DIR / f"failures_{run_name}.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    rows_cache: dict[str, dict[str, list[dict]]] = {}
    ranks_cache: dict[str, dict] = {}
    for season, preset in (cells if cells is not None else season_formats(SEASONS)):
        qpayload = json.loads(
            (QUESTIONS_DIR / f"breakoutbench_{season}_{preset}{suffix}.json").read_text())
        for family in FAMILIES:
            out_path = ANSWERS_DIR / f"qwen_{run_name}_{season}_{preset}_{family}.json"
            if out_path.exists():
                continue
            if spend["cost"] + MAX_REQUEST_COST > BUDGET_CAP:
                raise SystemExit(f"BUDGET CAP: ${spend['cost']:.3f} spent, next request "
                                 f"could cross ${BUDGET_CAP:.2f} — aborting")
            payload = dict(qpayload)
            if grounded:
                fam_rows = rows_cache.setdefault(preset, _family_rows(preset))
                pr = ranks_cache.setdefault(preset, _prior_ranks(preset))
                past = [r for r in fam_rows.get(family, []) if r["season"] < season]
                payload = {**qpayload, "questions": [
                    {**q, "packet": {**q["packet"],
                                     "historical_grounding": grounding_for(
                                         family, q["packet"], past, pr, season)}}
                    for q in qpayload["questions"]
                ]}
            messages, valid_ids = build_prompt(payload, family, grounded)
            task = f"{run_name}:{family}"
            try:
                text, usage = chat(messages, task=task)
                try:
                    answers = parse_answers(text, valid_ids)
                except Exception as e:  # one repair retry
                    messages += [{"role": "assistant", "content": text},
                                 {"role": "user", "content": f"Invalid ({e}). Return ONLY the JSON "
                                  "array, one object per candidate, covering every question_id."}]
                    text, usage2 = chat(messages, task=task + ":repair")
                    usage = {k: usage.get(k, 0) + usage2.get(k, 0) for k in set(usage) | set(usage2)}
                    answers = parse_answers(text, valid_ids)
            except Exception as e:  # skip-and-record
                failures.append({"run": run_name, "season": season, "format": preset,
                                 "family": family, "error": str(e)})
                failures_path.write_text(json.dumps(failures, indent=1) + "\n")
                print(f"  FAIL {run_name} {season} {preset} {family}: {e}", flush=True)
                continue
            spend["in"] += usage.get("prompt_tokens", 0)
            spend["out"] += usage.get("completion_tokens", 0)
            spend["cost"] = spend["in"] * PRICE_IN + spend["out"] * PRICE_OUT
            spend["requests"] += 1
            out_path.write_text(json.dumps(answers, indent=1) + "\n")
            print(f"  {run_name} {season} {preset} {family}: {len(answers)}/{len(valid_ids)} "
                  f"answered ({usage.get('prompt_tokens')}in/{usage.get('completion_tokens')}out, "
                  f"total ${spend['cost']:.3f})", flush=True)


# ---------------- report ----------------


def fmt_ci(ci, digits=3):
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else "—"


def overlap(a, b) -> bool:
    return not (a is None or b is None) and a[0] <= b[1] and b[0] <= a[1]


def verdict(name: str, a_label: str, a_ci, b_label: str, b_ci, digits=3) -> str:
    if a_ci is None or b_ci is None:
        return f"- **{name} — {a_label} vs {b_label}: no data.**"
    sig = "NOT significant (CIs overlap)" if overlap(a_ci, b_ci) else "significant (CIs do not overlap)"
    return (f"- **{name} — {a_label} vs {b_label}: {sig}.** "
            f"{a_label} {fmt_ci(a_ci, digits)} vs {b_label} {fmt_ci(b_ci, digits)}")


def write_report(results: dict[str, dict], spend: dict, n_failures: dict[str, int]) -> Path:
    def row(label: str, r: dict) -> str:
        p, ci = r["pooled"], r["ci"]
        cells = [label]
        for fam in FAMILIES:
            v = p["families"][fam]
            cells.append(f"{v['brier']:.4f} {fmt_ci(ci.get(f'brier_{fam}'))}" if v["n"] else "—")
        cells.append(f"{p['pooled_brier']:.4f} {fmt_ci(ci.get('pooled_brier'))}")
        cells.append(f"{p['lift']:.2f}x {fmt_ci(ci.get('lift'), 2)}")
        cells.append(str(r["answered"]))
        return "| " + " | ".join(cells) + " |"

    naked, harness = results["naked"], results["harness"]
    base, coin = results["base_rate"], results["coin"]
    total_q = base["n_total"]
    fam_counts = {f: base["pooled"]["families"][f]["n"] for f in FAMILIES}

    lines = [
        "# BreakoutBench v0.3 — expanded baseline battery",
        "",
        f"Model: `{MODEL}` via Prime Intellect serverless (`api.pinference.ai`), temp 0, "
        "anonymized structure track. Battery: every (season 2015-2024 × format "
        "standard/ppr/half_ppr with an ADP snapshot) × 3 families — "
        f"{total_q} scored judgments per run "
        f"({', '.join(f'{n} {f}' for f, n in fam_counts.items())}), "
        f"~{total_q // 100}x the original 100-pick run. "
        "95% CIs bootstrap over SEASONS (the independent unit), 10,000 resamples, seeded.",
        "",
        "| run | full_slate Brier [95% CI] | bust Brier [95% CI] | over_under Brier [95% CI] "
        "| pooled Brier [95% CI] | top-10 lift [95% CI] | answered |",
        "|---|---|---|---|---|---|---|",
        row("Qwen naked", naked),
        row("Qwen + harness", harness),
        row("base rate (no-peek)", base),
        row("coin (p=0.5)", coin),
        "",
        "Lift rows for the flat baselines are tie-broken by shuffled anon id — i.e. random "
        "selection at the slate base rate; their CIs bracket 1.0x by construction.",
        "",
        "## Per-era split",
        "",
        "| run | era | pooled Brier [95% CI] | top-10 lift [95% CI] | n |",
        "|---|---|---|---|---|",
    ]
    for label, r in (("Qwen naked", naked), ("Qwen + harness", harness),
                     ("base rate", base), ("coin", coin)):
        for era, er in r["eras"].items():
            ep = er["pooled"]
            lines.append(f"| {label} | {era} | {ep['pooled_brier']:.4f} "
                         f"{fmt_ci(er['ci'].get('pooled_brier'))} | {ep['lift']:.2f}x "
                         f"{fmt_ci(er['ci'].get('lift'), 2)} | {ep['n']} |")
    lines += [
        "",
        "## Verdicts (significant iff 95% CIs do not overlap)",
        "",
        verdict("Top-10 selection lift", "naked", naked["ci"].get("lift"),
                "harness", harness["ci"].get("lift"), 2),
        verdict("Top-10 selection lift", "naked", naked["ci"].get("lift"),
                "base rate/random", base["ci"].get("lift"), 2),
        verdict("Top-10 selection lift", "harness", harness["ci"].get("lift"),
                "base rate/random", base["ci"].get("lift"), 2),
        verdict("Pooled Brier", "naked", naked["ci"].get("pooled_brier"),
                "harness", harness["ci"].get("pooled_brier")),
        verdict("Pooled Brier", "naked", naked["ci"].get("pooled_brier"),
                "base rate", base["ci"].get("pooled_brier")),
        verdict("Pooled Brier", "harness", harness["ci"].get("pooled_brier"),
                "base rate", base["ci"].get("pooled_brier")),
        verdict("Pooled Brier", "harness", harness["ci"].get("pooled_brier"),
                "coin", coin["ci"].get("pooled_brier")),
        "",
        f"Failures skipped-and-recorded: naked {n_failures.get('naked', 0)}, "
        f"harness {n_failures.get('harness', 0)} "
        "(see evals/results/answers/failures_*.json).",
        f"This battery's marginal usage: {spend['in']}in/{spend['out']}out tokens across "
        f"{spend['requests']} requests ≈ ${spend['cost']:.3f} "
        f"(cap ${BUDGET_CAP:.2f}).",
        "",
    ]
    out = ROOT / "evals" / "results" / "expanded_baseline.md"
    out.write_text("\n".join(lines))
    return out


def append_ledger(spend: dict, what: str | None = None) -> None:
    ledger = ROOT / "docs" / "budget-ledger.md"
    text = ledger.read_text()
    what = what or ("BreakoutBench v0.3 expanded battery: naked + harness Qwen runs, "
                    f"{spend['requests']} requests over 27 season-formats × 3 families "
                    "(evals/run_expanded_qwen.py)")
    line = (f"| {date.today().isoformat()} | {what} | $0.18/M in, $0.54/M out | "
            f"{spend['in']:,} in / {spend['out']:,} out tok | ${spend['cost']:.3f} (API token counts) |")
    marker = "\n\nTotal to date:"
    old_total = float(re.search(r"Total to date: ~\$([\d.]+)", text).group(1))
    new_total = old_total + spend["cost"]
    text = text.replace(marker, f"\n{line}{marker}", 1)
    text = re.sub(r"Total to date: ~\$[\d.]+", f"Total to date: ~${new_total:.3f}", text)
    ledger.write_text(text)


V2_SEASONS = tuple(s for s in SEASONS if s >= 2016)  # v2 = news-covered seasons


def run_v2(presets: tuple[str, ...] = ("ppr",)) -> dict:
    """Enriched-packet batteries (naked_v2 / harness_v2) over the _v2 slates.

    Resume-safe like the structure battery; returns the spend dict. Called by
    evals/run_enriched_experiment.py (which writes the report)."""
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    cells = [(s, p) for s, p in season_formats(V2_SEASONS) if p in presets]
    spend = {"in": 0, "out": 0, "cost": 0.0, "requests": 0}
    run_battery("naked_v2", grounded=False, spend=spend, suffix="_v2", cells=cells)
    run_battery("harness_v2", grounded=True, spend=spend, suffix="_v2", cells=cells)
    if spend["requests"]:
        append_ledger(spend, "BreakoutBench enriched-packet (v2) battery: naked_v2 + "
                             f"harness_v2 Qwen runs, {spend['requests']} requests over "
                             f"{len(cells)} season-formats × 3 families "
                             "(evals/run_expanded_qwen.py v2)")
    return spend


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "v2":
        spend = run_v2()
        print(f"v2 batteries done: {spend['requests']} requests, ${spend['cost']:.3f}")
        return 0
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    spend = {"in": 0, "out": 0, "cost": 0.0, "requests": 0}
    run_battery("naked", grounded=False, spend=spend)
    run_battery("harness", grounded=True, spend=spend)

    # score everything (baseline answers regenerated for exact parity)
    baselines = write_baseline_answers(SEASONS, ("standard", "ppr", "half_ppr"))
    results, n_failures = {}, {}
    for run in ("naked", "harness"):
        paths = sorted(ANSWERS_DIR.glob(f"qwen_{run}_*_*.json"))
        results[run] = score_run(paths)
        fp = ANSWERS_DIR / f"failures_{run}.json"
        n_failures[run] = len(json.loads(fp.read_text())) if fp.exists() else 0
    results["base_rate"] = score_run(baselines["base_rate"])
    results["coin"] = score_run(baselines["coin"])

    out = write_report(results, spend, n_failures)
    if spend["requests"]:
        append_ledger(spend)
    print(out.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
