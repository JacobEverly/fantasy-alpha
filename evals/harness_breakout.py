#!/usr/bin/env python3
"""BreakoutBench B2-with-harness: same untrained model, two-stage harness.

Stage 1 (pure code, no model): for each anonymized candidate, compute
historical base rates from labels data STRICTLY LIMITED to seasons before
the eval season (hard leakage rule, asserted in code + unit-tested):
  - cohort_rate: past breakout rate at same position within ±8 adp_pos_rank
  - prior_rank_cohort_rate: past breakout rate at same position among
    candidates whose prior-season pos_rank was within ±10 of this one's
  - position_base_rate: overall past breakout rate for the position
All rates restricted to gate-eligible past candidates (the population the
breakout label is defined over). Appended as `historical_grounding`.

Stage 2: same model/prompt as scripts/run_qwen_slates.py, plus instructions
to anchor probabilities to the grounding. Picks written to
evals/demo/qwenharness_picks_<season>_ppr.json, scored via anon_demo.
"""
from __future__ import annotations

import csv
import json
import re
import ssl
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "processed" / "labels"
API = "https://api.pinference.ai/api/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-9B"
SEASONS = range(2015, 2025)
ADP_GATE = {"QB": 18, "TE": 18, "RB": 40, "WR": 40}

_ctx = ssl.create_default_context()
if not _ctx.cert_store_stats().get("x509_ca"):
    _ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")


def key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("PRIME_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no PRIME_API_KEY")


def chat(messages: list[dict]) -> tuple[str, dict]:
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2000,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json",
                 # bill the team account, not the (empty) personal balance
                 "X-Prime-Team-ID": "cmskuc3x7018j7g10tr01gukl",
                 # the API's WAF rejects urllib's default Python-urllib UA
                 "User-Agent": "fantasy-alpha-evals/0.1"},
    )
    with urllib.request.urlopen(req, timeout=180, context=_ctx) as resp:
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"], out.get("usage", {})


def parse_picks(text: str) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, flags=re.S)
    picks = json.loads(m.group(0) if m else text)
    assert isinstance(picks, list) and len(picks) == 10, f"bad picks shape: {len(picks) if isinstance(picks, list) else type(picks)}"
    return [{"anon_id": str(p["anon_id"]), "p_breakout": float(p["p_breakout"])} for p in picks]


# ---------------- grounding stage (pure code) ----------------

def load_breakout_rows(fmt: str = "ppr") -> list[dict]:
    """Gate-eligible candidate rows from breakouts.csv, minimally typed."""
    rows = []
    with open(LABELS / "breakouts.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["format"] != fmt or not r["adp_pos_rank"]:
                continue
            pos, rank = r["position"], int(r["adp_pos_rank"])
            if pos not in ADP_GATE or rank <= ADP_GATE[pos]:
                continue
            rows.append({
                "season": int(r["season"]),
                "player_id": r["player_id"],
                "position": pos,
                "adp_pos_rank": rank,
                "breakout": r["breakout"] == "True",
            })
    return rows


def load_prior_ranks(fmt: str = "ppr") -> dict[tuple[str, int], int]:
    """(player_id, season) -> positional season finish rank."""
    ranks: dict[tuple[str, int], int] = {}
    with open(LABELS / "season_points.csv", newline="") as f:
        for r in csv.DictReader(f):
            ranks[(r["player_id"], int(r["season"]))] = int(r[f"pos_season_rank_{fmt}"])
    return ranks


def past_rows(rows: list[dict], eval_season: int) -> list[dict]:
    """Hard leakage rule: only seasons strictly before the eval season."""
    past = [r for r in rows if r["season"] < eval_season]
    assert all(r["season"] < eval_season for r in past)
    return past


def _rate(subset: list[dict]) -> tuple[float | None, int]:
    n = len(subset)
    return (round(sum(r["breakout"] for r in subset) / n, 3) if n else None), n


def grounding_for(cand: dict, rows: list[dict], prior_ranks: dict[tuple[str, int], int],
                  eval_season: int) -> dict:
    """Compute historical_grounding for one slate candidate from past rows only."""
    assert all(r["season"] < eval_season for r in rows), "leakage: future season in grounding rows"
    pos = cand["position"]
    at_pos = [r for r in rows if r["position"] == pos]

    cohort = [r for r in at_pos if abs(r["adp_pos_rank"] - cand["adp_pos_rank"]) <= 8]
    cohort_rate, cohort_n = _rate(cohort)

    prior = cand.get("prior_season")
    if prior:
        cand_prior_rank = prior["pos_rank"]
        prior_cohort = []
        for r in at_pos:
            pr = prior_ranks.get((r["player_id"], r["season"] - 1))
            if pr is not None and abs(pr - cand_prior_rank) <= 10:
                prior_cohort.append(r)
        prior_rank_cohort_rate, prior_n = _rate(prior_cohort)
    else:
        prior_rank_cohort_rate, prior_n = None, 0

    base_rate, base_n = _rate(at_pos)
    return {
        "cohort_rate": cohort_rate, "cohort_n": cohort_n,
        "prior_rank_cohort_rate": prior_rank_cohort_rate, "prior_rank_cohort_n": prior_n,
        "position_base_rate": base_rate, "position_base_n": base_n,
    }


def ground_slate(slate: dict, season: int, rows: list[dict],
                 prior_ranks: dict[tuple[str, int], int]) -> list[dict]:
    past = past_rows(rows, season)
    grounded = []
    for c in slate["candidates"]:
        c = dict(c)
        c["historical_grounding"] = grounding_for(c, past, prior_ranks, season)
        grounded.append(c)
    return grounded


# ---------------- model stage + scoring ----------------

EXTRA_INSTRUCTIONS = (
    "The historical_grounding fields are real base rates computed from past seasons — "
    "anchor your probabilities to them and deviate only with a stated structural reason. "
    "Do not assign any p_breakout above 0.60 unless the grounding supports it; "
    "well-calibrated probabilities beat confident ones."
)


def main() -> int:
    rows = load_breakout_rows("ppr")
    prior_ranks = load_prior_ranks("ppr")
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    for season in SEASONS:
        picks_path = ROOT / "evals" / "demo" / f"qwenharness_picks_{season}_ppr.json"
        if picks_path.exists():
            continue
        slate = json.loads((ROOT / "evals" / "demo" / f"slate_{season}_ppr.json").read_text())
        grounded = ground_slate(slate, season, rows, prior_ranks)
        messages = [
            {"role": "system", "content": "You are a fantasy football analyst. Answer with strict JSON only."},
            {"role": "user", "content": slate["task"] + "\n\n" + EXTRA_INSTRUCTIONS
             + "\n\nCandidates:\n" + json.dumps(grounded)
             + '\n\nReturn a JSON array of exactly 10 objects: {"anon_id": str, "p_breakout": float 0-1}, most likely first. No prose.'},
        ]
        text, usage = chat(messages)
        try:
            picks = parse_picks(text)
        except Exception as e:
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"Invalid ({e}). Return ONLY the JSON array of exactly 10 objects."}]
            text, usage2 = chat(messages)
            usage = {k: usage.get(k, 0) + usage2.get(k, 0) for k in set(usage) | set(usage2)}
            picks = parse_picks(text)
        picks_path.write_text(json.dumps(picks, indent=1))
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)
        print(f"{season}: harness picks saved ({usage.get('prompt_tokens')}in/{usage.get('completion_tokens')}out)", flush=True)

    # score all seasons
    results, pooled = [], {"hits": 0, "expected": 0.0, "brier_sum": 0.0, "n": 0}
    for season in SEASONS:
        out = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "evals/anon_demo.py", "score", str(season), "ppr",
             f"evals/demo/qwenharness_picks_{season}_ppr.json"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout
        m = re.search(r"true breakouts among them: (\d+) \(base rate ([\d.]+)%; random@10 ≈ ([\d.]+)", out)
        h = re.search(r"model hits@10: (\d+).*Brier\(picks\): ([\d.]+)", out)
        n_break, rand10, hits, brier = int(m.group(1)), float(m.group(3)), int(h.group(1)), float(h.group(2))
        results.append((season, n_break, hits, rand10, brier))
        pooled["hits"] += hits
        pooled["expected"] += rand10
        pooled["brier_sum"] += brier * 10
        pooled["n"] += 10

    lift = pooled["hits"] / pooled["expected"] if pooled["expected"] else 0
    mean_brier = pooled["brier_sum"] / pooled["n"]
    lines = ["# BreakoutBench — untrained open model + quant harness (B2-with-harness)", "",
             f"Model: `{MODEL}` via Prime Intellect serverless (`api.pinference.ai`), temp 0, anonymized structure track.",
             "Harness: pure-code grounding stage appends `historical_grounding` (cohort / prior-rank-cohort / position "
             "base rates, computed strictly from seasons before the eval season) to each candidate; prompt instructs "
             "the model to anchor to those base rates (cap 0.60 without support).",
             "", "| season | breakouts in slate | hits@10 | random@10 | Brier |", "|---|---|---|---|---|"]
    for season, nb, hits, rand10, brier in results:
        lines.append(f"| {season} | {nb} | {hits} | {rand10:.1f} | {brier:.3f} |")
    lines += ["", f"**Pooled: {pooled['hits']} hits / {pooled['expected']:.1f} expected at random "
              f"(lift {lift:.1f}x) · mean Brier {mean_brier:.3f} · {pooled['n']} picks over 10 seasons**",
              "", "## Comparison (pooled, 100 picks)", "",
              "| run | hits | lift vs random | mean Brier |", "|---|---|---|---|",
              "| B2-naked (no harness) | 19 | 1.3x | 0.493 |",
              f"| B2-with-harness | {pooled['hits']} | {lift:.1f}x | {mean_brier:.3f} |",
              f"| random@10 | {pooled['expected']:.1f} | 1.0x | — |",
              "", f"This run's marginal usage: {total_usage['prompt_tokens']}in/{total_usage['completion_tokens']}out tokens."]
    outdir = ROOT / "evals" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "qwen_harness_breakout.md").write_text("\n".join(lines))
    print("\n".join(lines[6:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
