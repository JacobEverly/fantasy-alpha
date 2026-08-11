#!/usr/bin/env python3
"""SFT data pipeline v0 — teacher traces through the harness, then the
survivor-bias-proof filter (docs/training-risk-register.md risk 3).

Pilot: ~152 questions sampled across BreakoutBench full_slate/bust +
CalibBench season_threshold, NAMED track, seasons 2016-2023 ONLY
(2024 = SFT-val, held out of the pilot; 2025 = sealed gate holdout —
asserted in code and unit-tested in tests/test_sft_filter.py).

Per question, the harness (not the model) assembles:
  1. the question packet, re-named from the sequestered KEY;
  2. code-computed historical grounding (cohort base rates strictly from
     seasons before the eval season — same discipline as
     evals/harness_breakout.py, leakage asserted);
  3. up to 3 EvidenceStore lookups as of Sept 1 of the season
     (player_news / injury_status / depth_chart — the time gate lives in
     harness/evidence.py, never in the prompt).
The teacher writes a reasoning trace that cites this context and ends with
one FINAL PROBABILITY line. Temp 0.7, 2 samples/question, HARD budget cap
$8 tracked per call from API-reported usage.

Teacher: qwen/qwen3.5-397b-a17b via Prime Intellect serverless (see
training/README.md for the selection note + docs/budget-ledger.md for cost).

Commands (run from repo root with .venv/bin/python -m training.sft_datagen):
  ping      one tiny sanity call, prints usage + cost
  sample    build + print the pilot question sample (deterministic, no API)
  generate  teacher traces -> data/processed/sft/raw_traces_v0.jsonl
            (resumable: existing trace_ids are skipped; budget-capped)
  filter    run training/filter.py over the raw traces ->
            data/processed/sft/pilot_v0.jsonl + filter_report.md

Stdlib only. Python 3.11+.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import ssl
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.harness_breakout import (  # noqa: E402
    grounding_for, load_breakout_rows, load_prior_ranks, past_rows,
)
from evals.breakoutbench import _breakout_rows  # noqa: E402
from harness.evidence import EvidenceStore  # noqa: E402
from training import filter as trace_filter  # noqa: E402

QUESTIONS_DIR = ROOT / "evals" / "questions"
OUT_DIR = ROOT / "data" / "processed" / "sft"
RAW_TRACES = OUT_DIR / "raw_traces_v0.jsonl"
PILOT_OUT = OUT_DIR / "pilot_v0.jsonl"
REPORT_OUT = OUT_DIR / "filter_report.md"

API = "https://api.pinference.ai/api/v1/chat/completions"
TEACHER_MODEL = "qwen/qwen3.5-397b-a17b"
RATE_IN, RATE_OUT = 0.60, 3.60  # $/Mtok, from /models 2026-08-08
BUDGET_CAP_USD = 8.00
TEMPERATURE = 0.7
SAMPLES_PER_QUESTION = 2
MAX_TOKENS = 700

# Season discipline (unit-tested): pilot draws from 2016-2023 only.
PILOT_SEASONS = tuple(range(2016, 2024))
SFT_VAL_SEASON = 2024   # reserved for SFT validation, never in the pilot
HOLDOUT_SEASON = 2025   # sealed gate holdout (GAMEPLAN §9)
PRESET = "ppr"
# per season: 8 full_slate + 4 bust (BreakoutBench) + 7 season_threshold
# (CalibBench) = 19 -> 152 questions over 8 seasons.
PER_SEASON = {"full_slate": 8, "bust": 4, "season_threshold": 7}
SAMPLE_SEED = "sft_pilot_v0:20260808"

SYSTEM_PROMPT = (
    "You are Fantasy Alpha's draft analyst. Product voice, non-negotiable:\n"
    "- Verbal confidence in prose (unlikely / coin-flip / likely / near-lock), "
    "no percentage theater — the only numeric probability you emit is the "
    "final line.\n"
    "- Cite-or-don't-claim: every number you state must come verbatim from "
    "the packet, grounding, or evidence provided. You never invent numbers.\n"
    "- Base-rate anchoring: start from the code-computed historical grounding "
    "rates and move off them only for a stated, evidence-backed reason.\n"
    "- Well-calibrated beats confident. Bold calls need bold evidence.\n"
    "End your analysis with exactly one line of the form:\n"
    "FINAL PROBABILITY: <number between 0 and 1>"
)

_ctx = ssl.create_default_context()
if not _ctx.cert_store_stats().get("x509_ca"):
    _ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")


def _api_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("PRIME_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no PRIME_API_KEY in .env")


def chat(messages: list[dict], temperature: float = TEMPERATURE,
         max_tokens: int = MAX_TOKENS) -> tuple[str, dict]:
    body = {
        "model": TEACHER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_api_key()}",
                 "Content-Type": "application/json",
                 # bill the team account (evals/harness_breakout.py precedent)
                 "X-Prime-Team-ID": "cmskuc3x7018j7g10tr01gukl",
                 "User-Agent": "fantasy-alpha-sft-datagen/0.1"},
    )
    with urllib.request.urlopen(req, timeout=240, context=_ctx) as resp:
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"], out.get("usage", {})


def usage_cost(usage: dict) -> float:
    """Conservative per-call cost: max of the API-reported cost field and the
    catalog-rate estimate."""
    est = (usage.get("prompt_tokens", 0) * RATE_IN
           + usage.get("completion_tokens", 0) * RATE_OUT) / 1e6
    try:
        return max(est, float(usage.get("cost") or 0.0))
    except (TypeError, ValueError):
        return est


# ---------------------------------------------------------------------------
# grounding (pure code, strictly pre-eval-season — leakage asserted)


def _all_label_rows() -> list[dict]:
    """All drafted candidate rows (any ADP rank), ppr."""
    return [r for r in _breakout_rows() if r["format"] == PRESET]


def _rate(subset: list[dict], field) -> tuple[float | None, int]:
    n = len(subset)
    if not n:
        return None, 0
    return round(sum(field(r) for r in subset) / n, 3), n


def bust_grounding(cand: dict, rows_all: list[dict], season: int) -> dict:
    past = [r for r in rows_all if r["season"] < season]
    assert all(r["season"] < season for r in past), "leakage in bust grounding"
    top12 = [r for r in past if r["position"] == cand["position"]
             and r["adp_pos_rank"] <= 12]
    cohort = [r for r in top12
              if abs(r["adp_pos_rank"] - cand["adp_pos_rank"]) <= 3]
    c_rate, c_n = _rate(cohort, lambda r: r["bust"])
    b_rate, b_n = _rate(top12, lambda r: r["bust"])
    return {"cohort_rate": c_rate, "cohort_n": c_n,
            "position_top12_bust_rate": b_rate, "position_top12_bust_n": b_n}


def threshold_grounding(cand: dict, rows_all: list[dict], season: int,
                        threshold: int) -> dict:
    past = [r for r in rows_all if r["season"] < season
            and r["position"] == cand["position"]]
    assert all(r["season"] < season for r in past), "leakage in threshold grounding"

    def hit(r: dict) -> bool:
        return r["finish_pos_rank"] is not None and r["finish_pos_rank"] <= threshold

    cohort = [r for r in past
              if abs(r["adp_pos_rank"] - cand["adp_pos_rank"]) <= 4]
    wide = [r for r in past
            if abs(r["adp_pos_rank"] - cand["adp_pos_rank"]) <= 10]
    c_rate, c_n = _rate(cohort, hit)
    w_rate, w_n = _rate(wide, hit)
    return {"cohort_rate": c_rate, "cohort_n": c_n,
            "wide_cohort_rate": w_rate, "wide_cohort_n": w_n}


def pick_anchor(grounding: dict, order: tuple[str, ...],
                min_n: int = 20) -> tuple[float | None, int]:
    """First grounding rate (by preference order) with a usable sample;
    falls back to the largest-n non-null rate."""
    for key in order:
        rate, n = grounding.get(key), grounding.get(key.replace("rate", "n"), 0)
        if rate is not None and n >= min_n:
            return rate, n
    best = max(((grounding.get(k), grounding.get(k.replace("rate", "n"), 0))
                for k in order if grounding.get(k) is not None),
               key=lambda t: t[1], default=(None, 0))
    return best


ANCHOR_ORDER = {
    "full_slate": ("cohort_rate", "prior_rank_cohort_rate", "position_base_rate"),
    "bust": ("cohort_rate", "position_top12_bust_rate"),
    "season_threshold": ("cohort_rate", "wide_cohort_rate"),
}


# ---------------------------------------------------------------------------
# evidence (up to 3 EvidenceStore lookups, as_of-gated in the store)


def _clip(s: str | None, n: int = 280) -> str:
    s = (s or "").strip()
    return s[:n] + ("…" if len(s) > n else "")


def evidence_lookups(store: EvidenceStore, player: str) -> list[dict]:
    log: list[dict] = []

    def call(tool: str, fn, shrink):
        try:
            res = fn()
            log.append({"tool": tool, "args": {"player": player},
                        "ok": True, "result": shrink(res)})
        except Exception as e:  # unresolvable name, ambiguity, missing files
            log.append({"tool": tool, "args": {"player": player},
                        "ok": False, "error": f"{type(e).__name__}: {e}"})

    call("player_news", lambda: store.player_news(player, limit=3),
         lambda docs: [{"published": d["published_utc"], "source": d["source"],
                        "title": _clip(d["title"], 120),
                        "text": _clip(d["text"])} for d in docs])
    call("injury_status", lambda: store.injury_status(player),
         lambda r: None if r is None else
         {k: r[k] for k in ("published_utc", "report_status", "practice_status",
                            "primary_injury", "week", "season")})
    call("depth_chart", lambda: store.depth_chart(player),
         lambda rows: [{"snapshot": r["snapshot"], "team": r["team"],
                        "slot": r["depth_slot"], "rank": r["depth_rank"]}
                       for r in rows[:3]])
    return log


# ---------------------------------------------------------------------------
# question sampling (deterministic)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def build_pilot_sample() -> list[dict]:
    """~152 named questions across PILOT_SEASONS. Deterministic under
    SAMPLE_SEED. Each item: bench/family/season/question_id/player/packet/
    question/threshold?/outcome."""
    rng = random.Random(SAMPLE_SEED)
    sample: list[dict] = []
    for season in PILOT_SEASONS:
        assert season not in (SFT_VAL_SEASON, HOLDOUT_SEASON), \
            f"leakage: season {season} is reserved"
        # BreakoutBench (anonymized files; re-named via the sequestered KEY)
        bq = _load(QUESTIONS_DIR / f"breakoutbench_{season}_{PRESET}.json")
        bk = _load(QUESTIONS_DIR / f"breakoutbench_{season}_{PRESET}_KEY.json")["questions"]
        for family in ("full_slate", "bust"):
            pool = [q for q in bq["questions"] if q["family"] == family]
            for q in rng.sample(pool, min(PER_SEASON[family], len(pool))):
                key = bk[q["question_id"]]
                sample.append({
                    "bench": "breakoutbench", "family": family,
                    "season": season, "format": PRESET,
                    "question_id": q["question_id"],
                    "player": key["player"],
                    "question": q["question"],
                    "packet": q["packet"],
                    "outcome": int(key["outcome"]),
                })
        # CalibBench season_threshold (named variant)
        cq = _load(QUESTIONS_DIR / f"calibbench_{season}_{PRESET}.json")
        ck = _load(QUESTIONS_DIR / f"calibbench_{season}_{PRESET}_KEY.json")["questions"]
        pool = [q for q in cq["questions"] if q["family"] == "season_threshold"]
        for q in rng.sample(pool, min(PER_SEASON["season_threshold"], len(pool))):
            key = ck[q["question_id"]]
            sample.append({
                "bench": "calibbench", "family": "season_threshold",
                "season": season, "format": PRESET,
                "question_id": q["question_id"],
                "player": q["player"],
                "question": q["question"],
                "packet": q["packet"],
                "outcome": int(key["outcome"]),
            })
    return sample


# ---------------------------------------------------------------------------
# packet assembly -> teacher prompt


def grounding_for_item(item: dict, ctx: dict) -> dict:
    season, family, packet = item["season"], item["family"], item["packet"]
    cand = {"position": packet["position"],
            "adp_pos_rank": packet["adp_pos_rank"],
            "prior_season": packet.get("prior_season")}
    if family == "full_slate":
        past = past_rows(ctx["breakout_rows"], season)
        return grounding_for(cand, past, ctx["prior_ranks"], season)
    if family == "bust":
        return bust_grounding(cand, ctx["all_rows"], season)
    if family == "season_threshold":
        return threshold_grounding(cand, ctx["all_rows"], season,
                                   packet["threshold"])
    raise ValueError(family)


def build_user_message(item: dict, grounding: dict, tool_log: list[dict]) -> str:
    season = item["season"]
    named_packet = {"player": item["player"], **item["packet"]}
    return (
        f"Question ({item['bench']} / {item['family']}): {item['question']}\n\n"
        f"As-of date: {season}-09-01. Everything below was computed or "
        "retrieved strictly before this date; the archive contains nothing "
        "after it.\n\n"
        f"Candidate packet:\n{json.dumps(named_packet, indent=1)}\n\n"
        "Historical grounding (code-computed base rates over comparable past "
        f"candidates, seasons before {season}):\n"
        f"{json.dumps(grounding, indent=1)}\n\n"
        "Evidence lookups (time-gated archive):\n"
        f"{json.dumps(tool_log, indent=1)}\n\n"
        "Write a tight analyst's read of this question in under 220 words, "
        "citing the packet, grounding, and evidence. Follow the system rules, "
        "then end with the single FINAL PROBABILITY line."
    )


# ---------------------------------------------------------------------------
# generation


def generate(limit: float = BUDGET_CAP_USD, workers: int = 6) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = build_pilot_sample()
    ctx = {
        "breakout_rows": load_breakout_rows(PRESET),
        "prior_ranks": load_prior_ranks(PRESET),
        "all_rows": _all_label_rows(),
    }
    stores = {s: EvidenceStore(as_of=f"{s}-09-01") for s in PILOT_SEASONS}

    done: set[str] = set()
    if RAW_TRACES.exists():
        for line in RAW_TRACES.read_text().splitlines():
            done.add(json.loads(line)["trace_id"])

    jobs = []
    for item in sample:
        for k in range(SAMPLES_PER_QUESTION):
            tid = f"{item['bench']}:{item['season']}:{item['question_id']}:s{k}"
            if tid not in done:
                jobs.append((tid, k, item))
    print(f"{len(sample)} questions -> {len(jobs)} calls to make "
          f"({len(done)} already on disk); teacher {TEACHER_MODEL}, "
          f"temp {TEMPERATURE}, cap ${limit:.2f}")

    lock = threading.Lock()
    state = {"cost": 0.0, "in": 0, "out": 0, "n": 0, "stop": False}
    f = open(RAW_TRACES, "a")

    def run_one(job) -> None:
        tid, k, item = job
        with lock:
            if state["stop"]:
                return
        grounding = grounding_for_item(item, ctx)
        anchor_rate, anchor_n = pick_anchor(
            grounding, ANCHOR_ORDER[item["family"]])
        tool_log = evidence_lookups(stores[item["season"]], item["player"])
        user = build_user_message(item, grounding, tool_log)
        try:
            text, usage = chat([{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user}])
        except Exception as e:
            print(f"  {tid}: API error {type(e).__name__}: {e}", flush=True)
            return
        record = {
            "trace_id": tid, "bench": item["bench"], "family": item["family"],
            "season": item["season"], "format": item["format"],
            "question_id": item["question_id"], "player": item["player"],
            "as_of": f"{item['season']}-09-01",
            "grounding": grounding,
            "anchor_rate": anchor_rate, "anchor_n": anchor_n,
            "outcome": item["outcome"],
            "system": SYSTEM_PROMPT, "user": user, "assistant": text,
            "model": TEACHER_MODEL, "temperature": TEMPERATURE,
            "sample_idx": k, "usage": usage,
        }
        with lock:
            f.write(json.dumps(record) + "\n")
            f.flush()
            state["cost"] += usage_cost(usage)
            state["in"] += usage.get("prompt_tokens", 0)
            state["out"] += usage.get("completion_tokens", 0)
            state["n"] += 1
            if state["n"] % 20 == 0:
                print(f"  {state['n']}/{len(jobs)} calls · "
                      f"{state['in']}in/{state['out']}out tok · "
                      f"${state['cost']:.3f}", flush=True)
            # stop BEFORE exceeding the cap: leave headroom for in-flight calls
            per_call = state["cost"] / state["n"]
            if state["cost"] + per_call * (workers + 1) >= limit:
                state["stop"] = True
                print(f"BUDGET STOP at ${state['cost']:.3f} "
                      f"(cap ${limit:.2f})", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_one, jobs))
    f.close()
    print(f"done: {state['n']} calls, {state['in']}in/{state['out']}out tok, "
          f"est ${state['cost']:.3f} (rates {RATE_IN}/{RATE_OUT} $/Mtok) "
          f"-> {RAW_TRACES}")


# ---------------------------------------------------------------------------
# filter -> pilot corpus


def run_filter_cmd() -> None:
    traces = [json.loads(line) for line in RAW_TRACES.read_text().splitlines()]
    for t in traces:
        assert t["season"] in PILOT_SEASONS, \
            f"leakage: trace season {t['season']} outside pilot seasons"
    result = trace_filter.run_filter(traces)
    with open(PILOT_OUT, "w") as f:
        for t in result["kept"]:
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": t["system"]},
                    {"role": "user", "content": t["user"]},
                    {"role": "assistant", "content": t["assistant"]},
                ],
                "meta": {k: t[k] for k in (
                    "trace_id", "bench", "family", "season", "format",
                    "question_id", "player", "as_of", "p", "outcome",
                    "anchor_rate", "anchor_n", "brier_improvement",
                    "model", "temperature", "sample_idx")},
            }) + "\n")
    REPORT_OUT.write_text(trace_filter.render_report(
        result, title=f"SFT pilot v0 — filter report ({TEACHER_MODEL}, "
                      f"temp {TEMPERATURE}, seasons "
                      f"{PILOT_SEASONS[0]}-{PILOT_SEASONS[-1]})"))
    print(f"kept {len(result['kept'])}/{result['n_input']} traces -> {PILOT_OUT}")
    print(f"report -> {REPORT_OUT}")
    print("rejections:", result["reason_counts"])
    flagged = [r for r in result["band_audit"] if r["flagged"]]
    print("band audit:", "FLAGGED bands: " + str(flagged) if flagged
          else "clean (no survivorship signature)")


# ---------------------------------------------------------------------------
# T1 — ANONYMIZED-track generation at scale (risk register 3-4; GAMEPLAN
# season-split amendment 2026-08-09).
#
# The pilot above generated from NAMED packets; rule (a) catches numeric
# teacher-memory leaks but not narrative ones. T1 therefore generates from the
# anonymized question variants (BreakoutBench files are anonymized natively;
# CalibBench has *_anon.json), with the evidence surface restricted to the
# masked-mode pattern of envs/draftgym.py: free-text tools (player_news) are
# DISABLED, structured lookups (injury_status / depth_chart) run server-side
# against the sequestered KEY identity and return rows with the player name
# replaced by the anon id, teams dropped, and all absolute dates relativized
# to the as-of moment (a date string carries the year, and the year is masked).
# Grounding is unchanged: code-computed cohort base rates, strictly-prior
# seasons, leakage asserted.
#
# Season split (GAMEPLAN amendment, unit-tested in tests/test_sft_t1.py):
# train = 2008-2022 EXCLUDING 2013 and 2018 (sealed mid-era diagnostics);
# 2023-2024 = forward val (never SFT data); 2025 = sealed holdout.
# Preset per season: ppr where an FFC ppr board exists (2010+), standard for
# 2008-2009 (FFC ADP reaches 2008 in standard only).

MID_ERA_DIAGNOSTIC_SEASONS = (2013, 2018)  # sealed, never SFT data
FORWARD_VAL_SEASONS = (2023, 2024)         # checkpoint-selection val only

T1_SEASON_PRESET: dict[int, str] = {
    **{s: "standard" for s in (2008, 2009)},
    **{s: "ppr" for s in range(2010, 2023)
       if s not in MID_ERA_DIAGNOSTIC_SEASONS},
}
assert not set(T1_SEASON_PRESET) & set(MID_ERA_DIAGNOSTIC_SEASONS)
assert all(s < min(FORWARD_VAL_SEASONS) for s in T1_SEASON_PRESET), \
    "leakage: 2023+ can never be SFT data"

T1_SAMPLE_SEED = "sft_t1_v1:20260808"
# per season (capped by pool size): scaled from the pilot's 19/season to hit
# ~1,400 questions -> ~2,800 calls at 2 samples -> ~1,200-2,000 kept traces.
T1_PER_SEASON = {"full_slate": 30, "bust": 12,
                 "season_threshold": 30, "weekly_h2h": 40}
T1_BUDGET_CAP_USD = 9.00  # hard task cap $10; $1 headroom for pings/retries

T1_RAW_TRACES = OUT_DIR / "raw_traces_t1.jsonl"
T1_CORPUS_OUT = OUT_DIR / "t1_corpus_v1.jsonl"
T1_REPORT_OUT = OUT_DIR / "t1_filter_report.md"

from evals.calibbench import build_h2h_questions  # noqa: E402
from harness.evidence import _week_snapshot_date  # noqa: E402


# -- per-preset grounding contexts (the pilot hardcoded ppr) -----------------


def _label_rows_for(preset: str) -> list[dict]:
    return [r for r in _breakout_rows() if r["format"] == preset]


# Labels (breakouts.csv) start in 2008 and ppr boards in 2010, so the earliest
# eval seasons have no strictly-prior rows in their own format. Grounding
# therefore POOLS formats (precedent: evals/breakoutbench.family_base_rates —
# "pooled across formats — more data, same no-peek guarantee"); the leakage
# rule (strictly-prior seasons) is unchanged. 2008 itself has no prior labeled
# season at all, so season-level families start at 2009 (weekly_h2h keeps
# 2008 — its grounding comes from weekly labels, which reach back to 1999).
GROUNDING_POOL_PRESETS = ("standard", "ppr")
T1_SEASON_LEVEL_FLOOR = 2009


def _t1_ctx() -> dict:
    """Grounding inputs per preset, pooled across GROUNDING_POOL_PRESETS."""
    pooled_breakout: list[dict] = []
    pooled_all: list[dict] = []
    for p in GROUNDING_POOL_PRESETS:
        pooled_breakout.extend(load_breakout_rows(p))
        pooled_all.extend(_label_rows_for(p))
    ctx: dict = {}
    for preset in sorted(set(T1_SEASON_PRESET.values())):
        prior_ranks: dict = {}
        for p in GROUNDING_POOL_PRESETS:
            if p != preset:
                prior_ranks.update(load_prior_ranks(p))
        prior_ranks.update(load_prior_ranks(preset))  # item's preset wins
        ctx[preset] = {"breakout_rows": pooled_breakout,
                       "prior_ranks": prior_ranks,
                       "all_rows": pooled_all}
    return ctx


# -- weekly_h2h grounding (new family; same strictly-prior discipline) --------

H2H_HISTORY_FLOOR = 1999   # weekly labels start 1999
H2H_EDGE_TOL = 0.05        # cohort: past pairs with a similar ppg edge


def _h2h_edge(packet: dict) -> float:
    """Signed season-to-date ppg edge of A over B, as a fraction of the
    larger ppg (the same quantity the pair-builder tolerances)."""
    a, b = packet["a"]["ppg"], packet["b"]["ppg"]
    return (a - b) / max(a, b) if max(a, b) > 0 else 0.0


_H2H_HISTORY_CACHE: dict[tuple[str, int], list[tuple[float, int]]] = {}


def _h2h_history(preset: str, before_season: int) -> list[tuple[float, int]]:
    """(edge, outcome) for every h2h pair in seasons STRICTLY before
    ``before_season`` (leakage asserted). Built from the same deterministic
    pair generator as the benchmark itself."""
    key = (preset, before_season)
    if key not in _H2H_HISTORY_CACHE:
        pairs: list[tuple[float, int]] = []
        for s in range(H2H_HISTORY_FLOOR, before_season):
            assert s < before_season, "leakage in h2h grounding"
            for q in build_h2h_questions(s, preset):
                pairs.append((_h2h_edge(q["packet"]), int(q["outcome"])))
        _H2H_HISTORY_CACHE[key] = pairs
    return _H2H_HISTORY_CACHE[key]


def h2h_grounding(packet: dict, preset: str, season: int) -> dict:
    """Base rates for 'A outscores B' over comparable past pairs."""
    edge = round(_h2h_edge(packet), 4)
    history = _h2h_history(preset, season)
    cohort = [(e, y) for e, y in history if abs(e - edge) <= H2H_EDGE_TOL]
    sign = [(e, y) for e, y in history if (e >= 0) == (edge >= 0)]
    c_rate, c_n = _rate(cohort, lambda t: t[1])
    s_rate, s_n = _rate(sign, lambda t: t[1])
    b_rate, b_n = _rate(history, lambda t: t[1])
    return {"ppg_edge": edge,
            "cohort_rate": c_rate, "cohort_n": c_n,
            "edge_sign_rate": s_rate, "edge_sign_n": s_n,
            "base_rate": b_rate, "base_n": b_n}


ANCHOR_ORDER["weekly_h2h"] = ("cohort_rate", "edge_sign_rate", "base_rate")


# -- anonymized question sampling ---------------------------------------------


def _invert_calib_key(key_payload: dict) -> dict[str, dict]:
    return {q["anon_id"]: q for q in key_payload["questions"].values()}


def build_t1_sample() -> list[dict]:
    """Anonymized questions across T1_SEASON_PRESET, deterministic under
    T1_SAMPLE_SEED. Items carry NO player names — real identities stay in
    ``_identity`` (sequestered-KEY lookup data used server-side for masked
    evidence calls, never serialized into traces)."""
    rng = random.Random(T1_SAMPLE_SEED)
    sample: list[dict] = []
    for season in sorted(T1_SEASON_PRESET):
        preset = T1_SEASON_PRESET[season]
        assert season not in MID_ERA_DIAGNOSTIC_SEASONS
        assert season not in FORWARD_VAL_SEASONS
        assert season not in (SFT_VAL_SEASON, HOLDOUT_SEASON)
        # BreakoutBench: the question file IS the anonymized track
        bq = _load(QUESTIONS_DIR / f"breakoutbench_{season}_{preset}.json")
        bk = _load(QUESTIONS_DIR / f"breakoutbench_{season}_{preset}_KEY.json")["questions"]
        season_level = season >= T1_SEASON_LEVEL_FLOOR
        for family in ("full_slate", "bust") if season_level else ():
            pool = [q for q in bq["questions"] if q["family"] == family]
            for q in rng.sample(pool, min(T1_PER_SEASON[family], len(pool))):
                key = bk[q["question_id"]]
                sample.append({
                    "bench": "breakoutbench", "family": family,
                    "season": season, "format": preset,
                    "question_id": q["question_id"],
                    "question": q["question"], "packet": q["packet"],
                    "outcome": int(key["outcome"]),
                    "_identity": {"players": [key["player"]],
                                  "ids": [key["player_id"]]},
                })
        # CalibBench: the *_anon.json variant + inverted KEY
        cq = _load(QUESTIONS_DIR / f"calibbench_{season}_{preset}_anon.json")
        ck = _invert_calib_key(
            _load(QUESTIONS_DIR / f"calibbench_{season}_{preset}_KEY.json"))
        for family in (("season_threshold", "weekly_h2h") if season_level
                       else ("weekly_h2h",)):
            pool = [q for q in cq["questions"] if q["family"] == family]
            for q in rng.sample(pool, min(T1_PER_SEASON[family], len(pool))):
                key = ck[q["question_id"]]
                players = ([key["player"]] if family == "season_threshold"
                           else [key["player_a"], key["player_b"]])
                sample.append({
                    "bench": "calibbench", "family": family,
                    "season": season, "format": preset,
                    "question_id": q["question_id"],
                    "question": q["question"], "packet": q["packet"],
                    "outcome": int(key["outcome"]),
                    "_identity": {"players": players,
                                  "ids": [key.get("nflverse_id")]},
                })
    return sample


# -- masked evidence surface (draftgym._masked_dispatch pattern) ---------------

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _days_before(as_of_iso: str, published_iso: str) -> str:
    """Relativize an absolute timestamp to the as-of moment. Absolute dates
    carry the year and the year is masked on this track."""
    from datetime import date
    a = date.fromisoformat(as_of_iso[:10])
    p = date.fromisoformat(published_iso[:10])
    return f"{(a - p).days}d_before_as_of"


def _mask_injury(row: dict | None, anon_id: str, as_of: str) -> dict | None:
    if row is None:
        return None
    return {"player": anon_id,
            "published": _days_before(as_of, row["published_utc"]),
            "report_status": row.get("report_status"),
            "practice_status": row.get("practice_status"),
            "primary_injury": row.get("primary_injury"),
            "report_week": row.get("week")}


def _mask_depth(rows: list[dict], anon_id: str, as_of: str) -> list[dict]:
    return [{"player": anon_id,
             "snapshot": _days_before(as_of, r["published_utc"]),
             "slot": r.get("depth_slot"), "rank": r.get("depth_rank"),
             "unit": r.get("unit")} for r in rows[:3]]


def anon_evidence_lookups(store: EvidenceStore, players: list[str],
                          labels: list[str], as_of: str) -> list[dict]:
    """Structured lookups only (injury_status, depth_chart), dispatched
    server-side against the real identity, results re-masked to the anon
    label. Free-text tools are disabled exactly as in DraftGym masked mode;
    errors are sanitized so they can never de-anonymize."""
    log: list[dict] = [{
        "tool": "player_news", "ok": False,
        "error": ("tool disabled in anonymized mode (free-text results "
                  "reveal player names); structured lookups only"),
    }]
    for player, label in zip(players, labels):
        try:
            row = store.injury_status(player)
            log.append({"tool": "injury_status", "args": {"player": label},
                        "ok": True, "result": _mask_injury(row, label, as_of)})
        except Exception:
            log.append({"tool": "injury_status", "args": {"player": label},
                        "ok": False, "error": "lookup failed"})
        try:
            rows = store.depth_chart(player)
            log.append({"tool": "depth_chart", "args": {"player": label},
                        "ok": True, "result": _mask_depth(rows, label, as_of)})
        except Exception:
            log.append({"tool": "depth_chart", "args": {"player": label},
                        "ok": False, "error": "lookup failed"})
    return log


# -- anonymized prompt assembly ------------------------------------------------


def t1_as_of(item: dict) -> str:
    """Real as-of date (provenance + evidence gate). Sept 1 for season-level
    families; Thursday of week W (pre-kickoff) for weekly_h2h."""
    if item["family"] == "weekly_h2h":
        return _week_snapshot_date(item["season"], item["packet"]["week"]).isoformat()
    return f"{item['season']}-09-01"


def t1_grounding_for_item(item: dict, ctx_by_preset: dict) -> dict:
    season, family, packet = item["season"], item["family"], item["packet"]
    preset = item["format"]
    ctx = ctx_by_preset[preset]
    if family == "weekly_h2h":
        return h2h_grounding(packet, preset, season)
    cand = {"position": packet["position"],
            "adp_pos_rank": packet["adp_pos_rank"],
            "prior_season": packet.get("prior_season")}
    if family == "full_slate":
        past = past_rows(ctx["breakout_rows"], season)
        return grounding_for(cand, past, ctx["prior_ranks"], season)
    if family == "bust":
        return bust_grounding(cand, ctx["all_rows"], season)
    if family == "season_threshold":
        return threshold_grounding(cand, ctx["all_rows"], season,
                                   packet["threshold"])
    raise ValueError(family)


def build_anon_user_message(item: dict, grounding: dict,
                            tool_log: list[dict]) -> str:
    if item["family"] == "weekly_h2h":
        as_of_line = (f"As-of moment: Thursday of week {item['packet']['week']} "
                      "of the target season, pre-kickoff (year masked).")
    else:
        as_of_line = ("As-of moment: September 1 of the target season "
                      "(year masked).")
    msg = (
        f"Question ({item['bench']} / {item['family']}): {item['question']}\n\n"
        "This is an ANONYMIZED dossier: player names, teams, and the season "
        "year are masked. Reason from the packet features, the historical "
        "grounding, and the evidence — never from a guess at who this is.\n\n"
        f"{as_of_line} Everything below was computed or retrieved strictly "
        "before this moment; the archive contains nothing after it.\n\n"
        f"Candidate packet:\n{json.dumps(item['packet'], indent=1)}\n\n"
        "Historical grounding (code-computed base rates over comparable "
        "candidates from seasons strictly before the target season):\n"
        f"{json.dumps(grounding, indent=1)}\n\n"
        "Evidence lookups (time-gated archive, anonymized tool surface):\n"
        f"{json.dumps(tool_log, indent=1)}\n\n"
        "Write a tight analyst's read of this question in under 220 words, "
        "citing the packet, grounding, and evidence. Follow the system rules, "
        "then end with the single FINAL PROBABILITY line."
    )
    assert not _DATE_RE.search(msg), "anonymization leak: absolute date in prompt"
    assert str(item["season"]) not in msg.split("Candidate packet:")[0], \
        "anonymization leak: season year in preamble"
    return msg


# -- T1 generation (resumable; incremental writes; budget-capped) --------------


def t1_generate(limit: float = T1_BUDGET_CAP_USD, workers: int = 6) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = build_t1_sample()
    ctx_by_preset = _t1_ctx()
    # pre-warm h2h grounding history (thread-safety: build before the pool)
    for season in sorted(T1_SEASON_PRESET):
        _h2h_history(T1_SEASON_PRESET[season], season)
    stores: dict[str, EvidenceStore] = {}
    for item in sample:
        a = t1_as_of(item)
        if a not in stores:
            stores[a] = EvidenceStore(as_of=a)

    done: set[str] = set()
    spent_prior = 0.0  # the cap is per-CORPUS, not per-invocation: a resumed
    # run must count what earlier invocations already spent on disk
    if T1_RAW_TRACES.exists():
        for line in T1_RAW_TRACES.read_text().splitlines():
            rec = json.loads(line)
            done.add(rec["trace_id"])
            spent_prior += usage_cost(rec.get("usage") or {})

    jobs = []
    for item in sample:
        for k in range(SAMPLES_PER_QUESTION):
            tid = f"t1:{item['bench']}:{item['season']}:{item['question_id']}:s{k}"
            if tid not in done:
                jobs.append((tid, k, item))
    # deterministic shuffle: a budget/limit kill leaves coverage spread across
    # seasons and families instead of truncating the tail seasons
    random.Random(T1_SAMPLE_SEED + ":joborder").shuffle(jobs)
    print(f"{len(sample)} anon questions -> {len(jobs)} calls to make "
          f"({len(done)} already on disk, ${spent_prior:.3f} already spent); "
          f"teacher {TEACHER_MODEL}, temp {TEMPERATURE}, cap ${limit:.2f}")

    lock = threading.Lock()
    state = {"cost": spent_prior, "cost_new": 0.0,
             "in": 0, "out": 0, "n": 0, "stop": False}
    f = open(T1_RAW_TRACES, "a")

    def run_one(job) -> None:
        tid, k, item = job
        with lock:
            if state["stop"]:
                return
        as_of = t1_as_of(item)
        grounding = t1_grounding_for_item(item, ctx_by_preset)
        anchor_rate, anchor_n = pick_anchor(
            grounding, ANCHOR_ORDER[item["family"]])
        labels = ([item["question_id"]] if item["family"] != "weekly_h2h"
                  else [f"{item['question_id']}:A", f"{item['question_id']}:B"])
        tool_log = anon_evidence_lookups(
            stores[as_of], item["_identity"]["players"], labels, as_of)
        user = build_anon_user_message(item, grounding, tool_log)
        text = usage = None
        for attempt in (1, 2):
            try:
                text, usage = chat([{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": user}])
                break
            except Exception as e:
                print(f"  {tid}: API error {type(e).__name__} "
                      f"(attempt {attempt})", flush=True)
                if attempt == 2:
                    return
        record = {
            "trace_id": tid, "track": "anonymized",
            "bench": item["bench"], "family": item["family"],
            "season": item["season"], "format": item["format"],
            "question_id": item["question_id"],
            "nflverse_ids": item["_identity"]["ids"],
            "as_of": as_of,
            "grounding": grounding,
            "anchor_rate": anchor_rate, "anchor_n": anchor_n,
            "outcome": item["outcome"],
            "system": SYSTEM_PROMPT, "user": user, "assistant": text,
            "model": TEACHER_MODEL, "temperature": TEMPERATURE,
            "sample_idx": k, "usage": usage,
        }
        with lock:
            f.write(json.dumps(record) + "\n")
            f.flush()
            state["cost"] += usage_cost(usage)
            state["cost_new"] += usage_cost(usage)
            state["in"] += usage.get("prompt_tokens", 0)
            state["out"] += usage.get("completion_tokens", 0)
            state["n"] += 1
            if state["n"] % 50 == 0:
                print(f"  {state['n']}/{len(jobs)} calls · "
                      f"{state['in']}in/{state['out']}out tok · "
                      f"${state['cost']:.3f}", flush=True)
            per_call = state["cost_new"] / state["n"]
            if state["cost"] + per_call * (workers + 1) >= limit:
                state["stop"] = True
                print(f"BUDGET STOP at ${state['cost']:.3f} "
                      f"(cap ${limit:.2f})", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_one, jobs))
    f.close()
    print(f"done: {state['n']} calls, {state['in']}in/{state['out']}out tok, "
          f"est ${state['cost']:.3f} (rates {RATE_IN}/{RATE_OUT} $/Mtok) "
          f"-> {T1_RAW_TRACES}")


# -- T1 filter -> t1_corpus_v1.jsonl (anon kept + flagged named-pilot subset) --


def _named_pilot_rows() -> tuple[list[dict], list[int]]:
    """Pilot traces re-admitted under the season-split amendment: the pilot
    drew 2016-2023, but 2018 (mid-era diagnostic) and 2023 (forward val) can
    never be SFT data — those rows are dropped, and the drop is reported."""
    rows, dropped = [], []
    if not PILOT_OUT.exists():
        return rows, dropped
    for line in PILOT_OUT.read_text().splitlines():
        rec = json.loads(line)
        season = rec["meta"]["season"]
        if season in T1_SEASON_PRESET:
            rec["meta"]["track"] = "named_pilot"
            rows.append(rec)
        else:
            dropped.append(season)
    return rows, dropped


def run_t1_filter_cmd() -> None:
    traces = [json.loads(line) for line in T1_RAW_TRACES.read_text().splitlines()]
    for t in traces:
        assert t["season"] in T1_SEASON_PRESET, \
            f"leakage: trace season {t['season']} outside the T1 train split"
    result = trace_filter.run_filter(traces)

    named_rows, named_dropped = _named_pilot_rows()
    with open(T1_CORPUS_OUT, "w") as f:
        for t in result["kept"]:
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": t["system"]},
                    {"role": "user", "content": t["user"]},
                    {"role": "assistant", "content": t["assistant"]},
                ],
                "meta": {"track": "anonymized", **{k: t[k] for k in (
                    "trace_id", "bench", "family", "season", "format",
                    "question_id", "as_of", "p", "outcome",
                    "anchor_rate", "anchor_n", "brier_improvement",
                    "model", "temperature", "sample_idx")}},
            }) + "\n")
        for rec in named_rows:
            f.write(json.dumps(rec) + "\n")

    n_anon, n_named = len(result["kept"]), len(named_rows)
    mix = (f"\n## Corpus mix (t1_corpus_v1.jsonl)\n\n"
           f"- anonymized-track kept traces: **{n_anon}**\n"
           f"- named pilot traces (flagged `track=named_pilot`): **{n_named}** "
           f"of 162 — {len(named_dropped)} dropped for the season-split "
           f"amendment (seasons {sorted(set(named_dropped))}: 2018 is a "
           "sealed mid-era diagnostic, 2023 is forward val)\n"
           f"- named share: {n_named / max(1, n_anon + n_named):.1%}\n")
    T1_REPORT_OUT.write_text(trace_filter.render_report(
        result, title=f"SFT T1 v1 — filter report ({TEACHER_MODEL}, "
                      f"temp {TEMPERATURE}, ANONYMIZED track, seasons "
                      f"2008-2022 minus {{2013, 2018}})") + mix)
    print(f"kept {n_anon}/{result['n_input']} anon traces + {n_named} named "
          f"pilot -> {T1_CORPUS_OUT}")
    print(f"report -> {T1_REPORT_OUT}")
    print("rejections:", result["reason_counts"])
    flagged = [r for r in result["band_audit"] if r["flagged"]]
    print("band audit:", "FLAGGED bands: " + str(flagged) if flagged
          else "clean (no survivorship signature)")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SFT data pipeline v0")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping", help="one tiny sanity call to the teacher")
    sub.add_parser("sample", help="print the deterministic pilot sample")
    g = sub.add_parser("generate", help="teacher trace generation (budget-capped)")
    g.add_argument("--cap", type=float, default=BUDGET_CAP_USD)
    g.add_argument("--workers", type=int, default=6)
    sub.add_parser("filter", help="filter raw traces -> pilot_v0.jsonl + report")
    sub.add_parser("t1-sample", help="print the deterministic T1 anon sample")
    t1g = sub.add_parser("t1-generate",
                         help="T1 anonymized-track generation (budget-capped)")
    t1g.add_argument("--cap", type=float, default=T1_BUDGET_CAP_USD)
    t1g.add_argument("--workers", type=int, default=6)
    sub.add_parser("t1-filter",
                   help="filter T1 raw traces -> t1_corpus_v1.jsonl + report")
    args = ap.parse_args(argv)

    if args.cmd == "ping":
        text, usage = chat([{"role": "user", "content":
                             "Reply with the single word: ready"}],
                           temperature=0.0, max_tokens=10)
        print(f"{TEACHER_MODEL}: {text!r} · {usage} · ${usage_cost(usage):.6f}")
    elif args.cmd == "sample":
        sample = build_pilot_sample()
        by = {}
        for it in sample:
            by[(it["season"], it["family"])] = by.get((it["season"], it["family"]), 0) + 1
        print(f"{len(sample)} questions:")
        for (s, fam), n in sorted(by.items()):
            print(f"  {s} {fam:<18} {n}")
    elif args.cmd == "generate":
        generate(limit=args.cap, workers=args.workers)
    elif args.cmd == "filter":
        run_filter_cmd()
    elif args.cmd == "t1-sample":
        sample = build_t1_sample()
        by: dict = {}
        for it in sample:
            by[(it["season"], it["family"])] = by.get((it["season"], it["family"]), 0) + 1
        print(f"{len(sample)} anon questions across "
              f"{len(T1_SEASON_PRESET)} seasons:")
        for (s, fam), n in sorted(by.items()):
            print(f"  {s} ({T1_SEASON_PRESET[s]}) {fam:<18} {n}")
    elif args.cmd == "t1-generate":
        t1_generate(limit=args.cap, workers=args.workers)
    elif args.cmd == "t1-filter":
        run_t1_filter_cmd()
    return 0


if __name__ == "__main__":
    sys.exit(main())
