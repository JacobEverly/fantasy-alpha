#!/usr/bin/env python3
"""Frontier-model benchmark battery via OpenRouter (BreakoutBench anonymized track).

Runs the frontier contenders through the same protocol as the Qwen3.5-9B
baselines (evals/run_expanded_qwen.py, evals/canaries.py, envs/play_llm.py),
so every number lands on the existing leaderboard axes:

  Tier 1  naked anonymized battery, ppr grid (10 seasons x 3 families = 30
          requests/model), scored with breakoutbench.score_run (Brier + season-
          bootstrap CIs + top-10 lift).
  Tier 2  canary protocol v1 run (10 requests/model) + fingerprint probe
          reusing the tier-1 full_slate answers — the frontier memorization read.
  Tier 3  harness-grounded battery (same grid, grounding fields injected by
          run_expanded_qwen.grounding_for) for ALL contenders.
  Tier 4  DraftGym 2021: 2 named + 2 masked episodes for the top-2 tier-1
          models plus the big open Qwen — named-minus-masked reward deltas.

HARD BUDGET: $45.00 total (account holds $60). Cost tracking is exact:
`"usage": {"include": true}` on every request and OpenRouter's `usage.cost`
(credits == USD) accumulated per model; a conservative pre-flight ceiling is
reserved before each request and the run stops (gracefully, tier by tier)
before the cap can be crossed. Per-model actuals -> docs/budget-ledger.md.

Resume-safe: existing answers files skip their requests. One repair retry per
malformed response; per-model try/except so one provider outage doesn't kill
the battery; refusals/failures recorded to
evals/results/answers/failures_frontier.json.

Output: evals/results/frontier_battery.md (leaderboard vs the existing Qwen /
GBDT / base-rate / coin rows, canary gates, DraftGym deltas, per-model cost,
verdicts) and evals/results/frontier_draftgym.json (episode detail).
"""
from __future__ import annotations

import json
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.breakoutbench import (  # noqa: E402
    ANSWERS_DIR,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    ERAS,
    FAMILIES,
    QUESTIONS_DIR,
    SEASONS,
    score_run,
    season_formats,
)
from evals.canaries import (  # noqa: E402
    GATE_INDEX_CI_LOW,
    PRESET,
    _pair_credit,
    _pool_stats,
    bootstrap_index,
    load_answers_files,
    score_canaries,
)
from evals.canaries import build as build_canaries  # noqa: E402
from evals.telemetry import record as telemetry_record  # noqa: E402
from evals.run_expanded_qwen import (  # noqa: E402
    _family_rows,
    _prior_ranks,
    build_prompt,
    fmt_ci,
    grounding_for,
    overlap,
    parse_answers,
)

API = "https://openrouter.ai/api/v1/chat/completions"
MODELS_API = "https://openrouter.ai/api/v1/models"
HARD_BUDGET = 43.50  # $3.50 already spent + Jacob's $40 cap for the Fable extension
DEFAULT_MAX_TOKENS = 3000
REPAIR_MAX_TOKENS = 16000  # repair retry headroom when reasoning eats the cap
REQUEST_TIMEOUT = 300
UA = "fantasy-alpha-evals/0.1 (frontier battery)"
REFERER = "https://github.com/plantmangroup/fantasy-alpha"

RESULTS_MD = ROOT / "evals" / "results" / "frontier_battery.md"
DRAFTGYM_JSON = ROOT / "evals" / "results" / "frontier_draftgym.json"
FAILURES_PATH = ANSWERS_DIR / "failures_frontier.json"

# Contenders (ids verified against GET /models 2026-08-08; prices recorded
# at runtime from the live catalog into the report).
MODELS = [
    # Fable-5 rejects reasoning {"enabled": False} ("Reasoning is mandatory
    # for this endpoint"); effort "minimal" is the minimal working setting and
    # emitted 0 reasoning tokens in the pilot — closest available match to the
    # thinking-off protocol of every other row.
    {"slug": "fable5", "id": "anthropic/claude-fable-5",
     "label": "Claude Fable 5", "open": False,
     "reasoning": {"effort": "minimal"}, "max_tokens": 6000},
    {"slug": "gpt56lunapro", "id": "openai/gpt-5.6-luna-pro",
     "label": "GPT-5.6-Luna-Pro", "open": False},
    {"slug": "gemini31pro", "id": "google/gemini-3.1-pro-preview",
     "label": "Gemini 3.1 Pro", "open": False},
    {"slug": "dsv4pro", "id": "deepseek/deepseek-v4-pro",
     "label": "DeepSeek V4 Pro", "open": True},
    # reasoning {"enabled": False} for the two active contenders: at reasoning
    # effort "low" both models think for minutes per slate (5-15k reasoning
    # tokens, 8-10 min/request) and qwen35b returns empty content at any sane
    # max_tokens. Thinking OFF is also the Qwen3.5-9B baseline protocol
    # (enable_thinking=false), so the scale probe is apples-to-apples.
    {"slug": "dsv4flash", "id": "deepseek/deepseek-v4-flash",
     "label": "DeepSeek V4 Flash", "open": True,
     "reasoning": {"enabled": False}},
    {"slug": "qwen397b", "id": "qwen/qwen3.5-397b-a17b",
     "label": "Qwen3.5-397B-A17B", "open": True},
    {"slug": "qwen35b", "id": "qwen/qwen3.5-35b-a3b",
     "label": "Qwen3.5-35B-A3B", "open": True,
     "reasoning": {"enabled": False}},
    {"slug": "glm52", "id": "z-ai/glm-5.2",
     "label": "GLM-5.2", "open": True},
]

DRAFTGYM_SPECS = [
    dict(season=2021, preset="ppr", agent_slot=1, seed=0),
    dict(season=2021, preset="ppr", agent_slot=7, seed=0),
]
DRAFTGYM_MASKED = [dict(s, mask_names=True) for s in DRAFTGYM_SPECS]
MAX_REQUESTS_PER_PICK = 8

_ctx = ssl.create_default_context()
if not _ctx.cert_store_stats().get("x509_ca"):
    _ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")


def api_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no OPENROUTER_API_KEY in .env")


# ---------------------------------------------------------------------------
# Spend tracking (exact: OpenRouter usage.cost per response, lock-guarded)


class BudgetExhausted(RuntimeError):
    pass


SPEND_SNAPSHOT = ANSWERS_DIR / "frontier_spend.json"


class Spend:
    """Per-model exact spend (OpenRouter usage.cost), persisted after every
    request so a killed run never loses its accounting. `prior_unattributed`
    covers spend from before snapshotting existed (credits-endpoint delta)."""

    def __init__(self, cap: float = HARD_BUDGET):
        self.cap = cap
        self.lock = threading.Lock()
        self.by_model: dict[str, dict] = defaultdict(
            lambda: {"in": 0, "out": 0, "cost": 0.0, "requests": 0})
        self.prior_unattributed = 0.0
        if SPEND_SNAPSHOT.exists():
            snap = json.loads(SPEND_SNAPSHOT.read_text())
            self.prior_unattributed = snap.get("prior_unattributed", 0.0)
            for slug, m in snap.get("by_model", {}).items():
                self.by_model[slug] = m

    def _save(self) -> None:
        SPEND_SNAPSHOT.write_text(json.dumps(
            {"prior_unattributed": self.prior_unattributed,
             "by_model": dict(self.by_model)}, indent=1) + "\n")

    @property
    def total(self) -> float:
        return sum(m["cost"] for m in self.by_model.values()) + self.prior_unattributed

    def preflight(self, ceiling: float) -> None:
        with self.lock:
            if self.total + ceiling > self.cap:
                raise BudgetExhausted(
                    f"${self.total:.3f} spent; next request ceiling ${ceiling:.3f} "
                    f"could cross ${self.cap:.2f}")

    def add(self, slug: str, usage: dict, fallback_cost: float) -> None:
        cost = usage.get("cost")
        cost = float(cost) if cost is not None else fallback_cost
        with self.lock:
            m = self.by_model[slug]
            m["in"] += usage.get("prompt_tokens", 0)
            m["out"] += usage.get("completion_tokens", 0)
            m["cost"] += cost
            m["requests"] += 1
            self._save()


SPEND = Spend()
PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# OpenRouter client


def catalog() -> dict[str, dict]:
    req = urllib.request.Request(
        MODELS_API, headers={"User-Agent": UA, "HTTP-Referer": REFERER})
    with urllib.request.urlopen(req, timeout=60, context=_ctx) as resp:
        data = json.loads(resp.read())["data"]
    return {m["id"]: m for m in data}


def _ceiling(model: dict, messages: list[dict], max_tokens: int) -> float:
    est_in = len(json.dumps(messages)) / 3.5  # conservative chars->tokens
    est = est_in * model["price_in"] + max_tokens * model["price_out"]
    return max(1.5 * est, 0.02)


def chat(model: dict, messages: list[dict],
         max_tokens: int | None = None, task: str = "") -> tuple[str, str]:
    """One OpenRouter chat completion. Returns (content, finish_reason).
    Reserves budget, records exact usage.cost + per-request telemetry
    (evals/telemetry.py). Retries once on 429/5xx."""
    if max_tokens is None:
        max_tokens = model.get("max_tokens", DEFAULT_MAX_TOKENS)
    SPEND.preflight(_ceiling(model, messages, max_tokens))
    t0 = time.monotonic()
    body: dict = {
        "model": model["id"],
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if model.get("reasoning_capable"):
        body["reasoning"] = model.get("reasoning", {"effort": "low"})
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json",
                 "User-Agent": UA,
                 "HTTP-Referer": REFERER,
                 "X-Title": "fantasy-alpha frontier battery"})
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=_ctx) as resp:
                out = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            last_err = RuntimeError(f"HTTP {e.code}: {detail}")
            if e.code in (429, 500, 502, 503, 520, 524) and attempt == 0:
                time.sleep(15)
                continue
            raise last_err from None
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt == 0:
                time.sleep(10)
                continue
            raise
    usage = out.get("usage", {}) or {}
    fallback = (usage.get("prompt_tokens", 0) * model["price_in"]
                + usage.get("completion_tokens", 0) * model["price_out"])
    SPEND.add(model["slug"], usage, fallback)
    api_cost = usage.get("cost")
    telemetry_record(
        runner="run_frontier_battery", model=model["id"], task_family=task,
        latency_s=time.monotonic() - t0,
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("completion_tokens", 0),
        cost_usd=float(api_cost) if api_cost is not None else fallback,
        cost_source="api" if api_cost is not None else "computed")
    if out.get("error"):
        raise RuntimeError(f"provider error: {out['error']}")
    choice = out["choices"][0]
    content = choice["message"].get("content") or ""
    return content, choice.get("finish_reason") or ""


# ---------------------------------------------------------------------------
# failures ledger (shared, honest)

_FAIL_LOCK = threading.Lock()


def record_failure(**info) -> None:
    with _FAIL_LOCK:
        failures = (json.loads(FAILURES_PATH.read_text())
                    if FAILURES_PATH.exists() else [])
        failures.append(info)
        FAILURES_PATH.write_text(json.dumps(failures, indent=1) + "\n")


# ---------------------------------------------------------------------------
# battery request (shared by tiers 1-3): one (season, family) slate


def lenient_parse(text: str, valid_ids: set[str]) -> list[dict]:
    """Last-resort extraction when strict JSON parse fails twice (e.g. a
    missing comma mid-array): pull {"question_id": ..., "p": ...} pairs by
    regex. Same coverage bar (>=50%) as parse_answers."""
    pat = re.compile(r'"question_id"\s*:\s*"([^"]+)"\s*,\s*"p"\s*:\s*([0-9.eE+-]+)')
    answers, seen = [], set()
    for qid, p in pat.findall(text):
        if qid not in valid_ids or qid in seen:
            continue
        try:
            pv = float(p)
        except ValueError:
            continue
        answers.append({"question_id": qid, "p": min(max(pv, 0.0), 1.0)})
        seen.add(qid)
    if len(answers) / len(valid_ids) < 0.5:
        raise ValueError(f"lenient parse coverage {len(answers)}/{len(valid_ids)}")
    return answers


def answer_slate(model: dict, qpayload: dict, family: str, grounded: bool,
                 out_path: Path, tier: str, season: int) -> bool:
    """One slate request with the standard failure ladder. True on success."""
    if out_path.exists():
        return True
    messages, valid_ids = build_prompt(qpayload, family, grounded)
    task = f"{tier}:{family}"
    try:
        text, finish = chat(model, messages, task=task)
        try:
            answers = parse_answers(text, valid_ids)
        except Exception as e:  # one repair retry (bigger cap if truncated)
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"Invalid ({e}). Return ONLY the JSON "
                 "array, one object per candidate, covering every question_id."}]
            text, finish = chat(model, messages, max_tokens=REPAIR_MAX_TOKENS,
                                task=task + ":repair")
            try:
                answers = parse_answers(text, valid_ids)
            except Exception:
                answers = lenient_parse(text, valid_ids)
    except BudgetExhausted:
        raise
    except Exception as e:
        record_failure(model=model["id"], tier=tier, season=season,
                       family=family, error=str(e)[:500])
        log(f"  FAIL {model['slug']} {tier} {season} {family}: {str(e)[:200]}")
        return False
    out_path.write_text(json.dumps(answers, indent=1) + "\n")
    log(f"  {model['slug']} {tier} {season} {family}: "
        f"{len(answers)}/{len(valid_ids)} answered "
        f"(total ${SPEND.total:.2f})")
    return True


SLATE_WORKERS = 10  # concurrent slate requests per model


def _run_jobs(jobs: list[tuple]) -> None:
    """Run answer_slate jobs with a small pool; budget preflight is
    lock-guarded so the hard cap holds under concurrency."""
    with ThreadPoolExecutor(max_workers=SLATE_WORKERS) as pool:
        futs = [pool.submit(answer_slate, *j) for j in jobs]
        err = None
        for f in futs:
            try:
                f.result()
            except BudgetExhausted as e:
                err = e
        if err:
            raise err


def run_breakout_battery(model: dict, grounded: bool) -> None:
    tier = "harness" if grounded else "naked"
    rows = _family_rows(PRESET) if grounded else None
    ranks = _prior_ranks(PRESET) if grounded else None
    jobs = []
    for season, preset in season_formats(SEASONS):
        if preset != PRESET:
            continue
        qpayload = json.loads(
            (QUESTIONS_DIR / f"breakoutbench_{season}_{preset}.json").read_text())
        for family in FAMILIES:
            out_path = ANSWERS_DIR / (
                f"frontier_{model['slug']}_{tier}_{season}_{preset}_{family}.json")
            payload = qpayload
            if grounded:
                past = [r for r in rows.get(family, []) if r["season"] < season]
                payload = {**qpayload, "questions": [
                    {**q, "packet": {**q["packet"],
                                     "historical_grounding": grounding_for(
                                         family, q["packet"], past, ranks, season)}}
                    for q in qpayload["questions"]]}
            jobs.append((model, payload, family, grounded, out_path, tier, season))
    _run_jobs(jobs)


def run_canaries(model: dict) -> None:
    jobs = []
    for season, preset in season_formats(SEASONS):
        if preset != PRESET:
            continue
        out_path = ANSWERS_DIR / f"frontier_{model['slug']}_canary_{season}_{preset}.json"
        qpayload = json.loads(
            (QUESTIONS_DIR / f"canary_{season}_{preset}.json").read_text())
        jobs.append((model, qpayload, "full_slate", False, out_path, "canary", season))
    _run_jobs(jobs)


# ---------------------------------------------------------------------------
# fingerprint probe — canaries.fingerprint_probe generalized to our file names


def probe_for(slug: str) -> dict | None:
    fam_rows = _family_rows(PRESET)
    prior_ranks = _prior_ranks(PRESET)
    per_model: dict[int, dict] = {}
    per_ref: dict[int, dict] = {}
    for season, p in season_formats(SEASONS):
        if p != PRESET:
            continue
        apath = ANSWERS_DIR / f"frontier_{slug}_naked_{season}_{PRESET}_full_slate.json"
        kpath = QUESTIONS_DIR / f"canary_{season}_{PRESET}_KEY.json"
        if not apath.exists() or not kpath.exists():
            continue
        model_p = {str(a["question_id"]): float(a["p"])
                   for a in json.loads(apath.read_text())}
        orig = json.loads(
            (QUESTIONS_DIR / f"breakoutbench_{season}_{PRESET}.json").read_text())
        packets = {q["question_id"]: q["packet"] for q in orig["questions"]
                   if q["family"] == "full_slate"}
        past = [r for r in fam_rows.get("full_slate", []) if r["season"] < season]

        def ref_p(qid: str) -> float | None:
            g = grounding_for("full_slate", packets[qid], past, prior_ranks, season)
            for k in ("prior_rank_cohort_rate", "cohort_rate", "position_base_rate"):
                if g.get(k) is not None:
                    return g[k]
            return None

        key_payload = json.loads(kpath.read_text())
        pairs: dict[int, list[tuple[str, dict]]] = defaultdict(list)
        for qid, info in key_payload["questions"].items():
            if info["arm"] in ("control", "swapped"):
                pairs[info["pair_id"]].append((qid, info))
        m_credits, r_credits = [], []
        for pair_id in sorted(pairs):
            (qa, ia), (qb, ib) = pairs[pair_id]
            hi, lo = (qa, qb) if ia["outcome"] == 1 else (qb, qa)
            c = _pair_credit(model_p.get(hi), model_p.get(lo))
            if c is not None:
                m_credits.append(c)
            c = _pair_credit(ref_p(hi), ref_p(lo))
            if c is not None:
                r_credits.append(c)
        per_model[season] = {"control": m_credits, "swap_pf": []}
        per_ref[season] = {"control": r_credits, "swap_pf": []}
    if not per_model:
        return None
    out: dict = {"eras": {}}
    for era_name, era_seasons in ERAS:
        in_era = [s for s in per_model if s in era_seasons]
        if not in_era:
            continue
        sub_m = {s: per_model[s] for s in in_era}
        sub_r = {s: per_ref[s] for s in in_era}
        out["eras"][era_name] = {
            "model": _pool_stats(sub_m, in_era),
            "model_ci": bootstrap_index(sub_m, BOOTSTRAP_REPS, BOOTSTRAP_SEED),
            "reference": _pool_stats(sub_r, in_era),
            "reference_ci": bootstrap_index(sub_r, BOOTSTRAP_REPS, BOOTSTRAP_SEED),
        }
    return out


# ---------------------------------------------------------------------------
# DraftGym episodes (envs/play_llm.py driver, OpenRouter client)


def run_draftgym_episode(model: dict, spec: dict) -> dict:
    from envs.draftgym import DraftGym
    from envs.play_llm import autopick_action, build_messages, parse_action

    gym = DraftGym(**spec)
    obs = gym.reset()
    stats = {**spec, "model": model["id"], "model_requests": 0,
             "fallback_picks": 0, "parse_failures": 0, "illegal_picks": 0,
             "tool_calls": 0, "budget_truncated": False, "picks": []}
    reward, done, info = 0.0, False, {}
    requests_this_pick = 0
    while not done:
        action = None
        if requests_this_pick >= MAX_REQUESTS_PER_PICK:
            action = autopick_action(gym)
            stats["fallback_picks"] += 1
        else:
            messages = build_messages(obs)
            for attempt in range(2):
                try:
                    text, _finish = chat(model, messages, max_tokens=1500,
                                         task="draftgym")
                    stats["model_requests"] += 1
                    requests_this_pick += 1
                    action = parse_action(text)
                    break
                except BudgetExhausted:
                    stats["budget_truncated"] = True
                    break
                except ValueError as e:
                    stats["parse_failures"] += 1
                    if attempt == 0:
                        messages = messages + [
                            {"role": "assistant", "content": text},
                            {"role": "user", "content": f"Invalid ({e}). Reply with "
                             'ONLY one JSON action object: {"pick": "<player_id>"} '
                             'or {"tool": {"name": ..., "arguments": {...}}}.'}]
                except Exception as e:
                    record_failure(model=model["id"], tier="draftgym",
                                   season=spec["season"], family=str(spec),
                                   error=str(e)[:300])
                    break
            if action is None:
                action = autopick_action(gym)
                stats["fallback_picks"] += 1
        if "tool" in action:
            stats["tool_calls"] += 1
            obs, reward, done, info = gym.step(action)
            continue
        was_pick = obs["pick_number"]
        try:
            obs, reward, done, info = gym.step(action)
        except ValueError:
            stats["illegal_picks"] += 1
            stats["fallback_picks"] += 1
            obs, reward, done, info = gym.step(autopick_action(gym))
        stats["picks"].append({"pick_number": was_pick, "action": action})
        requests_this_pick = 0
    stats["reward"] = reward
    stats["agent_points_realistic"] = info.get("agent_points_realistic")
    log(f"  {model['slug']} draftgym {spec['season']} slot {spec['agent_slot']} "
        f"masked={bool(spec.get('mask_names'))}: reward {reward:+.1f} "
        f"(total ${SPEND.total:.2f})")
    return stats


def run_draftgym(model: dict) -> dict:
    existing = (json.loads(DRAFTGYM_JSON.read_text())
                if DRAFTGYM_JSON.exists() else {})
    if model["slug"] in existing:  # resume-safe
        return existing[model["slug"]]
    episodes = [run_draftgym_episode(model, s)
                for s in DRAFTGYM_SPECS + DRAFTGYM_MASKED]
    named = [e for e in episodes if not e.get("mask_names")]
    masked = [e for e in episodes if e.get("mask_names")]
    deltas = []
    for m in masked:
        twin = next((n for n in named
                     if (n["season"], n["agent_slot"], n["seed"]) ==
                        (m["season"], m["agent_slot"], m["seed"])), None)
        if twin:
            deltas.append(round(twin["reward"] - m["reward"], 2))
    result = {
        "model": model["id"],
        "named_rewards": [e["reward"] for e in named],
        "masked_rewards": [e["reward"] for e in masked],
        "named_minus_masked_paired_deltas": deltas,
        "fallback_picks": sum(e["fallback_picks"] for e in episodes),
        "illegal_picks": sum(e["illegal_picks"] for e in episodes),
        "tool_calls": sum(e["tool_calls"] for e in episodes),
        "episodes": episodes,
    }
    existing[model["slug"]] = result
    DRAFTGYM_JSON.write_text(json.dumps(existing, indent=1) + "\n")
    return result


# ---------------------------------------------------------------------------
# scoring + report


def score_glob(pattern: str) -> dict | None:
    paths = sorted(p for p in ANSWERS_DIR.glob(pattern) if "_v2_" not in p.name)
    if not paths:
        return None
    return score_run(paths)


def is_complete(r: dict | None) -> bool:
    """A run counts as complete only when it covers every season in the grid.
    Partial rows (from the killed pre-restriction run) keep zero-width or
    degenerate CIs and must stay out of the verdict machinery."""
    return r is not None and set(r.get("seasons", ())) >= set(SEASONS)


def leaderboard_row(label: str, r: dict | None, cost: str = "—") -> str:
    if r is None:
        return f"| {label} | — | — | — | — | {cost} |"
    if not is_complete(r):
        ss = r.get("seasons", [])
        label += (f" (PARTIAL: {len(ss)} season{'s' if len(ss) != 1 else ''}, "
                  "pre-restriction run — not comparable, excluded from verdicts)")
    p, ci = r["pooled"], r["ci"]
    fs = p["families"]["full_slate"]
    fs_cell = (f"{fs['brier']:.4f} {fmt_ci(ci.get('brier_full_slate'))}"
               if fs["n"] else "—")
    return (f"| {label} | {fs_cell} | {p['pooled_brier']:.4f} "
            f"{fmt_ci(ci.get('pooled_brier'))} | {p['lift']:.2f}x "
            f"{fmt_ci(ci.get('lift'), 2)} | {r['answered']}/{r['n_total']} | {cost} |")


def sig(a, b) -> str:
    if a is None or b is None:
        return "n/a"
    return "no (CIs overlap)" if overlap(a, b) else "YES (CIs separate)"


def write_report(scored: dict, canary: dict, probes: dict, draftgym: dict,
                 prices: dict, budget_stopped: list[str]) -> Path:
    base = scored.get("base_rate")
    lines = [
        "# Frontier battery — BreakoutBench anonymized track via OpenRouter",
        "",
        f"Run date {date.today().isoformat()}. Protocol identical to the Qwen3.5-9B "
        "expanded battery (evals/run_expanded_qwen.py): ppr grid, seasons 2015-2024, "
        "3 families per season, one request per (season, family), temp 0, "
        "reasoning/thinking DISABLED (the Qwen3.5-9B baseline protocol — "
        "`enable_thinking=false`; a reasoning-effort-low pilot thought for 5-15k "
        "tokens and 8-10 min per slate and was discarded). Exception: Fable-5 "
        "rejects disabled reasoning (mandatory on that endpoint) and runs at "
        "effort `minimal`, which emitted 0 reasoning tokens in the pilot. "
        "max_tokens 3000 (repair "
        "retry 16000), one repair retry per malformed response. 95% CIs bootstrap over "
        "seasons (10,000 resamples, seeded). Cost is OpenRouter's exact per-response "
        "`usage.cost`. 2025 remains the untouched holdout.",
        "",
        "## Contenders (exact ids + live catalog prices, $/M tokens)",
        "",
        "| model | id | in | out | open weights |",
        "|---|---|---|---|---|",
    ]
    for m in MODELS:
        pr = prices.get(m["id"], {})
        lines.append(f"| {m['label']} | `{m['id']}` | {pr.get('in', '—')} | "
                     f"{pr.get('out', '—')} | {'yes' if m['open'] else 'no'} |")
    lines += [
        "",
        "## Leaderboard — anonymized ppr track (2015-2024, 1,849 judgments/run)",
        "",
        "Existing rows re-scored on the identical ppr-only subset for exact "
        "comparability (the published expanded_baseline.md rows pool all three "
        "formats). GBDT row from evals/results/gbdt_baseline.md (ppr full_slate "
        "walk-forward; no bust/over_under judgments, so no pooled Brier).",
        "",
        "| run | full_slate Brier [95% CI] | pooled Brier [95% CI] | "
        "top-10 lift [95% CI] | answered | cost |",
        "|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        c = SPEND.by_model.get(m["slug"], {}).get("cost")
        lines.append(leaderboard_row(
            f"{m['label']} naked", scored.get(f"{m['slug']}_naked"),
            f"${c:.2f} total" if c else "—"))
    for m in MODELS:
        lines.append(leaderboard_row(
            f"{m['label']} + harness", scored.get(f"{m['slug']}_harness")))
    lines += [
        leaderboard_row("Qwen3.5-9B naked (ppr subset)", scored.get("qwen9b_naked"),
                        "$0.42 (all-format run)"),
        leaderboard_row("Qwen3.5-9B + harness (ppr subset)", scored.get("qwen9b_harness")),
        "| GBDT walk-forward (boring-ML bar) | 0.1270 [0.103, 0.151] | — | "
        "1.02x [0.77, 1.30] | 539 full_slate | ~$0 |",
        leaderboard_row("base rate (no-peek)", scored.get("base_rate")),
        leaderboard_row("coin p=0.5 / random selection", scored.get("coin")),
        "",
        "Cost column: per-model total across ALL tiers (tier 1 + canary + harness "
        "+ DraftGym where run) — exact OpenRouter usage.cost sums.",
        "",
        "## Canary gate + fingerprint probe (memorization read)",
        "",
        "Gate: FAIL iff pooled swap-index 95% CI lower bound > "
        f"{GATE_INDEX_CI_LOW} (identity-channel leak). Fingerprint excess = "
        "model-minus-code-reference within-pair accuracy on unswapped ADP-adjacent "
        "pairs, per era; recent-era-only excess is the fingerprint-recall signature.",
        "",
        "| model | gate | swap index [95% CI] | fp excess 2015-19 | fp excess 2020-24 |",
        "|---|---|---|---|---|",
    ]
    for m in MODELS:
        c = canary.get(m["slug"])
        pr = probes.get(m["slug"])
        if c is None:
            lines.append(f"| {m['label']} | not run | — | — | — |")
            continue
        g = c["gate"]
        exc = {}
        if pr:
            for era, er in pr["eras"].items():
                mm, rr = er["model"]["a_control"], er["reference"]["a_control"]
                if mm is not None and rr is not None:
                    exc[era] = mm - rr
        idx = c["pooled"]["index"]
        ici = c["ci"].get("index")
        lines.append(
            f"| {m['label']} | {'PASS' if g['pass'] else '**FAIL**'} | "
            f"{idx:+.3f} {fmt_ci(ici)} | "
            f"{exc.get('2015-2019', float('nan')):+.3f} | "
            f"{exc.get('2020-2024', float('nan')):+.3f} |"
            if idx is not None else
            f"| {m['label']} | {'PASS' if g['pass'] else '**FAIL**'} | — | — | — |")
    lines += [
        "",
        "Qwen3.5-9B reference (evals/results/canaries.md): gate PASS, index "
        "-0.145 [-0.390, 0.081], fp excess +0.000 / -0.085.",
        "",
        "## DraftGym 2021 — named vs masked (memorization delta)",
        "",
        "2 named + 2 masked episodes (slots 1 and 7, seed 0, 12-team ppr, 15 "
        "rounds), identical configs; delta = named reward − masked reward on the "
        "paired config. Qwen3.5-9B reference deltas (14-episode baseline): "
        "+403.9, +139.6.",
        "",
        "| model | named rewards | masked rewards | paired deltas | fallbacks | tools |",
        "|---|---|---|---|---|---|",
    ]
    if draftgym:
        for slug, d in draftgym.items():
            label = next((m["label"] for m in MODELS if m["slug"] == slug), slug)
            lines.append(
                f"| {label} | {', '.join(f'{r:+.1f}' for r in d['named_rewards'])} | "
                f"{', '.join(f'{r:+.1f}' for r in d['masked_rewards'])} | "
                f"{', '.join(f'{x:+.1f}' for x in d['named_minus_masked_paired_deltas'])} | "
                f"{d['fallback_picks']} | {d['tool_calls']} |")
    else:
        lines.append("| (tier 4 not reached) | — | — | — | — | — |")
    lines += ["", "## Per-model cost (exact OpenRouter usage.cost)", "",
              "| model | requests | tokens in/out | cost |", "|---|---|---|---|"]
    for m in MODELS:
        s = SPEND.by_model.get(m["slug"])
        if s:
            lines.append(f"| {m['label']} | {s['requests']} | "
                         f"{s['in']:,} / {s['out']:,} | ${s['cost']:.3f} |")
    if SPEND.prior_unattributed:
        lines.append(f"| pre-restriction burn (8-model tier-1 partial, killed on "
                     f"scope change; OpenRouter credits-endpoint delta, not "
                     f"attributable per model) | ~29 slates | — | "
                     f"${SPEND.prior_unattributed:.3f} |")
    lines.append(f"| **total** |  |  | **${SPEND.total:.3f}** (cap ${HARD_BUDGET:.2f}) |")
    if budget_stopped:
        lines += ["", f"Budget-stopped tiers: {', '.join(budget_stopped)}."]

    # ---- verdicts -----------------------------------------------------
    lines += ["", "## Verdicts", ""]
    base_lift = base["ci"].get("lift") if base else None
    base_brier = base["ci"].get("pooled_brier") if base else None
    beat_lift = []
    beat_brier = []
    briers = {}
    for m in MODELS:
        for tier in ("naked", "harness"):
            r = scored.get(f"{m['slug']}_{tier}")
            if not is_complete(r):
                continue
            briers[f"{m['label']} {tier}"] = (r["pooled"]["pooled_brier"],
                                              r["ci"].get("pooled_brier"))
            lci = r["ci"].get("lift")
            if lci and base_lift and not overlap(lci, base_lift) and \
                    r["pooled"]["lift"] > base["pooled"]["lift"]:
                beat_lift.append(f"{m['label']} {tier} ({r['pooled']['lift']:.2f}x "
                                 f"{fmt_ci(lci, 2)})")
            bci = r["ci"].get("pooled_brier")
            if bci and base_brier and not overlap(bci, base_brier) and \
                    r["pooled"]["pooled_brier"] < base["pooled"]["pooled_brier"]:
                beat_brier.append(f"{m['label']} {tier}")
    if beat_lift or beat_brier:
        lines.append(
            "- **(a) Benchmark headroom:** YES — "
            + (f"significant top-10 lift over base-rate/random: {'; '.join(beat_lift)}. "
               if beat_lift else "")
            + (f"Significant pooled-Brier beat over base rate: {'; '.join(beat_brier)}."
               if beat_brier else ""))
    else:
        lines.append(
            "- **(a) Benchmark headroom:** NO frontier model beats the no-peek "
            "base-rate floor with statistical significance on anonymized selection "
            "(all top-10 lift and pooled-Brier CIs overlap the base-rate row). The "
            "anonymized track continues to look information-ceilinged, not "
            "capability-ceilinged.")
    # (b) memorization
    mem_bits = []
    for m in MODELS:
        c = canary.get(m["slug"])
        pr = probes.get(m["slug"])
        if not c:
            continue
        gate_fail = not c["gate"]["pass"]
        rec = None
        if pr and "2020-2024" in pr["eras"]:
            er = pr["eras"]["2020-2024"]
            mm, rr = er["model"]["a_control"], er["reference"]["a_control"]
            rec = (mm - rr) if (mm is not None and rr is not None) else None
        if gate_fail or (rec is not None and rec > 0.10):
            mem_bits.append(f"{m['label']} (gate {'FAIL' if gate_fail else 'pass'}, "
                            f"recent-era fp excess {rec:+.3f})" if rec is not None
                            else f"{m['label']} (gate FAIL)")
    dg_bits = []
    if draftgym:
        for slug, d in draftgym.items():
            label = next((m["label"] for m in MODELS if m["slug"] == slug), slug)
            if d["named_minus_masked_paired_deltas"]:
                mean_d = sum(d["named_minus_masked_paired_deltas"]) / \
                    len(d["named_minus_masked_paired_deltas"])
                dg_bits.append(f"{label} mean delta {mean_d:+.1f}")
    lines.append(
        "- **(b) Frontier memorization vs 9B Qwen:** "
        + (f"stronger signatures in: {'; '.join(mem_bits)}. " if mem_bits else
           "no frontier model fails the canary gate or shows a large recent-era "
           "fingerprint excess — same verdict class as 9B Qwen. ")
        + (f"DraftGym named-minus-masked (Qwen9B ref +403.9/+139.6): "
           f"{'; '.join(dg_bits)}." if dg_bits else ""))
    if briers:
        best = min(briers.items(), key=lambda kv: kv[1][0])
        lines.append(
            f"- **(c) Competitive calibration bar:** best pooled Brier is "
            f"{best[0]} at {best[1][0]:.4f} {fmt_ci(best[1][1])}; the no-peek "
            f"base rate sits at {base['pooled']['pooled_brier']:.4f} "
            f"{fmt_ci(base_brier)} and GBDT full-slate at 0.1270 [0.103, 0.151]. "
            "The product bar: match base-rate calibration everywhere and add "
            "selection lift on top — nothing here moves that bar materially.")
    q397 = scored.get("qwen397b_naked")
    q35 = scored.get("qwen35b_naked")
    q9 = scored.get("qwen9b_naked")
    if not is_complete(q397):
        q397 = None  # partial pre-restriction row — not a valid probe point
    if is_complete(q35) and is_complete(q9):
        a35, b9 = q35["ci"].get("pooled_brier"), q9["ci"].get("pooled_brier")
        l35, l9 = q35["ci"].get("lift"), q9["ci"].get("lift")
        three_way = (
            f"Qwen scale probe (same prompts, same slates, thinking off) — 9B: "
            f"Brier {q9['pooled']['pooled_brier']:.4f} {fmt_ci(b9)}, lift "
            f"{q9['pooled']['lift']:.2f}x {fmt_ci(l9, 2)}; 35B-A3B: "
            f"{q35['pooled']['pooled_brier']:.4f} {fmt_ci(a35)}, lift "
            f"{q35['pooled']['lift']:.2f}x {fmt_ci(l35, 2)}"
            + (f"; 397B-A17B: {q397['pooled']['pooled_brier']:.4f} "
               f"{fmt_ci(q397['ci'].get('pooled_brier'))}, lift "
               f"{q397['pooled']['lift']:.2f}x {fmt_ci(q397['ci'].get('lift'), 2)}."
               if q397 else
               "; 397B-A17B not yet run to completion (1 pre-restriction season "
               "only — no valid probe point).")
        )
        same = (a35 and b9 and overlap(a35, b9) and l35 and l9 and overlap(l35, l9))
        lines.append(
            f"- **(d) Does scale alone buy selection skill?** {three_way} "
            + ("The 35B-vs-9B contrast does not separate (all CIs overlap): "
               "within the Qwen family, ~4x active-parameter scale buys nothing "
               "on this track — consistent with an information ceiling rather "
               "than a capability ceiling. The 397B point is still needed for "
               "the full scale read."
               if same else
               "the 35B-vs-9B contrast RESOLVES — see CIs above."))
    lines.append("")
    RESULTS_MD.write_text("\n".join(lines))
    return RESULTS_MD


def append_ledger() -> None:
    ledger = ROOT / "docs" / "budget-ledger.md"
    text = ledger.read_text()
    rows = []
    if SPEND.prior_unattributed and "pre-restriction burn" not in text:
        rows.append(
            f"| {date.today().isoformat()} | Frontier battery pre-restriction burn "
            "— 8-model tier-1 partial via OpenRouter, killed when Jacob narrowed "
            "scope to dsv4flash+qwen35b (~29 slates + sanity pings; "
            "credits-endpoint delta, per-model split unavailable without a "
            "management key) | OpenRouter usage.cost | ~29 requests | "
            f"${SPEND.prior_unattributed:.3f} (API-reported) |")
    for m in MODELS:
        s = SPEND.by_model.get(m["slug"])
        if not s or not s["requests"]:
            continue
        if f"`{m['id']}`" in text:
            continue  # already ledgered by an earlier invocation
        rows.append(
            f"| {date.today().isoformat()} | Frontier battery via OpenRouter — "
            f"`{m['id']}`: naked + canary + harness ppr batteries"
            + (" + 4 DraftGym episodes" if m["slug"] in
               (json.loads(DRAFTGYM_JSON.read_text()).keys()
                if DRAFTGYM_JSON.exists() else []) else "")
            + f" (evals/run_frontier_battery.py) | OpenRouter usage.cost | "
            f"{s['in']:,} in / {s['out']:,} out tok, {s['requests']} req | "
            f"${s['cost']:.3f} (API-reported) |")
    if not rows:
        return
    m = re.search(r"Total to date: ~\$([\d.]+)", text)
    new_total = float(m.group(1)) + SPEND.total
    marker = "\nTotal to date:"
    text = text.replace(marker, "\n" + "\n".join(rows) + "\n" + marker, 1)
    text = re.sub(r"Total to date: ~\$[\d.]+", f"Total to date: ~${new_total:.3f}", text)
    ledger.write_text(text)


# ---------------------------------------------------------------------------
# main


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Frontier battery via OpenRouter")
    ap.add_argument("--models", default=None,
                    help="comma-separated slugs to RUN (others still scored "
                         "from any answers already on disk)")
    ap.add_argument("--no-draftgym", action="store_true",
                    help="skip tier 4 entirely")
    args = ap.parse_args()
    active_slugs = (set(args.models.split(",")) if args.models
                    else {m["slug"] for m in MODELS})
    unknown = active_slugs - {m["slug"] for m in MODELS}
    if unknown:
        ap.error(f"unknown slugs: {unknown}")
    active = [m for m in MODELS if m["slug"] in active_slugs]

    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    cat = catalog()
    prices = {}
    for m in MODELS:
        entry = cat.get(m["id"])
        if entry is None:
            raise SystemExit(f"model id {m['id']} not in OpenRouter catalog")
        p = entry["pricing"]
        m["price_in"] = float(p["prompt"])
        m["price_out"] = float(p["completion"])
        m["reasoning_capable"] = "reasoning" in entry.get("supported_parameters", [])
        prices[m["id"]] = {"in": f"${m['price_in'] * 1e6:.3f}",
                           "out": f"${m['price_out'] * 1e6:.3f}"}
        log(f"contender {m['id']}: ${m['price_in']*1e6:.3f}/{m['price_out']*1e6:.3f} "
            f"per M, reasoning={m['reasoning_capable']}")

    if not (QUESTIONS_DIR / f"canary_{SEASONS[0]}_{PRESET}.json").exists():
        build_canaries(SEASONS)

    budget_stopped: list[str] = []

    def guarded(fn, *args, tier: str) -> None:
        try:
            fn(*args)
        except BudgetExhausted as e:
            budget_stopped.append(tier)
            log(f"BUDGET STOP in {tier}: {e}")
        except Exception as e:  # provider outage etc. — keep the battery alive
            record_failure(tier=tier, error=str(e)[:500],
                           model=args[0]["id"] if args else "?")
            log(f"TIER ERROR {tier}: {str(e)[:300]}")

    # Tier 1: naked batteries + Tier 2: canaries (parallel across models)
    with ThreadPoolExecutor(max_workers=max(1, len(active))) as pool:
        futs = []
        for m in active:
            def tier12(model=m):
                guarded(run_breakout_battery, model, False,
                        tier=f"{model['slug']}:naked")
                guarded(run_canaries, model, tier=f"{model['slug']}:canary")
            futs.append(pool.submit(tier12))
        for f in futs:
            f.result()
    log(f"tiers 1-2 done: ${SPEND.total:.2f} spent")

    # Tier 3: harness batteries for the active contenders
    with ThreadPoolExecutor(max_workers=max(1, len(active))) as pool:
        futs = [pool.submit(guarded, run_breakout_battery, m, True,
                            tier=f"{m['slug']}:harness") for m in active]
        for f in futs:
            f.result()
    log(f"tier 3 done: ${SPEND.total:.2f} spent")

    # score tiers 1+3
    scored: dict[str, dict | None] = {}
    for m in MODELS:
        scored[f"{m['slug']}_naked"] = score_glob(f"frontier_{m['slug']}_naked_*_ppr_*.json")
        scored[f"{m['slug']}_harness"] = score_glob(f"frontier_{m['slug']}_harness_*_ppr_*.json")
    scored["qwen9b_naked"] = score_glob("qwen_naked_*_ppr_*.json")
    scored["qwen9b_harness"] = score_glob("qwen_harness_*_ppr_*.json")
    scored["base_rate"] = score_glob("base_rate_*_ppr.json")
    scored["coin"] = score_glob("coin_*_ppr.json")

    # canary scoring + probes
    canary: dict[str, dict] = {}
    probes: dict[str, dict] = {}
    for m in MODELS:
        paths = sorted(ANSWERS_DIR.glob(f"frontier_{m['slug']}_canary_*_ppr.json"))
        if paths:
            canary[m["slug"]] = score_canaries(load_answers_files(paths))
        pr = probe_for(m["slug"])
        if pr:
            probes[m["slug"]] = pr

    # Tier 4: DraftGym for top-2 tier-1 models (by pooled Brier) + big Qwen
    remaining = HARD_BUDGET - SPEND.total
    draftgym: dict[str, dict] = (json.loads(DRAFTGYM_JSON.read_text())
                                 if DRAFTGYM_JSON.exists() else {})
    if args.no_draftgym:
        budget_stopped.append("draftgym (skipped by --no-draftgym)")
    elif remaining >= 8.0:
        ranked = sorted(
            (m for m in active if scored.get(f"{m['slug']}_naked")),
            key=lambda m: scored[f"{m['slug']}_naked"]["pooled"]["pooled_brier"])
        chosen = ranked[:2]
        big_qwen = next((m for m in active if m["slug"] == "qwen397b"), None)
        if big_qwen and big_qwen not in chosen:
            chosen.append(big_qwen)
        log(f"tier 4 contenders: {[m['slug'] for m in chosen]} "
            f"(${remaining:.2f} remaining)")
        for m in chosen:
            try:
                draftgym[m["slug"]] = run_draftgym(m)
            except BudgetExhausted as e:
                budget_stopped.append(f"{m['slug']}:draftgym")
                log(f"BUDGET STOP in draftgym: {e}")
                break
            except Exception as e:
                record_failure(tier="draftgym", model=m["id"], error=str(e)[:500])
                log(f"DRAFTGYM ERROR {m['slug']}: {str(e)[:300]}")
    else:
        budget_stopped.append("draftgym (under $8 remaining)")

    out = write_report(scored, canary, probes, draftgym, prices, budget_stopped)
    append_ledger()
    log(f"\nwrote {out}\ntotal spend ${SPEND.total:.3f} of ${HARD_BUDGET:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
