#!/usr/bin/env python3
"""Per-request timing + cost telemetry for the eval runners.

Every LLM request an eval runner makes gets one JSONL row appended to
data/processed/telemetry/eval_requests.jsonl (append-only; the directory is
created on first write). This is the raw feed for the two standard
benchmark-presentation axes we could not backfill: time-to-task (wall-clock
latency) and cost-per-task, per model, per task family.

Row schema (one JSON object per line):

    ts            ISO-8601 UTC timestamp of the request completion
    runner        which script made the request (e.g. "run_frontier_battery")
    model         provider model id (e.g. "anthropic/claude-fable-5")
    task_family   what the request was for (e.g. "naked:full_slate",
                  "harness:bust", "canary:full_slate", "draftgym")
    latency_s     wall-clock seconds for the request (float, includes retries
                  only if the caller times them together)
    tokens_in     prompt tokens (API-reported; 0 if unavailable)
    tokens_out    completion tokens (API-reported; 0 if unavailable)
    cost_usd      cost in USD, or null when not knowable
    cost_source   "api" (provider-reported, e.g. OpenRouter usage.cost),
                  "computed" (tokens x known catalog rate), or null
    extra         optional dict for runner-specific context (season, tier...)

Usage in a runner loop (minimal diff pattern):

    from evals.telemetry import record
    t0 = time.monotonic()
    ... make the request ...
    record(runner="run_x", model=model_id, task_family=f"{tier}:{family}",
           latency_s=time.monotonic() - t0,
           tokens_in=usage.get("prompt_tokens", 0),
           tokens_out=usage.get("completion_tokens", 0),
           cost_usd=usage.get("cost"), cost_source="api")

Telemetry must never kill an eval run: record() swallows I/O errors after
printing a one-line warning.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TELEMETRY_PATH = ROOT / "data" / "processed" / "telemetry" / "eval_requests.jsonl"

REQUIRED_FIELDS = ("ts", "runner", "model", "task_family", "latency_s",
                   "tokens_in", "tokens_out", "cost_usd", "cost_source")

_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(runner: str, model: str, task_family: str, latency_s: float,
           tokens_in: int = 0, tokens_out: int = 0,
           cost_usd: float | None = None, cost_source: str | None = None,
           extra: dict | None = None, path: Path | None = None) -> dict:
    """Append one telemetry row. Returns the row dict (also on I/O failure)."""
    if cost_usd is not None:
        cost_usd = float(cost_usd)
        if cost_source is None:
            cost_source = "api"
    row = {
        "ts": _now_iso(),
        "runner": runner,
        "model": model,
        "task_family": task_family,
        "latency_s": round(float(latency_s), 3),
        "tokens_in": int(tokens_in or 0),
        "tokens_out": int(tokens_out or 0),
        "cost_usd": cost_usd,
        "cost_source": cost_source,
    }
    if extra:
        row["extra"] = extra
    target = path or TELEMETRY_PATH
    try:
        with _LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
    except OSError as e:  # telemetry must never kill an eval run
        print(f"telemetry: write failed ({e}); row dropped", flush=True)
    return row


def load(path: Path | None = None) -> list[dict]:
    """Read all telemetry rows (skipping blank lines)."""
    target = path or TELEMETRY_PATH
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text().splitlines() if line.strip()]


def validate_row(row: dict) -> list[str]:
    """Return a list of schema problems ([] = valid)."""
    problems = [f"missing field: {f}" for f in REQUIRED_FIELDS if f not in row]
    if problems:
        return problems
    if not isinstance(row["latency_s"], (int, float)) or row["latency_s"] < 0:
        problems.append("latency_s must be a non-negative number")
    for f in ("tokens_in", "tokens_out"):
        if not isinstance(row[f], int) or row[f] < 0:
            problems.append(f"{f} must be a non-negative int")
    if row["cost_usd"] is not None and not isinstance(row["cost_usd"], (int, float)):
        problems.append("cost_usd must be a number or null")
    if row["cost_source"] not in (None, "api", "computed"):
        problems.append("cost_source must be 'api', 'computed', or null")
    if row["cost_usd"] is not None and row["cost_source"] is None:
        problems.append("cost_source required when cost_usd is set")
    return problems


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Per-model aggregates: n, mean latency, total tokens, total cost."""
    out: dict[str, dict] = {}
    for r in rows:
        m = out.setdefault(r["model"], {"n": 0, "latency_total": 0.0,
                                        "tokens_in": 0, "tokens_out": 0,
                                        "cost_usd": 0.0, "cost_known_n": 0})
        m["n"] += 1
        m["latency_total"] += r["latency_s"]
        m["tokens_in"] += r["tokens_in"]
        m["tokens_out"] += r["tokens_out"]
        if r["cost_usd"] is not None:
            m["cost_usd"] += r["cost_usd"]
            m["cost_known_n"] += 1
    for m in out.values():
        m["latency_mean_s"] = round(m["latency_total"] / m["n"], 3)
    return out
