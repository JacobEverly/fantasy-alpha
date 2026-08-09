"""evals/telemetry.py — per-request timing+cost recorder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.telemetry import (  # noqa: E402
    REQUIRED_FIELDS,
    load,
    record,
    summarize,
    validate_row,
)


def _rec(tmp_path, **kw):
    defaults = dict(runner="test", model="test/model", task_family="naked:bust",
                    latency_s=1.25, tokens_in=100, tokens_out=20,
                    cost_usd=0.001, cost_source="api",
                    path=tmp_path / "t.jsonl")
    defaults.update(kw)
    return record(**defaults)


def test_record_appends_jsonl(tmp_path):
    p = tmp_path / "t.jsonl"
    _rec(tmp_path)
    _rec(tmp_path, model="other/model", latency_s=2.0)
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["model"] == "test/model"
    assert row["latency_s"] == 1.25
    assert row["cost_usd"] == 0.001
    assert row["cost_source"] == "api"


def test_record_rows_pass_schema(tmp_path):
    _rec(tmp_path)
    _rec(tmp_path, cost_usd=None, cost_source=None)  # cost unknown is legal
    _rec(tmp_path, cost_usd=0.5, cost_source="computed")
    for row in load(tmp_path / "t.jsonl"):
        assert validate_row(row) == [], row


def test_record_defaults_cost_source_to_api(tmp_path):
    row = _rec(tmp_path, cost_usd=0.002, cost_source=None)
    assert row["cost_source"] == "api"


def test_validate_row_catches_problems():
    good = _row_template()
    assert validate_row(good) == []
    for field in REQUIRED_FIELDS:
        bad = dict(good)
        del bad[field]
        assert validate_row(bad), f"missing {field} not caught"
    assert validate_row({**good, "latency_s": -1})
    assert validate_row({**good, "tokens_in": "many"})
    assert validate_row({**good, "cost_source": "guess"})
    # a cost without a source is not allowed
    assert validate_row({**good, "cost_usd": 0.1, "cost_source": None})


def _row_template() -> dict:
    return {"ts": "2026-08-09T00:00:00+00:00", "runner": "r", "model": "m",
            "task_family": "f", "latency_s": 1.0, "tokens_in": 1,
            "tokens_out": 1, "cost_usd": None, "cost_source": None}


def test_load_missing_file_is_empty(tmp_path):
    assert load(tmp_path / "nope.jsonl") == []


def test_summarize_per_model(tmp_path):
    _rec(tmp_path, latency_s=1.0, cost_usd=0.01)
    _rec(tmp_path, latency_s=3.0, cost_usd=0.03)
    _rec(tmp_path, model="other/model", latency_s=2.0, cost_usd=None,
         cost_source=None)
    s = summarize(load(tmp_path / "t.jsonl"))
    assert s["test/model"]["n"] == 2
    assert s["test/model"]["latency_mean_s"] == 2.0
    assert abs(s["test/model"]["cost_usd"] - 0.04) < 1e-9
    assert s["other/model"]["cost_known_n"] == 0


def test_runners_are_instrumented():
    """Both eval runners import and call the telemetry recorder."""
    for name in ("run_frontier_battery.py", "run_expanded_qwen.py"):
        src = (ROOT / "evals" / name).read_text()
        assert "telemetry" in src and "telemetry_record(" in src, name
