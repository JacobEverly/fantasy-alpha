"""evals/report_dashboard.py — checkpoint-log schema + artifact generation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.report_dashboard import (  # noqa: E402
    CHECKPOINTS_PATH,
    LEADERBOARD,
    METRIC_META,
    cost_per_1k,
    generate,
    load_checkpoints,
    load_spend,
    validate_checkpoint_row,
)

GOOD_ROW = {
    "run_id": "b3-sft-v0", "phase": "B3", "step": 500,
    "tokens_trained": 1_000_000, "cost_cumulative": 12.5,
    "metrics": {"pooled_brier": 0.19, "draftgym_masked_reward": 10.0,
                "league_index": 88.0},
    "timestamp": "2026-08-09T00:00:00+00:00",
}


# ---------------- checkpoint-log schema ----------------


def test_good_checkpoint_row_validates():
    assert validate_checkpoint_row(GOOD_ROW) == []


def test_checkpoint_row_missing_fields_caught():
    for field in ("run_id", "phase", "step", "tokens_trained",
                  "cost_cumulative", "metrics", "timestamp"):
        bad = {k: v for k, v in GOOD_ROW.items() if k != field}
        assert validate_checkpoint_row(bad), f"missing {field} not caught"


def test_checkpoint_row_bad_values_caught():
    assert validate_checkpoint_row({**GOOD_ROW, "phase": "B7"})
    assert validate_checkpoint_row({**GOOD_ROW, "step": -1})
    assert validate_checkpoint_row({**GOOD_ROW, "step": 1.5})
    assert validate_checkpoint_row({**GOOD_ROW, "tokens_trained": -5})
    assert validate_checkpoint_row({**GOOD_ROW, "cost_cumulative": -0.1})
    assert validate_checkpoint_row({**GOOD_ROW, "metrics": [1, 2]})
    assert validate_checkpoint_row(
        {**GOOD_ROW, "metrics": {"pooled_brier": "low"}})


def test_checkpoint_metrics_allow_null():
    row = {**GOOD_ROW, "metrics": {"pooled_brier": None, "league_index": 80.0}}
    assert validate_checkpoint_row(row) == []


def test_committed_checkpoint_log_is_valid():
    """The seed rows in training/checkpoints_log.jsonl pass the schema and
    carry the two real B2-proxy points at step 0."""
    rows = load_checkpoints(CHECKPOINTS_PATH)
    assert len(rows) >= 2
    run_ids = {r["run_id"] for r in rows}
    assert {"b2-proxy-qwen9b-naked", "b2-proxy-qwen9b-harness"} <= run_ids
    step0 = [r for r in rows if r["step"] == 0]
    assert all(r["phase"] == "B2" for r in step0)
    naked = next(r for r in rows if r["run_id"] == "b2-proxy-qwen9b-naked")
    assert naked["metrics"]["pooled_brier"] == pytest.approx(0.1992)


def test_load_checkpoints_rejects_bad_line(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    p.write_text(json.dumps({**GOOD_ROW, "phase": "nope"}) + "\n")
    with pytest.raises(ValueError):
        load_checkpoints(p)


# ---------------- cost backfill ----------------


def test_cost_per_1k_from_real_spend():
    spend = load_spend()
    fable = cost_per_1k("fable5", spend)
    flash = cost_per_1k("dsv4flash", spend)
    # exact backfill from frontier_spend.json / judgment counts
    assert fable == pytest.approx(13.7325 / 3694 * 1000, rel=1e-3)
    assert flash == pytest.approx(0.13316 / 3698 * 1000, rel=1e-3)
    # the quality-vs-cost headline: ~two orders of magnitude apart
    assert 50 < fable / flash < 200
    assert cost_per_1k("base_rate", spend) == 0.0


# ---------------- artifact generation smoke ----------------


def test_generate_renders_both_artifacts(tmp_path):
    md, html = generate(out_md=tmp_path / "cmp.md",
                        out_html=tmp_path / "dash.html",
                        telemetry_path=tmp_path / "empty_telemetry.jsonl")
    text = md.read_text()
    page = html.read_text()
    # markdown: table with every leaderboard row + honest latency marker
    assert "| model | config |" in text
    assert "n/a (pre-instrumentation)" in text
    for r in LEADERBOARD:
        assert r["label"] in text
    assert "cost / 1,000 judgments" in text
    # html: both views, self-contained inline SVG, design tokens, both themes
    assert "<!doctype html>" in page
    assert "Quality vs cost" in page and "Training progress" in page
    assert page.count("<svg") >= 1 + len(METRIC_META) - 1  # scatter + charts
    assert "base-rate par" in page
    assert "b2-proxy-qwen9b-naked" in page
    assert "#10B981" in page and "IBM Plex Mono" in page
    assert "prefers-color-scheme" in page and 'data-theme="light"' in page
    assert "tablewrap" in page and "overflow-x: auto" in page
    # no external assets
    for needle in ("http://", "https://", "src=", "@import"):
        assert needle not in page, f"external reference found: {needle}"


def test_generate_uses_telemetry_when_present(tmp_path):
    tele = tmp_path / "tele.jsonl"
    from evals.telemetry import record
    record(runner="run_frontier_battery", model="anthropic/claude-fable-5",
           task_family="naked:full_slate", latency_s=12.5,
           tokens_in=100, tokens_out=10, cost_usd=0.05, path=tele)
    md, _ = generate(out_md=tmp_path / "cmp.md", out_html=tmp_path / "d.html",
                     telemetry_path=tele)
    assert "12.5 s mean (1 req)" in md.read_text()
