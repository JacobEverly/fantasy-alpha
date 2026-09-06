import json
from decimal import Decimal
from pathlib import Path

from training.tinker_backend import ROOT, sha256_file


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_leakage_report_matches_materialized_split_boundaries():
    report = json.loads(
        (ROOT / "training/datasets/tool_decision_v1/leakage-report.json").read_text()
    )
    rows = {
        "train": _rows(ROOT / "training/datasets/tool_decision_v1/train.jsonl"),
        "development": _rows(
            ROOT / "training/datasets/tool_decision_v1/development.jsonl"
        ),
        "heldout": _rows(ROOT / "evals/frozen/tool_decision_v1/heldout.jsonl"),
    }
    assert report["status"] == "pass"
    assert sum(map(len, rows.values())) == report["rows"]
    for left, right in (
        ("train", "development"),
        ("train", "heldout"),
        ("development", "heldout"),
    ):
        for key in ("episode_group_id", "matched_group_id"):
            assert not ({row[key] for row in rows[left]} & {row[key] for row in rows[right]})
        left_sources = {
            source for row in rows[left] for source in row["provenance"]["source_trace_ids"]
        }
        right_sources = {
            source for row in rows[right] for source in row["provenance"]["source_trace_ids"]
        }
        assert not left_sources & right_sources
    assert all(
        row["input"]["observation"]["anonymized"] is True
        for split_rows in rows.values() for row in split_rows
    )


def test_spend_audit_is_exact_and_source_hashed():
    audit = json.loads(
        (ROOT / "artifacts/tool-decision-supervisor-v1/spend-audit.json").read_text()
    )
    total = sum(Decimal(str(item["cost_usd"])) for item in audit["line_items"])
    assert total == Decimal(str(audit["incremental_experiment_usd"]))
    assert (
        Decimal(str(audit["cumulative_tinker_before_experiment_usd"])) + total
        == Decimal(str(audit["cumulative_tinker_after_experiment_usd"]))
    )
    for item in audit["line_items"]:
        assert sha256_file(ROOT / item["source"]) == item["source_sha256"]
    provider = audit["provider_snapshot"]
    assert sha256_file(ROOT / provider["path"]) == provider["sha256"]
    assert audit["target_under_five_usd"] is True
    assert audit["hard_cap_under_ten_usd"] is True
    assert audit["sealed_seasons_opened"] == []
    assert audit["rl_started"] is False
