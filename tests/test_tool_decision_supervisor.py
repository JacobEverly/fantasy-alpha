import json

import pytest

from harness.tool_supervisor import (
    AlwaysActSupervisor,
    HardCodedToolSupervisor,
    LearnedToolSupervisor,
)
from training import tool_decision_supervisor as trainer
from training.tool_decision_dataset import load_split


def test_trainer_rejects_heldout_rows():
    with pytest.raises(ValueError, match="heldout"):
        trainer._extract(load_split("heldout")[:1])


def test_static_baseline_metrics_have_expected_failure_modes():
    rows = load_split("development")
    always = trainer.evaluate_supervisor(rows, AlwaysActSupervisor())
    assert always["required_tool_recall"] == 0.0
    assert always["unsafe_direct_action_rate"] > 0
    rules = trainer.evaluate_supervisor(rows, HardCodedToolSupervisor())
    assert rules["required_tool_recall"] > always["required_tool_recall"]
    assert rules["structured_action_rate"] == 1.0


def test_materialized_learned_artifact_reloads_without_heldout_fit():
    if not trainer.ARTIFACT_PATH.exists():
        pytest.skip("train the learned tool supervisor first")
    report = json.loads(trainer.REPORT_PATH.read_text())
    learned = LearnedToolSupervisor(trainer.ARTIFACT_PATH)
    assert learned.metadata["heldout_rows_seen"] == 0
    assert report["artifact"]["reload_verified"] is True
    result = trainer.evaluate_supervisor(load_split("development"), learned)
    assert result["rows"] == 164
