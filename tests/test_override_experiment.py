import hashlib
import json
from pathlib import Path

from training.override_dataset import CORPUS_PATH, ROOT


ARTIFACT = ROOT / "artifacts/high-confidence-override-v1"
SPEC = ROOT / "training/override-eval-spec-v1.json"


def load(path: Path):
    return json.loads(path.read_text())


def test_frozen_override_result_matches_protocol_and_paid_run():
    spec = load(SPEC)
    result = load(ARTIFACT / "holdout-2025.json")
    run = load(ARTIFACT / "tinker-run/run-summary.json")
    assert result["status"] == run["status"] == "complete"
    assert result["spec_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest()
    assert spec["holdout"]["season"] == 2025
    assert run["config"]["model"] == spec["training"]["base_model"]
    assert run["config"]["lora_rank"] == spec["training"]["lora_rank"]
    assert run["config"]["epochs"] == spec["training"]["epochs"] == 1
    assert run["reload_verified"] is True
    assert Path(ROOT / run["archive"]["path"]).exists()
    assert len(result["base"]["episodes"]) == len(result["adapter"]["episodes"]) == 15
    assert result["base"]["behavior"]["decisions"] == 225
    assert result["adapter"]["behavior"]["decisions"] == 225
    expected_cost = (
        run["cost"]["computed_usd"]
        + result["base"]["cost"]["computed_usd"]
        + result["adapter"]["cost"]["computed_usd"]
    )
    assert abs(result["incremental_cost_usd"] - expected_cost) < 1e-12
    assert result["incremental_cost_usd"] < spec["training"]["new_spend_hard_cap_usd"]


def test_negative_decision_is_supported_and_training_corpus_excludes_holdout():
    result = load(ARTIFACT / "holdout-2025.json")
    analysis = load(ARTIFACT / "failure-analysis.json")
    assert result["gates"]["pass"] is False
    assert result["adapter"]["behavior"]["teacher_override_recall"] == 0
    assert result["adapter"]["product"]["mean_points_above_adp"] < 0
    assert result["teacher"]["product"]["mean_points_above_adp"] < 0
    assert analysis["decision"] == "stop_this_sft_path_and_retain_adp_as_product_fallback"
    for line in CORPUS_PATH.read_text().splitlines():
        assert json.loads(line)["meta"]["season"] != 2025


def test_experiment_artifacts_do_not_persist_credentials():
    markers = ("tml-", "TINKER_API_KEY=", "dtn_", "sk-")
    for path in ARTIFACT.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md", ".txt"}:
            text = path.read_text(errors="replace")
            assert not any(marker in text for marker in markers), path
