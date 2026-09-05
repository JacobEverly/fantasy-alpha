from training.t11_failure_analysis import prediction_diagnostics


def test_prediction_diagnostics_counts_new_overconfident_miss():
    base = {
        "q": {
            "benchmark": "calibbench", "season": 2024,
            "family": "season_threshold", "question_id": "q",
            "p": 0.4, "y": 1,
        }
    }
    adapter = {
        "q": {
            "benchmark": "calibbench", "season": 2024,
            "family": "season_threshold", "question_id": "q",
            "p": 0.1, "y": 1,
        }
    }
    result = prediction_diagnostics(base, adapter)
    assert result["base_high_confidence_wrong"] == 0
    assert result["adapter_high_confidence_wrong"] == 1
    assert result["adapter_more_extreme"] == 1
    assert result["more_extreme_and_worse"] == 1
    assert result["mean_brier_improvement"] < 0
