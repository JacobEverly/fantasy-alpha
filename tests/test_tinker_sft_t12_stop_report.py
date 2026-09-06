from evals import tinker_sft_t12_stop_report as report


def test_stop_report_proves_gate_failure_and_no_full_run():
    result = report.build_stop_scorecard()
    assert result["decision"]["verdict"] == "stop_additional_sft"
    assert result["decision"]["full_t12_training_started"] is False
    assert result["decision"]["frozen_t12_arm_evaluated"] is False
    assert result["decision"]["rl_started"] is False
    assert result["canaries"]["v2"]["behavior"]["structured_coverage"] == 1.0
    assert result["canaries"]["v2"]["behavior"]["tool_valid_rate"] == 0.0
    assert result["spend"]["t12_incremental_total_usd"] < 7
    assert result["sealed_2025_untouched"] is True
