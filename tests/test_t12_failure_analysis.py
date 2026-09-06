from training.t12_failure_analysis import analyze


def test_t12_failure_analysis_uses_only_development_artifacts():
    result = analyze()
    assert result["uses_existing_artifacts_only"] is True
    assert result["sealed_2025_untouched"] is True
    assert result["prompt_conflicts"] == {
        "overlapping_prompts": 120,
        "different_targets": 120,
        "resolution": "T1.2 supersedes the old T1 answer with the T1.1 correction",
    }
    assert result["supervision_balance"]["unweighted_t11_target_share"] < 0.04
    assert result["observed_product_failures"]["tool_calls"] == 0
    assert result["observed_product_failures"]["zero_reward_draft_episodes"] == 21
