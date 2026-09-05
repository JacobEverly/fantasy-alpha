from evals import tinker_sft_t11_scorecard as sc


def test_t11_spec_uses_24_development_drafts_and_keeps_2025_sealed():
    spec = sc.build_spec()
    assert spec["seasons"] == [2018, 2023, 2024]
    assert 2025 not in spec["seasons"]
    assert spec["sealed_2025_untouched"] is True
    assert len(spec["draftgym"]["episodes"]) == 24
    assert {x["agent_slot"] for x in spec["draftgym"]["episodes"]} == {1, 4, 7, 10}
    assert {x["seed"] for x in spec["draftgym"]["episodes"]} == {11, 29}
    assert spec["cost"]["estimated_incremental_total_usd"] < 10


def test_tool_opportunity_is_visible_and_conservative():
    obs = {
        "round": 6, "lookups_remaining": 5, "tool_results": [],
        "top_available": [
            {"player_id": "B001", "adp": 70, "adp_stdev": 4,
             "prev_season_points": 200},
            {"player_id": "B002", "adp": 80, "adp_stdev": 12,
             "prev_season_points": 0},
        ],
    }
    assert sc._opportunity_ids(obs) == {"B002"}
    obs["tool_results"] = [{"response": {"ok": True, "result": []}}]
    assert sc._opportunity_ids(obs) == set()


def _episodes(rewards):
    return [
        {"season": 2018 + i % 3, "agent_slot": 1 + i, "seed": 11, "reward": reward}
        for i, reward in enumerate(rewards)
    ]


def test_draft_comparison_reports_median_wins_and_outlier_robustness():
    reference = _episodes([0] * 10)
    contender = _episodes([10, 10, 10, 10, 10, 10, 10, -2, -2, 1000])
    result = sc._draft_comparison(contender, reference, "base")
    assert result["win_rate"] == 0.8
    assert result["median_delta"] == 10
    assert result["trimmed_mean_delta"] > 0
    assert len(result["episode_bootstrap_95ci"]) == 2


def _arm(*, draft_reward=10.0, brier=0.20, log_loss=0.60, ece=0.05):
    return {
        "breakout": {"pooled": {"pooled_brier": brier}},
        "calib": {"overall": {"brier": brier, "log_loss": log_loss, "ece": ece}},
        "canary": {"gate": {"pass": True}},
        "structured_coverage": 1.0,
        "draftgym": {
            "mean_reward": draft_reward, "fallback_rate": 0.0,
            "illegal_picks": 0, "tool_precision": 1.0, "tool_recall": 0.5,
            "useful_tool_result_rate": 1.0, "unnecessary_tool_rate": 0.0,
        },
    }


def test_precommitted_decision_requires_both_pairwise_comparisons():
    results = {"base": _arm(), "t1": _arm(), "t11": _arm(draft_reward=20)}
    good = {"win_rate": 0.7, "median_delta": 2, "trimmed_mean_delta": 2}
    bad = {"win_rate": 0.5, "median_delta": 2, "trimmed_mean_delta": 2}
    comparisons = {"draftgym": {"base": good, "t1": bad}}
    result = sc._decision(results, comparisons, sc.build_spec())
    assert result["criteria"]["draft_beats_base"] is True
    assert result["criteria"]["draft_beats_t1"] is False
    assert result["verdict"] != "proceed_to_small_draftgym_rl_pilot"
