from evals import tinker_sft_t12_scorecard as sc


def test_t12_spec_reuses_old_arms_and_keeps_2025_sealed():
    spec = sc.build_spec()
    assert spec["seasons"] == [2018, 2023, 2024]
    assert 2025 not in spec["seasons"]
    assert spec["reuse"]["base_t1_t11_not_regenerated"] is True
    assert len(spec["draftgym"]["episodes"]) == 24
    assert spec["cost"]["estimated_total_usd"] < 7


def _arm(*, reward=20.0, coverage=1.0, zero_rate=0.0):
    return {
        "breakout": {"pooled": {"pooled_brier": 0.2}},
        "calib": {"overall": {"brier": 0.2, "log_loss": 0.6, "ece": 0.05}},
        "canary": {"gate": {"pass": True}},
        "structured_coverage": coverage,
        "draftgym": {
            "mean_reward": reward, "illegal_picks": 0, "tool_calls": 5,
            "tool_precision": 1.0, "tool_recall": 0.5,
            "useful_tool_result_rate": 1.0, "unnecessary_tool_rate": 0.0,
            "terminal_control_reward_match_rate": zero_rate,
        },
    }


def test_t12_decision_requires_draft_and_retention_gates(monkeypatch):
    spec = sc.build_spec()
    monkeypatch.setattr(sc, "_json", lambda path: {"gate": {"pass": True}})
    results = {"base": _arm(), "t12": _arm(reward=30)}
    comparisons = {"draftgym": {"base": {
        "win_rate": 0.7, "median_delta": 5, "trimmed_mean_delta": 5,
    }}}
    decision = sc._decision(results, comparisons, spec)
    assert decision["verdict"] == "proceed_to_small_draftgym_rl_pilot"
    results["t12"]["structured_coverage"] = 0.5
    decision = sc._decision(results, comparisons, spec)
    assert decision["verdict"] != "proceed_to_small_draftgym_rl_pilot"
