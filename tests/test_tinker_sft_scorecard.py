import json

from evals import tinker_sft_scorecard as sc


def test_frozen_spec_is_current_and_keeps_2025_sealed():
    spec = sc.verify_spec()
    assert spec["seasons"] == [2018, 2023, 2024]
    assert 2025 not in spec["seasons"]
    assert spec["frozen_before_t1_training"] is True
    assert spec["model"] == "Qwen/Qwen3.5-9B"


def test_calib_prompt_is_anonymized_and_rectangular():
    payload = json.loads(
        (sc.QUESTIONS_DIR / "calibbench_2024_ppr_anon.json").read_text()
    )
    messages, valid = sc._calib_prompt(payload, "weekly_h2h")
    assert len(valid) == payload["families"]["weekly_h2h"]
    assert "Season and player identities are masked" in messages[1]["content"]
    assert "EXACTLY one object per question" in messages[1]["content"]


def _fake_results(*, breakout_gain=0.0, calib_gain=0.0, draft_gain=0.0,
                  canary=True, coverage_gain=0.0):
    def arm(brier_b, brier_c, draft, coverage):
        return {
            "breakout": {"pooled": {"pooled_brier": brier_b}},
            "calib": {"overall": {"brier": brier_c}},
            "draftgym": {"mean_reward": draft},
            "structured_coverage": coverage,
            "canary": {"gate": {"pass": canary}},
        }
    return {
        "base": arm(0.20, 0.25, 0.0, 1.0),
        "adapter": arm(0.20 - breakout_gain, 0.25 - calib_gain,
                       draft_gain, 1.0 + coverage_gain),
    }


def test_decision_rule_requires_two_material_improvements_to_proceed():
    spec = sc.verify_spec()
    one = sc._decision(_fake_results(draft_gain=30), spec)
    assert one["verdict"] == "revise_sft_before_rl"
    two = sc._decision(_fake_results(draft_gain=30, calib_gain=0.01), spec)
    assert two["verdict"] == "proceed_to_draftgym_rl"


def test_decision_rule_stops_on_canary_failure_or_major_regression():
    spec = sc.verify_spec()
    assert sc._decision(_fake_results(canary=False, draft_gain=30), spec)["verdict"] == "stop_sft_training"
    assert sc._decision(_fake_results(breakout_gain=-0.02, draft_gain=30), spec)["verdict"] == "stop_sft_training"
