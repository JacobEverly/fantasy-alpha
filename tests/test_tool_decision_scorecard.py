from evals import tool_decision_scorecard as scorecard


def test_frozen_episode_panel_has_one_internal_heldout_episode_per_season():
    assert [row["season"] for row in scorecard.FROZEN_EPISODES] == [
        2015, 2016, 2017, 2019, 2020, 2021, 2022,
    ]
    assert all(row["mask_names"] is True for row in scorecard.FROZEN_EPISODES)
    assert all(row["enable_evidence"] is True for row in scorecard.FROZEN_EPISODES)


def test_usage_cost_matches_pinned_rates():
    usage = {
        "prefill_tokens": 1_000_000,
        "cached_prefill_tokens": 1_000_000,
        "sample_tokens": 1_000_000,
        "training_tokens": 0,
        "checkpoint_count": 0,
    }
    assert scorecard._usage_cost(usage) == (
        scorecard.PREFILL_USD_PER_MTOK
        + scorecard.CACHED_PREFILL_USD_PER_MTOK
        + scorecard.SAMPLE_USD_PER_MTOK
    )


def test_candidate_manifest_verification_when_materialized():
    if not scorecard.CANDIDATE_MANIFEST.exists():
        return
    manifest = scorecard.verify_candidates()
    assert manifest["heldout_rows_seen_before_freeze"] == 0
    assert manifest["contrastive_sft_status"] == "rejected_on_development"
