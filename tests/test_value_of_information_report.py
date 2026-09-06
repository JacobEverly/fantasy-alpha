from evals.value_of_information_report import (
    _sum_usage,
    provider_sampling_usage,
    reviewed_pairs,
)


def _row(index, label, delta):
    return {
        "example_id": f"example-{index}",
        "episode_id": f"season2020:slot1:seed{index}",
        "split": "development",
        "input": {"proposed_action": {"pick": f"B{index:03d}"}},
        "intervention": {
            "tool": {"name": "injury_status", "arguments": {"player": f"B{index:03d}"}},
            "resolved": True,
            "post_tool_model_action": {"pick": f"B{index + 1:03d}"},
        },
        "outcome": {"action_changed": True, "reward_difference": delta},
        "label": {"value": label, "reason_code": "test"},
    }


def test_reviewed_pairs_is_deterministic_and_balanced_when_available():
    rows = []
    labels = ("HELPFUL", "UNNECESSARY", "HARMFUL", "UNRESOLVABLE")
    for index in range(16):
        rows.append(_row(index, labels[index % 4], float(index)))
    first = reviewed_pairs(rows)
    second = reviewed_pairs(list(reversed(rows)))
    assert first == second
    assert len(first) == 12
    counts = {label: sum(row["label"]["value"] == label for row in first) for label in labels}
    assert counts == {label: 3 for label in labels}


def test_provider_sampling_usage_excludes_storage_and_earlier_work():
    events = [
        {
            "bucket_start": "2026-09-06T19:00:00+00:00",
            "type": "sampling_sample",
            "token_count": 99,
        },
        {
            "bucket_start": "2026-09-06T20:00:00+00:00",
            "type": "sampling_prefill",
            "token_count": 100,
            "cached": False,
        },
        {
            "bucket_start": "2026-09-06T20:00:00+00:00",
            "type": "sampling_prefill",
            "token_count": 40,
            "cached": True,
        },
        {
            "bucket_start": "2026-09-06T21:00:00+00:00",
            "type": "sampling_sample",
            "token_count": 5,
        },
        {
            "bucket_start": "2026-09-06T21:00:00+00:00",
            "type": "storage",
            "gigabyte_hours": 10,
        },
    ]
    assert provider_sampling_usage(
        events, ending_before="2026-09-06T22:00:00+00:00",
    ) == {
        "training_tokens": 0,
        "prefill_tokens": 100,
        "cached_prefill_tokens": 40,
        "sample_tokens": 5,
        "checkpoint_count": 0,
    }


def test_sum_usage_accepts_one_pass_iterators():
    rows = (
        {"prefill_tokens": value, "sample_tokens": 1} for value in (10, 20)
    )
    assert _sum_usage(rows) == {
        "training_tokens": 0,
        "prefill_tokens": 30,
        "cached_prefill_tokens": 0,
        "sample_tokens": 2,
        "checkpoint_count": 0,
    }
