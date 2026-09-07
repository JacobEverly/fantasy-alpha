import json

import pytest

from evals import sparse_reversible_canary as canary


def test_panel_is_development_only_and_unique():
    panel = canary.evaluation_panel()
    assert len(panel) == 30
    assert len({row["episode_id"] for row in panel}) == 30
    assert not ({row["season"] for row in panel} & canary.SEALED_SEASONS)


def test_protocol_freeze_is_immutable(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    digest = tmp_path / "spec.sha256"
    monkeypatch.setattr(canary, "SPEC_PATH", spec)
    monkeypatch.setattr(canary, "SPEC_SHA_PATH", digest)
    frozen = canary.freeze_protocol()
    assert frozen["frozen_before_paid_outcomes"] is True
    assert canary.verify_protocol()["panel"] == canary.evaluation_panel()
    payload = json.loads(spec.read_text())
    payload["controller"]["last_round"] = 5
    spec.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        canary.verify_protocol()


def test_scorecard_rejects_record_outside_frozen_panel(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    digest = tmp_path / "spec.sha256"
    records = tmp_path / "records.jsonl"
    scorecard = tmp_path / "scorecard.json"
    monkeypatch.setattr(canary, "SPEC_PATH", spec)
    monkeypatch.setattr(canary, "SPEC_SHA_PATH", digest)
    monkeypatch.setattr(canary, "RECORDS_PATH", records)
    monkeypatch.setattr(canary, "SCORECARD_PATH", scorecard)
    canary.freeze_protocol()
    records.write_text(
        json.dumps(
            {
                "protocol": canary.PROTOCOL,
                "arm": "base",
                "episode_id": "season2025:slot1:seed1",
                "spec": {"season": 2025},
                "outcome": {},
                "cost_usd": 0.0,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="outside the frozen panel"):
        canary.materialize_scorecard(require_complete=False)
