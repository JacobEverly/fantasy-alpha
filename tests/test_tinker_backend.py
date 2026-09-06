import json
from pathlib import Path

import pytest

from training import tinker_backend as tb


def _row(i: int, family: str = "full_slate", season: int = 2017, sample: int = 0):
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"user {i}"},
            {"role": "assistant", "content": f"answer {i}"},
        ],
        "meta": {
            "track": "anonymized", "trace_id": f"t:{i}:{sample}",
            "bench": "breakoutbench", "family": family, "season": season,
            "question_id": f"Q{i}", "sample_idx": sample,
        },
    }


def test_frozen_corpus_identity_and_contract():
    rows = tb.load_corpus()
    assert len(rows) == 811
    assert tb.sha256_file(tb.DEFAULT_CORPUS) == tb.CORPUS_SHA256
    assert {r["meta"]["season"] for r in rows} <= tb.TRAIN_SEASONS
    assert not ({r["meta"]["season"] for r in rows} & tb.SEALED_SEASONS)


def test_existing_t0_corpus_identity_and_contract():
    rows = tb.load_t0_corpus()
    assert len(rows) == 162
    assert tb.sha256_file(tb.T0_CORPUS) == tb.T0_CORPUS_SHA256
    assert all([m["role"] for m in row["messages"]] == [
        "system", "user", "assistant"
    ] for row in rows)


def test_existing_t0_corpus_identity_and_chat_shape():
    rows = tb.load_t0_corpus()
    assert len(rows) == 162
    assert tb.sha256_file(tb.T0_CORPUS) == tb.T0_CORPUS_SHA256
    assert all([m["role"] for m in r["messages"]] ==
               ["system", "user", "assistant"] for r in rows)


def test_grouped_split_keeps_variants_together_and_is_stable():
    rows = []
    for i in range(80):
        family = "full_slate" if i % 2 else "bust"
        rows.extend([_row(i, family, sample=0), _row(i, family, sample=1)])
    a_train, a_dev = tb.grouped_development_split(rows)
    b_train, b_dev = tb.grouped_development_split(rows)
    assert [r["meta"]["trace_id"] for r in a_train] == [r["meta"]["trace_id"] for r in b_train]
    assert [r["meta"]["trace_id"] for r in a_dev] == [r["meta"]["trace_id"] for r in b_dev]
    train_q = {r["meta"]["question_id"] for r in a_train}
    dev_q = {r["meta"]["question_id"] for r in a_dev}
    assert train_q.isdisjoint(dev_q)


def test_load_rejects_sealed_season_and_changed_hash(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps(_row(1, season=2024)) + "\n")
    with pytest.raises(ValueError, match="season 2024"):
        tb.load_corpus(path, exact=False)
    path.write_text(json.dumps(_row(1)) + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        tb.load_corpus(path, exact=True)


def test_cost_ledger_uses_cached_and_uncached_prices():
    class Seq:
        tokens = [1, 2, 3]

    class Response:
        prompt_cache_hit_tokens = 40
        sequences = [Seq()]

    ledger = tb.CostLedger(training_tokens=100)
    ledger.add_sample(100, Response())
    assert ledger.prefill_tokens == 60
    assert ledger.cached_prefill_tokens == 40
    assert ledger.sample_tokens == 3
    expected = (
        100 * tb.TRAIN_USD_PER_MTOK + 60 * tb.PREFILL_USD_PER_MTOK
        + 40 * tb.CACHED_PREFILL_USD_PER_MTOK + 3 * tb.SAMPLE_USD_PER_MTOK
    ) / 1_000_000
    assert ledger.usd == pytest.approx(expected)


def test_safe_text_redacts_tinker_tokens():
    assert "tml-" not in tb.safe_text("failed tml-thisisasecretvalue123")


def test_render_rows_applies_optional_loss_weight_only_to_assistant_tokens():
    row = tb.load_corpus()[0]
    baseline, _, _ = tb.render_rows([row], tb.RunConfig(run_name="baseline"))
    weighted_row = json.loads(json.dumps(row))
    weighted_row["meta"]["loss_weight"] = 3.5
    weighted, _, _ = tb.render_rows(
        [weighted_row], tb.RunConfig(run_name="weighted")
    )
    before = baseline[0].loss_fn_inputs["weights"].data
    after = weighted[0].loss_fn_inputs["weights"].data
    assert [value == 0 for value in before] == [value == 0 for value in after]
    assert sum(after) == pytest.approx(sum(before) * 3.5)


def test_no_credential_literal_in_source():
    paths = list((tb.ROOT / "training").glob("tinker*.py"))
    paths.append(tb.ROOT / "evals/tinker_sft_scorecard.py")
    for path in paths:
        text = path.read_text()
        if path.name == "tinker_backend.py":
            assert "TINKER_API_KEY" in text
        assert not tb._SECRET_RE.search(text)
