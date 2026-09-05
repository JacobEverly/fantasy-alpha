from dataclasses import dataclass

from evals.tinker_sft_t11_resilient_runner import _ask_json_array_conservative
from training.tinker_backend import CostLedger


@dataclass
class _FakeSampler:
    ledger: CostLedger
    calls: int = 0

    def chat(self, messages, *, max_tokens, temperature, seed):
        self.calls += 1
        self.ledger.prefill_tokens += 10
        self.ledger.sample_tokens += 3
        return {
            "content": (
                '[{"question_id":"x","p":0.5}] '
                '[{"question_id":"x","p":0.5}]'
            )
        }


def test_terminal_parse_failure_is_zero_coverage_after_one_repair():
    sampler = _FakeSampler(CostLedger())
    answers, record = _ask_json_array_conservative(
        sampler,
        [{"role": "user", "content": "answer"}],
        {"x"},
        max_tokens=20,
        seed=7,
    )
    assert answers == []
    assert sampler.calls == 2
    assert record["repaired"] is True
    assert record["parse_success"] is False
    assert record["coverage"] == 0.0
    assert record["usage_delta"]["prefill_tokens"] == 20
    assert record["usage_delta"]["sample_tokens"] == 6
