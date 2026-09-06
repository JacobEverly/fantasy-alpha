"""Conservative continuation for terminal T1.1 structured-output failures.

The frozen evaluator allows one strict-JSON repair but originally aborts when
that repair is also malformed.  This runner leaves prompts, sampling settings,
repair count, answer parsing, and scoring unchanged.  It only converts a
terminal parse failure into zero answered questions so the frozen arm can
finish and the failure is charged against structured coverage.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any

import evals.tinker_sft_scorecard as shared
from evals.tinker_sft_t11_scorecard import run_arm, safe_text


def _ask_json_array_conservative(
    sampler: Any,
    messages: list[dict],
    valid_ids: set[str],
    *,
    max_tokens: int,
    seed: int,
) -> tuple[list[dict], dict[str, Any]]:
    before = asdict(sampler.ledger)
    result = sampler.chat(messages, max_tokens=max_tokens, temperature=0.0, seed=seed)
    repaired = False
    terminal_error: str | None = None
    try:
        answers = shared.parse_answers(result["content"], valid_ids)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        repaired = True
        repair_messages = messages + [
            {"role": "assistant", "content": result["content"]},
            {
                "role": "user",
                "content": (
                    f"Invalid ({safe_text(exc)}). Return ONLY the JSON array, "
                    "one object per question_id."
                ),
            },
        ]
        result = sampler.chat(
            repair_messages, max_tokens=max_tokens, temperature=0.0, seed=seed
        )
        try:
            answers = shared.parse_answers(result["content"], valid_ids)
        except (AssertionError, KeyError, TypeError, ValueError) as exc2:
            answers = []
            terminal_error = safe_text(exc2)
    after = asdict(sampler.ledger)
    return answers, {
        "valid_ids": len(valid_ids),
        "answered": len(answers),
        "coverage": len(answers) / len(valid_ids),
        "repaired": repaired,
        "parse_success": terminal_error is None,
        "terminal_parse_error": terminal_error,
        "usage_delta": {key: after[key] - before[key] for key in before},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("t11",), required=True)
    args = parser.parse_args()
    shared._ask_json_array = _ask_json_array_conservative
    print(shared.json.dumps(run_arm(args.arm), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
