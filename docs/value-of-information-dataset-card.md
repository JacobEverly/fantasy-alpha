# Fantasy Alpha value-of-information dataset v1

This dataset measures whether one relevant evidence lookup changed a Qwen3.5-9B
draft decision and improved its realized DraftGym reward. It is an intervention
dataset, not a policy-imitation dataset.

## Composition

- Development: 70 points from 35 complete episodes; labels `{"HARMFUL": 6, "HELPFUL": 6, "UNNECESSARY": 54, "UNRESOLVABLE": 4}`.
- Held out: 70 points from 35 complete episodes; labels `{"HARMFUL": 11, "HELPFUL": 7, "UNNECESSARY": 48, "UNRESOLVABLE": 4}`.
- All inputs are masked PPR states from 2015–2017 and 2019–2022.
- Seasons 2013, 2018, and 2023–2025 remain sealed.

Each point replays the exact same action prefix. The control accepts the proposed
pick. The treatment performs one structured lookup and lets the same model
reconsider once. Both branches then use deterministic AutopickADP.

## Labels

`HELPFUL` and `HARMFUL` require a changed action and at least 25 points of
terminal reward movement, except that preventing an illegal action is directly
helpful. Empty or failed evidence is `UNRESOLVABLE`; everything else is
`UNNECESSARY`.

## Intended use and limits

Use development rows to fit evidence-value supervisors. Held-out rows are for one
fit-free evaluation only. The small panel supports a first routing experiment,
not a general claim about every fantasy season, model, tool, or league format.
Terminal reward is deterministic conditional on the frozen state and continuation,
but it remains a noisy proxy for the local decision's true value.

Hashes and leakage checks are in `artifacts/value-of-information-v1/dataset-audit.json`. Twelve
deterministically selected review rows are in `artifacts/value-of-information-v1/reviewed-pairs.jsonl`.
