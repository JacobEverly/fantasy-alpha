# Fantasy Alpha tool-decision dataset v1

A matched-state, outcome-blind dataset for deciding whether a proposed
DraftGym pick may proceed, requires a structured evidence lookup, or must
be blocked because evidence is unavailable.

## Composition

- 948 examples in 237 matched groups.
- 98 whole episodes; episodes never cross splits.
- Split rows: `{"development": 164, "heldout": 220, "train": 564}`.
- Scenario counts: `{"context_sufficient": 237, "post_tool_resolved": 237, "tool_required": 237, "unresolved_no_budget": 237}`.
- Tool labels: `{"depth_chart": 177, "injury_status": 60}`.
- All states are anonymized and use only training seasons.
- 2013, 2018, 2023, 2024, and 2025 are absent.

## Matched design

Each evidence group contributes a pre-tool decision, its successful
post-tool state, a failed/no-budget state, and a genuine nearby state from
the same episode where the expert policy acts without a lookup. Splitting
by full episode prevents nearly adjacent draft states from leaking across
training and evaluation.

## Label provenance

Labels come from the outcome-blind SurvivalSequencer workflow and strictly
pre-as-of EvidenceStore responses. Realized season outcomes, verifier
rewards, and future evidence never enter either inputs or labels.

The machine-readable leakage audit is at
`training/datasets/tool_decision_v1/leakage-report.json`. It verifies zero
episode, matched-group, or source-trace overlap between splits; zero duplicate
decision inputs; anonymized observations throughout; and no sealed seasons.

## Intended use

Train and compare a narrow tool-decision supervisor. Held-out rows are for
one frozen internal evaluation only and must never be passed to a trainer.
This dataset does not establish that tool use improves season points; that
requires a later paired DraftGym run.

## Known limitations

- Labels encode a conservative outcome-blind expert policy, not observed product lift.
- ACT_NOW has two surfaces (post-tool and context-sufficient), so class counts are 2:1:1 while scenario counts are balanced.
- The current tool surface covers structured depth-chart and injury lookups; named free-text search remains outside this masked benchmark.
- Missing ADP/ranking evidence, local roster arithmetic, illegal or irrelevant
  tool requests, and inconclusive result envelopes are represented only in the
  separate 12-case post-hoc adversarial challenge. That challenge is never
  training data or part of the primary frozen gate.
