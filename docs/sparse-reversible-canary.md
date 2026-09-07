# Sparse reversible intervention canary

- **Completed:** 2026-09-07
- **Model:** `Qwen/Qwen3.5-9B` through Tinker
- **Decision:** revise the controller before collecting training data

## Question

Can one carefully bounded evidence lookup improve a complete draft without
creating the repeated-tool and downstream-state failures seen in the earlier
supervisor experiments?

I froze a simple controller before running the comparison. In rounds one
through four, it could inspect one candidate when the draft market showed high
uncertainty. It kept the model's original pick, requested one structured piece
of evidence, and accepted a different pick only when the evidence was adverse
and the replacement passed market and roster checks. Otherwise it used the
original pick.

The evaluation used 30 development-only episode pairs. Base and treatment runs
shared the season, draft seat, seed, opponent policy, and model settings.
Seasons 2013, 2018, and 2023–2025 remained sealed.

## Result

| Measure | Result |
|---|---:|
| Matched pairs | 30 |
| Better / worse / tied drafts | 3 / 3 / 24 |
| Mean reward change | -4.92 |
| 10% trimmed mean change | 0.00 |
| Mean without the largest gain | -10.15 |
| Paired bootstrap 95% interval | [-31.30, 14.86] |
| Revisions accepted / rejected | 4 / 26 |
| Request-level spend | $1.237063 |

The controller did not pass its frozen signal gate. Mean performance was
negative, wins did not exceed losses, and the result was not positive after
removing the largest gain. The uncertainty interval also spans zero.

All 60 runs completed. The treatment used exactly one lookup per episode, never
accepted an illegal action, and preserved the matched action prefix. No fitting,
threshold changes, or prompt changes were made after outcomes were observed.

## What this clarified

The hard safety boundary worked: intervention count, tool count, legality, and
the immediate fallback action were controlled. That is useful, but it is not
the same as improving the draft.

Four evidence-backed changes produced two gains and two losses. More subtly,
two of the 26 rejected revisions still changed later picks. The controller
restored the original pick and DraftGym cleared the tool result, but the extra
reconsideration call advanced the turn counter used to seed later model calls.
In other words, the action and environment state were reversible while the
sampling path was not.

That distinction explains why this is not yet suitable training data for a
policy. A rejected intervention should recover both the original action and the
same future sampling schedule. Until that is true, the comparison mixes the
value of the accepted action with variation caused by an extra model call.

## Next experiment

Separate action-decision seeds from intervention-call seeds, assert that a
rejected branch returns to the same observable state and next action seed, then
preregister a second small paired run. If that version cannot beat untouched
Qwen on full drafts, stop this line of supervision. If it does, the matched
trajectories become candidates for preference or policy training.

## Reproduction trail

- Frozen protocol: `training/sparse-reversible-canary-v1.json`
- Protocol hash: `training/sparse-reversible-canary-v1.sha256`
- Matched records: `artifacts/sparse-reversible-canary-v1/matched-records.jsonl`
- Scorecard: `artifacts/sparse-reversible-canary-v1/scorecard.json`
- Spend audit: `artifacts/sparse-reversible-canary-v1/spend-audit.json`
- Billing snapshot: `artifacts/sparse-reversible-canary-v1/billing.json`
