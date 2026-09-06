# T1 failure analysis — what T1.1 must fix

This analysis uses only the already-opened 2018/2023/2024 development
results. Player examples remain anonymized. The 2025 gate was not opened.

## Root finding

The clearest issue is supervision mismatch: T1 had **811**
forecast traces but **0 model-authored
draft actions** and **0 tool calls**. Evidence
was pre-fetched into the user message, so the model never learned when to ask
for it. The draft improvement therefore cannot be credited to learned draft
process from this corpus.

## Error taxonomy

| observed failure | evidence | T1.1 intervention |
|---|---|---|
| Unstable draft reward | mean delta +81.15, median -7.48; 2/6 wins | more matched drafts plus explicit action supervision |
| QB/TE bench overinvestment | T1 had 3 episodes with >2 QBs and 1 with >2 TEs | roster guardrail and recovery traces |
| Late-round misses | 15 T1 late picks had negative realized-capture diagnostics | late-round opportunity-cost examples |
| Unsupported confidence tails | high-confidence misses 40→80; log loss 0.6861→0.6993 | outcome-blind shrinkage and confidence correction |
| No evidence seeking | 0 tool calls in T1 evaluation | paired selective tool-call/post-result traces |

Realized capture is used below only to explain completed drafts. It is hindsight
attribution, not a training input and not proof that an earlier choice caused the
terminal reward difference.

## Six paired drafts

| season/slot | delta | first matched-state divergence | T1 position mix | best T1-only capture diagnostic |
|---|---:|---|---|---|
| 2018/slot 1 | -6.78 | round 2: B028 → B022 | QB4, RB4, TE1, WR5 | B132 (round 12, +86.1) |
| 2018/slot 7 | -42.14 | round 4: B039 → B046 | QB5, RB3, TE3, WR3 | B103 (round 9, +139.8) |
| 2023/slot 1 | -70.74 | round 4: B050 → B058 | QB2, RB6, TE1, WR6 | B130 (round 13, +89.2) |
| 2023/slot 7 | +331.72 | round 1: B007 → B008 | QB4, RB3, TE1, WR7 | B130 (round 11, +89.2) |
| 2024/slot 1 | +283.04 | round 8: B088 → B108 | QB2, RB3, TE2, WR8 | B157 (round 15, +107.0) |
| 2024/slot 7 | -8.18 | round 3: B030 → B039 | QB2, RB3, TE2, WR8 | B160 (round 14, +98.9) |

The two large wins contain valuable examples, but they do not establish a
repeatable policy: one first diverged immediately and the other only in round
eight, while four paired drafts regressed. T1.1 therefore teaches the process
directly and precommits to median and paired-win criteria rather than relying on
another favorable mean.

## Evidence versus hypotheses

Observed facts:

- T1 contains no model-authored draft or tool actions.
- Neither evaluated arm called a tool.
- T1 won only two of six paired drafts and had a negative median delta.
- Both arms repeatedly over-drafted QB/TE bench depth.
- T1 produced more extreme high-confidence misses and worse log loss.

Testable hypotheses for T1.1:

- Explicit roster/timing demonstrations may reduce avoidable bench imbalance.
- Conservative outcome-blind shrinkage may reduce damaging probability tails.
- Paired tool-call and post-result demonstrations may teach selective evidence use.
