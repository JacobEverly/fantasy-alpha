# Fantasy Alpha T1.1 targeted dataset card

This 300-trace development-only corpus is a controlled replacement for T1,
not an append-only pile of generic examples. It directly addresses the three
observed T1 gaps: unstable draft choices, overconfident probability tails, and
zero evidence-tool use. All assistant tokens are trainable; system/user context
remains loss-masked by the existing Tinker backend.

## Composition

| subtype | rows | observable behavior |
|---|---:|---|
| calibration correction | 120 | conservative, grounded probability |
| direct draft pick | 120 | timing, roster and late-round action |
| evidence-tool call | 30 | query a specific uncertain candidate |
| post-tool pick | 30 | use structured depth evidence in the next action |

Corpus SHA-256: `0ebbb365f1be9fb76804256cbbdf1eccca300a31c01beca0cd1ff446fc6a81b1`.

The draft labels come from the previously benchmarked market-only
`SurvivalSequencer(scarcity_ratio=1.5)` policy. It uses ADP survival to the
next snake turn, positional scarcity, and QB/TE roster guardrails. It does not
see realized outcomes. Forecast targets shrink the old teacher's directional
update toward a strictly-prior historical anchor and cap unsupported tails.

## Coverage

| behavior | traces |
|---|---:|
| calibrated probability | 120 |
| confidence correction | 120 |
| contradictory evidence | 120 |
| draft timing | 150 |
| evidence tool use | 60 |
| late round strategy | 48 |
| league arithmetic | 270 |
| mistake prevention | 180 |
| qb te guardrail | 76 |
| recovery | 24 |
| roster construction | 82 |
| tool avoidance | 128 |
| tool precision | 60 |

## Leakage and quality checks

- Seasons are limited to the existing training split; 2013, 2018, 2023,
  2024, and 2025 are absent.
- Draft prompts expose only the environment's existing pre-draft observation
  fields. Reward, realized points, outcomes, and verifier data are rejected.
- Tool results are time-gated by `EvidenceStore`; each included call has a
  successful, non-empty structured depth-chart response.
- Picks must be legal, model-visible board IDs. Tool actions must use the
  masked structured interface and have a paired post-result decision.
- Exact and effectively identical prompt/target duplicates are rejected.
- Forecast numeric claims pass the existing cite-or-don't-claim and grounding
  checks. Outcomes are never used to select or construct T1.1 targets.

## Intended use and limitations

This corpus tests whether a smaller amount of behavior-specific supervision
beats T1's larger probability-heavy corpus. The draft teacher is a disciplined
rule policy, not an oracle; its historical edge over ADP is small and unresolved.
Tool-use labels encode an explicit product policy (inspect uncertain candidates
with structured depth evidence), not a claim that every successful lookup changes
the final roster. Evaluation therefore reports both task reward and policy-alignment
metrics. The 2025 season remains sealed for a later one-shot gate.
