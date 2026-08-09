# Memorization canaries — canary protocol v1

Model: `Qwen/Qwen3.5-9B` via Prime Intellect serverless, temp 0, naked (no harness), over canary variants of the 10 ppr full_slate sets (seasons 2015-2024). Protocol: same-position, ADP-adjacent, different-outcome pairs; the swapped arm exchanges ENTIRE feature packets between the pair (anon ids stable, swap recorded only in the sequestered canary KEY); the control arm is untouched. Ties score 0.5; 95% CIs bootstrap over seasons (10,000 resamples, seeded).

## Canary sensitivity index = A_control − A_swap_pf

A_control = pair accuracy on control pairs (higher p on the candidate that broke out). A_swap_pf = packet-following pair accuracy on swapped pairs (higher p on the question carrying the breakout packet, wherever the swap moved it). A packet-driven model scores index ≈ 0; a model betting on the hidden identity behind the anon id collapses on swapped pairs and scores index >> 0.

| scope | A_control [95% CI] | n | A_swap_pf [95% CI] | n | index [95% CI] |
|---|---|---|---|---|---|
| pooled 2015-2024 | 0.434 [0.289, 0.586] | 38 | 0.580 [0.423, 0.736] | 44 | -0.145 [-0.390, 0.081] |
| era 2015-2019 | 0.342 [0.192, 0.533] | 19 | 0.682 [0.412, 0.854] | 22 | -0.340 [-0.619, 0.056] |
| era 2020-2024 | 0.526 [0.294, 0.722] | 19 | 0.477 [0.333, 0.643] | 22 | 0.049 [-0.172, 0.184] |

## Canary gate (standard for every future checkpoint)

**Gate: FAIL iff the pooled index 95% CI lower bound > 0.05.** A failing checkpoint's anonymized-track numbers are void (they measure id-recall, not reasoning) and cannot be claimed in any eval report. The gate runs with every BreakoutBench battery (`python evals/canaries.py run`) and its verdict line ships next to the score.

This run: **PASS** — index -0.145 CI [-0.390, 0.081] does not resolve above 0.05.

## Scope note + fingerprint probe (the era-anomaly question)

A full-packet swap moves the profile fingerprint WITH the packet, so pretraining recall keyed on the packet content itself is swap-invariant: the swap index detects identity leakage through the anon-id channel (e.g. eval questions leaked into training data), not fingerprint recall. The fingerprint probe below addresses the era anomaly directly: within-pair accuracy on the UNSWAPPED originals (existing expanded-battery answers, no new inference) for the model vs the pure-code no-peek grounding reference. Pairs are ADP-adjacent so legitimate feature headroom is small and era-stable; model-over-reference excess concentrated in 2020-2024 is the fingerprint-recall signature.

| era | Qwen naked pair acc [95% CI] | n | code reference pair acc [95% CI] | n |
|---|---|---|---|---|
| 2015-2019 | 0.500 [0.389, 0.607] | 41 | 0.500 [0.372, 0.675] | 41 |
| 2020-2024 | 0.476 [0.350, 0.603] | 41 | 0.561 [0.424, 0.705] | 41 |

## Verdict — era-anomaly hypothesis

- 2020-2024 swap index 0.049 [-0.172, 0.184] does NOT resolve above the gate threshold: no detectable identity-channel leak (and none is expected pre-training-contamination — see scope note).
- Era contrast of the index: 2015-2019 [-0.619, 0.056] vs 2020-2024 [-0.172, 0.184] — CIs overlap; no resolved era difference.
- Fingerprint probe, model-minus-reference within-pair excess: 2015-2019: +0.000, 2020-2024: -0.085.
- The model shows NO within-pair discrimination edge over the pure-code feature reference in 2020-2024 — the fingerprint-recall signature is absent at pair level. **Verdict: the era anomaly is not explained by memorization detectable under protocol v1**; the era-asymmetric lift of the flat baselines (random selection already lifts >1x in 2020-2024) points to era composition/base-rate structure instead.

## Per-season detail

| season | A_control | n | A_swap_pf | n |
|---|---|---|---|---|
| 2015 | 0.500 | 5 | 0.300 | 5 |
| 2016 | 0.250 | 6 | 0.857 | 7 |
| 2017 | 0.750 | 2 | 0.667 | 3 |
| 2018 | 0.250 | 4 | 0.900 | 5 |
| 2019 | 0.000 | 2 | 0.500 | 2 |
| 2020 | 0.500 | 2 | 0.333 | 3 |
| 2021 | 0.500 | 4 | 0.300 | 5 |
| 2022 | 1.000 | 1 | 1.000 | 2 |
| 2023 | 0.200 | 5 | 0.400 | 5 |
| 2024 | 0.714 | 7 | 0.571 | 7 |

Run cost: 71216in/11893out tokens across 10 requests ≈ $0.019 (cap $1.00); failures: 0.
