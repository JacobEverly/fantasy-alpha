# Fantasy Alpha targeted Tinker SFT T1.1 — frozen development evaluation

Decision: **revise sft before rl**.

This is a three-way matched comparison of the untouched Qwen3.5-9B base,
the original T1 adapter, and the targeted T1.1 adapter. The 2018/2023/2024
suite is development evidence because it was opened during T1. The 2025
one-shot gate remains sealed.

## Scorecard

| metric | base | T1 | T1.1 |
|---|---:|---:|---:|
| BreakoutBench Brier ↓ | 0.1981 | 0.1991 | 0.1656 |
| CalibBench Brier ↓ | 0.2441 | 0.2458 | 0.2423 |
| CalibBench log loss ↓ | 0.6845 | 0.7095 | 0.6796 |
| CalibBench ECE ↓ | 0.0819 | 0.0850 | 0.1448 |
| BreakoutBench top-10 lift ↑ | 1.4214 | 1.2437 | 1.7333 |
| Masked DraftGym mean reward ↑ | 12.1125 | -6.0875 | 3.0367 |
| Masked DraftGym median reward ↑ | 13.7200 | -30.1500 | 0.0000 |
| Structured answer coverage ↑ | 1.0000 | 1.0000 | 0.5214 |

## Paired draft result

| comparison | wins | win rate | median delta | trimmed mean | episode 95% CI | season-clustered 95% CI |
|---|---:|---:|---:|---:|---|---|
| T1.1 vs base | 12/24 | 50.0% | -13.72 | -10.36 | [-61.58666666666667, 44.884166666666665] | [-62.32750000000001, 31.395] |
| T1.1 vs t1 | 13/24 | 54.2% | +30.15 | +22.29 | [-59.89916666666667, 74.11166666666666] | [-28.415000000000003, 76.8175] |

Every paired episode:

| season/slot/seed | base | T1 | T1.1 | T1.1−base | T1.1−T1 |
|---|---:|---:|---:|---:|---:|
| 2018/1/11 | -9.14 | +170.96 | +0.00 | +9.14 | -170.96 |
| 2018/1/29 | -137.24 | +13.16 | +0.00 | +137.24 | -13.16 |
| 2018/4/11 | +161.64 | -109.44 | +0.00 | -161.64 | +109.44 |
| 2018/4/29 | +38.26 | -148.72 | +0.00 | -38.26 | +148.72 |
| 2018/7/11 | -25.34 | +133.08 | +0.00 | +25.34 | -133.08 |
| 2018/7/29 | -161.06 | +6.22 | +0.00 | +161.06 | -6.22 |
| 2018/10/11 | +121.94 | -103.68 | +0.00 | -121.94 | +103.68 |
| 2018/10/29 | -18.70 | +265.74 | +0.00 | +18.70 | -265.74 |
| 2023/1/11 | +46.58 | +2.28 | +0.00 | -46.58 | -2.28 |
| 2023/1/29 | +92.52 | +378.12 | +0.00 | -92.52 | -378.12 |
| 2023/4/11 | -40.82 | -64.38 | +0.00 | +40.82 | +64.38 |
| 2023/4/29 | +45.54 | -92.20 | +0.00 | -45.54 | +92.20 |
| 2023/7/11 | +88.78 | -22.84 | +0.00 | -88.78 | +22.84 |
| 2023/7/29 | -136.74 | -192.74 | -18.52 | +118.22 | +174.22 |
| 2023/10/11 | +347.66 | +323.36 | +0.00 | -347.66 | -323.36 |
| 2023/10/29 | +36.58 | -181.88 | +0.00 | -36.58 | +181.88 |
| 2024/1/11 | +193.50 | -162.00 | +92.10 | -101.40 | +254.10 |
| 2024/1/29 | -18.78 | +48.78 | -0.70 | +18.08 | -49.48 |
| 2024/4/11 | -240.80 | -172.20 | +0.00 | +240.80 | +172.20 |
| 2024/4/29 | -257.82 | +93.70 | +0.00 | +257.82 | -93.70 |
| 2024/7/11 | -44.20 | -151.14 | +0.00 | +44.20 | +151.14 |
| 2024/7/29 | +153.16 | +77.72 | +0.00 | -153.16 | -77.72 |
| 2024/10/11 | -41.48 | -37.46 | +0.00 | +41.48 | +37.46 |
| 2024/10/29 | +96.66 | -220.54 | +0.00 | -96.66 | +220.54 |

## Mistake-prevention diagnostics

| diagnostic | base | T1 | T1.1 |
|---|---:|---:|---:|
| early qb or te before round 5 | 17 | 18 | 13 |
| episodes with more than two qbs | 8 | 14 | 5 |
| episodes with more than two tes | 5 | 1 | 0 |
| late round qb or te | 27 | 24 | 25 |

These are predeclared process diagnostics. They identify roster/timing mistakes
prevented or introduced without using post-season outcomes as labels.

## Evidence-tool behavior

T1.1 made 0 calls across 246 predeclared
opportunities: precision 0.0%, recall 0.0%,
useful-result rate 0.0%, and unnecessary-call rate
0.0%. These are policy-alignment metrics under the
frozen opportunity definition, not a claim that every lookup causally improved reward.

## What the experiment actually learned

The adapter learned the targeted corpus extremely well in the narrow training-loss
sense: development NLL fell from 1.7267 to 0.1113. That did **not** translate into
a stronger product policy. T1.1 returned a zero DraftGym reward in 21 of 24 episodes,
which means it usually reproduced the control draft rather than finding extra value.
It reduced some roster-construction mistakes relative to T1 (for example, episodes
with more than two quarterbacks fell from 14 to 5), but it did not beat the untouched
base model and introduced three illegal picks.

The larger failure was multi-task brittleness. Eight of 18 probability-output families
still emitted multiple JSON values after the one allowed repair, leaving only 52.1%
structured coverage. The apparently better Brier score is therefore not a clean win:
it is calculated only on the surviving answers, while nearly half of the required
answers are missing. ECE also regressed by 0.063 versus base. T1.1 made zero tool calls
across 246 opportunities, so the targeted tool examples did not become a usable tool
policy.

This is evidence of over-specialization, not evidence that targeted SFT is useless.
The next SFT revision should mix the broad T1 corpus with the targeted corrections,
give each output schema explicit development gates, use a lower learning rate or
earlier checkpoint, and require tool-call behavior to pass before DraftGym evaluation.
RL should not start yet: it would optimize a brittle policy and make the failure harder
to diagnose.

## Evaluation recovery note

The frozen runner originally aborted when both the initial strict-JSON response and
its single repair were malformed. Prompts, decoding, repair count, keys, thresholds,
and scoring were left frozen. A conservative continuation recorded each terminal
failure as zero answers and continued; all eight failures count against structured
coverage. Three aborted sessions were retained in the spend audit rather than hidden.

## Spending

Exact T1.1 incremental spend: **$4.489987**.
This includes **$0.129938** for the three failed/retried inference sessions.
Exact cumulative Tinker workload spend: **$9.616091**.
Ongoing checkpoint storage is separate.

## Precommitted decision

- FAIL — draft beats base
- FAIL — draft beats t1
- FAIL — prediction non regressive vs base
- FAIL — prediction non regressive vs t1
- FAIL — tool policy
- FAIL — structured output and legality
- PASS — canary
- PASS — contamination and point in time

The JSON scorecard contains every episode, paired delta, prediction regression,
tool event, bootstrap interval, and exact price-based cost. No RL was started.
