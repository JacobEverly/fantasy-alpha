# BreakoutBench — untrained open model + quant harness (B2-with-harness)

Model: `Qwen/Qwen3.5-9B` via Prime Intellect serverless (`api.pinference.ai`), temp 0, anonymized structure track.
Harness: pure-code grounding stage appends `historical_grounding` (cohort / prior-rank-cohort / position base rates, computed strictly from seasons before the eval season) to each candidate; prompt instructs the model to anchor to those base rates (cap 0.60 without support).

| season | breakouts in slate | hits@10 | random@10 | Brier |
|---|---|---|---|---|
| 2015 | 10 | 2 | 1.8 | 0.229 |
| 2016 | 12 | 5 | 2.5 | 0.298 |
| 2017 | 5 | 1 | 1.0 | 0.229 |
| 2018 | 9 | 2 | 1.7 | 0.247 |
| 2019 | 4 | 1 | 0.7 | 0.262 |
| 2020 | 5 | 2 | 0.9 | 0.223 |
| 2021 | 8 | 1 | 1.2 | 0.223 |
| 2022 | 3 | 2 | 0.9 | 0.215 |
| 2023 | 10 | 1 | 1.6 | 0.225 |
| 2024 | 14 | 1 | 2.3 | 0.256 |

**Pooled: 18 hits / 14.6 expected at random (lift 1.2x) · mean Brier 0.241 · 100 picks over 10 seasons**

## Comparison (pooled, 100 picks)

| run | hits | lift vs random | mean Brier |
|---|---|---|---|
| B2-naked (no harness) | 19 | 1.3x | 0.493 |
| B2-with-harness | 18 | 1.2x | 0.241 |
| random@10 | 14.6 | 1.0x | — |

This run's marginal usage: 111030in/2139out tokens.