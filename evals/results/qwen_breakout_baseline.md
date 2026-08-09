# BreakoutBench — untrained open-model baseline (B2-naked)

Model: `Qwen/Qwen3.5-9B` via Prime Intellect serverless (`api.pinference.ai`), temp 0, anonymized structure track.

| season | breakouts in slate | hits@10 | random@10 | Brier |
|---|---|---|---|---|
| 2015 | 10 | 2 | 1.8 | 0.454 |
| 2016 | 12 | 3 | 2.5 | 0.439 |
| 2017 | 5 | 1 | 1.0 | 0.524 |
| 2018 | 9 | 2 | 1.7 | 0.497 |
| 2019 | 4 | 1 | 0.7 | 0.579 |
| 2020 | 5 | 1 | 0.9 | 0.485 |
| 2021 | 8 | 1 | 1.2 | 0.531 |
| 2022 | 3 | 3 | 0.9 | 0.437 |
| 2023 | 10 | 2 | 1.6 | 0.553 |
| 2024 | 14 | 3 | 2.3 | 0.431 |

**Pooled: 19 hits / 14.6 expected at random (lift 1.3x) · mean Brier 0.493 · 100 picks over 10 seasons**

Comparison: Claude (Fable, in-session demo, 2019 only, partial de-anonymization disclosed): 3 hits, 4.0x lift, Brier 0.192.

This run's marginal usage: 29173in/848out tokens.