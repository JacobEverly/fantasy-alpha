# BreakoutBench — GBDT baseline (boring ML bar)

Generated 2026-08-08 by `evals/gbdt_baseline.py` (seed 20260808, scikit-learn HistGradientBoostingClassifier + CalibratedClassifierCV).

Protocol: strict walk-forward. To predict season S, the model trains only
on gate-eligible candidates from seasons 2011..S-1 (features rebuilt
with the exact `evals/anon_demo.py` slate definitions; labels from
`data/processed/labels/breakouts.csv`). Feature set is exactly the anonymized
slate packet — no extra information. Calibration (sigmoid <500 rows, isotonic
otherwise) is fit inside each fold, so it also sees only seasons <= S-1.
Format: ppr. Eval seasons: 2015-2024. Top-10 picks per slate.

## Per-season results

| Season | Slate n | Breakouts | Base rate | Hits@10 | Lift | Brier (full slate) | Brier (top-10) |
|---|---|---|---|---|---|---|---|
| 2015 | 56 | 10 | 17.9% | 3 | 1.7x | 0.147 | 0.222 |
| 2016 | 48 | 12 | 25.0% | 2 | 0.8x | 0.193 | 0.163 |
| 2017 | 48 | 5 | 10.4% | 1 | 1.0x | 0.105 | 0.113 |
| 2018 | 52 | 9 | 17.3% | 2 | 1.2x | 0.143 | 0.160 |
| 2019 | 54 | 4 | 7.4% | 1 | 1.4x | 0.075 | 0.099 |
| 2020 | 58 | 5 | 8.6% | 1 | 1.2x | 0.084 | 0.099 |
| 2021 | 67 | 8 | 11.9% | 2 | 1.7x | 0.104 | 0.159 |
| 2022 | 33 | 3 | 9.1% | 0 | 0.0x | 0.086 | 0.032 |
| 2023 | 61 | 10 | 16.4% | 1 | 0.6x | 0.145 | 0.113 |
| 2024 | 62 | 14 | 22.6% | 2 | 0.9x | 0.178 | 0.137 |

## Pooled (2015-2024) with season-resampled bootstrap 95% CI (10,000 draws)

| Metric | Value | 95% CI |
|---|---|---|
| Hits@10 (of 100) | 15 | [10, 20] |
| Lift vs random | 1.02x | [0.77x, 1.30x] |
| Brier, full slate | 0.127 | [0.103, 0.151] |
| Brier, top-10 subset | 0.130 | [0.100, 0.159] |

## GBDT vs measured Qwen (same slates, 2015-2024)

Note: the Qwen runs reported Brier over their 10 picks only, so the
apples-to-apples GBDT column is the top-10-subset Brier, not full-slate.

| Model | Hits (of 100) | Lift | Brier full slate | Brier top-10 |
|---|---|---|---|---|
| **GBDT baseline** | **15** | **1.0x** | 0.127 | **0.130** |
| Qwen naked (top-10-only Brier) | 19 | 1.3x | n/a | 0.493 |
| Qwen + harness (top-10-only Brier) | 18 | 1.2x | n/a | 0.241 |
| Random | ~15 | 1.0x | n/a | n/a |

## Reliability (10-bin, pooled full-slate predictions)

**ECE = 0.017** over 539 candidate predictions.

| Bin | n | Mean p | Hit rate |
|---|---|---|---|
| [0.0, 0.1) | 46 | 0.081 | 0.109 |
| [0.1, 0.2) | 411 | 0.155 | 0.146 |
| [0.2, 0.3) | 77 | 0.227 | 0.182 |
| [0.3, 0.4) | 2 | 0.314 | 0.000 |
| [0.4, 0.5) | 3 | 0.434 | 0.333 |
| [0.5, 0.6) | 0 | — | — |
| [0.6, 0.7) | 0 | — | — |
| [0.7, 0.8) | 0 | — | — |
| [0.8, 0.9) | 0 | — | — |
| [0.9, 1.0) | 0 | — | — |
