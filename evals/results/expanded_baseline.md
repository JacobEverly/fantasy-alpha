# BreakoutBench v0.3 — expanded baseline battery

Model: `Qwen/Qwen3.5-9B` via Prime Intellect serverless (`api.pinference.ai`), temp 0, anonymized structure track. Battery: every (season 2015-2024 × format standard/ppr/half_ppr with an ADP snapshot) × 3 families — 5030 scored judgments per run (1480 full_slate, 1296 bust, 2254 over_under), ~50x the original 100-pick run. 95% CIs bootstrap over SEASONS (the independent unit), 10,000 resamples, seeded.

| run | full_slate Brier [95% CI] | bust Brier [95% CI] | over_under Brier [95% CI] | pooled Brier [95% CI] | top-10 lift [95% CI] | answered |
|---|---|---|---|---|---|---|
| Qwen naked | 0.1344 [0.111, 0.163] | 0.1941 [0.171, 0.220] | 0.2482 [0.245, 0.251] | 0.2008 [0.191, 0.213] | 1.14x [0.90, 1.40] | 5030 |
| Qwen + harness | 0.1271 [0.107, 0.152] | 0.1761 [0.159, 0.193] | 0.2452 [0.238, 0.252] | 0.1927 [0.183, 0.203] | 1.09x [0.80, 1.35] | 5030 |
| base rate (no-peek) | 0.1256 [0.105, 0.151] | 0.1754 [0.158, 0.193] | 0.2438 [0.241, 0.247] | 0.1914 [0.182, 0.202] | 1.07x [0.91, 1.21] | 5030 |
| coin (p=0.5) | 0.2500 [0.250, 0.250] | 0.2500 [0.250, 0.250] | 0.2500 [0.250, 0.250] | 0.2500 [0.250, 0.250] | 1.07x [0.91, 1.21] | 5030 |

Lift rows for the flat baselines are tie-broken by shuffled anon id — i.e. random selection at the slate base rate; their CIs bracket 1.0x by construction.

## Per-era split

| run | era | pooled Brier [95% CI] | top-10 lift [95% CI] | n |
|---|---|---|---|---|
| Qwen naked | 2015-2019 | 0.2063 [0.192, 0.227] | 0.77x [0.66, 0.96] | 2224 |
| Qwen naked | 2020-2024 | 0.1964 [0.183, 0.213] | 1.44x [1.29, 1.70] | 2806 |
| Qwen + harness | 2015-2019 | 0.1946 [0.186, 0.206] | 0.99x [0.44, 1.50] | 2224 |
| Qwen + harness | 2020-2024 | 0.1911 [0.176, 0.209] | 1.17x [0.95, 1.34] | 2806 |
| base rate | 2015-2019 | 0.1943 [0.185, 0.207] | 0.94x [0.67, 1.14] | 2224 |
| base rate | 2020-2024 | 0.1891 [0.174, 0.206] | 1.17x [1.06, 1.38] | 2806 |
| coin | 2015-2019 | 0.2500 [0.250, 0.250] | 0.94x [0.67, 1.14] | 2224 |
| coin | 2020-2024 | 0.2500 [0.250, 0.250] | 1.17x [1.06, 1.38] | 2806 |

## Verdicts (significant iff 95% CIs do not overlap)

- **Top-10 selection lift — naked vs harness: NOT significant (CIs overlap).** naked [0.90, 1.40] vs harness [0.80, 1.35]
- **Top-10 selection lift — naked vs base rate/random: NOT significant (CIs overlap).** naked [0.90, 1.40] vs base rate/random [0.91, 1.21]
- **Top-10 selection lift — harness vs base rate/random: NOT significant (CIs overlap).** harness [0.80, 1.35] vs base rate/random [0.91, 1.21]
- **Pooled Brier — naked vs harness: NOT significant (CIs overlap).** naked [0.191, 0.213] vs harness [0.183, 0.203]
- **Pooled Brier — naked vs base rate: NOT significant (CIs overlap).** naked [0.191, 0.213] vs base rate [0.182, 0.202]
- **Pooled Brier — harness vs base rate: NOT significant (CIs overlap).** harness [0.183, 0.203] vs base rate [0.182, 0.202]
- **Pooled Brier — harness vs coin: significant (CIs do not overlap).** harness [0.183, 0.203] vs coin [0.250, 0.250]

Failures skipped-and-recorded: naked 0, harness 0 (see evals/results/answers/failures_*.json).
This battery's marginal usage: 1667423in/221302out tokens across 162 requests ≈ $0.420 (cap $3.00).
