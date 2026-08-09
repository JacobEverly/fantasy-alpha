# Enriched-packet experiment — does the evidence channel add rankable signal?

Frame: seasons 2016-2024 (news coverage), format ppr, all three families; **full_slate is the primary question** (483 judgments). Enrichment = packet-v2 fields with honest historical coverage (news 90-day window counts + keyword flags, S-1 season-end depth rank, age, draft capital, opportunities/game + target share + YoY), joined from `data/processed/packet_features/`; the as-of-null channels (depth movement, vacated opportunity, coaching) are excluded. Structure-only rows are the SAME contenders re-scored on this identical frame (subset of the expanded battery), so every comparison is like-for-like. 95% CIs: season-resampled bootstrap (10,000 draws, seeded). GBDT rows are full_slate-only by construction.

## Side-by-side: structure-only vs enriched

| contender | full_slate Brier [95% CI] | pooled Brier [95% CI] | top-10 lift [95% CI] | n answered |
|---|---|---|---|---|
| GBDT structure (train 2011..S-1) | 0.1250 [0.100, 0.151] | — | 0.93x [0.70, 1.18] | 483 |
| GBDT structure, short history (train 2015..S-1) | 0.1276 [0.102, 0.155] | — | 0.78x [0.51, 1.02] | 483 |
| **GBDT enriched v2** (train 2015..S-1) | 0.1273 [0.102, 0.154] | — | 0.93x [0.59, 1.26] | 483 |
| Qwen naked, structure | 0.1320 [0.102, 0.165] | 0.1998 [0.192, 0.210] | 1.21x [0.89, 1.52] | 1662 |
| **Qwen naked, enriched v2** | 0.1288 [0.100, 0.161] | 0.2131 [0.199, 0.228] | 0.98x [0.50, 1.39] | 1662 |
| Qwen harness, structure | 0.1281 [0.104, 0.154] | 0.1931 [0.186, 0.201] | 1.06x [0.57, 1.41] | 1662 |
| **Qwen harness, enriched v2** | 0.1264 [0.102, 0.153] | 0.1933 [0.186, 0.201] | 1.28x [0.86, 1.57] | 1662 |
| base rate (no-peek, feature-blind) | 0.1273 [0.101, 0.156] | 0.1923 [0.183, 0.203] | 1.06x [0.76, 1.31] | 1662 |
| coin (p=0.5) | 0.2500 [0.250, 0.250] | 0.2500 [0.250, 0.250] | 1.06x [0.76, 1.31] | 1662 |

Paired season-bootstrap deltas (enriched − structure, same seasons; negative Brier delta = enrichment helped, positive lift delta = enrichment helped):

| contender | Δ full-frame Brier [95% CI] | Δ top-10 lift [95% CI] |
|---|---|---|
| GBDT (v2 − v1-short) | -0.0002 [-0.0015, 0.0007] | +0.16 [-0.29, 0.56] |
| Qwen naked (v2 − v1) | +0.0132 [0.0011, 0.0297] | -0.23 [-0.88, 0.37] |
| Qwen harness (v2 − v1) | +0.0002 [-0.0014, 0.0017] | +0.23 [0.00, 0.49] |

(GBDT delta pools full_slate only; Qwen deltas pool all three families for Brier and full_slate for lift — same convention as the headline table.)

## Per-era split

| contender | era | Brier [95% CI] | top-10 lift [95% CI] | n |
|---|---|---|---|---|
| GBDT structure | 2016-2019 | 0.1277 [0.089, 0.171] | 1.00x [0.85, 1.23] | 202 |
| GBDT structure | 2020-2024 | 0.1230 [0.090, 0.156] | 0.87x [0.45, 1.32] | 281 |
| GBDT structure-short | 2016-2019 | 0.1335 [0.094, 0.181] | 0.67x [0.31, 1.08] | 202 |
| GBDT structure-short | 2020-2024 | 0.1233 [0.092, 0.157] | 0.87x [0.50, 1.11] | 281 |
| GBDT enriched v2 | 2016-2019 | 0.1323 [0.094, 0.177] | 0.83x [0.40, 1.44] | 202 |
| GBDT enriched v2 | 2020-2024 | 0.1238 [0.092, 0.158] | 1.02x [0.50, 1.42] | 281 |
| qwen_naked (structure) | 2016-2019 | 0.2043 [0.192, 0.225] | 0.80x [0.31, 1.07] | 721 |
| qwen_naked (structure) | 2020-2024 | 0.1963 [0.189, 0.206] | 1.57x [1.32, 1.88] | 941 |
| qwen_naked_v2 (enriched) | 2016-2019 | 0.2151 [0.201, 0.230] | 1.13x [0.79, 1.62] | 721 |
| qwen_naked_v2 (enriched) | 2020-2024 | 0.2115 [0.192, 0.236] | 0.86x [0.00, 1.48] | 941 |
| qwen_harness (structure) | 2016-2019 | 0.1954 [0.189, 0.207] | 1.13x [0.00, 1.69] | 721 |
| qwen_harness (structure) | 2020-2024 | 0.1913 [0.180, 0.203] | 1.00x [0.54, 1.26] | 941 |
| qwen_harness_v2 (enriched) | 2016-2019 | 0.1947 [0.187, 0.206] | 1.45x [0.56, 1.80] | 721 |
| qwen_harness_v2 (enriched) | 2020-2024 | 0.1921 [0.181, 0.204] | 1.14x [0.67, 1.37] | 941 |
| base_rate | 2016-2019 | 0.1963 [0.184, 0.214] | 1.29x [1.03, 1.62] | 721 |
| base_rate | 2020-2024 | 0.1892 [0.177, 0.202] | 0.86x [0.41, 1.18] | 941 |

## Verdicts

**(i) Does enrichment give GBDT rankable signal?** GBDT-v2 top-10 lift 0.93x [0.59, 1.26] — NO — CI brackets 1.0. Paired Brier delta vs the structure-only control (same training history): -0.0002 [-0.0015, 0.0007] — CI includes 0, no detectable change.

**(ii) Does enrichment give Qwen any?** naked_v2 lift 0.98x [0.50, 1.39] — NO — CI brackets 1.0; harness_v2 lift 1.28x [0.86, 1.57] — NO — CI brackets 1.0. Paired deltas vs their structure-only runs: naked Δlift -0.23 [-0.88, 0.37], harness Δlift +0.23 [0.00, 0.49]. Note: naked Qwen's pooled Brier delta (+0.0132 [0.0011, 0.0297]) excludes 0 on the WRONG side — enrichment measurably worsened its calibration (it overreacts to the added evidence).

**(iii) Does the LLM gain more from enrichment than GBDT (co-design)?** GBDT Δlift +0.16 [-0.29, 0.56] vs Qwen naked Δlift -0.23 [-0.88, 0.37] / harness Δlift +0.23 [0.00, 0.49] — the delta CIs overlap heavily; no evidence either learner extracts more from the channel.

## GBDT-v2 permutation importances (mean held-out Brier increase, n-weighted across folds)

| feature | Δ Brier when shuffled |
|---|---|
| target_share_s1 (enrichment) | +0.00078 |
| opps_pg_s1 (enrichment) | +0.00028 |
| two_ago_games | +0.00015 |
| injury_mention_flags (enrichment) | +0.00013 |
| two_ago_pos_rank | +0.00006 |
| opps_pg_yoy (enrichment) | +0.00004 |
| two_ago_ppg | +0.00004 |
| seasons_of_data | +0.00003 |
| pos_rank_s1_end (enrichment) | +0.00002 |
| age_at_sept1 (enrichment) | +0.00002 |
| prior_missing | +0.00000 |
| undrafted (enrichment) | +0.00000 |
| pos_TE | +0.00000 |
| opps_pg_s1_missing (enrichment) | +0.00000 |
| target_share_s1_missing (enrichment) | +0.00000 |
| opps_pg_yoy_missing (enrichment) | +0.00000 |
| pos_rank_s1_end_missing (enrichment) | +0.00000 |
| age_at_sept1_missing (enrichment) | +0.00000 |

Summed importance — structure columns -0.00104, enrichment columns +0.00058. (Negative sums mean shuffling those columns on average IMPROVED held-out Brier — noise, not signal.)

## Cost

Qwen v2 batteries this run: 54 requests, 1021536in/72993out tokens ≈ $0.223 (cap $3.00; logged in docs/budget-ledger.md). GBDT/baselines: local CPU, $0.
