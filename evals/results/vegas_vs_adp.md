# Vegas vs ADP — does the week-1 implied-total signal survive ADP conditioning at the player level?

Generated 2026-08-09 by `evals/vegas_packet_test.py` (seed 20260808, 10,000 season-resampled bootstrap draws; GBDT + base-rate machinery only, no API calls).

Setup: `evals/results/vegas_history.md` measured the week-1 implied team total as additive over naive prior-season team stats (+0.23 [+0.11, +0.32] incremental r, 2016-2024) at the TEAM level. This test asks whether that edge survives at the PLAYER level once ADP — the drafting market's player consensus — is conditioned on. Frame: seasons 2016-2024, ppr, walk-forward (train 2015..S-1 for every arm). Team vegas joined per packet_features conventions (as-of team, else S-1 season-end depth team; era codes normalized to current franchise codes). Join coverage: 312/483 (65%) of gate-eligible candidates carry a week-1 implied total (the rest are rookies/no-S-1-depth players — encoded as missing, never zero).

## 1. Full-slate GBDT arms (paired, same folds, same seasons)

| arm | features | Brier (full slate) [95% CI] | top-10 hits (of 90) | top-10 lift [95% CI] |
|---|---|---|---|---|
| GBDT v1s | structure only (ADP + prior seasons) | 0.1276 [0.1021, 0.1549] | 11 | 0.85x [0.37, 1.45] |
| GBDT v2 | v1 + enrichment (news/depth/usage/pedigree) | 0.1272 [0.1019, 0.1543] | 13 | 1.01x [0.71, 1.34] |
| GBDT v3 | **v2 + team vegas (week-1 implied, prior-season mean implied)** | 0.1275 [0.1019, 0.1551] | 11 | 0.85x [0.57, 1.08] |

Paired season-bootstrap deltas (negative Brier delta = vegas/enrichment helped; positive lift delta = helped):

| comparison | Δ Brier [95% CI] | Δ top-10 lift [95% CI] | Δ hits [95% CI] | read |
|---|---|---|---|---|
| **v3 − v2** (vegas over everything) | +0.0003 [-0.0014, +0.0022] | -0.16 [-0.61, +0.26] | -2 [-7, +4] | CI includes 0 — no detectable change |
| v3 − v1s (vegas + enrichment over structure) | -0.0001 [-0.0011, +0.0009] | +0.00 [-0.71, +0.50] | +0 [-9, +7] | CI includes 0 — no detectable change |
| v2 − v1s (enrichment over structure) | -0.0004 [-0.0017, +0.0009] | +0.16 [-0.37, +0.67] | +2 [-5, +8] | CI includes 0 — no detectable change |

## 2. Bust / over_under families (v3 − v2, same machinery)

| family | v2 Brier | v3 Brier | base-rate Brier | Δ Brier (v3 − v2) [95% CI] |
|---|---|---|---|---|
| bust (n=432) | 0.1750 | 0.1766 | 0.1756 | +0.0015 [-0.0002, +0.0041] |
| over_under (n=747) | 0.2489 | 0.2497 | 0.2440 | +0.0008 [-0.0012, +0.0031] |

## 3. Permutation importances (v3 model, mean held-out Brier increase, n-weighted across folds)

All 47 columns ranked; the two vegas columns:

| vegas column | Δ Brier when shuffled | rank of 47 |
|---|---|---|
| team_week1_implied_total | +0.00011 | 5 |
| team_prior_season_mean_implied_total | +0.00025 | 2 |
| team_week1_implied_total_missing | +0.00000 | 27 |
| team_prior_season_mean_implied_total_missing | +0.00000 | 28 |

Top 12 columns overall (for scale):

| feature | Δ Brier when shuffled |
|---|---|
| target_share_s1 | +0.00037 |
| team_prior_season_mean_implied_total **(vegas)** | +0.00025 |
| years_since_first_season | +0.00021 |
| two_ago_games | +0.00017 |
| team_week1_implied_total **(vegas)** | +0.00011 |
| opps_pg_yoy | +0.00006 |
| seasons_of_data | +0.00005 |
| camp_promotion_flags | +0.00002 |
| two_ago_total | +0.00001 |
| pos_rank_s1_end | +0.00000 |
| pos_QB | +0.00000 |
| two_ago_missing | +0.00000 |

Summed vegas-column importance: +0.00036 (negative = shuffling vegas on average IMPROVED held-out Brier — noise, not signal).

## 4. Direct stratified read (the fame-control pattern from source_alpha)

Among gate-eligible candidates, breakout rate by positional-ADP band x per-season week-1 implied-total tercile. If Vegas carries player-selection signal beyond ADP, high-implied-total offenses should hit more WITHIN each band.

### draft-day-legal attach (packet_features team convention) — 312 candidate rows, week-1 implied vs ADP mean r = -0.00

| band | T1 low (rate, n) | T2 mid (rate, n) | T3 high (rate, n) | T3/T1 ratio [95% CI] |
|---|---|---|---|---|
| near_gate | 20.9% (14/67) | 19.0% (12/63) | 17.0% (9/53) | 0.81 [0.46, 1.37] |
| mid | 15.8% (6/38) | 13.2% (5/38) | 12.2% (6/49) | 0.78 [0.40, 1.68] |
| deep | 0.0% (0/1) | 0.0% (0/3) | — | — [—] |
| all | 18.9% (20/106) | 16.3% (17/104) | 14.7% (15/102) | 0.78 [0.54, 1.08] |

### robustness: actual season-S week-1 team (post-gate source standing in for legal roster knowledge) — 462 candidate rows, week-1 implied vs ADP mean r = -0.01

| band | T1 low (rate, n) | T2 mid (rate, n) | T3 high (rate, n) | T3/T1 ratio [95% CI] |
|---|---|---|---|---|
| near_gate | 20.7% (19/92) | 18.0% (16/89) | 16.7% (14/84) | 0.81 [0.37, 1.67] |
| mid | 12.9% (8/62) | 12.7% (8/63) | 7.6% (5/66) | 0.59 [0.26, 1.10] |
| deep | 0.0% (0/2) | 0.0% (0/3) | 0.0% (0/1) | — [—] |
| all | 17.3% (27/156) | 15.5% (24/155) | 12.6% (19/151) | 0.73 [0.42, 1.17] |

## Verdict

**Vegas is NOT additive over ADP for player selection — treat it as already priced for drafting purposes.** The team-level week-1 signal (vegas_history.md, +0.23 incremental r over naive stats) does not survive ADP conditioning at the player level: adding the implied-total pair moves neither Brier nor top-10 lift by a detectable amount on any family, and it does not stratify breakout rate within ADP bands (point estimates even lean below 1.0). Within this pool the week-1 implied total is nearly orthogonal to ADP (r = -0.00), so this is not literal pricing — it is dilution: knowing an offense will score does not identify WHICH of its discounted players captures the points.

- Headline paired delta (v3 − v2): Brier +0.0003 [-0.0014, +0.0022], top-10 lift -0.16 [-0.61, +0.26] — CI includes 0 — no detectable change.
- Vegas permutation importance sums to +0.00036 (vs adp_overall at -0.00017).
- Stratified T3/T1 ratios within ADP bands: near_gate 0.81 [0.46, 1.37]; mid 0.78 [0.40, 1.68]; deep —.
- Week-1 implied total vs overall ADP among candidates: mean per-season r = -0.00 (restriction of range — the gate-eligible pool is all late-ADP players, so this measures orthogonality within the pool, not across the whole draft board).

## Caveats

- Team attach for historical seasons is the S-1 season-end depth team (no dated pre-Sept-1 snapshots exist before 2025), so offseason movers are attributed to their OLD team's line in the draft-day-legal arm; the true-team robustness arm above bounds how much that attenuates the read.
- Join coverage is 65% of candidates; the uncovered tail is rookie-heavy, where team context arguably matters most — this test cannot speak to them.
- Week-1 implied totals are CLOSING week-1 lines (games.csv); August look-ahead lines are slightly staler, so any live edge would be smaller still (same caveat as vegas_history.md).
- 2025 remains the untouched holdout; nothing here touches it.
