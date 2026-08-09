# Slot-value curves (expected realized VORP by overall draft slot)

Generated 2026-08-08 by `evals/slot_values.py`. Canonical league: 12-team QB1/RB2/WR3/TE1/FLEX2 (2qb format: QB2, standard points). Seasons pooled per format below; 2025 holdout untouched.

Same-season join is deliberate: this is a scoring curve (slot price axis for the pick-timing family), never packet/feature data.

## ppr

Seasons 2011-2024 (14 pooled), 2167 drafted players matched, 37 unmatched (scored as 0-point busts).

| bucket | n | mean VORP | median | p25 | p75 | smoothed @ bucket start |
|---|---|---|---|---|---|---|
| 1-3 | 42 | 99.67 | 103.05 | 62.7 | 166.4 | 109.85 |
| 4-6 | 42 | 112.54 | 125.94 | 55.55 | 194.97 | 109.85 |
| 7-9 | 42 | 103.23 | 94.3 | 49.12 | 167.7 | 101.38 |
| 10-12 | 42 | 88.03 | 93.7 | 24.27 | 157.62 | 90.95 |
| 13-24 | 168 | 80.68 | 89.15 | 38.62 | 143.33 | 90.95 |
| 25-36 | 168 | 53.81 | 58.0 | 6.83 | 102.89 | 65.46 |
| 37-48 | 168 | 46.7 | 56.23 | 4.72 | 92.82 | 55.12 |
| 49-60 | 168 | 14.39 | 18.6 | -40.5 | 66.88 | 34.32 |
| 61-72 | 168 | 14.92 | 20.77 | -39.91 | 72.17 | 14.47 |
| 73-84 | 168 | -0.62 | 0.0 | -39.12 | 45.8 | 0.0 |
| 85-96 | 160 | -4.69 | -0.05 | -50.75 | 33.78 | 0.0 |
| 97-108 | 146 | -4.18 | 2.8 | -55.63 | 46.75 | 0.0 |
| 109-120 | 137 | -18.39 | -12.68 | -69.2 | 24.8 | 0.0 |
| 121-132 | 139 | -10.24 | -11.14 | -60.45 | 43.2 | 0.0 |
| 133-144 | 129 | -35.75 | -36.9 | -89.9 | 3.0 | 0.0 |
| 145-156 | 106 | -39.78 | -41.15 | -87.3 | 6.58 | 0.0 |
| 157-168 | 115 | -40.43 | -45.72 | -92.87 | 9.55 | 0.0 |
| 169-180 | 96 | -35.06 | -35.55 | -82.25 | 0.0 | 0.0 |

Face validity: picks 1-3 mean VORP 99.67 vs round-8 (85-96) mean -4.69 -> PASS. Smoothed curve is monotone non-increasing by construction.

## half-ppr

Seasons 2018-2024 (7 pooled), 1082 drafted players matched, 18 unmatched (scored as 0-point busts).

| bucket | n | mean VORP | median | p25 | p75 | smoothed @ bucket start |
|---|---|---|---|---|---|---|
| 1-3 | 21 | 82.35 | 87.32 | 16.3 | 170.16 | 105.68 |
| 4-6 | 21 | 118.78 | 102.2 | 62.2 | 187.06 | 105.68 |
| 7-9 | 21 | 94.38 | 93.4 | 61.3 | 143.18 | 105.68 |
| 10-12 | 21 | 107.94 | 102.5 | 76.54 | 158.7 | 105.68 |
| 13-24 | 84 | 78.31 | 84.6 | 28.02 | 139.62 | 105.68 |
| 25-36 | 84 | 50.93 | 60.6 | 12.18 | 88.43 | 57.21 |
| 37-48 | 84 | 44.6 | 50.1 | 1.5 | 79.39 | 51.9 |
| 49-60 | 84 | 17.29 | 10.6 | -19.12 | 61.42 | 22.13 |
| 61-72 | 84 | 17.8 | 28.7 | -21.83 | 65.35 | 22.03 |
| 73-84 | 84 | 4.12 | 8.5 | -33.43 | 42.35 | 7.71 |
| 85-96 | 83 | -1.99 | -2.2 | -43.5 | 35.25 | 0.0 |
| 97-108 | 83 | -15.47 | -9.6 | -52.9 | 25.9 | 0.0 |
| 109-120 | 71 | 1.29 | -1.38 | -39.3 | 50.35 | 0.0 |
| 121-132 | 67 | -12.85 | -19.6 | -55.45 | 33.1 | 0.0 |
| 133-144 | 59 | -22.9 | -20.26 | -56.95 | 8.25 | 0.0 |
| 145-156 | 40 | -43.1 | -52.94 | -84.07 | -18.95 | 0.0 |
| 157-168 | 51 | -41.55 | -36.8 | -89.65 | 1.15 | 0.0 |
| 169-180 | 58 | -37.28 | -35.2 | -94.97 | 0.0 | 0.0 |

Face validity: picks 1-3 mean VORP 82.35 vs round-8 (85-96) mean -1.99 -> PASS. Smoothed curve is monotone non-increasing by construction.

## standard

Seasons 2011-2024 (14 pooled), 2189 drafted players matched, 38 unmatched (scored as 0-point busts).

| bucket | n | mean VORP | median | p25 | p75 | smoothed @ bucket start |
|---|---|---|---|---|---|---|
| 1-3 | 42 | 80.11 | 80.55 | 25.55 | 153.22 | 88.11 |
| 4-6 | 42 | 96.12 | 96.5 | 46.09 | 144.97 | 88.11 |
| 7-9 | 42 | 81.3 | 83.55 | 26.07 | 136.6 | 84.06 |
| 10-12 | 42 | 82.64 | 84.0 | 52.82 | 123.3 | 84.06 |
| 13-24 | 168 | 68.15 | 73.05 | 26.89 | 108.1 | 82.94 |
| 25-36 | 168 | 44.43 | 48.7 | 0.0 | 75.28 | 56.9 |
| 37-48 | 168 | 29.48 | 36.1 | -6.15 | 63.58 | 33.05 |
| 49-60 | 168 | 21.61 | 20.58 | -18.01 | 63.53 | 25.94 |
| 61-72 | 168 | 12.94 | 17.6 | -18.45 | 49.42 | 15.27 |
| 73-84 | 168 | 0.51 | -3.17 | -32.23 | 35.4 | 10.1 |
| 85-96 | 160 | 4.42 | 1.0 | -35.97 | 34.25 | 0.0 |
| 97-108 | 157 | -11.48 | -6.8 | -43.5 | 31.1 | 0.0 |
| 109-120 | 144 | -3.73 | -0.15 | -39.95 | 32.25 | 0.0 |
| 121-132 | 149 | -14.86 | -9.6 | -53.7 | 19.84 | 0.0 |
| 133-144 | 133 | -29.85 | -31.74 | -63.0 | 5.1 | 0.0 |
| 145-156 | 102 | -28.42 | -24.1 | -66.09 | 8.12 | 0.0 |
| 157-168 | 115 | -21.78 | -25.0 | -54.1 | 10.38 | 0.0 |
| 169-180 | 91 | -33.82 | -35.3 | -66.88 | -1.27 | 0.0 |

Face validity: picks 1-3 mean VORP 80.11 vs round-8 (85-96) mean 4.42 -> PASS. Smoothed curve is monotone non-increasing by construction.

## 2qb

Seasons 2014-2024 (11 pooled), 1764 drafted players matched, 35 unmatched (scored as 0-point busts).

| bucket | n | mean VORP | median | p25 | p75 | smoothed @ bucket start |
|---|---|---|---|---|---|---|
| 1-3 | 33 | 91.98 | 103.0 | 26.6 | 170.9 | 110.9 |
| 4-6 | 33 | 125.54 | 119.66 | 75.6 | 179.56 | 110.9 |
| 7-9 | 33 | 91.59 | 94.64 | 29.6 | 153.2 | 98.04 |
| 10-12 | 33 | 86.83 | 92.7 | 54.6 | 131.9 | 91.28 |
| 13-24 | 132 | 81.41 | 85.36 | 44.73 | 124.45 | 91.28 |
| 25-36 | 132 | 62.45 | 61.7 | 19.61 | 108.22 | 75.03 |
| 37-48 | 132 | 46.68 | 53.5 | 16.47 | 85.53 | 58.44 |
| 49-60 | 132 | 34.59 | 38.85 | -3.42 | 65.13 | 41.77 |
| 61-72 | 132 | 27.86 | 30.8 | -20.68 | 74.22 | 32.01 |
| 73-84 | 132 | 14.78 | 21.22 | -23.4 | 53.22 | 19.78 |
| 85-96 | 132 | 4.25 | 1.4 | -30.25 | 38.8 | 9.39 |
| 97-108 | 128 | -1.62 | -5.35 | -37.52 | 35.68 | 0.26 |
| 109-120 | 123 | 4.07 | 0.0 | -38.0 | 39.52 | 0.26 |
| 121-132 | 115 | -13.18 | -11.0 | -47.45 | 14.18 | 0.0 |
| 133-144 | 106 | -9.13 | -3.85 | -47.38 | 23.53 | 0.0 |
| 145-156 | 91 | -15.78 | -9.8 | -60.05 | 23.47 | 0.0 |
| 157-168 | 90 | -14.53 | -17.12 | -43.8 | 15.67 | 0.0 |
| 169-180 | 90 | -22.68 | -25.0 | -59.3 | 8.68 | 0.0 |

Face validity: picks 1-3 mean VORP 91.98 vs round-8 (85-96) mean 4.25 -> PASS. Smoothed curve is monotone non-increasing by construction.

## Worked timing-score examples (ppr curve)

### (a) 8th-round-ADP conviction call (adp 90, stdev 8, realized VORP 120)
- acted at pick 1: act_pick=1, adp=90, stdev=8, realized_vorp=120 -> survival=1.000, slot_expected_vorp=109.85, **timing_score=10.15**
- acted at pick 78: act_pick=78, adp=90, stdev=8, realized_vorp=120 -> survival=0.933, slot_expected_vorp=0.00, **timing_score=111.98**
- acted at pick 100: act_pick=100, adp=90, stdev=8, realized_vorp=120 -> survival=0.106, slot_expected_vorp=0.00, **timing_score=12.68**

The round-7 action (pick 78) wins: nearly full survival odds at a slot priced far below the player's realized value. Pick 1 pays a first-round opportunity cost for the same player; pick 100 is after the market -- survival collapses and the credit evaporates.

### (b) Undrafted flier acted at pick 175 (adp 200, stdev 15, realized VORP 80)
- acted at pick 175: act_pick=175, adp=200, stdev=15, realized_vorp=80 -> survival=0.952, slot_expected_vorp=0.00, **timing_score=76.18**

A last-round slot prices near replacement, so a hit there is almost pure capture.

### (c) First-rounder taken at his ADP (adp 5, stdev 2, realized VORP 150)
- acted at pick 5: act_pick=5, adp=5, stdev=2, realized_vorp=150 -> survival=0.500, slot_expected_vorp=101.38, **timing_score=-26.38**

At your own ADP survival is a coin flip and the slot price is steep -- taking the market's pick at the market's price earns roughly nothing, exactly the intended economics.
