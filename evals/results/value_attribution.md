# Value ledger — per-pick value capture vs the slot-price curve

Generated 2026-08-08 by `evals/value_attribution.py`.

Per pick: `capture = realized_season_vorp(player) - slot_expected_vorp(pick, fmt)` (cost axis: evals/slot_values.py; realized VORP via the same canonical 12-team QB1/RB2/WR3/TE1/FLEX2 replacement machinery, hindsight full-season totals — see the module docstring's hindsight-vs-realistic note). **Attribution/telemetry only — never reward**: DraftGym's terminal reward already prices value capture implicitly; shaping with this signal would double-count it.

Autopick IS the market's base rate of value capture (everyone lands some +50 picks by luck). Every policy is therefore reported relative to its same-seed autopick control: the claim format is "finds MORE +50 players than the market does", mean paired per-draft delta with a 10k-resample bootstrap 95% CI.

Grid policies: 2015-2024 x standard/half_ppr/ppr x slots 1/6/12 x 5 seeds (half_ppr ADP exists 2018+; n=405 drafts). LLM policies: the stored DraftGym episodes replayed exactly from seeds + recorded picks (evals/normalize.py rescore pattern); small n — read the CIs.

## Policy comparison — value capture per draft (<=15 picks; thin QB/RB/WR/TE boards can end drafts a pick early)

| policy | drafts | +25/draft | **+50/draft** | +100/draft | Δ+50 vs market [95% CI] | ≥1 +50 | ≥2 +50 | capture mean | median | p10 | p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| autopick_adp | 405 | 4.33 | **2.70** | 0.78 | 0 (is the market) | 91% | 76% | -7.3 | -5.0 | -91.8 | +76.8 |
| greedy_vorp | 405 | 3.35 | **1.62** | 0.34 | -1.07 [-1.25, -0.89] | 77% | 50% | -18.9 | -16.5 | -92.7 | +55.3 |
| bestball_adp | 405 | 4.33 | **2.69** | 0.77 | -0.01 [-0.05, +0.04] | 91% | 75% | -7.2 | -4.9 | -91.8 | +76.3 |
| survival_seq | 405 | 4.54 | **2.68** | 0.61 | -0.01 [-0.14, +0.10] | 93% | 75% | -6.0 | -1.9 | -90.2 | +72.2 |
| survival_seq_bands55_85 | 405 | 4.54 | **2.69** | 0.62 | -0.00 [-0.13, +0.12] | 93% | 75% | -5.9 | -2.0 | -90.2 | +72.2 |
| survival_seq_scarcity1.5 | 405 | 4.41 | **2.72** | 0.73 | +0.03 [-0.05, +0.10] | 92% | 75% | -6.6 | -4.2 | -91.8 | +75.0 |
| survival_seq_vorp | 405 | 4.11 | **2.18** | 0.45 | -0.52 [-0.69, -0.35] | 89% | 67% | -7.8 | -3.7 | -83.9 | +64.3 |
| qwen3.5-9b_named | 12 | 6.33 | **4.00** | 1.83 | +0.83 [+0.00, +1.67] | 100% | 92% | +8.5 | +8.8 | -81.6 | +106.3 |
| qwen3.5-9b_masked | 2 | 4.00 | **2.00** | 1.00 | -1.50 [-2.00, -1.00] | 100% | 50% | -17.2 | -27.2 | -99.9 | +63.9 |
| claude-fable-5_named | 2 | 8.00 | **7.00** | 4.00 | +6.00 [+6.00, +6.00] | 100% | 100% | +41.0 | +29.5 | -61.4 | +163.7 |
| claude-fable-5_masked | 2 | 3.50 | **2.00** | 0.50 | +1.00 [+0.00, +2.00] | 100% | 100% | -11.8 | -8.3 | -88.3 | +56.4 |

Named-vs-masked LLM rows are the memorization story told per player: a named model can "find" +50 players by remembering how the season went; masked boards force the capture rate back toward what features alone support.

## Where captures come from

+50 captures by round bucket (counts across each policy's drafts; grid policies n=405 drafts, LLM policies far fewer — rates not directly comparable across different n):

| policy | 1-3 | 4-6 | 7-9 | 10-12 | 13+ |
|---|---|---|---|---|---|
| autopick_adp | 290 | 294 | 245 | 203 | 60 |
| greedy_vorp | 166 | 145 | 158 | 138 | 51 |
| bestball_adp | 292 | 291 | 243 | 202 | 62 |
| survival_seq | 294 | 292 | 226 | 209 | 65 |
| survival_seq_bands55_85 | 295 | 292 | 229 | 207 | 68 |
| survival_seq_scarcity1.5 | 292 | 307 | 244 | 198 | 61 |
| survival_seq_vorp | 203 | 256 | 210 | 164 | 49 |
| qwen3.5-9b_named | 13 | 10 | 10 | 12 | 3 |
| qwen3.5-9b_masked | 2 | 2 | 0 | 0 | 0 |
| claude-fable-5_named | 1 | 4 | 5 | 1 | 3 |
| claude-fable-5_masked | 1 | 1 | 0 | 1 | 1 |

+50 captures by position:

| policy | QB | RB | WR | TE |
|---|---|---|---|---|
| autopick_adp | 110 | 416 | 517 | 49 |
| greedy_vorp | 73 | 171 | 404 | 10 |
| bestball_adp | 95 | 427 | 528 | 40 |
| survival_seq | 76 | 197 | 762 | 51 |
| survival_seq_bands55_85 | 76 | 196 | 765 | 54 |
| survival_seq_scarcity1.5 | 94 | 398 | 553 | 57 |
| survival_seq_vorp | 75 | 130 | 640 | 37 |
| qwen3.5-9b_named | 5 | 10 | 29 | 4 |
| qwen3.5-9b_masked | 1 | 1 | 2 | 0 |
| claude-fable-5_named | 0 | 5 | 9 | 0 |
| claude-fable-5_masked | 1 | 1 | 2 | 0 |

## Freak-injury caveat (luck, visible separately)

Rows join `data/processed/labels/injury_context.csv`; `season_ending_acute` players are flagged so a -50 'loss' that was a torn ACL in week 2 (luck) is never silently read as drafting skill, and a +50 on a flagged player (value banked before the injury) is visible too:

| policy | +50 on flagged | of +50 total | -50 on flagged | of -50 total |
|---|---|---|---|---|
| autopick_adp | 10 | 1092 | 259 | 1448 |
| greedy_vorp | 0 | 658 | 212 | 1487 |
| bestball_adp | 12 | 1090 | 258 | 1430 |
| survival_seq | 13 | 1086 | 259 | 1407 |
| survival_seq_bands55_85 | 13 | 1091 | 260 | 1403 |
| survival_seq_scarcity1.5 | 8 | 1102 | 248 | 1441 |
| survival_seq_vorp | 1 | 882 | 288 | 1217 |
| qwen3.5-9b_named | 0 | 48 | 9 | 37 |
| qwen3.5-9b_masked | 0 | 4 | 2 | 8 |
| claude-fable-5_named | 1 | 14 | 2 | 4 |
| claude-fable-5_masked | 0 | 4 | 2 | 7 |

## Example top captures

- **Cooper Kupp** (WR, 2021 ppr): pick 43 (round 4, ADP 41.3) -> realized VORP +298.6 vs slot cost 34.3 = **capture +264.3** [bestball_adp]
- **Deebo Samuel Sr.** (WR, 2021 ppr): pick 97 (round 9, ADP 89.8) -> realized VORP +198.1 vs slot cost 0.0 = **capture +198.1** [autopick_adp]
- **Brandon Marshall** (WR, 2015 ppr): pick 54 (round 5, ADP 50.5) -> realized VORP +205.6 vs slot cost 14.5 = **capture +191.1** [autopick_adp]

## Replay exactness

All LLM episode replays reproduced their stored rewards exactly.

## Product note

DraftGym now emits this table as terminal telemetry (`info["value_ledger"]`, reward math untouched). This is the post-draft user receipt: "your best value: X at pick 68, +87 vs slot" — per-pick rows with name, slot cost, realized VORP, capture, and the injury-luck flag, ready for the app's draft recap surface (timestamps/freshness per DESIGN.md when live).

Reproduce: `python -m evals.value_attribution`
