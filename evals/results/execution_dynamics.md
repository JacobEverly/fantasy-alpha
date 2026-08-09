# Execution dynamics vs the market (DraftBench)

Does draft *execution* alone — survival-conditional sequencing, turn
awareness, scarcity discipline, QB/TE guardrails (research board:
draft-board.md, edge-framework.md §10) — beat ADP autopick, with no
projection skill at all?

- `survival_seq*`: market-consensus values (ADP order) + execution logic.
- `survival_seq_vorp`: same execution, GreedyVORP's naive prev-season
  projections as values (plain GreedyVORP: ~-141 vs market).
- Grid: 2015–2024 × standard/ppr/half_ppr × slots 1/6/12 × 5 seeds,
  paired same-seed AutopickADP control. half_ppr ADP exists 2018+.
- Cells: mean ± std of points-above-control, paired bootstrap 95% CI
  (10k resamples) in brackets.

| agent | standard | ppr | half_ppr | overall | beats market |
|---|---|---|---|---|---|
| survival_seq | -68.4 ± 178.2 [-96.4, -39.4] (n=150) | +28.4 ± 182.8 [-0.3, +58.1] (n=150) | -3.1 ± 189.7 [-38.8, +32.2] (n=105) | -15.6 ± 187.3 [-33.9, +2.1] (n=405) | **unresolved** |
| survival_seq_bands55_85 | -69.5 ± 183.6 [-98.2, -40.0] (n=150) | +22.1 ± 186.6 [-7.2, +52.6] (n=150) | -1.4 ± 183.6 [-35.6, +33.1] (n=105) | -17.9 ± 188.7 [-36.4, +0.0] (n=405) | **unresolved** |
| survival_seq_scarcity1.5 | -0.7 ± 91.2 [-15.3, +14.0] (n=150) | +4.0 ± 110.1 [-12.8, +21.6] (n=150) | +12.4 ± 105.9 [-7.3, +33.6] (n=105) | +4.5 ± 102.2 [-5.4, +14.4] (n=405) | **unresolved** |
| survival_seq_vorp | -67.6 ± 226.9 [-103.6, -30.8] (n=150) | -37.2 ± 241.4 [-74.9, +0.8] (n=150) | -73.0 ± 230.2 [-117.7, -30.0] (n=105) | -57.8 ± 233.2 [-80.7, -34.8] (n=405) | **no** |

Verdicts use the overall paired bootstrap 95% CI: yes = CI entirely
above 0, no = entirely below 0, unresolved otherwise.

Rows: data/processed/draftbench/results_execution.csv
## Verdicts (2026-08-08 run, seeds 0-4)

- survival_seq (bands 0.40/0.70, scarcity 1.0) — beats market: **unresolved**
  (-15.6, CI [-33.9, +2.1]). Points-above-median is +26, but against the
  paired control it does not separate from zero.
- survival_seq_bands55_85 — beats market: **unresolved** (-17.9,
  CI [-36.4, +0.0]). Widening the take band changes almost nothing.
- survival_seq_scarcity1.5 — beats market: **unresolved** (+4.5,
  CI [-5.4, +14.4]). The winning config: eager scarcity makes the agent
  hew closer to the board (std 102 vs 187), removing the losing deviations
  the survival bands introduce, but the residual edge is ~+4 points — noise.
- survival_seq_vorp — beats market: **no** (-57.8, CI [-80.7, -34.8]).
  BUT execution discipline recovers ~84 points of plain GreedyVORP's -141
  (same grid, results_v0.csv): sequencing/guardrails rescue roughly 60% of
  the damage bad projections do, without fixing the projections.

## Reading

Execution dynamics alone do NOT beat the market on DraftBench v0. That is
the expected null under this simulator: opponents ARE the noisy-ADP market,
so market-consensus values + timing tricks can only reshuffle which
ADP-priced players you get, and the survival model mostly re-derives what
AutopickADP does implicitly. Format split is suggestive (ppr +28 for the
default config, standard -68) but its CI still straddles zero. The real
value of the execution layer shows up in the VORP variant: it is a large,
significant improvement over the same values without execution discipline.
Execution is a damage-limiter for imperfect projections, not an independent
source of alpha. Next probe: pair it with a projection source that has any
skill (GBDT baseline) rather than prev-season points.
