# evals/

Label builders and benchmarks (GAMEPLAN §4). Everything here is stdlib-only
and runs from repo data snapshots. **2025 is the untouched holdout** — no
eval defaults include it; it is evaluated once per contender at gate time.

- `build_labels.py` — weekly/season fantasy points + finish ranks per scoring
  preset (`data/processed/labels/{weekly,season}_points.csv`).
- `build_breakout_labels.py` — ADP-vs-finish alpha/breakout/bust labels
  (`data/processed/labels/breakouts.csv`, see `docs/breakoutbench-design.md`).
- `names.py` — cross-source player-name normalization used by every join.
- `draftbench.py` — DraftBench v0, below.

## DraftBench v0 (`draftbench.py`)

Draft-replay benchmark: 12-team snake drafts over historical FFC ADP boards
(QB/RB/WR/TE), roster QB1/RB2/WR3/TE1/FLEX2/BENCH6, 15 rounds. Opponents are
noisy-ADP autopickers (`score = adp + Normal(0, player's FFC stdev)`) that
respect roster feasibility; a league-level scarcity guard keeps every roster
legal even on boards thinner than 180 picks. Finished rosters are scored by
realized weekly points with hindsight-optimal lineups, weeks 1–17, plus a
weeks 15–17 "fantasy playoffs" subtotal.

Metrics per run: roster points, points above the league-median opponent, and
**points above an AutopickADP control drafted in the same seat with the same
seed** — the market baseline every agent must beat.

Baseline agents: `autopick_adp` (lowest-ADP feasible — IS the market),
`greedy_vorp` (naive projection = last season's points; max VORP via
`harness.valuation`), `bestball_adp` (autopick that refuses QB/TE before
round 5). Custom agents implement
`pick(board, my_roster, league, pick_number) -> player_id`.

```
.venv/bin/python -m evals.draftbench                  # full default grid
.venv/bin/python -m evals.draftbench --seasons 2018-2022 --formats ppr --slots 1,6,12 --seeds 5
```

Output: `data/processed/draftbench/results_v0.csv` (one row per
agent × season × format × slot × seed) plus a printed summary table.
Defaults: seasons 2015–2024, formats standard/half_ppr/ppr (half_ppr ADP
exists 2018+), slots 1/6/12, 5 seeds.

First run (2026-08-08, n=405 per agent, mean ± std points above control):
`greedy_vorp` **−141.5 ± 271.2** (last-year's-points projections lose to the
market, as expected), `bestball_adp` **−1.5 ± 65.8** (structure-only tweak is
market-neutral). Negative results are published, not hidden — the bar for a
real agent is beating the autopick control, and this pins where the bar sits.

v0 simplifications (v1 candidates): no positional-run contagion in the
opponent model, no schedule Monte Carlo / league win rate, hindsight-optimal
lineup setting (no start/sit noise), name+position join to realized outcomes
(~97% match), no anonymized variant yet (GAMEPLAN §4 requires it before any
frontier-model comparison on historical seasons).

## CalibBench v0 (`calibbench.py`)

Calibration benchmark (GAMEPLAN §4.2): dated forecast questions with realized
outcomes, generated deterministically per (season, format) from the ADP
snapshots + label CSVs. No model calls live here — models plug in via an
answers JSON, same as `anon_demo.py`. Two families:

- **season_threshold** — "Will this player finish top-12 (QB/TE) / top-24
  (RB/WR) at position this season?" One per QB/RB/WR/TE on the season's ADP
  board. Packet: prior-season stats only (strictly seasons < target) plus the
  target season's ADP (`anon_demo`-style as-of discipline).
- **weekly_h2h** — "Will player A outscore player B in week W?" ~50 pairs per
  season, weeks 5–16: same position, both played week W, ≥3 games and ≥5.0
  season-to-date ppg entering W, ppg within 20% of each other (deliberately
  coin-flip-ish, so calibration is what separates models). Packet per side:
  games/total/ppg strictly from weeks < W.

`build` writes to `evals/questions/`: a **named** variant (internal
regression only — pretraining-contaminated, never published), an
**anonymized** variant (shuffled `Q####` ids, no names, season masked — the
structure track), and a sequestered **KEY** file holding outcomes + the
anon-id map. Question files contain no outcome fields; the scorer accepts
named or anon ids interchangeably.

Metrics (`score`): Brier, log-loss, 10-bin reliability table (n / mean p /
realized freq), ECE — per family and overall. `baseline` scores two reference
forecasters: **coin** (p=0.5) and **base-rate** (per-family base rate over
2015–2023 *excluding the scored season* — it never peeks, and never touches
2024–2025).

```
.venv/bin/python -m evals.calibbench build --season 2019 --format ppr
.venv/bin/python -m evals.calibbench score --season 2019 --format ppr --answers picks.json
.venv/bin/python -m evals.calibbench baseline --season 2019 --format ppr
```

Answers JSON: `[{"question_id": ..., "p": <0..1>}]`. **2025 is the untouched
holdout** — every subcommand refuses it without `--include-holdout`
(gate-time only).

First baselines (2026-08-08, ppr; Brier / ECE):

| season | family | n | coin | base-rate |
|---|---|---|---|---|
| 2019 | season_threshold | 170 | 0.2500 / 0.1235 | 0.2347 / 0.0009 |
| 2019 | weekly_h2h | 50 | 0.2500 / 0.0200 | 0.2499 / 0.0175 |
| 2022 | season_threshold | 146 | 0.2500 / 0.0685 | 0.2489 / 0.0600 |
| 2022 | weekly_h2h | 50 | 0.2500 / 0.0600 | 0.2503 / 0.0625 |

Reading: the h2h family is base-rate ~0.5 by construction, so coin ≈
base-rate there and any model Brier < 0.25 is real skill. On the season
family the base rate (~0.37) buys a few Brier points over coin; a model must
beat ~0.23, not 0.25. Season base rates move with board depth (2022's board
carries fewer gate-band players), so ECE for the cross-season base rate is
honest, not zero.

v0 simplifications (v1 candidates): no breakout-family questions (that is
BreakoutBench / `anon_demo.py` for now), no difficulty-adjusted Brier
(ForecastBench-style, adopted in `docs/breakoutbench-design.md` §8), no
memorization canary in the anonymized variant, ties in h2h score as "no".
