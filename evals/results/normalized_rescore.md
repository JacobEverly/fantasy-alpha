# Normalized rescore — League Index + championship odds

Every headline points result translated onto the league-anchored scale
(evals/normalize.py). League Index: 100 = tied with the best OPPOSING
team in the same simulated league (same seed, same lineup policy);
>100 = outscored everyone. Title/playoff odds: schedule Monte Carlo
(2000 sims/league, weeks 1-14 regular season, top-4 playoff,
semi wk 15, final wks 16+17; deterministic under seed). Raw
points-above-control stays the statistical backbone; this is the
human-readable layer.

## DraftBench heuristic grids (hindsight-optimal lineups, 2015-2024, n=405 each)

Exact re-runs: same simulator, same seeds — simulation-local, no API calls.
Cells are grid means; the control's own means sit in the last columns
(the autopick control is NOT index 100 on average — even the league's
best drafter usually trails somebody in an 11-opponent league).

| agent | pts vs control | League Index | playoff% | title% | control index | control title% |
|---|---|---|---|---|---|---|
| autopick_adp | +0.0 | 87.0 | 39% | 9% | 87.0 | 9% |
| greedy_vorp | -141.5 | 79.6 | 17% | 3% | 87.0 | 9% |
| bestball_adp | -1.5 | 87.0 | 39% | 9% | 87.0 | 9% |
| survival_seq | -15.6 | 86.1 | 37% | 10% | 87.0 | 9% |
| survival_seq_bands55_85 | -17.9 | 86.2 | 36% | 10% | 87.0 | 9% |
| survival_seq_scarcity1.5 | +4.5 | 87.3 | 40% | 10% | 87.0 | 9% |
| survival_seq_vorp | -57.8 | 84.2 | 26% | 6% | 87.0 | 9% |

## DraftGym LLM episodes (realistic-manager lineups)

Reconstructed by replaying each stored episode's recorded picks against
the seed-deterministic opponents (envs/draftgym.py) — the full league is
recovered exactly; a row is 'exact' when the replayed reward matches the
stored one to 0.01. Control = same-seed AutopickADP league, same seat.

| contender | episode | reward (pts) | League Index | rank | playoff% | title% | control index | control title% | exact |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B named | 2018 ppr slot 1 seed 0 | -6.3 | 86.3 | 5/12 | 29% | 27% | 86.7 | 1% | yes |
| Qwen3.5-9B named | 2018 ppr slot 1 seed 1 | +37.6 | 86.8 | 6/12 | 31% | 0% | 81.3 | 0% | yes |
| Qwen3.5-9B named | 2018 ppr slot 7 seed 0 | +165.4 | 108.0 | 1/12 | 96% | 53% | 100.5 | 58% | yes |
| Qwen3.5-9B named | 2018 ppr slot 7 seed 1 | +183.9 | 98.9 | 3/12 | 85% | 8% | 92.5 | 0% | yes |
| Qwen3.5-9B named | 2021 ppr slot 1 seed 0 | +327.4 | 93.9 | 2/12 | 89% | 5% | 78.3 | 0% | yes |
| Qwen3.5-9B named | 2021 ppr slot 1 seed 1 | +245.6 | 109.8 | 1/12 | 98% | 24% | 99.6 | 0% | yes |
| Qwen3.5-9B named | 2021 ppr slot 7 seed 0 | +427.8 | 85.3 | 7/12 | 17% | 1% | 64.9 | 0% | yes |
| Qwen3.5-9B named | 2021 ppr slot 7 seed 1 | +293.1 | 97.5 | 3/12 | 89% | 10% | 83.3 | 3% | yes |
| Qwen3.5-9B named | 2024 ppr slot 1 seed 0 | +140.8 | 69.1 | 10/12 | 1% | 0% | 64.3 | 0% | yes |
| Qwen3.5-9B named | 2024 ppr slot 1 seed 1 | -20.4 | 77.7 | 12/12 | 0% | 0% | 79.9 | 0% | yes |
| Qwen3.5-9B named | 2024 ppr slot 7 seed 0 | -1.8 | 95.6 | 4/12 | 74% | 3% | 99.0 | 18% | yes |
| Qwen3.5-9B named | 2024 ppr slot 7 seed 1 | +256.4 | 92.0 | 3/12 | 66% | 17% | 78.4 | 0% | yes |
| Qwen3.5-9B masked | 2021 ppr slot 1 seed 0 | -76.5 | 71.7 | 9/12 | 12% | 0% | 78.3 | 0% | yes |
| Qwen3.5-9B masked | 2024 ppr slot 7 seed 0 | -141.4 | 85.4 | 9/12 | 24% | 2% | 99.0 | 18% | yes |
| anthropic/claude-fable-5 named | 2021 ppr slot 1 seed 0 | +459.1 | 107.2 | 1/12 | 99% | 48% | 78.3 | 0% | yes |
| anthropic/claude-fable-5 named | 2021 ppr slot 7 seed 0 | +796.3 | 106.4 | 1/12 | 91% | 91% | 64.9 | 0% | yes |
| anthropic/claude-fable-5 masked | 2021 ppr slot 1 seed 0 | -108.4 | 71.5 | 10/12 | 18% | 0% | 78.3 | 0% | yes |
| anthropic/claude-fable-5 masked | 2021 ppr slot 7 seed 0 | +379.8 | 86.4 | 10/12 | 10% | 1% | 64.9 | 0% | yes |
| Qwen3.5-9B named (mean, n=12) | — | +170.8 | 91.7 | — | 56% | 12% | 84.1 | 7% | — |
| Qwen3.5-9B masked (mean, n=2) | — | -108.9 | 78.6 | — | 18% | 1% | 88.7 | 9% | — |

## Approximation ledger

None — every LLM episode replay reproduced its stored reward exactly, and
the heuristic grids are exact deterministic re-runs. No approximated rows.
