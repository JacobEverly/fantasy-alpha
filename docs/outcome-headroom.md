# Outcome headroom before model training

Fantasy Alpha treats a snake draft as a finite-horizon stochastic allocation game. The terminal objective is the roster's realized season points under a deterministic weekly manager that ranks players using prior weeks only. Subtracting the same-seat, same-opponents, same-seed legal ADP roster is only the evaluation layer.

The deployable planner uses ADP, ADP dispersion, roster constraints, positional scarcity, and distance to the next snake turn. It never sees the target season's outcomes. The hindsight arm does see them and is an unattainable greedy ceiling reference—not a training input and not a claim of global optimality.

## Result

| panel | n | mean | 10% trimmed | W-L-T | no largest gain | 95% bootstrap CI | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| selection (2015–2020, sealed years excluded) | 180 | +8.09 | +5.58 | 63-57-60 | +6.06 | [-5.12, +21.81] | pass |
| later validation (2021–2022) | 90 | +4.66 | +3.32 | 26-29-35 | +1.32 | [-13.95, +23.35] | fail |
| all development | 270 | +6.95 | +4.79 | 89-86-95 | +5.59 | [-4.15, +17.93] | pass |
| hindsight reference | 270 | +694.22 | +686.75 | 270-0-0 | +691.75 | [+668.45, +720.12] | reference |

The planner has positive average value and the full development panel narrowly passes the mechanical gate. The later-season replication does not: it wins 26 and loses 29 drafts, and its interval includes zero. The honest conclusion is provisional headroom, not a repeatable win. Paid LLM training remains gated until the rollout planner or simple ranker clears the later-season test.

Machine-readable rows and exact metrics are in `artifacts/outcome-headroom-v1/`.
