# Fantasy Alpha Tinker SFT T1.2 — stopped at the canary gate

Decision: **stop additional SFT; do not begin RL**.

T1.2 did not proceed to full training or the paid four-way evaluation. This is
the intended behavior of the frozen gate, not an incomplete run. Both canaries
completed their full lifecycle, reloaded, sampled, and exported successfully.

## What happened

| gate | first canary | one allowed revision |
|---|---:|---:|
| training rows | 54 | 230 |
| final development NLL | 0.9605 | 0.7939 |
| structured coverage | 100.0% | 100.0% |
| target match | 43.8% | 65.6% |
| valid tool-call rate | 0.0% | 0.0% |
| illegal/wrong-type actions | 3 | 3 |

The revision doubled tool-call learning weight from 5% to 10% and expanded the
one-epoch canary from 54 to 230 rows. It raised target matching from 43.8% to
65.6% and preserved perfect parseability, but the model still chose a pick on
all three unseen states where the correct action was a depth-chart lookup.

## Interpretation

SFT learned surface formats and some targets, but it did not learn the conditional
policy boundary between acting and gathering evidence. More loss reduction is not
evidence that a full adapter would solve that boundary. The next useful experiment
would redesign tool supervision as contrastive paired decisions or evaluate a
hard-coded tool router; it should not spend on DraftGym RL yet.

## Spending

Exact incremental T1.2 spend: **$1.239102**.
Exact cumulative Tinker workload: **$10.855193**.
Ongoing checkpoint storage is separate. The $7 target and $10 hard cap were respected.

2025 stayed sealed. No full T1.2 adapter and no RL run were started.
