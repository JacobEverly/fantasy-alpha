# Budget ledger — BreakoutBench untrained baseline run

Hard budget: $15.00. All spend via Prime Intellect serverless inference API
(`https://api.pinference.ai/api/v1`), billed to team `Plantmangroup`
(`cmskuc3x7018j7g10tr01gukl`) via `PRIME_API_KEY`. No pods provisioned.

| Date (UTC) | What | Rate | Duration / volume | Est. cost |
|---|---|---|---|---|
| 2026-08-08 | Sanity ping 1, Qwen/Qwen3.5-9B (thinking on, hit personal balance first — failed, $0) | $0.18/M in, $0.54/M out | 27 in / 50 out tok | $0.0001 (API-reported) |
| 2026-08-08 | Sanity ping 2, Qwen/Qwen3.5-9B (enable_thinking=false) | $0.18/M in, $0.54/M out | 29 in / 6 out tok | $0.0001 (API-reported) |
| 2026-08-08 | Eval run: 10 slates (2015-2024), 1 request each, temp 0 (6 by agent, 4 by resume script; X-Prime-Team-ID + UA headers required) | $0.18/M in, $0.54/M out | ~60k in / ~2k out tok total | ~$0.012 |
| 2026-08-08 | B2-with-harness eval run: 10 grounded slates (2015-2024), 1 request each, temp 0, no retries needed (evals/harness_breakout.py) | $0.18/M in, $0.54/M out | 111,030 in / 2,139 out tok total | ~$0.021 |
| 2026-08-08 | BreakoutBench v0.3 expanded battery: naked + harness Qwen runs, 162 requests over 27 season-formats × 3 families (evals/run_expanded_qwen.py) | $0.18/M in, $0.54/M out | 1,667,423 in / 221,302 out tok | $0.420 (API token counts) |
| 2026-08-08 | BreakoutBench enriched-packet (v2) battery: naked_v2 + harness_v2 Qwen runs, 54 requests over 9 season-formats × 3 families (evals/run_expanded_qwen.py v2) | $0.18/M in, $0.54/M out | 1,021,536 in / 72,993 out tok | $0.223 (API token counts) |
| 2026-08-08 | Memorization canary run (protocol v1): naked Qwen over 10 canary ppr full_slate sets (evals/canaries.py) | $0.18/M in, $0.54/M out | 71,216 in / 11,893 out tok | $0.019 (API token counts) |
| 2026-08-08 | DraftGym untrained-baseline: 14 episodes (2018/2021/2024 × slots 1,7 × 2 seeds, 12-team ppr, 15 rounds), Qwen/Qwen3.5-9B temp 0, pick-or-tool prompt (envs/play_llm.py) | $0.18/M in, $0.54/M out | 484,717 in / 2,056 out tok | ~$0.088 |
| 2026-08-08 | SFT pilot teacher selection: catalog query + sanity ping, qwen/qwen3.5-397b-a17b (strongest permissive-licensed model on the serverless catalog; selection note in training/README.md) | $0.60/M in, $3.60/M out | 19 in / 1 out tok | $0.0001 (API-reported) |
| 2026-08-08 | SFT pilot v0 trace generation (training/sft_datagen.py): 152 questions (breakoutbench full_slate/bust + calibbench season_threshold, named, 2016–2023) × 2 samples, temp 0.7, cap $8 — 304 calls incl. 9 rate-limit retries | $0.60/M in, $3.60/M out | 373,540 in / 88,545 out tok | ~$0.547 (max of API-reported and catalog-rate) |

| 2026-08-08 | Frontier battery pre-restriction burn — 8-model tier-1 partial via OpenRouter, killed when Jacob narrowed scope to dsv4flash+qwen35b (~29 slates + killed-in-flight requests + sanity/diagnostic pings; credits-endpoint delta, per-model split unavailable without a management key) | OpenRouter usage.cost | ~35 requests | $2.910 (API-reported; includes Fable protocol pilots) |

| 2026-08-08 | Frontier battery via OpenRouter — `deepseek/deepseek-v4-flash`: naked + canary + harness ppr batteries (evals/run_frontier_battery.py) | OpenRouter usage.cost | 919,083 in / 185,641 out tok, 106 req | $0.133 (API-reported) |
| 2026-08-08 | Frontier battery via OpenRouter — `qwen/qwen3.5-35b-a3b`: naked + canary + harness ppr batteries (evals/run_frontier_battery.py) | OpenRouter usage.cost | 1,031,359 in / 399,053 out tok, 110 req | $0.670 (API-reported) |

| 2026-08-08 | Frontier battery via OpenRouter — `anthropic/claude-fable-5`: naked + canary + harness ppr batteries + 4 DraftGym episodes (evals/run_frontier_battery.py) | OpenRouter usage.cost | 967,253 in / 81,200 out tok, 131 req | $13.733 (API-reported) |

## T1 SFT corpus generation (2026-08-09/11, task cap $10)

| Date (UTC) | What | Rate | Duration / volume | Est. cost |
|---|---|---|---|---|
| 2026-08-11 | Teacher sanity ping, qwen/qwen3.5-397b-a17b | $0.60/M in, $3.60/M out | 19 in / 1 out tok | $0.0001 (API-reported) |
| 2026-08-09/11 | T1 anonymized-track trace generation (training/sft_datagen.py t1-generate): 1,354 anon questions (full_slate/bust/season_threshold/weekly_h2h, seasons 2008-2022 minus {2013, 2018}) × 2 samples, temp 0.7, cap $9 — first invocation killed by org API limit at 1,522/2,708 calls, resumed (resumable by design) | $0.60/M in, $3.60/M out | 1,427,723 in / 449,935 out tok through 1,522 calls; resume in flight | $2.495 through 1,522 calls (max of API-reported and catalog-rate); final figure below |
| 2026-08-11 | T1 resume completion — see final corpus stats in data/processed/sft/t1_filter_report.md | $0.60/M in, $3.60/M out | pending | pending |

## T0 infra smoke test (2026-08-09, Jacob's explicit go, hard cap $50)

| Date (UTC) | What | Rate | Duration / volume | Est. cost |
|---|---|---|---|---|
| 2026-08-09 | T0 pod: 1×H100_80GB SXM5 spot (datacrunch FIN-01 via Prime, pod 5cdcf034a2804e7e859d2f55e4a96d67, cloudId 1H100.80S.30V_SPOT, secure_cloud, ubuntu_22_cuda_12, 150GB disk) — created 02:23:20 UTC, ACTIVE 02:25, prime-rl installed 02:29, SFT launched 02:32; **driving agent died (Claude org spend limit), pod auto-terminated 05:20:23 UTC (3h)**; API confirms TERMINATED. Milestones proven: provision→install→SFT-launch. Unproven: checkpoint/resume/generation/download/run-template — T0b redo needed (~$5) | $1.6661/hr (API-reported) | 2h57m | **~$4.92** |

| 2026-08-09 | T0b pod: 1×A100_80GB PCIe spot (datacrunch FIN-02 via Prime, pod 76ce0cb1ca0f4e8fb5c3f3cd16dac6d0) — created 13:58:15 UTC; **T0 COMPLETE**: LoRA SFT ran (Qwen3-4B-Instruct-2507, loss 3.05→1.31, val 3.06→1.55, 0 NaN), checkpoints at steps 12/24/36/40 + trainer resume-states, generation from step_40 verified (grounded-reasoning style confirmed), LoRA adapters + metrics + configs downloaded to training/checkpoints/t0/. Driver agent stalled 2×; main session disarmed self-destruct 2 min before artifact loss, finished verification inline, terminated pod 17:14 UTC (API-verified TERMINATED) | $0.9361/hr (API-reported) | ~3h16m | **~$3.06** |

## Tinker Qwen3.5-9B SFT experiment (2026-09-05)

Prices are the published Qwen3.5-9B Tinker rates recorded in
`training/tinker_backend.py`. Training and sampling token counts come from the
Tinker responses. The provider billing-event export is retained as an
independent check, but Tinker documents that recent events can lag; adapter
storage is ongoing and not included in the fixed workload total below.

| Date (UTC) | What | Volume | Exact price-based cost |
|---|---|---|---:|
| 2026-09-05 | Small paid T1-format canary, rank 32, balanced 24-row slice, 12 optimizer steps, save/reload/export + two samples | 222,456 train/forward + 1,786 prefill + 552 output tokens | **$0.327733** |
| 2026-09-05 | Exact existing T0 `pilot_v0` parity smoke, rank 32, 162 rows, 20 optimizer steps, save/reload/export + two samples | 520,581 train/forward + 2,466 prefill + 616 output tokens | **$0.764466** |
| 2026-09-05 | T1 managed-LoRA run, `Qwen/Qwen3.5-9B`, rank 32, 811 rows (728 train/83 development), 2 epochs/46 steps, two samples | 2,155,863 train/forward + 2,466 prefill + 594 output tokens | **$3.156840** |
| 2026-09-05 | Frozen untouched-base evaluation, 18 prediction cells + 6 masked DraftGym episodes | 502,720 uncached + 40,064 cached prefill + 48,836 output tokens | **$0.434511** |
| 2026-09-05 | Frozen selected-adapter evaluation, identical workload | 518,477 uncached + 23,680 cached prefill + 48,738 output tokens | **$0.442553** |
|  | **Tinker experiment workload total** |  | **$5.126104** |

Tinker billing snapshot: `artifacts/tinker-sft-v1/billing.json`. The snapshot
also records 42.63 GB-hours of checkpoint storage accrued by query time
(approximately $0.0059 at $0.10/GB-month); storage continues until checkpoints
are removed and is therefore reported separately from the completed workload.

## Targeted Tinker T1.1 (preregistered 2026-09-05)

Hard incremental cap: **$10.00**. Frozen estimate: **$4.51** ($0.20 final-format
canary, $1.71 controlled rank-32 training run, and $2.60 for identical base/T1/T1.1
evaluation across prediction benches and 24 masked DraftGym episodes). The 2025
gate remains sealed. The completed provider-reconciled workload was:

| Date (UTC) | What | Exact price-based cost |
|---|---|---:|
| 2026-09-05 | T1.1 final-format canary | **$0.243131** |
| 2026-09-05 | T1.1 rank-32 training, 300 targeted traces, 2 epochs/18 steps | **$1.712584** |
| 2026-09-05 | Frozen untouched-base evaluation, 18 prediction families + 24 DraftGym episodes | **$0.808163** |
| 2026-09-05 | Frozen T1 evaluation, identical workload | **$0.807874** |
| 2026-09-05 | Completed T1.1 evaluation session | **$0.788297** |
| 2026-09-05 | Three failed/retried T1.1 strict-JSON sessions, provider billing reconciled | **$0.129938** |
|  | **T1.1 incremental total** | **$4.489987** |

The detailed token/session reconciliation is in
`artifacts/tinker-sft-t11/evaluation/spend-audit.json`. Exact cumulative Tinker
workload spend is **$9.616091**. Ongoing checkpoint storage remains separate.

## Tinker T1.2 broad-plus-targeted canary gate (2026-09-05)

Target incremental spend: **under $7**. Hard cap: **$10**. The preregistered
protocol required stopping before full training when the representative format
canary failed. One dataset-weighting/canary-size revision was permitted.

| Date (UTC) | What | Exact price-based cost |
|---|---|---:|
| 2026-09-05 | First T1.2 54-row canary: one epoch, save/reload/export | **$0.241897** |
| 2026-09-05 | First held-out 32-prompt schema/tool behavior gate | **$0.040929** |
| 2026-09-05 | One allowed revised 230-row canary: one epoch, save/reload/export | **$0.915515** |
| 2026-09-05 | Revised held-out 32-prompt schema/tool behavior gate | **$0.040761** |
|  | **T1.2 incremental total** | **$1.239102** |

Both canaries had 100% structured coverage but 0/3 valid tool decisions. Full
T1.2 training, frozen T1.2 inference, and RL were not started. Request token
counts and published model rates establish the exact price-based total. The
retained Tinker billing snapshot was still three hourly buckets behind the last
canary when queried; ongoing checkpoint storage is separate.

## Tool-decision supervisor experiment (2026-09-06)

Target incremental spend: under **$5**. Hard cap: **$10**. Exact price-based
cost uses provider-returned token counts and the pinned Qwen3.5-9B rates. The
provider billing event export is retained at
`artifacts/tool-decision-supervisor-v1/billing.json` as an independent check;
ongoing checkpoint storage is excluded.

| Date (UTC) | What | Exact price-based cost |
|---|---|---:|
| 2026-09-06 | Untouched base, 164-row development decision evaluation | **$0.229297** |
| 2026-09-06 | Contrastive rank-32 canary v1, 160 rows, five steps, reload/export | **$0.684672** |
| 2026-09-06 | Canary v1 development behavior evaluation | **$0.225781** |
| 2026-09-06 | One allowed rank-32 canary v2 revision, 320 rows, nine steps, reload/export | **$1.333664** |
| 2026-09-06 | Canary v2 development behavior evaluation | **$0.225304** |
| 2026-09-06 | Untouched base, frozen 220-row held-out decision evaluation | **$0.301981** |
| 2026-09-06 | Frozen DraftGym integration: 21 episodes across three arms | **$0.456939** |
|  | **Tool-decision experiment total** | **$3.457638** |

Both SFT canaries failed their precommitted development gates, so the full
adapter and RL were not run. Exact cumulative completed Tinker workload is
**$14.312830878**. Total project spend is now approximately **$41.07** (prior
historical estimate ~$26.76 plus exact Tinker workload; ongoing storage
excluded). The exact nine-decimal line items are in
`artifacts/tool-decision-supervisor-v1/spend-audit.json`.

## Outcome-linked value-of-information supervisor (2026-09-06)

Target incremental spend: under **$5**. Hard cap: **$10**. The experiment used
the same pinned Qwen3.5-9B sampling prices. One learned-arm canary was
interrupted after exposing a repeated-tool loop; Tinker's provider-authoritative
20:00–22:00 UTC billing window was reconciled against all completed request
ledgers to capture its server-side work exactly.

| Date (UTC) | What | Exact price-based cost |
|---|---|---:|
| 2026-09-06 | Development matched-counterfactual collection, 35 episodes / 70 decisions | **$0.730879155** |
| 2026-09-06 | Held-out matched-counterfactual collection, 35 episodes / 70 decisions | **$0.729943500** |
| 2026-09-06 | Frozen four-arm DraftGym evaluation, 30 episodes per arm / 120 runs | **$2.742848826** |
| 2026-09-06 | Interrupted learned-arm canary, provider reconciled | **$0.057486843** |
|  | **Outcome-linked experiment total** | **$4.261158324** |

Exact cumulative completed Tinker workload is **$18.573989202**. Approximate
whole-project spend is now **$45.33** (historical estimate ~$26.76 plus exact
Tinker workload), excluding ongoing checkpoint storage. The exact token and
session reconciliation is in
`artifacts/value-of-information-v1/spend-audit.json`.

## Sparse reversible intervention canary (2026-09-07)

Target incremental spend: under **$2**. Hard cap: **$5**. The frozen comparison
used 30 matched base/treatment pairs with at most one Tinker-backed evidence
intervention in each treatment episode.

| Date (UTC) | What | Exact price-based cost |
|---|---|---:|
| 2026-09-07 | Untouched base, 30 masked DraftGym episodes | **$0.611265309** |
| 2026-09-07 | Sparse reversible controller, 30 matched episodes | **$0.625797729** |
|  | **Sparse reversible canary total** | **$1.237063038** |

Exact cumulative completed Tinker workload is **$19.811052240**. Approximate
whole-project spend is now **$46.57** (historical estimate ~$26.76 plus exact
Tinker workload), excluding ongoing checkpoint storage. The provider billing
snapshot is retained at `artifacts/sparse-reversible-canary-v1/billing.json`;
queries through 02:03 UTC still preceded provider settlement, so the
provider-returned request token ledger is the current authoritative total.

## Outcome-aligned draft planner and ranker (2026-09-07)

All counterfactual draft simulation, dataset generation, gradient-boosted
ranker training, and frozen 2021–2022 validation ran locally. The validation
failed the outlier-robustness and catastrophic-draft gates, so no Tinker
training or inference was launched.

| Date (UTC) | What | Exact price-based cost |
|---|---|---:|
| 2026-09-07 | 270-draft headroom panel, 1,243-state outcome dataset, season-held-out ranker, and 30-draft frozen validation | **$0.000000000** |

Exact cumulative completed Tinker workload remains **$19.811052240**.
Approximate whole-project spend remains **$46.57**, excluding ongoing
checkpoint storage.

## High-confidence ADP-override SFT (2026-09-07)

Hard incremental cap: **$10.00**. The one-epoch rank-16 run and frozen 2025
comparison used the pinned Qwen3.5-9B Tinker rates and provider-returned token
counts. The exported adapter archive is stored locally and excluded from Git.

| Date (UTC) | What | Exact price-based cost |
|---|---|---:|
| 2026-09-07 | Rank-16 LoRA training, development forward passes, reload/export, and lifecycle samples | **$1.128817490** |
| 2026-09-07 | Frozen untouched-base 2025 evaluation, 15 full drafts / 225 decisions | **$0.072957633** |
| 2026-09-07 | Frozen adapter 2025 evaluation, identical 15 drafts / 225 decisions | **$0.072626205** |
|  | **High-confidence override experiment total** | **$1.274401328** |

Exact cumulative completed Tinker workload is **$21.085453568**. Approximate
whole-project spend is now **$47.84**, excluding ongoing checkpoint storage.
