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

Total to date: ~$26.76 (T0 total incl. first attempt: ~$7.98, vs $50 cap)
