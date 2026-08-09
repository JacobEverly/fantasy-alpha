# Frontier battery — BreakoutBench anonymized track via OpenRouter

Run date 2026-08-08. Protocol identical to the Qwen3.5-9B expanded battery (evals/run_expanded_qwen.py): ppr grid, seasons 2015-2024, 3 families per season, one request per (season, family), temp 0, reasoning/thinking DISABLED (the Qwen3.5-9B baseline protocol — `enable_thinking=false`; a reasoning-effort-low pilot thought for 5-15k tokens and 8-10 min per slate and was discarded). Exception: Fable-5 rejects disabled reasoning (mandatory on that endpoint) and runs at effort `minimal`, which emitted 0 reasoning tokens in the pilot. max_tokens 3000 (repair retry 16000), one repair retry per malformed response. 95% CIs bootstrap over seasons (10,000 resamples, seeded). Cost is OpenRouter's exact per-response `usage.cost`. 2025 remains the untouched holdout.

## Contenders (exact ids + live catalog prices, $/M tokens)

| model | id | in | out | open weights |
|---|---|---|---|---|
| Claude Fable 5 | `anthropic/claude-fable-5` | $10.000 | $50.000 | no |
| GPT-5.6-Luna-Pro | `openai/gpt-5.6-luna-pro` | $0.100 | $0.600 | no |
| Gemini 3.1 Pro | `google/gemini-3.1-pro-preview` | $2.000 | $12.000 | no |
| DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | $0.435 | $0.870 | yes |
| DeepSeek V4 Flash | `deepseek/deepseek-v4-flash` | $0.140 | $0.280 | yes |
| Qwen3.5-397B-A17B | `qwen/qwen3.5-397b-a17b` | $0.500 | $3.600 | yes |
| Qwen3.5-35B-A3B | `qwen/qwen3.5-35b-a3b` | $0.140 | $1.000 | yes |
| GLM-5.2 | `z-ai/glm-5.2` | $0.112 | $0.352 | yes |

## Leaderboard — anonymized ppr track (2015-2024, 1,849 judgments/run)

Existing rows re-scored on the identical ppr-only subset for exact comparability (the published expanded_baseline.md rows pool all three formats). GBDT row from evals/results/gbdt_baseline.md (ppr full_slate walk-forward; no bust/over_under judgments, so no pooled Brier).

| run | full_slate Brier [95% CI] | pooled Brier [95% CI] | top-10 lift [95% CI] | answered | cost |
|---|---|---|---|---|---|
| Claude Fable 5 naked | 0.1303 [0.103, 0.159] | 0.2000 [0.190, 0.210] | 1.27x [0.84, 1.73] | 1848/1849 | $13.73 total |
| GPT-5.6-Luna-Pro naked (PARTIAL: 3 seasons, pre-restriction run — not comparable, excluded from verdicts) | 0.1741 [0.126, 0.230] | 0.2201 [0.191, 0.236] | 0.72x [0.56, 0.96] | 539/539 | — |
| Gemini 3.1 Pro naked (PARTIAL: 3 seasons, pre-restriction run — not comparable, excluded from verdicts) | 0.1597 [0.097, 0.229] | 0.2069 [0.144, 0.233] | 0.54x [0.00, 1.92] | 460/539 | — |
| DeepSeek V4 Pro naked (PARTIAL: 1 season, pre-restriction run — not comparable, excluded from verdicts) | 0.1671 [0.167, 0.167] | 0.1671 [0.167, 0.167] | 0.56x [0.56, 0.56] | 56/187 | — |
| DeepSeek V4 Flash naked | 0.1347 [0.107, 0.163] | 0.2052 [0.195, 0.216] | 1.20x [0.63, 1.83] | 1849/1849 | $0.13 total |
| Qwen3.5-397B-A17B naked (PARTIAL: 1 season, pre-restriction run — not comparable, excluded from verdicts) | 0.1798 [0.180, 0.180] | 0.1798 [0.180, 0.180] | 0.56x [0.56, 0.56] | 56/187 | — |
| Qwen3.5-35B-A3B naked | 0.1340 [0.104, 0.164] | 0.2122 [0.203, 0.222] | 0.87x [0.58, 1.30] | 1849/1849 | $0.67 total |
| GLM-5.2 naked (PARTIAL: 1 season, pre-restriction run — not comparable, excluded from verdicts) | 0.1555 [0.156, 0.156] | 0.2185 [0.218, 0.218] | 0.56x [0.56, 0.56] | 187/187 | — |
| Claude Fable 5 + harness | 0.1282 [0.104, 0.153] | 0.1918 [0.184, 0.200] | 1.40x [0.88, 1.74] | 1846/1849 | — |
| GPT-5.6-Luna-Pro + harness | — | — | — | — | — |
| Gemini 3.1 Pro + harness | — | — | — | — | — |
| DeepSeek V4 Pro + harness | — | — | — | — | — |
| DeepSeek V4 Flash + harness | 0.1298 [0.105, 0.155] | 0.1930 [0.185, 0.201] | 1.33x [0.68, 1.85] | 1849/1849 | — |
| Qwen3.5-397B-A17B + harness | — | — | — | — | — |
| Qwen3.5-35B-A3B + harness | 0.1299 [0.107, 0.154] | 0.1931 [0.186, 0.200] | 1.27x [0.64, 1.69] | 1849/1849 | — |
| GLM-5.2 + harness | — | — | — | — | — |
| Qwen3.5-9B naked (ppr subset) | 0.1321 [0.107, 0.160] | 0.1992 [0.191, 0.210] | 1.21x [0.93, 1.48] | 3137/3137 | $0.42 (all-format run) |
| Qwen3.5-9B + harness (ppr subset) | 0.1280 [0.106, 0.155] | 0.1949 [0.186, 0.205] | 1.01x [0.67, 1.29] | 3137/3137 | — |
| GBDT walk-forward (boring-ML bar) | 0.1270 [0.103, 0.151] | — | 1.02x [0.77, 1.30] | 539 full_slate | ~$0 |
| base rate (no-peek) | 0.1258 [0.104, 0.153] | 0.1922 [0.184, 0.203] | 1.13x [0.96, 1.26] | 3137/3137 | — |
| coin p=0.5 / random selection | 0.2500 [0.250, 0.250] | 0.2500 [0.250, 0.250] | 1.13x [0.96, 1.26] | 3137/3137 | — |

Cost column: per-model total across ALL tiers (tier 1 + canary + harness + DraftGym where run) — exact OpenRouter usage.cost sums.

## Canary gate + fingerprint probe (memorization read)

Gate: FAIL iff pooled swap-index 95% CI lower bound > 0.05 (identity-channel leak). Fingerprint excess = model-minus-code-reference within-pair accuracy on unswapped ADP-adjacent pairs, per era; recent-era-only excess is the fingerprint-recall signature.

| model | gate | swap index [95% CI] | fp excess 2015-19 | fp excess 2020-24 |
|---|---|---|---|---|
| Claude Fable 5 | PASS | -0.194 [-0.333, -0.045] | +0.012 | +0.000 |
| GPT-5.6-Luna-Pro | not run | — | — | — |
| Gemini 3.1 Pro | not run | — | — | — |
| DeepSeek V4 Pro | not run | — | — | — |
| DeepSeek V4 Flash | PASS | +0.033 [-0.223, 0.218] | -0.012 | -0.159 |
| Qwen3.5-397B-A17B | not run | — | — | — |
| Qwen3.5-35B-A3B | PASS | -0.039 [-0.168, 0.127] | -0.012 | -0.122 |
| GLM-5.2 | not run | — | — | — |

Qwen3.5-9B reference (evals/results/canaries.md): gate PASS, index -0.145 [-0.390, 0.081], fp excess +0.000 / -0.085.

## DraftGym 2021 — named vs masked (memorization delta)

2 named + 2 masked episodes (slots 1 and 7, seed 0, 12-team ppr, 15 rounds), identical configs; delta = named reward − masked reward on the paired config. Qwen3.5-9B reference deltas (14-episode baseline): +403.9, +139.6.

| model | named rewards | masked rewards | paired deltas | fallbacks | tools |
|---|---|---|---|---|---|
| Claude Fable 5 | +459.1, +796.3 | -108.4, +379.8 | +567.5, +416.4 | 0 | 1 |

## Per-model cost (exact OpenRouter usage.cost)

| model | requests | tokens in/out | cost |
|---|---|---|---|
| Claude Fable 5 | 131 | 967,253 / 81,200 | $13.733 |
| DeepSeek V4 Flash | 106 | 919,083 / 185,641 | $0.133 |
| Qwen3.5-35B-A3B | 110 | 1,031,359 / 399,053 | $0.670 |
| unattributed (8-model tier-1 partial killed on scope change + killed-in-flight requests + protocol pilots; OpenRouter credits-endpoint delta, not attributable per model) | ~35 requests | — | $2.911 |
| **total** |  |  | **$17.446** (cap $43.50; reconciled to the OpenRouter credits endpoint: total_usage $17.4464) |

DraftGym tier ran for Claude Fable 5 only (Jacob's frontier-anchor scope); the dsv4flash/qwen35b stacks stopped after tier 3 per the earlier scope restriction.

## Verdicts

- **(a) Benchmark headroom:** NO frontier model beats the no-peek base-rate floor with statistical significance on anonymized selection (all top-10 lift and pooled-Brier CIs overlap the base-rate row). The anonymized track continues to look information-ceilinged, not capability-ceilinged.
- **(b) Frontier memorization vs 9B Qwen:** on the ANONYMIZED track, no frontier model fails the canary gate or shows a recent-era fingerprint excess (Fable's swap index is actually negative, -0.194 [-0.333, -0.045] — it follows the packet, not the hidden id, and its fp excess is ~0 in both eras). But on NAMED DraftGym the frontier signature is stronger: Fable's named-minus-masked paired deltas are +567.5 / +416.4 (mean +492.0) vs Qwen9B's +403.9 / +139.6 (mean +271.8) — Fable extracts MORE value from seeing real player names than 9B Qwen does, i.e. deeper name-keyed knowledge of 2021 outcomes, while remaining clean once identities are masked. Fable also posted the strongest named episode on record here (+796.3, slot 7).
- **(c) Competitive calibration bar:** best pooled Brier is Claude Fable 5 harness at 0.1918 [0.184, 0.200]; the no-peek base rate sits at 0.1922 [0.184, 0.203] and GBDT full-slate at 0.1270 [0.103, 0.151]. The product bar: match base-rate calibration everywhere and add selection lift on top — nothing here moves that bar materially.
- **(d) Does scale alone buy selection skill?** Qwen scale probe (same prompts, same slates, thinking off) — 9B: Brier 0.1992 [0.191, 0.210], lift 1.21x [0.93, 1.48]; 35B-A3B: 0.2122 [0.203, 0.222], lift 0.87x [0.58, 1.30]; 397B-A17B not yet run to completion (1 pre-restriction season only — no valid probe point). The 35B-vs-9B contrast does not separate (all CIs overlap): within the Qwen family, ~4x active-parameter scale buys nothing on this track — consistent with an information ceiling rather than a capability ceiling. The 397B point is still needed for the full scale read.
