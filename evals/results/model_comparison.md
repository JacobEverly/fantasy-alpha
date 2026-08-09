# Model comparison — quality x cost x latency

Generated 2026-08-09 02:44 UTC by evals/report_dashboard.py. Quality rows: anonymized ppr track 2015-2024, 1,849 judgments/run (evals/results/frontier_battery.md; Qwen3.5-9B cost basis is its all-format 5,030-judgment battery). Costs: exact API-reported spend (evals/results/answers/frontier_spend.json + docs/budget-ledger.md). Latency: data/processed/telemetry/eval_requests.jsonl (evals/telemetry.py) — pre-instrumentation runs have no recoverable per-request timing and are marked n/a, not estimated. Partial pre-restriction rows (GPT-5.6-Luna-Pro, Gemini 3.1 Pro, DeepSeek V4 Pro, Qwen3.5-397B, GLM-5.2) are excluded, as in the battery verdicts.

| model | config | pooled Brier [95% CI] | top-10 lift [95% CI] | canary verdict | cost / 1,000 judgments | cost / request | latency / request |
|---|---|---|---|---|---|---|---|
| Claude Fable 5 | naked | 0.2000 [0.190, 0.210] | 1.27x [0.84, 1.73] | PASS (swap idx -0.194 [-0.333, -0.045]) | $3.718 (cost basis includes canary tier + 4 DraftGym episodes) | $0.105 | n/a (pre-instrumentation) |
| Claude Fable 5 | + harness | 0.1918 [0.184, 0.200] | 1.40x [0.88, 1.74] | PASS (swap idx -0.194 [-0.333, -0.045]) | $3.718 (cost basis includes canary tier + 4 DraftGym episodes) | $0.105 | n/a (pre-instrumentation) |
| DeepSeek V4 Flash | naked | 0.2052 [0.195, 0.216] | 1.20x [0.63, 1.83] | PASS (swap idx +0.033 [-0.223, 0.218]) | $0.036 (cost basis includes canary tier) | $0.001 | n/a (pre-instrumentation) |
| DeepSeek V4 Flash | + harness | 0.1930 [0.185, 0.201] | 1.33x [0.68, 1.85] | PASS (swap idx +0.033 [-0.223, 0.218]) | $0.036 (cost basis includes canary tier) | $0.001 | n/a (pre-instrumentation) |
| Qwen3.5-35B-A3B | naked | 0.2122 [0.203, 0.222] | 0.87x [0.58, 1.30] | PASS (swap idx -0.039 [-0.168, 0.127]) | $0.181 (cost basis includes canary tier) | $0.006 | n/a (pre-instrumentation) |
| Qwen3.5-35B-A3B | + harness | 0.1931 [0.186, 0.200] | 1.27x [0.64, 1.69] | PASS (swap idx -0.039 [-0.168, 0.127]) | $0.181 (cost basis includes canary tier) | $0.006 | n/a (pre-instrumentation) |
| Qwen3.5-9B (ours, untrained) | naked | 0.1992 [0.191, 0.210] | 1.21x [0.93, 1.48] | PASS (swap idx -0.145 [-0.390, 0.081]) | $0.042 (all-format expanded battery, 3 scoring formats) | $0.003 | n/a (pre-instrumentation) |
| Qwen3.5-9B (ours, untrained) | + harness | 0.1949 [0.186, 0.205] | 1.01x [0.67, 1.29] | PASS (swap idx -0.145 [-0.390, 0.081]) | $0.042 (all-format expanded battery, 3 scoring formats) | $0.003 | n/a (pre-instrumentation) |
| GBDT walk-forward | boring-ML bar | — (full_slate Brier 0.1270 [0.103, 0.151]; no bust/over_under judgments, so no pooled Brier) | 1.02x [0.77, 1.30] | n/a (code) | ~$0 | — | n/a (local compute) |
| base rate (no-peek) | flat baseline | 0.1922 [0.184, 0.203] | 1.13x [0.96, 1.26] | n/a (code) | ~$0 | — | n/a (local compute) |
| coin p=0.5 | flat baseline | 0.2500 [0.250, 0.250] | 1.13x [0.96, 1.26] | n/a (code) | ~$0 | — | n/a (local compute) |

Cost / 1,000 judgments = the model's total recorded API spend divided by its scored naked+harness judgments; per-model spend also covers the canary tier (and Fable's 4 DraftGym episodes), so LLM rows are slight overestimates of pure battery cost. A model's naked and harness rows share one cost basis. Cost / request = spend / API requests (a request answers a whole question batch — one (season, family) slate of ~40-90 questions, or one DraftGym pick).

## Quality-vs-cost reading

**DeepSeek V4 Flash delivers frontier-tied calibration at roughly 1/103 of Fable's cost per judgment** — harness pooled Brier 0.1930 [0.185, 0.201] vs Fable's 0.1918 [0.184, 0.200] (CIs overlap: statistically tied) at $0.036 vs $3.718 per 1,000 judgments. (Fable's cost basis also carries its DraftGym episodes, so the true battery-only ratio is somewhat smaller but stays around two orders of magnitude.)

The quality axis is compressed: the best model row (Claude Fable 5 + harness, 0.1918) is statistically tied with the free no-peek base rate (0.1922) — on this anonymized track, paying more buys latency and cost, not calibration. Cost is currently the only axis that separates the field.

## Keeping this current

- New battery runs record latency/cost per request automatically via evals/telemetry.py (wired into run_frontier_battery.py and run_expanded_qwen.py).
- After a battery run, transcribe any new/changed leaderboard rows from frontier_battery.md into LEADERBOARD in evals/report_dashboard.py, then `python evals/report_dashboard.py` to regenerate this file and dashboard.html.
