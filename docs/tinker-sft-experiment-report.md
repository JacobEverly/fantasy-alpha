# Fantasy Alpha Tinker SFT v1 — frozen evaluation

Decision: **revise sft before rl**.

The comparison uses the untouched `Qwen/Qwen3.5-9B` and the development-selected rank-32 adapter under the same prompts, renderer, decoding, seeds, tools, and verifiers. Seasons 2018, 2023, and 2024 are held out; 2025 remains untouched.

## Run record

The exact existing 162-row T0 corpus completed 20 optimizer steps, reduced development NLL from 1.302 to 1.011, reloaded its saved state, sampled the base and adapter, and exported a 378.4 MB archive. A separate 24-row balanced T1-format canary completed first; its post-training metadata check needed one client-side retry, with zero repeated optimizer steps.

The full 811-trace run used 728 grouped training rows and 83 grouped development rows for two epochs (46 steps). Training-batch NLL moved from 1.289 to 0.739; development NLL moved 1.257 → 0.860 → 0.815. The frozen development-only rule selected epoch 2. The saved state reloaded and the exported adapter hash was verified.

Unlike the existing Prime/prime-rl backend, Tinker exposed managed forward/backward, optimizer, checkpoint, and sampler clients: there was no GPU pod to provision or abandon. Both backends remain available and consume the same system/user/assistant data shape with assistant-only loss.

## Frozen scorecard

| Measure | Base | Fine-tuned | Fine-tuned minus base |
|---|---:|---:|---:|
| BreakoutBench Brier ↓ | 0.1982 | 0.1991 | +0.0009 |
| CalibBench Brier ↓ | 0.2449 | 0.2436 | -0.0013 |
| CalibBench ECE ↓ | 0.0842 | 0.0795 | -0.0047 |
| Masked DraftGym mean reward ↑ | -6.68 | 74.48 | +81.15 |
| Structured answer coverage ↑ | 100.00% | 100.00% | +0.00% |

Canary gate: **PASS**. Paired examples: 1,998; season-clustered 95% CI for mean Brier improvement [-0.004188524590163939, 0.008963650306748465].

Exact price-based incremental workload spend: **$5.1261** (ongoing checkpoint storage reported separately in the budget ledger).

## What changed

BreakoutBench top-10 lift was unchanged at 1.42×. CalibBench log loss moved from 0.6861 to 0.6993, despite the small Brier/ECE gains—evidence that some probability changes became too confident.

The masked-draft mean improved by 81.15, but the adapter won only 2/6 paired drafts; the median change was -7.48, and the season-clustered 95% interval was [-24.46, 137.43]. Two very large wins drive the mean, so this is a promising signal rather than a robust result.

Both arms covered 100% of requested predictions with no repair, fallback, or illegal picks. Neither arm used an evidence tool (0 adapter calls), so SFT did not teach evidence-seeking behavior.

## Decision

The frozen decision above follows the preregistered thresholds: **revise the SFT approach before RL**. The JSON scorecard contains family-level results, every DraftGym episode, and the 15 largest paired improvements and regressions. These results measure behavior on this suite; they do not establish broad football expertise or prove that the model will generalize to the still-sealed 2025 season. The logical revision is to add development-only tool-use and calibrated-decision traces, then repeat with a newly frozen validation slice before spending on RL. No RL was started in this milestone.
