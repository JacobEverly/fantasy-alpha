# Model card — Fantasy Alpha Qwen3.5-9B Tinker SFT v1

## Summary

This is a rank-32 LoRA adapter for `Qwen/Qwen3.5-9B`, trained through Tinker on
Fantasy Alpha's frozen T1 corpus. It is an experiment, not a production model.
The frozen decision is **revise SFT before RL**: the adapter learned the training
format and changed draft decisions, but the held-out evidence does not show a
robust overall product improvement.

## Intended use

- Research on calibrated fantasy-football probabilities and masked draft choices.
- Comparison against the untouched Qwen3.5-9B base under the frozen Fantasy Alpha suite.
- Development-only diagnosis of what SFT data should be added next.

Do not present this checkpoint as a proven forecasting edge, deploy it as an
autonomous draft manager, or use it for betting or other financial decisions.

## Training data

- Source: `data/processed/sft/t1_corpus_v1.jsonl`
- Rows: 811 (695 anonymized, 116 flagged named-pilot traces)
- SHA-256: `b68020f1209c78c085d492a486d0905fa5e74631c33755c6a5bfd3f68a5d0f2f`
- Seasons: 2008–2022 excluding 2013 and 2018
- Excluded: 2013/2018 diagnostics, 2023/2024 forward evaluation, sealed 2025
- Deterministic question-group split: 728 train / 83 development; zero group overlap
- Rendered tokens: 1,024,901 per epoch; 238,677 assistant target tokens per epoch
- Format: system/user/assistant; only the final assistant message receives loss

The dataset's filters reject unsupported numbers and outcome-conditioned
survivorship. User-role evidence remains loss-masked. These controls reduce but
cannot eliminate teacher bias, historical-data artifacts, or pretraining recall.

## Training configuration

| Field | Value |
|---|---|
| Provider / SDK | Tinker 0.27.1; tinker-cookbook 0.5.7 |
| Base | `Qwen/Qwen3.5-9B` |
| Renderer | `qwen3_5_disable_thinking` |
| LoRA rank | 32 |
| Batch / epochs | 32 / 2 |
| Learning rate | linear 3e-4 → 3e-5 |
| Optimizer | Adam β1=.9, β2=.95, ε=1e-8, clip norm 1.0 |
| Seed | 20260808 |
| Context limit used | 4,096 (no truncated examples) |
| Steps | 46 |

Training NLL moved 1.289→0.739 across first/last batches. Development NLL
moved 1.257→0.860→0.815. The preregistered development-only selection rule
therefore selected epoch 2. No held-out result influenced checkpoint selection.

## Checkpoint and reproducibility

- Tinker run: `027360a8-fff7-5774-b005-29e854b7a464:train:0`
- Selected sampler: `tinker://027360a8-fff7-5774-b005-29e854b7a464:train:0/sampler_weights/fantasy-alpha-t1-qwen35-9b-r32-final-sampler`
- Exported adapter: `artifacts/tinker-sft-v1/t1-full/final-adapter.tar.gz`
- Archive size: 378,419,200 bytes
- Archive SHA-256: `4c99aa04e9dae2ad4fbdc3f7e3a91fa2f5bae86a3f1539070d8eb12a1ff90202`
- Frozen eval specification SHA-256: `5a9344a298bdb2ef0bc59eb7bb2586839bbe44d766fbf123f2f2423426b97a43`

The remote state and sampler were reloaded successfully. The local archive is
gitignored because of its size; the summary, hashes, loss history, scorecard,
and checkpoint address are retained as experiment artifacts.

## Frozen evaluation

The untouched base and selected adapter used identical no-thinking rendering,
temperature 0, seeds, prompts, tools, and scoring on held-out 2018/2023/2024.
The suite contained 1,998 matched probability judgments and six paired masked
DraftGym episodes per arm. 2025 was never opened.

| Measure | Base | Adapter | Reading |
|---|---:|---:|---|
| BreakoutBench pooled Brier ↓ | 0.1982 | 0.1991 | Tiny regression; all three families slightly worse |
| BreakoutBench top-10 lift ↑ | 1.42× | 1.42× | No change |
| CalibBench Brier ↓ | 0.2449 | 0.2436 | Tiny improvement, below frozen materiality threshold |
| CalibBench ECE ↓ | 0.0842 | 0.0795 | Small improvement |
| CalibBench log loss ↓ | 0.6861 | 0.6993 | Regression from overconfident misses |
| Masked DraftGym mean reward ↑ | -6.68 | 74.48 | +81.15, but only 2/6 episode wins |
| Structured coverage ↑ | 100% | 100% | No regression; no repair needed |
| Canary | pass | pass | No detected identity-channel contamination |

Across all paired probability judgments, mean Brier improvement was 0.00067
with a season-clustered 95% interval of [-0.00419, 0.00896]. Masked DraftGym's
median paired change was -7.48 and its season-clustered interval was
[-24.46, 137.43]; two large wins drove the positive mean. Neither model used an
evidence tool. Full family results and paired examples are in
`artifacts/tinker-sft-v1/evaluation/scorecard.json`.

## What it can and cannot do

The adapter can reliably emit the required structured formats, imitate the T1
teacher corpus better than the base, and produce materially different masked
draft choices. It cannot yet be said to forecast better, calibrate consistently,
seek evidence, prevent more mistakes across varied drafts, or generalize to a
new season. The apparent DraftGym gain is too unstable to justify RL from this
checkpoint without revising the SFT data first.

## Decision and next experiment

Do not begin RL from this checkpoint. Add development-only traces that explicitly
teach valid tool use, probability calibration under uncertainty, and late-round
roster/position constraints. Freeze a fresh validation slice, run a small
rank/epoch ablation selected only on development data, and repeat this same
base-versus-adapter battery. Proceed to DraftGym RL only if that revision yields
a stable masked-draft gain without predictive or calibration regression.

## Spend and known failures

The completed Tinker workload cost exactly $5.126104 at the recorded token
rates: $0.327733 paid T1-format canary, $0.764466 exact existing-T0 parity
smoke, $3.156840 T1 training, $0.434511 base evaluation, and $0.442553 adapter
evaluation. Ongoing checkpoint storage is separate. The exact existing-T0 run
reduced train NLL 1.263→0.811 and development NLL 1.302→1.011, reloaded the
checkpoint, sampled both arms, and exported a 378,419,200-byte adapter.

The earlier 24-row canary's first post-training finalization attempt used the wrong SDK metadata field
for LoRA rank. The checkpoint had already saved; recovery repeated zero optimizer
steps, then completed reload, sampling, and export. T1 and both frozen evaluation
arms had no failed or retried model requests.
