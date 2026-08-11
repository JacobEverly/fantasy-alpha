# Base-model decision memo — 2026-08-11

**Decision: T1 trains on `Qwen/Qwen3.5-9B` (dense, instruct). The Qwen3.8-27B remains a pre-approved swap candidate the day official weights + license appear.**

## Situation
Qwen3.8-27B weights were promised for the week of Aug 10; as of Aug 11 nothing official exists on HF/ModelScope (community placeholders only, no license). Per the pre-committed rule (GAMEPLAN: model-agnostic pipeline, never block on a release), the calendar wins: drafts peak Aug 24–Sep 1; T1 must run this week to leave room for T2 and the app.

## Why Qwen3.5-9B
1. **It is the most-measured model in the project** — the full B2 baseline already exists for it: expanded battery (naked + harness, CIs), canary gate PASS, fingerprint probe, DraftGym named/masked episodes, cost/latency profile. T1's delta will be cleanly attributable — no new-model confounds.
2. **Dense** — the risk register's MoE trainer/inference-numerics caveat avoided.
3. **Proven trainable in our pipeline** — T0 ran the same family (Qwen3-4B) through prime-rl LoRA end-to-end.
4. **Economics** — 9B+harness already ties Fable+harness on calibration; LoRA on 1×A100 (~$1/hr); serving at $0.18/$0.54 per Mtok serverless.
5. **License** — Apache-2.0 lineage, no ambiguity (unlike 3.8's undisclosed license).

## Cost estimate (revised down from the $300 T1 budget, which was sized for 14–32B)
LoRA SFT on ~1,000+ filtered traces, ≤3 epochs, A100 spot: **~$10–25 pod time**, plus checkpoint evals (CalibBench/BreakoutBench slices + canary per checkpoint, serverless): **~$5–15**. Total **~$20–40**, cap $100. Runbook: `training/t1-run-template.md` (T0-resolved configs + season-split amendment + grad-spike batch logging + checkpoint pruning).

## Swap protocol for Qwen3.8-27B (when real)
License check → same T1 recipe on the same corpus (~$50–150 at 27B scale) → both checkpoint lines on the dashboard → keep the winner by the standard gates. LoRA makes base swaps a re-run, not a redesign.

## Checkpoint eval plan (the dashboard dots)
Every saved checkpoint: (1) CalibBench pooled Brier vs B2 0.1949/base-rate 0.1922; (2) BreakoutBench full_slate Brier; (3) canary gate (must PASS or the checkpoint's numbers are void); (4) format/tool-taking smoke on DraftGym observations (B2 baseline: zero tool calls); (5) append to training/checkpoints_log.jsonl.
