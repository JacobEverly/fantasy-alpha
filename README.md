# Fantasy Alpha

An AI fantasy-football analyst: a chatbot + draft advisor backed by a real quantitative engine, with an open model trained (SFT → RL) against verifiable outcomes — real NFL box scores, real draft markets. Built on the [Prime Intellect](https://www.primeintellect.ai/) stack (prime-rl, verifiers) with a model-agnostic harness.

**What we sell:** drafting better — decision quality, pick timing, league-native math, calibrated odds — *demonstrated publicly every season* via a frozen, timestamped prediction ledger, a benchmark leaderboard against frontier models, and per-user "points above your league's baseline" receipts.

**What we deliberately don't sell:** breakout clairvoyance. We benchmarked GBDT, Qwen (9B/35B/397B), DeepSeek V4, and Claude Fable 5 on ten years of anonymized player dossiers — **nobody beats the market's base rate at picking outliers from historical information** (it's an information ceiling, not a capability ceiling), and models that *look* brilliant with player names visible collapse below the market when names are masked (+492 points of pure memorization for the strongest model tested). Every "our AI predicted the breakouts" backtest you've seen is, measurably, one of those two failure modes. Our edge lives where the data says it can: information speed, calibration honesty, and draft-room decision quality.

## Post-training case study: optimize the outcome, not the proxy

This repository documents a complete post-training loop around
`Qwen/Qwen3.5-9B`, Tinker LoRA, and a real tool-using DraftGym harness. The work
did not stop when training loss improved—or when the first classifier looked
accurate:

| Stage | What the evidence said | Decision |
|---|---|---|
| Tinker SFT on 811 frozen traces | Lower NLL and a reloadable rank-32 adapter did not produce a reliable product win | Revise the data objective |
| Tool-policy supervisor | 99.1% held-out policy accuracy, but successful lookups made full drafts worse | Reject policy imitation as the target |
| Outcome-linked supervisor | +2.64 points per isolated held-out decision, but **−15.75 points per complete paired draft** across 30 episodes | Stop before RL; local value did not survive trajectory feedback |

The latest experiment contains 140 matched counterfactual decisions, 120
full-episode arm runs, episode-grouped splits, frozen hashes, a reloadable
supervisor, paired bootstrap uncertainty, exact token/cost ledgers, and explicit
stop gates. Its most useful finding is the gap between local offline lift and
end-to-end agent performance: a credible supervisor must be trained and judged
on the trajectory it changes.

Start with the [outcome-linked experiment report](docs/value-of-information-experiment.md),
then inspect the [dataset card](docs/value-of-information-dataset-card.md),
[frozen protocol](training/value-of-information-spec-v1.json), and prior
[Tinker SFT report](docs/tinker-sft-experiment-report.md).

## Orientation

| Read | For |
|---|---|
| [`docs/status-*.md`](docs/) (latest) | Where things stand + next steps — **start here to resume work** |
| [`GAMEPLAN.md`](GAMEPLAN.md) | The operating document: locked decisions, eval gates, budget, schedule |
| [`docs/benchmarks-review-guide.md`](docs/benchmarks-review-guide.md) | How every benchmark works, with challengeable decisions marked |
| [`docs/benchmark-results-complete.md`](docs/benchmark-results-complete.md) | All results + what they mean for a consumer |
| [`docs/training-risk-register.md`](docs/training-risk-register.md) | 17 training failure modes and the specific countermeasure for each |
| [`docs/budget-ledger.md`](docs/budget-ledger.md) | Every dollar, as it was spent |

## Layout

```
harness/    League-agnostic engine (any scoring/roster/format), time-gated evidence store
evals/      DraftBench · BreakoutBench · CalibBench · FP-accuracy replication · canaries ·
            slot values · value attribution · results/ (all reports) · dashboard
envs/       DraftGym — draft RL environment (verifiers v1 adapter, masked-board mode,
            realistic-manager reward)
training/   SFT datagen + survivor-bias-proof filter · T0 artifacts · T1 run template
scripts/    Data collectors (stdlib-first): nflverse, ADP history, evidence archiver, odds
data/       raw/ immutable dated snapshots (gitignored) · processed/ derived tables (gitignored)
docs/       Design docs, research reports, risk register, status
tests/      477 tests — leakage poison-tests, determinism, holdout guards, invariants
```

## Setup & verification

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q                      # full suite
.venv/bin/python scripts/archive_evidence.py   # daily evidence capture (run nightly)
```

Copy `.env.example` → `.env` for local development keys. Tinker credentials are
read only from `TINKER_API_KEY`; managed Tinker workers or rented Prime GPUs do
the training, never the laptop. See `training/README.md` and
`training/t1-run-template.md`.

## The measurement discipline (non-negotiable)

- **2025 is sealed** — every tool refuses it without `--include-holdout`; it's spent once per contender at gate time. 2026 is proven live via the public ledger (predictions frozen before kickoff).
- **Walk-forward everywhere** — no feature, base rate, or training row from the scored season or later; tests poison future data and assert nothing moves.
- **Every number ships with its null** — random, coin, no-peek base rate, the market, and boring-ML are built-in baselines; bootstrap CIs resample seasons.
- **Memorization is detected, not assumed away** — canary gates void any checkpoint whose anonymized-track scores are identity recall; DraftGym trains on masked boards.
- **Money**: $5k ceiling, warn before any single spend ≥$50, every resource logged at provisioning time, kill criteria preregistered before every training run.

## Status (2026-09-06)

The first outcome-linked evidence supervisor experiment is complete. The
learned policy improved reward on 70 isolated held-out decision points, then
failed the more important 30-episode paired test: 12 wins, 15 losses, 3 ties,
and a −15.75-point mean change versus base Qwen. It also made 12 critical
regressions versus 8 critical improvements while calling tools 412 times.
Decision: **stop treating the current supervisor as the primary performance
lever and do not begin RL**. This is a trajectory-distribution failure, not a
tool-execution failure: all 412 forced lookups succeeded. Sealed seasons remain
untouched. See [the current status](docs/status-2026-09-06.md) and the
[prior policy-imitation experiment](docs/tool-decision-supervisor-experiment.md).

---

*Private repository. The upstream research foundation lives in [`fantasy-football-alpha-2026`](https://github.com/JacobEverly/fantasy-football-alpha-2026) (separate repo, gitignored here as a working clone).*
