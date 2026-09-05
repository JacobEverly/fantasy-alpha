# Fantasy Alpha

An AI fantasy-football analyst: a chatbot + draft advisor backed by a real quantitative engine, with an open model trained (SFT → RL) against verifiable outcomes — real NFL box scores, real draft markets. Built on the [Prime Intellect](https://www.primeintellect.ai/) stack (prime-rl, verifiers) with a model-agnostic harness.

**What we sell:** drafting better — decision quality, pick timing, league-native math, calibrated odds — *demonstrated publicly every season* via a frozen, timestamped prediction ledger, a benchmark leaderboard against frontier models, and per-user "points above your league's baseline" receipts.

**What we deliberately don't sell:** breakout clairvoyance. We benchmarked GBDT, Qwen (9B/35B/397B), DeepSeek V4, and Claude Fable 5 on ten years of anonymized player dossiers — **nobody beats the market's base rate at picking outliers from historical information** (it's an information ceiling, not a capability ceiling), and models that *look* brilliant with player names visible collapse below the market when names are masked (+492 points of pure memorization for the strongest model tested). Every "our AI predicted the breakouts" backtest you've seen is, measurably, one of those two failure modes. Our edge lives where the data says it can: information speed, calibration honesty, and draft-room decision quality.

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
tests/      340+ tests — leakage poison-tests, determinism, holdout guards, invariants
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

## Status (2026-09-05)

Measurement stack complete and validated · both Prime and Tinker LoRA backends
work end-to-end · T1 and T1.1 are fully evaluated · T1.2 corrected conflicting
targets and froze a 70/30 broad/targeted mix, but both representative canaries
made zero valid tool calls on unseen tool-opportunity states · the frozen gate
therefore stopped the full run after **$1.239102** · evidence-backed decision:
**stop additional SFT and do not begin RL** until tool choice is redesigned ·
2025 remains sealed · exact price-based Tinker workload $10.855193 · historical
project spend approximately $37.62 of $5k, excluding ongoing storage.

---

*Private repository. The upstream research foundation lives in [`fantasy-football-alpha-2026`](https://github.com/JacobEverly/fantasy-football-alpha-2026) (separate repo, gitignored here as a working clone).*
