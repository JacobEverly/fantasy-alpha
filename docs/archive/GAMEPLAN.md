# GAMEPLAN — Fantasy Alpha

**Dated 2026-08-08. This is the operating document for the one-week sprint (Aug 8–15) and the phase after it.**

Product: an AI fantasy-football analyst — a chatbot that answers player questions and produces league-format-aware draft recommendations, backed by a real quantitative engine and, ultimately, our own trained model. Revenue: credits/subscription (deferred until after launch).

## 0. Thesis

Mainstream fantasy advice (ESPN-tier) is unsophisticated relative to what a disciplined quant layer plus modern AI can do. Fantasy is a nearly ideal domain for **model–harness co-design**: outcomes are objective and dated, rewards are verifiable (real weekly scores), simulation is cheap, and the market price (ADP) gives a clean baseline to beat. We win by (1) building a harness whose tools are a real projection/value engine, and (2) training an open model — SFT then RL — against verifiable fantasy outcomes, so the model gets better at exactly the decisions users pay for. Distribution: a chatbot with a public, timestamped prediction ledger as proof.

## 1. Locked decisions (2026-08-08)

| Decision | Call |
|---|---|
| Pace | One-week accelerated sprint; all workstreams in flight simultaneously |
| Architecture | **Eval-driven.** Harness first, then train the model; the model WILL be trained (SFT + RL). All ship decisions settled by the eval suite, not by opinion |
| Budget | ~$5k compute + data through first serious checkpoint |
| League input | Manual entry v1 (~7 fields: teams, scoring type, QB/RB/WR/TE/FLEX counts, bench, draft slot). Platform imports deferred |
| Base model | Decide after the Qwen 3.8 open-weights drop (~Aug 10). 3.8-Max (2.4T MoE) is NOT a fine-tune target; the rumored 27B is. Fallback: Qwen3.5 small dense/MoE. Pipeline stays model-agnostic |
| Training stack | verifiers (environments) + prime-rl (SFT/RL) + Environments Hub; Prime Agent's continual-harness pattern as the design reference for the self-learning loop |
| Training runs | None launched without an explicit go; all runs on rented GPUs, never the Mac |
| Model access (2026-08-08 eve) | **Self-hosted open models only** on our Prime Intellect account (key live in `.env`). No frontier API keys until a real product exists; dev-time frontier work happens inside Claude Code itself |
| SFT teacher (2026-08-08 eve) | Self-hosted permissive open model (Qwen-class) served on our own pods — sidesteps distillation clauses entirely |
| Web design (2026-08-08 eve) | Follow Jacob's website-generator-skills pipeline: capture-inspo → design-brief → site-gen → site-loop → site-refine, on the Boundless base theme |

## 2. Division of labor: model vs. tools (the architecture question, plainly)

When a user asks *"Should I take Bijan Robinson at 1.02?"*:

- **Tools compute the numbers.** The quant engine (lifted from the research repo) produces the projection distribution, value over replacement for THIS league's roster shape, the probability he survives to the next pick, and the risk flags. LLMs are bad at calibrated arithmetic; classical code is perfect at it and free.
- **The model makes the decision and the case.** It weighs tool outputs against evidence tools can't see well — camp reports, injury context, scheme change, roster construction strategy — decides, and explains in language a player trusts.
- **Training makes the model good at that job**, not at emitting stat lines: SFT teaches it the workflow and voice; RL against DraftGym teaches it which considerations actually move realized outcomes. This is the co-design: the harness defines what the model can touch, and the reward comes from what actually happened in real NFL seasons.

So "train the harness first, then the model" is exactly right, and is the plan.

## 3. System architecture

```
Web app (chat + draft mode + ledger)
   └─ Harness (orchestrator; model-agnostic router)
        ├─ Model slot: frontier API today → our checkpoint when it wins evals
        └─ Tools:
            ├─ League engine (scoring, VORP, replacement, turn/survival math — any format)
            ├─ Projection & signals layer (from research repo: projections, SOS,
            │   coaching priors, availability, alpha-vs-market, avoid logic)
            ├─ Market data (ADP current + historical)
            └─ Evidence lookups (news/reporter feeds, injury status)

Training loop (offline):
   DraftGym + CalibBench envs (verifiers) ──rollouts──▶ prime-rl (SFT → GRPO)
        ▲                                                      │
        └────────────── eval suite gates ◀── checkpoints ──────┘

Self-learning loop (in-season):
   Public ledger predictions → scored weekly vs. reality →
   harness memory/prompt/skill updates (Continual-Harness pattern) + periodic RL refresh
```

## 4. Eval suite — the north star

Every contender runs the same gauntlet. Contenders: **B1** strongest self-hosted open model + harness (the dev-time bar; frontier-API models join the public leaderboard at launch, when keys exist) · **B2** our chosen base + harness · **B3** B2 + SFT · **B4** B3 + RL.

1. **DraftBench** — simulated drafts across seasons 2015–2024 (train) with **2025 as untouched holdout**, across a grid of formats and slots. Opponents: ADP-noise autopickers with positional-run dynamics. Metric: realized season points of the drafted roster (from actual weekly scores, weekly lineup setting) and league win rate via schedule Monte Carlo, reported as points-above-ADP-autopick. **Anonymized variant** (names stripped, features only) controls for the model having memorized 2015–2025 outcomes in pretraining.
2. **CalibBench** — dated forecast questions with realized outcomes (weekly finishes, season finishes vs. positional ADP, breakout labels pinned in `evals/labels.md`). Metrics: Brier score, log-loss, calibration curves. This is also the in-season self-learning signal.
3. **AdviceBench** — curated start/sit, trade, and pick-here questions; rubric-graded (LLM judge + human spot checks) for groundedness, use of tool evidence, and actionability.

**Ship gate:** the serving model swaps from B1 to ours only when it wins DraftBench and CalibBench and ties AdviceBench (≥95%). The business case is B4 ≥ B1 quality at a fraction of serving cost.

**Marketing honesty rule:** historical backtest claims are contamination-suspect by construction; the anonymized variant is the internal check, and the **public timestamped 2026 ledger is the only external proof we lead with.**

## 5. DraftGym (the RL environment)

- **State:** league config, full board with ADP + quant signals, both rosters' needs, picks until next turn.
- **Action:** the pick (plus a short rationale channel for SFT-compatible traces).
- **Opponents:** ADP samplers — pick ~Normal(ADP, σ·round) with need-based positional-run contagion, fitted to real FFC draft distributions (we have `stdev`, `high`, `low` per player-year).
- **Reward:** realized fantasy points of the final roster over the actual season (weekly lineups set by a fixed manager policy), normalized against the ADP-autopick baseline for that seed. Optional shaped variant: league win rate from H2H schedule Monte Carlo.
- **Cheap and parallel:** env steps are pure table lookups; only our agent's turns cost tokens. Thousands of drafts per hour per node.
- Built on **verifiers v1** (tasksets/harnesses/traces — v0 `MultiTurnEnv` is deprecated per the risk register), published to the Environments Hub, trained with prime-rl GRPO.
- **RL-reward caveat (risk register #7):** the gym's reward must use fixed-manager-policy lineups, NOT DraftBench's hindsight-optimal lineups — hindsight scoring rewards unharvestable variance and is a reward-hacking vector. DraftBench keeps hindsight for benchmarking; the gym must not.

## 6. Data plan

| Source | What | Years | Method | Status |
|---|---|---|---|---|
| nflverse releases | Weekly player stats | 2011–2025 | direct CSV download, stdlib collector | **running today** |
| nflverse releases | Snap counts, injuries | 2012–2025 / 2011–2025 | same | **running today** |
| nflverse releases | Combine, draft picks | all available | same | **running today** |
| FantasyFootballCalculator API | ADP history (std/ppr/half/2QB × 8–14 teams) + per-player draft variance | 2010–2026 | JSON API, records *returned* meta (API coerces team counts) | **running today** |
| Research repo pipeline | 2026 projections, VORP, SOS, coaching priors, availability, alpha signals | current | `make refresh` reuse | working, needs generalization |
| Reporter feeds registry | Forward news archive (2/team verified feeds) | 2026→ | RSS snapshotter to build | this sprint |
| Labels | Weekly/season finish ranks, breakout definitions, points-above-ADP | 2011–2025 | derived in `evals/` | this sprint |

ToS rules carried over from the research repo: FantasyPros only within permitted preview access and never load-bearing for product rankings; free/open sources for anything the product depends on; preserve dated snapshots, never overwrite market history.

## 7. Training plan & budget (ceiling $5k, H100 ≈ $2–3/hr)

| Phase | What | Est. cost |
|---|---|---|
| T0 | Infra smoke test: tiny Qwen3.5, LoRA SFT, 1×H100 spot, end-to-end through prime-rl | ~$50 |
| T1 | SFT: harness-distilled traces (teacher = self-hosted open model on our PI pods, over DraftGym states + dossier-style analyses + calibration-formatted forecasts), LoRA on 14–32B | ~$300 |
| T2 | RL pilot: GRPO on DraftGym subset, one 8×H100 node, 1–2 days | ~$1,000–1,500 |
| T3 | Serious RL: winning recipe on ~27–30B (3.8-27B if real, else Qwen3.5 MoE), multi-day | ~$2,000–2,500 |
| — | Reserve for failures/retries | ~$700 |

Serving starts serverless per-token; dedicated GPU (~$1–2k/mo) only after traffic justifies it. Every run gets logged to `docs/budget-ledger.md` before launch. **No run starts without an explicit go.**

**Season-split amendment (2026-08-09, Jacob's era-drift flag):** training seasons = 2008–2022 **excluding 2013 and 2018**, which are sealed as mid-era robustness diagnostics (run-heavier and pass-boom regimes respectively); forward val = 2023–2024 (primary checkpoint selection); test = 2025 (sealed, single-shot); live = 2026. Mid-era diagnostics are evaluated in masked mode (weakest player-persistence leak channel) and interpreted as drift detectors, never selection criteria — a checkpoint strong on forward-val but weak on 2013/2018 has learned an era, not a skill. Random interleaving of eval seasons is prohibited (player-persistence leakage: training on a season after the eval year lets entity trajectories leak backward). Note: `envs/verifiers_v1_adapter.py` currently asserts the old 2015–2022 split — update it in the T1 run template.

Run-template requirements from `docs/training-risk-register.md`: checkpointing ON (`--ckpt` — prime-rl defaults it off), `max_off_policy_steps` 2–4 to start (cheap rollouts), stock Dr.GRPO advantage + length/repetition filters, val envs (2023–2024) wired as live eval sources, preregistration entry in the ledger before any gate attempt (2025 attempts capped at 2 for the B4 line). Base-model choice leans **dense** over MoE (trainer/inference numerics risk).

## 8. Seven-day schedule (Aug 8–15)

- **D0 Fri (today):** game plan, scaffold, task board, historical data collectors running. ✅
- **D1 Sat:** league-agnostic engine lifted from research repo (any teams/scoring/roster/slot); label derivation; DraftBench skeleton replaying real ADP.
- **D2 Sun:** DraftGym runnable end-to-end with heuristic agents; harness v0 with quant tools wired; baseline B1 measurable.
- **D3 Mon:** Qwen 3.8 drop assessment → base-model decision memo; SFT data generation at scale; chatbot backend (harness behind an API).
- **D4 Tue:** web app MVP — manual league entry, chat, draft-day board; CalibBench + AdviceBench filled out.
- **D5 Wed:** T0 + T1 (smoke + SFT) pending go; eval B1 vs B2 vs B3.
- **D6 Thu:** T2 RL pilot pending go; env/reward iteration from results.
- **D7 Fri:** ship gate — deploy on whichever model wins; publish **ledger v1** (2026 sleepers/avoids with timestamps + methodology post).

Ship-critical for D7: app + ledger + B1 quality. Not ship-critical: T3, in-season loop polish.

## 9. Risks & mitigations

- **Backtest contamination** (models know 2015–2025 outcomes) → anonymized DraftBench variant internally; lead externally with the live ledger only.
- **Qwen 3.8 small never materializes / bad license** → model-agnostic pipeline; eval decides among Qwen3.5 / GLM-class fallbacks; nothing blocks on Monday.
- **LLM numeric weakness** → numbers only ever come from tools; the model never invents a projection.
- **One-week overrun** → ship gates ranked; app + ledger beat training milestones if forced to choose.
- **ToS/legal** → open-data-only for load-bearing logic; entertainment/analysis framing, no gambling advice; revisit for DFS later.
- **2025 holdout leaks into iteration** → 2025 gets evaluated once per contender at gate time, never during development.

## 10. Deferred (explicitly not this sprint)

Sleeper/ESPN/Yahoo league import · payments/credits · mobile apps · DFS and betting props · dynasty/keeper modes · trade-market values · auction drafts · production hardening beyond a basic deploy.
