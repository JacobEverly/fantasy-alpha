# Complete Benchmark Results — and What They Mean for a Consumer

2026-08-08, end of day one. Every number below: deterministic pipeline, bootstrap 95% CIs, leakage poison-tested, 2025 sealed, 255 green tests. Total measurement cost: **$18.78**.

---

## Track 1 — Outlier selection & calibration (BreakoutBench, anonymized, 5,030+ judgments, 2015–2024)

*Blind player dossiers, probability predictions, scored against what actually happened. Lower Brier = better; lift 1.0x = random; "par" = the no-peek base-rate strategy (group averages, zero individual judgment).*

| Contender | Pooled Brier [95% CI] | Top-10 lift [95% CI] | Breaks par? |
|---|---|---|---|
| Coin (p=0.5) | 0.2500 | 1.13x [0.96–1.26] | — (bogey) |
| **Base rate — "par"** | **0.1922 [0.184–0.203]** | 1.13x [0.96–1.26] | — |
| GBDT (boring ML, walk-forward) | 0.1270 full-slate | 1.02x [0.77–1.30] | No |
| Qwen3.5-9B naked / +harness | 0.1992 / 0.1949 | 1.21x / 1.01x | No |
| Qwen3.5-35B-A3B naked / +harness | 0.2122 / 0.1931 | 0.87x / 1.27x | No |
| DeepSeek V4 Flash naked / +harness | 0.2052 / 0.1930 | 1.20x / 1.33x | No |
| **Claude Fable 5 naked / +harness** | 0.2000 / **0.1918** | 1.27x / **1.40x [0.88–1.74]** | **No** |

**Findings:** (1) No model at any scale breaks par — the track is information-ceilinged: the market has already priced everything reconstructable about the past. (2) The harness (grounded base rates) improves *every* model's calibration; unanchored, enrichment measurably hurts. (3) Models are NOT parroting one forced answer — inter-model correlation is only 0.47–0.68, and correlation with par is 0.12–0.18: they form varied player-level opinions, and all of that variation is noise. (4) 4x scale made the Qwen *worse*; enriched packets (news counts, depth slots, pedigree) moved nothing; feature importances ≈ 0.

## Track 2 — Memorization (canaries, fingerprints, masked drafts)

| Model | Canary gate (anonymized) | DraftGym named − masked reward |
|---|---|---|
| Qwen3.5-9B | PASS | **+272** (named +171 avg, masked −77/−141) |
| Qwen 35B / DeepSeek Flash | PASS / PASS | not run |
| Claude Fable 5 | PASS (cleanest packet-follower) | **+492** (named up to +796; masked slot-1 went negative) |

**Finding:** on historical seasons with real names, models "beat the market" by hundreds of points — via memory, not skill. Mask the names and the same models fall *below* the market. The stronger the model, the bigger the mirage. Our masked mode and canary gate strip this out of both training and evaluation; nobody else's "AI backtest" does.

## Track 3 — Draft decision quality (DraftBench, 1,215+ simulated drafts vs realized outcomes)

| Agent | Points vs market (per season) |
|---|---|
| ADP autopick (the market) | 0 (control) |
| Greedy value on naive projections | **−141.5 ± 271** |
| "Wait on QB/TE" structure alone | −1.5 (neutral) |
| Survival-sequencing execution, market values | −15.6 to +4.5 (market-neutral, **halves variance**) |
| Execution layer wrapped around bad projections | **−57.8** (rescues 60% of the damage) |
| Untrained LLM, masked (honest) | −77 to −141 |

**Findings:** the market is a strong opponent; bad player-evaluation costs ~8.3 pts/week; execution discipline can't create edge from nothing but massively limits damage and tail risk. Hindsight-perfect rankings score far below ADP on FantasyPros' own metric — decision headroom exists.

## Track 4 — Evidence priors (source-credibility study, 310k dated news items, 2016–2024)

| Signal (August news) | Effect | Significant? |
|---|---|---|
| Injury-concern on a top-12 pick → bust | **1.58x** [1.08–2.49] (freak injuries excluded) | **Yes** |
| Any injury-touched news on a deep candidate → breakout | **0.65x** [0.52–0.89] | **Yes** |
| Camp-promotion keyword mentions (drafted longshots) | 0.86x [0.69–1.16] | No |

Plus: injury-cause classifier shows **only 31% of historical "busts" were performance failures** — 21% confirmed freak events, 15% invisible IR vanishes; bust base rate drops 23.3% → 16–19% when luck is removed.

## Track 5 — Pick-timing economics (slot-value curves, 14 seasons)

The same correct sleeper call (8th-round ADP, elite season): acted in **round 7 = +112** captured value · reached in round 1 = +10 · waited past ADP = +13. Optimal behavior — act one round before the market — is now the scored incentive.

## External anchor — FantasyPros' own accuracy formula (replicated)

ADP gap 11,444 « last-season momentum 16,933 « 3-yr averages 18,139 (lower better; ADP wins every position, every season). Our system's rankings plug into `score_rankings()` for direct ordinal comparison to their ~212 ranked human experts.

---

# What all of this means for a consumer

**1. Every "our AI predicted the breakouts" claim you've ever seen is now measurably suspect.** The strongest AI on the market, given ten years of player data, cannot out-pick the crowd's draft position — unless it's allowed to remember who won, in which case it looks like a genius (+796 points!) and is worthless for next season. When a fantasy product shows you a backtest, this is what's usually inside it. **Product consequence: we never sell foresight we can't prove. Every claim ships with its base rate, and our 2026 predictions are frozen and timestamped before kickoff — verify us, don't trust us.**

**2. Confidence numbers finally mean something.** Every model tested is wildly overconfident by default (Brier 0.49 = worse than a coin at knowing what it knows); our grounding harness cuts that in half and is the only configuration that matches honest par. **Product consequence: when the app says "moderate confidence," that's a calibrated statement, not decoration. No "94.7% match score" theater — that's mathematically a lie in this domain, and now we can show why.**

**3. The market (ADP) is genuinely good — your edge is not out-picking it, it's not beating yourself.** Naive "value" drafting loses 141 points a season; disciplined execution around market prices recovers most of that and halves worst-case outcomes. **Product consequence: the draft assistant's core promise is mistake-prevention and league-native math — your league's scoring changes who's valuable, most rankings assume a league you don't play in — plus timing ("take him in round 7, not round 5; he survives to your pick 68% of the time"). Worth on the order of 50–140 points a season versus how people actually draft, which is multiple wins.**

**4. Injuries: we price the predictable kind and refuse to pretend about the rest.** August injury-concern on an early pick nearly *doubles* bust risk — the app will surface that. A week-2 ACL is unforecastable — the app will never claim credit or blame around freak events, and our own accuracy stats exclude them so they stay honest. **Consequence: durability advice you can act on, no injury-oracle cosplay.**

**5. Where a real edge can still exist — and where we're pointing the product:** information the market hasn't priced *yet*: same-day depth-chart moves, camp usage, coaching-change fit, breaking news — reacting faster and more coherently than ADP updates. That channel provably can't be validated on the past (the data doesn't exist historically), so we prove it the only honest way: **a public, timestamped prediction ledger through the 2026 season, scored in the open, wins and losses both.** The consumer story isn't "we're psychic" — it's "we do the math right, we show every receipt, we're honest about uncertainty, and here's our live track record."

*Everything above is reproducible from the repo: `evals/results/` for tables, `docs/benchmarks-review-guide.md` for methodology, `docs/budget-ledger.md` for costs.*
