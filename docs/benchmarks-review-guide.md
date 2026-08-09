# How the benchmarks work — reviewer's guide

Written 2026-08-08 for Jacob's review. Plain-language mechanics of each benchmark, with the challengeable design decisions called out explicitly. Flag anything — decisions marked ⚖️ are judgment calls where reasonable people could choose differently.

**The mental model:** three benchmarks measure three different skills, plus one external anchor.

| Benchmark | Skill measured | One-line mechanic |
|---|---|---|
| DraftBench | Can you draft a better team than the market? | Simulated drafts, scored by the roster's REAL season points |
| BreakoutBench | Can you spot the outliers? | Blind player dossiers → probability picks → scored vs what happened |
| CalibBench | Do your probabilities mean anything? | Forecast questions → Brier score + reliability curves |
| FP replication | How do we compare to named human experts? | FantasyPros' own accuracy formula, implemented on our data |

---

## 1. DraftBench (`evals/draftbench.py`)

**What happens, step by step:**
1. Build a league (default: 12 teams, QB1/RB2/WR3/TE1/FLEX2/BENCH6, 15 rounds) for a historical season, say 2021 PPR.
2. The board is that year's real ADP pool (FantasyFootballCalculator, captured with per-player mean/stdev/high/low).
3. Eleven opponents draft by "noisy ADP": each pick they take the best player after jittering ADP by his real draft-position variance. They respect roster feasibility.
4. The contender (an agent: a heuristic, later an LLM policy) drafts from one seat.
5. When the draft ends, we compute what that roster **actually scored** that season using real weekly stats, setting lineups each week.
6. The metric: contender's points **minus an ADP-autopick control that drafted from the same seat with the same random seed**. Same luck, same opponents — the difference is pure skill.

**Measured so far:** naive value-drafting (greedy VORP on last-season points) loses to the market by −141 pts/season. The "wait on QB/TE" heuristic is dead neutral. ADP itself beats noisy opponents by +39.

**⚖️ Decisions you might challenge:**
1. Opponents are ADP-noise robots with no positional-run "panic" behavior. Real rooms have runs. (A run-contagion opponent family is designed but not built.)
2. Lineup-setting is **hindsight-optimal** — we award the best possible weekly lineup. Real managers bench breakouts for weeks. This inflates everyone equally as a benchmark, but the risk register flagged it as a reward-hacking vector for RL training, where it must switch to a realistic manager policy.
3. Seasons pool 2015–2024 — a 2015 draft room and a 2024 one are treated as equally informative.
4. One projection-free agent design so far; agents that use projections will inherit projection-source bias.

## 2. BreakoutBench (`evals/anon_demo.py`, expanding into `evals/breakoutbench.py`)

**What happens:**
1. "Breakout" is preregistered: finished top-12 (QB/TE) / top-24 (RB/WR) at position while drafted outside top-18 (QB/TE) / top-40 (RB/WR). ~8–20 per season. Busts and undrafted-in-pool breakouts (the Pukas) are also labeled.
2. For a season, every gate-eligible drafted player becomes a **blind dossier**: position, ADP + market disagreement (stdev), years in league, last two seasons' games/points/rank. Names stripped, IDs shuffled, answer key sequestered in a separate file.
3. The contender picks 10 with probabilities (expanding to: probability on EVERY candidate, plus bust and over/under-vs-ADP families — ~15x more judgments).
4. Scoring: hits@10 vs the published base rate, lift vs random, Brier on the probabilities. Bootstrap CIs resample *seasons* (the honest independence unit).

**Measured so far (pooled, 100 picks, 2015–2024):** random 1.0x by construction · GBDT 1.02x but Brier 0.130 (perfectly calibrated, can't rank) · Qwen naked 1.3x but Brier 0.493 (overconfident garbage) · Qwen + harness grounding 1.2x, Brier 0.241. Claude one-season demo: 4.0x, with honestly disclosed partial de-anonymization.

**⚖️ Decisions you might challenge:**
1. The breakout gates (top-24 finish / outside top-40 ADP for RB/WR) are defensible but arbitrary — moving them changes who counts. They were calibrated to realized base rates BEFORE any model ran, and are versioned (v0.2).
2. Anonymization removes names but a distinctive stat line can still be fingerprinted by a model with memorized history — that's why canaries (counterfactually swapped dossiers) are in the design; they are **not built yet**.
3. Historical PPR ADP pools resolve to 8-team boards (the API's coercion) — shallower pools, fewer gate-eligible candidates in some years (2022: only 33).
4. Dossier features are deliberately thin (structure-only). The GBDT result says these features contain almost no rankable signal — which we read as "the market already priced them," and motivates the evidence-channel enrichment. You could instead read it as "the benchmark can't measure selection skill at all until packets are richer." Both readings imply the same next step.

## 3. CalibBench (`evals/calibbench.py`)

**What happens:**
1. Two question families. **Season threshold**: "will this player finish top-24 at WR?" — one per ADP-pool player, same blind dossiers. **Weekly head-to-head**: "will A outscore B in week 13?" — pairs matched to be genuinely uncertain (same position, similar per-game scoring entering that week), so the base rate is ~50% by construction.
2. Contender answers a probability per question. Key files hold outcomes; question files are unit-tested to contain zero outcome fields.
3. Scoring: Brier, log-loss, a 10-bin reliability table (when you say 70%, does it happen 70% of the time?), and ECE.
4. Two built-in reference rows: a coin (p=0.5 → Brier 0.2500 exactly) and a no-peek base rate (computed from other seasons only). **The bar is the base rate (~0.235 on season questions), not the coin.**

**⚖️ Decisions you might challenge:**
1. Weekly H2H pairs require ≥3 games and ≥5 ppg entering the week — excludes early-season and fringe players by design.
2. The 20%-similarity matching makes questions hard on purpose; a looser band would let skill show up more easily but measure less.
3. 300 pairs/season (was 50) is a parameter, not a law.

## 4. FantasyPros accuracy replication (`evals/fp_accuracy.py`)

Their published formula (verified this week against their methodology FAQ), implemented with leakage guards: rankings → slot-value curves from the prior 3 years → weighted |expected − actual| gaps over a pool of (consensus top-N ∪ actual top-N). Scored so far: ADP 11,444 · last-season momentum 16,933 · 3-yr average 18,139 (lower = better; ADP wins everywhere, every year).

**⚖️ Honest limits:** FP publishes no raw scores (ordinal leaderboards only), so external comparison is ordinal; we proxy their ECR with FFC ADP and their half-PPR with PPR; nine assumptions are itemized in the report. The clean external play is entering their 2027 competition.

## Cross-cutting rules (apply to everything)

- **2025 is sealed.** Every tool refuses it without an explicit `--include-holdout` flag. It gets used once per contender, at gate time.
- **Walk-forward always**: no feature, base rate, or training row may come from the season being scored or later. Tests poison future data and assert predictions don't move.
- **Determinism**: same seed → byte-identical outputs, everywhere.
- **Baselines built in**: random, coin, base-rate, market, boring-ML. A number without its null is not a result.
- **Contamination honesty**: named-historical results are never published; anonymized track for iteration; the timestamped 2026 ledger (predictions frozen by Sep 4) is the only external proof.
- 88 tests currently enforce the above.

## Questions worth asking as you review

1. Do the breakout gate thresholds match what YOU mean by "breakout"? (This is the most consequential preregistered choice.)
2. Is points-above-autopick the right DraftBench metric, or should it be league win rate (schedule Monte Carlo — designed, not built)?
3. Should the anonymized dossiers include age and draft capital? (Currently absent — years-in-league is a weak proxy. Adding them helps models AND helps fingerprinting.)
4. Are 8-team ADP boards acceptable for PPR history, or should standard-format boards (12-team, deeper) drive the slates?
5. Is hindsight-optimal lineup scoring acceptable for the *benchmark* (it's banned for RL reward already)?
6. Weekly H2H at ~50% base rate measures pure calibration — do you also want an easier "any-pair" family that measures ranking skill?
7. The 10-seasons ceiling: comfortable treating 2015–2019 and 2020–2024 as one pooled regime, or should headline numbers be 2020+ only?

## Current consolidated scoreboard

| Contender | BreakoutBench lift | Brier (top-10) | DraftBench vs market | FP gap |
|---|---|---|---|---|
| Market (ADP) | — | — | 0 (control) | 11,444 |
| Random | 1.0x | — | — | — |
| GBDT | 1.02x [0.77–1.30] | 0.130 | — | — |
| Naive momentum | — | — | −141/season (greedy-VORP variant) | 16,933 |
| Qwen3.5-9B naked | 1.3x | 0.493 | not yet run | — |
| Qwen3.5-9B + harness | 1.2x | 0.241 | not yet run | — |

Everything above cost $0.03 of inference and runs on a laptop. The gaps in this table (Qwen on DraftBench, enriched packets, canaries, expanded CIs) are the current work queue.

## Normalized scoring (answers open question #2)

Jacob's call: **report both — lead with the normalized layer, keep raw points-above-control as the statistical backbone.** Raw season totals ("Fable 2900 vs ours 3050") are unrelatable and not comparable across eras or scoring formats; `evals/normalize.py` translates any evaluated roster onto a league-anchored scale:

- **League Index** = 100 × roster season points / best *opposing* team's season points, in the same simulated league (same seed, same lineup policy). 100 = tied with the league-best opponent; >100 = outscored everyone. Emitted with league rank (1..12) and percentile.
- **Championship odds** = schedule Monte Carlo over the teams' realized weekly scores (2000 sims, deterministic under seed): weeks 1–14 random weekly pairings, standings by W-L with points-for tiebreak, top-4 playoff — semifinal week 15, final weeks 16+17 summed (pre-2021 data without a week 17: final = weeks 15+16, week 15 shared with the semi by documented convention). Output: expected wins, playoff%, title%.

Both DraftBench result rows (`league_index`, `league_rank`, `playoff_pct`, `title_pct` columns) and DraftGym terminal info now carry these fields — reporting only, never reward. The full record translated onto this scale lives in `evals/results/normalized_rescore.md`. Headline translation: untrained Qwen named (+171 pts) reads as **index 92, title 12%** vs its autopick controls' index 84, title 7%; Fable named episodes read as **index 107, title 48–91%** vs controls at index 65–78, title 0%; the masked variants of both drop to index 72–86, title ≤2% — the memorization gap is just as visible on the normalized scale. The autopick control itself averages index 87, title 9% (not 100 — even a market-consensus drafter usually trails somebody in an 11-opponent league; a title-odds base rate of 1/12 ≈ 8.3% is the "no skill" anchor).
