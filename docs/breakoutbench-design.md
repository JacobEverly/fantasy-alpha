# BreakoutBench — outlier prediction benchmark (design)

**v0.2, 2026-08-08.** Slots into the eval suite (GAMEPLAN §4) as the breakout module of CalibBench, plus a public frontier-model leaderboard. Labels built: `data/processed/labels/breakouts.csv` (2011–2025 × std/half/ppr, 98%+ resolved).

v0.2 amendments (pre-first-run calibration against realized base rates):
- QB/TE ADP gate 24 → 18 (draft pools rarely carry 24 QBs/TEs; 24 made QB/TE breakouts near-impossible by construction).
- **Undrafted synthesis**: top finishers absent from the ADP pool entirely (Puka Nacua 2023, Kyren Williams 2023) are the deepest breakouts and are synthesized into the labels with `match_status=undrafted_in_pool`, `adp_pos_rank = pool_depth + 1`.
- Observed base rates under v0.2: 12–20 breakouts/season (PPR, 2022–2024), published with every result.

v0.3 amendment (2026-08-08, Jacob's review flag — freak injuries): a player's own **acute traumatic event** (ACL/Achilles/fracture class, from nflverse injury-report text) is unforecastable noise; chronic/soft-tissue durability risk is predictable skill. Therefore: (a) busts split into `performance_bust` vs `acute_injury_bust` via an injury-cause classifier; (b) every metric reports two panels — all-outcomes and freak-excluded; (c) breakouts enabled by *someone else's* injury remain valid (handcuff/beneficiary structure is skill) but get an opportunity-source annotation; (d) durability-history features belong in the packets so models can price chronic risk.

## 1. What we're measuring

"Predicting outliers" = identifying players whose realized season wildly beats (breakout) or misses (bust) their market price. Market price is ADP; realized outcome is positional finish. Both are already on disk (FFC ADP 2010–2026; season labels 2011–2025), so every definition below is computable, dated, and reproducible.

**Pinned definitions (preregistered before any model sees them):**
- `alpha(player, season, format) = adp_positional_rank − realized_positional_rank` (season totals; ppg variant with ≥8 games for injury robustness).
- **Breakout**: finished top-12 (QB/TE) or top-24 (RB/WR) at position while drafted outside the top-24 (QB/TE) / top-40 (RB/WR) at position. Roughly 8–15 players/season league-wide — a real needle-in-haystack task.
- **Bust**: drafted top-12 positional, finished outside top-24 (min 8 games played, else "injury-bust" label tracked separately).
- Base rates get published with every result — hit-rate@K means nothing without them.

## 2. The contamination problem decides the design

Any 2026-era model has memorized who broke out in 2011–2025. A retrospective benchmark run naively against frontier models measures recall, not foresight. Three regimes, used for different purposes:

| Regime | What it is | Clean for frontier comparison? | Use |
|---|---|---|---|
| **Prospective** | Freeze 2026 predictions before kickoff (~Sep 10); score against reality weekly | **Yes — the only fully clean regime** | The public leaderboard + our marketing ledger |
| **Anonymized retrospective** | Strip names/teams; present feature dossiers (age, draft capital, usage history, efficiency, depth-chart change, ADP) for a historical season; predict alpha | Mostly (structure-only reasoning) | Fast iteration on our model + harness; nightly evals |
| **Named retrospective** | Historical seasons with names | No — memorization | Internal regression tests only, never published |

Plus a **memorization canary**: in the anonymized set, a small subset of dossiers gets counterfactually perturbed (swap two players' usage profiles). A model whose picks follow the hidden identity rather than the features is leaking memory — quantifiable and reportable.

## 2b. Two evaluation modes (added 2026-08-08, Jacob's review)

- **Closed-packet mode** (default, everything in §3): identical information per contender → isolates reasoning-over-evidence; the attribution/diagnostic layer.
- **Open-tool mode**: the contender runs as an agent with retrieval over the **time-gated evidence corpus** (`published < as-of` enforced in the tool, never in the prompt) — measures the full agentic loop including deciding what to look up. Retrospective open-tool runs are named-track/internal-only (pretraining contamination via real news text); the clean open-tool comparison is prospective 2026. DraftGym inherits the same retrieval tool in its action space so RL trains the lookup skill it will be measured on.

## 3. Task battery (same packet for every model)

Every contender gets an identical, dated information packet per player: prior-season stats and usage, career trajectory, age/experience, draft capital, team/coaching changes, depth-chart summary, current ADP by format. No live browsing during the eval. Tasks:

1. **Breakout top-K**: name K=10 breakout candidates per position group with probability estimates.
2. **Bust top-K**: same for busts.
3. **Over/under vs market**: for a fixed 150-player slate, call each player's finish OVER or UNDER their ADP-implied rank, with confidence.
4. **Candidate ranking**: rank a curated 25-player "sleeper slate" (ADP 75–200) by expected alpha.
5. **Pick-timing / value capture** (v0.4, 2026-08-08, Jacob's review): for each conviction call the contender also states the round/pick it would act. Score = `survival(act_pick | adp, stdev) × realized_VORP − slot_expected_VORP(act_pick)`, where slot-expected VORP comes from the historical slot-value curve (evals/slot_values.py). Encodes the economics: over-reaching is punished by opportunity cost (taking an 8th-round ADP player in round 1 spends ~a first-rounder's expected value); acting after ADP earns ~nothing (survival collapses); the maximum sits just ahead of market price. Undrafted fliers cost a last-round slot (~zero), so hits there are pure capture. Note: DraftBench/DraftGym already price this end-to-end by construction (a reach displaces the better player from the roster); this family is the fast diagnostic equivalent.
6. *(In-season, later)*: weekly waiver-gem and start/sit variants scored weekly.

## 4. Metrics

- **Hit-rate@K vs base rate** (lift): of the K breakout calls, how many hit, vs. random-at-that-ADP-band.
- **Brier score / log-loss** on all probabilistic calls — punishes overconfidence, rewards calibration.
- **Spearman rank correlation** on the sleeper-slate ranking vs realized alpha.
- **Alpha-points captured**: mean realized `alpha` of top-K picks — measures magnitude, not just direction.
- **Portfolio ROI**: draft the model's calls at their ADP in simulated rosters; measure realized points vs ADP-autopick (connects directly to DraftBench).

## 5. Baselines (a benchmark without baselines is a demo)

1. **Market**: ADP itself (null hypothesis: no alpha exists).
2. **Naive momentum**: last season's finish repeats.
3. **Age-curve heuristic**: year-2/3 WR bump, RB cliff at 27, etc. — the "podcast wisdom" baseline.
4. **GBDT**: gradient-boosted trees on our feature set — the "boring ML" baseline any LLM must beat to justify itself.
5. **Consensus experts**: FantasyPros ECR-implied finishes (licensed access permitting).

## 6. Frontier comparison harness

Same packet, same JSON output schema, temperature 0, N=3 seeds, majority/mean aggregation: GPT (latest), Gemini (latest), Claude, Qwen 3.8-Max API, DeepSeek, plus **our harness+model** and the naked open base model (measures how much our training + harness add). Estimated cost per full battery: $50–200 in API calls. Results in two tracks:
- **Open-book track** (named, prospective 2026): measures the full system including world knowledge.
- **Structure track** (anonymized retrospective): measures reasoning over evidence, memorization-controlled.

## 7. 2026 prospective protocol (time-sensitive)

1. Freeze the packet snapshot and all model predictions by **Sep 4, 2026** (kickoff Sep 10).
2. Publish every prediction with timestamps + a hash of the packet (provably pre-season).
3. Score publicly: first read at Week 6, midseason at Week 10, final at Week 18.
4. This IS the public ledger from GAMEPLAN §4 — the benchmark and the marketing asset are the same artifact. "We benchmarked GPT/Gemini/Claude/ours on 2026 breakouts — live leaderboard" is the launch post.

## 8. Prior art & adopted methodology (landscape research 2026-08-08)

Full report: `docs/benchmark-landscape.md`. Verdict: **no purpose-built fantasy football benchmark for AI systems exists** — we would be first. What we adopt from adjacent work:

- **FantasyPros Accuracy Gap replication** becomes its own track: their draft/weekly rankings scoring (rank-slot → 3-yr-average points, |projected − actual| gaps, pool = ECR-top-N ∪ actual-top-N, positional weighting) is public and replicable offline — scoring our system with THEIR method yields direct comparability to ~212 named human experts and a decade of leaderboards. Stretch goal: enter the actual 2027 competition as an AI entrant.
- **Baseline hierarchy hardened** (decade of evidence that consensus beats individuals): naive last-season/rolling baselines scored with **MASE**, then market (ADP, ECR, Vegas-derived), then experts. ECR consensus is the must-beat bar, not individual experts.
- **Scoring rules extended**: CRPS when we output point distributions (Big Data Bowl precedent); difficulty-adjusted Brier (ForecastBench 2026 methodology) across heterogeneous question sets; head-to-head peer scores with significance tests (Metaculus) for model-vs-model claims.
- **Two-track split** (KellyBench precedent — frontier models that recite bankroll theory still go broke executing it): Track 1 = one-shot prediction accuracy (this doc); Track 2 = **sequential decision quality** (draft rooms, season-long waiver/start-sit management) — which is exactly DraftGym, so the RL environment and the decision benchmark are one artifact.
- **Published comparables to beat**: IBM Watson×ESPN 72% boom/bust accuracy, 6.78 weekly RMSE (arXiv:2111.02874); best-source season MAE ≈ QB 61 / RB 52 / WR 40 / TE 31 (FantasyFootballAnalytics). Amazon's production NFL assistant is evaluated only on analyst agreement, not outcomes — outcome accuracy is the open flank in industry.

## 9. Governance

- Definitions in this doc are frozen before first run; changes require a new benchmark version number.
- 2025 season stays the untouched holdout for model development (GAMEPLAN §9); BreakoutBench retrospective sets use 2015–2024.
- Publish negative results too — credibility compounds; cherry-picking is how the incumbents lost sophisticated users' trust.
