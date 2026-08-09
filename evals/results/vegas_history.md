# Historical Vegas channel — free game lines + archived-props probe (2026-08-08)

Extends `odds_channel_scoping.md` ("player props history: purchase-only") with
the two FREE paths. Path 1 (nflverse game lines, 1999→) is fully built and
validated; Path 2 (Wayback/article recovery of season player props) is a
time-boxed feasibility probe with a yield verdict.

Code: `scripts/collect_vegas_lines.py` (collector),
`evals/vegas_features.py` (features + walk-forward study).
Tests: `tests/test_collect_vegas_lines.py`, `tests/test_vegas_features.py`.
Data: `data/raw/vegas/` (immutable dated snapshots),
`data/processed/vegas/` (join-ready tables),
`data/raw/vegas/archived_props/` (Path-2 extracted lines).

## Path 1 — nflverse game lines (built)

### Source & capture

Lee Sharpe's `games.csv` (github.com/nflverse/nfldata): per-game closing
`spread_line` + `total_line` + scores, **1999→present, 100% REG-season line
coverage 1999–2025**, plus 52 already-posted 2026 look-ahead lines (week 1
included). Free, no key, stdlib download. Snapshot captured 2026-08-09 (UTC):
`data/raw/vegas/games_2026-08-09.csv` (2.2MB, 7,548 games, sha256 in
`data/raw/vegas/manifest.json`). Dated snapshots are immutable; the collector
refuses to overwrite and validates schema + line coverage before writing.

### Timing honesty (what is knowable when)

Closing lines are **per-week**, set days before each game. There is no
preseason aggregate of closing lines — the season-mean implied total for
season S exists only after season S ends. Feature-by-feature:

| Feature | Knowable | Legal uses |
|---|---|---|
| (team, week) implied total | a few days before that week's game | in-season features, weekly context |
| (team, season) mean implied total | after the season | retrospective environment score, eval labels — **never** in a same-season draft packet |
| week-1 implied total (season S) | months before week 1 (look-ahead lines; posted for 2026 already) | draft-day packets |
| prior-season (S-1) mean implied total | before season S | draft-day packets |

Implied team total = `total_line/2 ± spread_line/2` (`spread_line` = home
expected margin, positive = home favored).

### Emitted tables (join keys for the packet builder)

- `data/processed/vegas/implied_team_totals.csv` — 14,038 rows, keys
  **(season, week, team)**; columns opponent, is_home, gameday, total_line,
  team_spread (negative = favored), implied_total. Pre-game knowable.
- `data/processed/vegas/team_season_vegas.csv` — 893 rows, keys
  **(season, team)**; `mean_implied_total` (retrospective) plus the
  draft-day-legal pair `week1_implied_total` +
  `prior_season_mean_implied_total`.

Team codes are normalized to **current franchise codes** (SD→LAC, STL→LA,
OAK→LV, JAC→JAX), matching how `data/processed/labels/*.csv` and
`packet_features` code historical seasons — tables join directly on
(season, team) with no mapping. `packet_features.py` was not modified
(other agents' module); joining these tables in is a follow-up wiring task.

### Validation study — is Vegas additive over naive?

Walk-forward, 793 team-seasons 2000–2024 (**2025 holdout untouched**).
Predict season-S team output — (a) summed player half-PPR points (weeks
1–18, from `labels/weekly_points.csv`), (b) team offensive TDs (rushing +
receiving TDs from nflverse stats) — from features knowable before season S.
Naive baseline: prior-season realized team fantasy points. Pearson r with
2,000-draw bootstrap 95% CIs (seeded). "Incremental" = correlation of the
Vegas feature with the naive model's residual — the additivity test.
Reproduce: `python -m evals.vegas_features --study`.

Outcome = team fantasy points:

| Era (n) | prior-mean implied | week-1 implied | naive baseline | prior-mean incr. | **week-1 incr.** |
|---|---|---|---|---|---|
| 2000–2007 (251) | +0.49 | +0.40 | +0.54 | +0.06 [-0.07,+0.18] | **+0.12 [+0.00,+0.23]** |
| 2008–2015 (256) | +0.51 | +0.51 | +0.55 | +0.03 [-0.09,+0.15] | **+0.18 [+0.07,+0.28]** |
| 2016–2024 (286) | +0.42 | +0.44 | +0.44 | +0.05 [-0.06,+0.17] | **+0.23 [+0.11,+0.32]** |
| pooled (793)   | +0.53 | +0.51 | +0.59 | +0.03 [-0.03,+0.10] | **+0.17 [+0.11,+0.23]** |

Outcome = team offensive TDs: same pattern, week-1 incremental
+0.17/+0.19/+0.21 per era (all CIs exclude 0), prior-mean incremental
+0.06 to +0.09 (CIs straddle 0 in every era, pooled +0.06 [-0.01,+0.13]).

**Verdict: yes for the week-1 line, no for the prior-season mean.**
- The **week-1 implied total is additive** over prior-season realized points
  in every era, and the edge is *growing* (+0.12 → +0.23 across eras) —
  consistent with the market pricing offseason information (QB changes,
  coaching, roster moves) that last year's box scores can't see. This is the
  feature to put in the draft-day packet.
- The **prior-season mean implied total adds nothing** once you have prior
  realized points — closing lines from season S-1 encode roughly the same
  information as season S-1 results. Keep it as a retrospective environment
  descriptor, not a packet feature.
- Head-to-head levels: naive ≥ either Vegas feature alone in most eras; the
  market's value here is *incremental*, not substitutional.

Caveat: week-1 implied totals used are the **closing** week-1 lines (games.csv
carries closings); August look-ahead lines are slightly staler than what we
scored, so the live draft-day edge is likely a bit smaller than the +0.17–0.23
measured. The 2026 capture (look-aheads, running daily via the odds channel)
will measure that gap prospectively.

## Path 2 — archived season-props probe (time-boxed)

Question: how many season player-prop O/U **lines** (pass/rush/rec yards etc.)
are recoverable free from archived articles for 2015–2024?

### What the probe found

- **One proven "anchor dump":** Sports Insights "Bovada's NFL Prop Bet
  Motherload" (2017-09-06, still live) republishes Bovada's full season board
  in clean HTML tables. **Extracted programmatically: 314 fantasy-relevant
  player-lines** (pass yds 28, pass TDs 27, INTs 27, rush yds 37, rec yds 71,
  rec TDs 67, receptions 9, combo markets 48) → stored as facts + source URL
  (no article text) at
  `data/raw/vegas/archived_props/bovada_2017_season_props_sportsinsights.csv`.
  ~50% of lines carry one-sided juice (e.g. `u-140`); the rest are bare
  lines — **de-vig is impossible without both prices**, so most of this is
  median-line data, exactly the marginal-median trap `vegas-arbitrage.md`
  warns about. Usable for rank/median studies, not calibrated (mean, sigma).
- **2016:** only league-leader markets (OddsShark "most passing yards" odds)
  and rookie-specific props surfaced — no O/U yardage board. Estimate **<30
  usable lines**.
- **2019:** DraftKings released 6 season markets; the Action Network article
  covering it is prose commentary (~10 lines with numbers) even in its
  unpaywalled 2019-08-21 Wayback capture; MyBookie's "full odds" page is
  JS-rendered with no content in HTML. Estimate **30–80 lines** recoverable by
  stitching 5–10 pick-articles (PFF, theScore, sportsbettingdime...).
- **2022:** same genre mix (SportsbookReview "22 futures", sportsbettingdime
  picks) — **20–60 lines** per stitching effort. BettingPros had full boards
  but is FantasyPros property → excluded per CLAUDE.md constraint.
- Wayback CDX of sportsbook pages themselves (Bovada/DK/FD boards): dynamic
  pages, effectively uncaptured.

### Yield & effort estimate for a full 2015–2024 harvest

Optimistically 1–3 anchor years (2017 proven; 2018/2021-style "motherload"
posts may exist — unverified) at 200–350 lines each, plus 20–80
article-stitched lines for other years at ~2–4 hours/season of manual
search + extraction. Total ≈ **800–1,500 heterogeneous player-lines across
the decade**, mixed books, mixed capture dates (opening vs. September),
mostly without two-sided prices.

### Verdict: NOT worth a full harvest

As a load-bearing backtest dataset this fails on all three axes the channel
needs: depth (most seasons <100 lines vs ~350 needed), pricing (no de-vig for
the majority of lines), and consistency (book/date heterogeneity across
seasons). Recommended posture:

1. **Keep the opportunistic anchors only** — the 2017 Bovada board is banked
   (~1 hour each for any further anchors discovered in passing); n=1–3 seasons
   of median-line data is a useful *sanity cross-check* against the purchased
   data, nothing more.
2. The **$59 one-month Odds API purchase** (scoping doc §3) remains the only
   credible player-level history path (2023–2024, real two-sided prices).
3. The deep backtest lever stays team-level: Path-1 game lines (this doc,
   1999→) + SportsOddsHistory win totals.

Copyright posture followed: extracted lines are uncopyrightable facts; stored
lines + source URLs only, never article text.
