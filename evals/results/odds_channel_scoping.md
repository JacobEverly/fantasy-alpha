# Betting-odds channel — source scoping & de-vig design (2026-08-08)

Jacob's prior: de-vigged betting markets beat fantasy-site projections. This doc
wires that prior in as a **measured source, not doctrine** — the market becomes
(a) a raw evidence stream we archive daily, (b) a benchmark contender scored by
the same evals as every other source, and (c) a credibility row that earns or
loses weight as outcomes accrue. Method background:
`fantasy-football-alpha-2026/docs/vegas-arbitrage.md` (de-vig procedure,
marginal-median trap, confidence policy) — that doc's conversion rules are
adopted here unchanged.

Capture code: `scripts/collect_odds.py` (standalone or as the `odds` source in
`scripts/archive_evidence.py`). Tests: `tests/test_collect_odds.py`.

## 1. What is obtainable right now (Aug 2026)

### Free, no key, captured daily as of today

| Source | What | Verdict |
|---|---|---|
| **ESPN core API NFL futures** (`sports.core.api.espn.com/v2/.../seasons/2026/futures`) | 23 DraftKings-priced season markets: MVP/OPOY/ROY awards, **season leaders in pass/rush/rec yards** (player-level, priced), division + conference + Super Bowl winners, "most regular-season wins". Public unauthenticated JSON. | **Captured** → `data/raw/evidence/<date>/odds/espn_futures.json`. No per-team win-total O/U lines and no player O/U props — it's winner-market implied strength, not stat lines. Useful for the vegas-arbitrage doc's "team environment disagreement" check. |
| **FFC ADP** (already archived) | Market of drafters, not books | Already the archiver's ADP leg; listed for completeness. |

First run (2026-08-08): 23 futures markets, 136KB, provenance-stamped;
`odds` rolled into the day's `manifest.json` alongside the other five sources.

### Free with a signup key (implemented, awaiting key)

**The Odds API** (`the-odds-api.com`) — free tier: email signup → API key, no
card, **500 credits/month**. Covered markets: NFL h2h/spreads/totals (featured,
1 credit per market×region), active NFL futures/outrights (Super Bowl winner
etc., discovered from the 0-credit sports index), and **per-game player props
in-season** (pass/rush/rec yards+TDs+receptions and more, event-level calls).
No season-long player props and no per-team win-total market on any tier.
Historical odds: paid tiers only.

`collect_odds.py` is implemented against their v4 schema: reads `ODDS_API_KEY`
from `.env` (documented in `.env.example`), skips gracefully with status
`skipped` when absent, redacts the key from stored `_source` URLs, and records
`x-requests-remaining` in the manifest for budget tracking.

- **Daily cost as coded:** sports index (0) + game odds (3) + ≤8 futures (≤8)
  ≈ ≤11 credits/day ≈ ~340/month worst case — inside the free tier.
- **Weekly player props once the season starts:** event-level calls cost
  markets×regions each. 16 games × 6 core markets × 1 region ≈ 96
  credits/week ≈ 384+/month — **does not fit free tier alongside the daily
  capture**. Ask: **$30/month (20K credits)** when Week 1 props post, if we
  want weekly props. That is the entire paid ask for live capture.

### Reachable but NOT automatable (ToS) — the aggregator findings

**StartWho (startwho.com/nfl/draft) is almost certainly the "start who" site**
— it is already cited in `docs/vegas-arbitrage.md` and its 2026-08-03 snapshot
lives at `fantasy-football-alpha-2026/data/manual/vegas_derived_rankings_2026-08-03.csv`.
Scoping probe (one-off, manual) found the site is a Next.js app over an open
JSON endpoint (`/api/projections/season?sport=NFL&format=Half-PPR&position=QB`)
that returns, per player, **exactly the raw schema the vegas-arbitrage doc
demands**: per-book season prop lines with over/under prices (FanDuel +
"Consensus (Action Network)"), a `source: vegas|projected` flag per component
(imputation made explicit!), de-vigged EV values, and league-format points.
Today's coverage: 154 players (QB 29 / RB 37 / WR 62 / TE 26), 8 prop types,
591 component props (356 vegas-priced, 235 model-imputed), refreshed daily.

**But their ToS explicitly prohibits bots/automated access, scraping, and any
non-personal/commercial use.** So: **not added to the archiver.** Options, in
order of preference:
1. **Ask permission** — small operation (one X account, an iOS app); a
   data-partnership email is cheap and they may welcome attribution/traffic.
   Until then it stays a *manual* research lead: Jacob saving a page/CSV by
   hand for personal research (the 08-03 precedent) is the defensible mode.
2. Replicate their pipeline ourselves from raw books via The Odds API paid
   props (below) — which we need anyway for anything load-bearing.

Other aggregators checked:
- **BettingPros** (bettingpros.com/nfl/props) — season-long prop analyzer with
  7-book consensus. **FantasyPros property** → falls under the CLAUDE.md
  FantasyPros constraint (preview-only, never load-bearing). Skip.
- **Win With Odds** (winwithodds.com) and **First Down Studio**
  (firstdown.studio/season-rankings) — same genre as StartWho (props →
  league-scored season projections). Research leads for cross-checking a
  manual StartWho snapshot; not automation targets without a ToS read.
- **Stokastic** — subscription DFS tooling, weekly-focused. Skip.
- **Sportsbook site endpoints** (DraftKings sportsbook/Pick6, FanDuel internal
  JSON) — technically reachable, ToS-prohibited surfaces. **Not scraped.**
  DK Pick6 remains a *manual discovery surface* per the vegas-arbitrage doc.

### Historical depth — the verdict

- **Season win totals: solved, free.** SportsOddsHistory (now hosted at
  Covers.com, `sportsoddshistory.com/nfl-regular-season-win-total-results-by-team/`)
  publishes preseason Vegas win totals + results per team back to the 1990s,
  freely usable for research/citation; ESPN/FiveThirtyEight cite it. nfelo
  (`nfeloapp.com/nfl-power-ratings/nfl-win-totals/`) has totals+odds+results
  tables as a cross-check. Wayback captures of VegasInsider/Covers give
  point-in-time August lines where SOH only has one preseason number. This is
  enough to backtest team-environment signals over 10+ seasons.
- **Player props history: purchase-only, and shallow everywhere.**
  - **The Odds API historical** — game lines from mid-2020, **player props
    only from May 2023**; $30–249/month tiers (historical included on all
    paid tiers). Cheapest credible option: one month of $59 (100K credits)
    to pull prop snapshots for 2023 + 2024 preseason/weekly, then cancel.
  - **SportsDataIO historical odds** — props and futures from 2020,
    ~$500–1k+/month tier (same vendor as the news Vault; could ride one
    trial/negotiation).
  - **Odds Warehouse** (oddswarehouse.com) — NFL game odds databases
    2009–2025, one-time purchase per season or bundle (game lines, not a
    props archive).
  - **Big data resellers** (BigDataBall etc.) — game lines + DFS salaries,
    not season props.
  - **Season-long player prop archives effectively do not exist** at any
    vendor before ~2020, and *season* (vs weekly) props are patchy even
    after; nobody sells "August 2018 FanDuel season receiving-yards lines".
    Wayback of aggregator pages (RotoWire prop screens, VegasInsider) is the
    only partial recovery path and is capture-sparse.

**Honest validation limit:** the prior "de-vigged markets beat site
projections" can be backtested *deeply* only at the team level (win totals,
1990s→). At the player level we can assemble at most **2023 and 2024**
preseason prop sets (purchased), since 2025 is the untouched holdout. n=2
seasons of player-level evidence is a sanity check, not proof — the real test
is prospective: archive 2026 lines daily (running as of today), score them
against 2026 outcomes, and let the source-credibility table decide. That is
why the channel enters as a *contender*, not a prior baked into the engine.

## 2. De-vig pipeline design (props → league points)

Primitives live in `scripts/collect_odds.py` today (`american_to_prob`,
`devig_pair`, unit-tested); the pipeline below lands in `harness/` as
`harness/odds.py` when the first keyed capture exists.

```
raw snapshot (data/raw/evidence/<date>/odds/*.json, or manual StartWho CSV)
  └─ 1. normalize   → observations: (observed_at, operator, player, market,
                       line, over_price, under_price, source_url)
  └─ 2. de-vig      → devig_pair(over, under) → p_over_fair per book;
                       multi-book: median of fair probabilities, never of
                       raw prices; record overround per book (data-quality)
  └─ 3. implied line → expected stat, not the threshold: line is where
                       P(X > line) = p_over_fair. Model X per component
                       (yards ~ normal w/ positional sigma; TDs/INTs ~
                       Poisson-ish with push handling at integer lines);
                       shift the distribution until the de-vigged tail
                       probability matches. Preserve (mean, sigma) — never
                       report the bare line as the mean (vegas-arbitrage §2).
  └─ 4. fill policy  → components without a priced market are imputed
                       (explicitly flagged, confidence lowered) or left
                       missing → rank-only mode (vegas-arbitrage §4).
                       Correlate components; never sum marginal medians
                       into a "median season" (§3).
  └─ 5. score        → harness.scoring.ScoringConfig.score(stats, position)
                       over the nflverse stat vocabulary (passing_yards,
                       rushing_tds, receptions, ...) — the same engine that
                       scores projections and historical weeks, so league
                       rules (half-PPR, TE premium, per-position overrides)
                       apply for free.
  └─ 6. emit         → per-player: points_mean, points_sigma, components
                       (each tagged vegas|imputed), coverage tier from the
                       vegas-arbitrage confidence-policy table.
```

### Plug-in (a): harness tool

`odds_implied_projection(player, date?) -> {points, sigma, components,
coverage_tier, observed_at, books}` — a read-only tool over the archived
snapshots, same shape as the projection tools, so the chatbot can cite "the
market's implied line" with a freshness stamp and *verbal* confidence mapped
from the coverage tier. The LLM never converts odds itself; the tool does
(hard rule: LLM never invents numbers).

### Plug-in (b): benchmark contender

"Vegas-implied rankings" become one more contender through the existing
machinery, zero new eval code:

- **FP-accuracy / rank quality:** order players per position by implied
  points → `evals.fp_accuracy.score_rankings(rankings, season, position,
  scoring)` (lower Accuracy Gap is better), next to ADP / momentum / avg3 /
  site contenders. Players below the props-coverage cutoff (~154 today) fall
  back to consensus tail — score both "pure" (covered players only, via the
  rank-disagreement lens) and "blended" variants so shallow coverage isn't
  mistaken for skill or penalized as ignorance.
- **CalibBench:** the de-vig step emits (mean, sigma) → P(player exceeds X
  points) questions score directly; the market's *stated* uncertainty is
  exactly what calibration evals exist to test.
- Season gates: 2026 prospectively; 2023–2024 retrospectively iff props
  history is purchased; **2025 stays holdout, evaluated once at gate time.**

### Plug-in (c): source-credibility row

The `evals/source_alpha.py` / credibility table gets a `vegas_implied` row
per season×position: accuracy gap vs consensus, calibration error, and
coverage tier distribution. Weighting of the odds channel inside any blended
projection is *earned from this row*, never assumed — that is the whole
"measured source, not doctrine" contract.

## 3. Exact asks (money / keys / actions)

| Ask | Cost | Unblocks |
|---|---|---|
| Sign up at the-odds-api.com, put key in `.env` as `ODDS_API_KEY` | free, 2 min | Daily game odds + futures capture (code already live, skips today) |
| The Odds API $30/mo from ~Sept | $30/mo in-season | Weekly player-prop capture, all books |
| One month of The Odds API $59 tier | $59 one-time | Historical 2023–2024 prop snapshots for the retrospective sanity check |
| Email StartWho re: data access | free | Legitimate automated season-prop aggregate (or a "no", which also settles it) |
| SportsDataIO trial (already planned for news Vault) | free trial | Probe their 2020+ props/futures archive depth in the same trial |

Everything above the first row is optional; the channel is already capturing
daily without any of it.
