# Host default ranks: scoping + first capture (edge family #1, host-rank anchoring)

**Date:** 2026-08-11 · **Collector:** `scripts/collect_host_ranks.py` · **Data:** `data/raw/host_ranks/<YYYY-MM-DD>/` · **Tests:** `tests/test_collect_host_ranks.py` (7 passing)

**Thesis.** Drafters in ESPN/Yahoo rooms pick from the host's visible default queue,
so `host_rank − sharp ADP` deltas predict in-room falls. This is an *execution*
edge (who is still on the board when), not a priced signal — the market (FFC ADP)
already knows the player's value; the host queue distorts *when the room sees him*.

## 1. What's legitimately reachable (live)

| Source | Endpoint | Auth | What it exposes | Verdict |
|---|---|---|---|---|
| ESPN | `lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leaguedefaults/3?view=kona_player_info` + `X-Fantasy-Filter` | none | The exact payload ESPN's own draft room renders: per-player `draftRanksByRankType` (STANDARD / PPR / ELIMINATION / SUPERFLEX rank + auction value = the default queue), plus ESPN ownership ADP/%owned and per-source expert `rankings` | **CAPTURE** — 1 request/day, 500 players |
| Yahoo | `pub-api-ro.fantasysports.yahoo.com/fantasy/v2/game/nfl/players;start=N;count=100;sort=OR/draft_analysis?format=json` | none | Players in Yahoo **O-Rank order** (the draft-room default queue — the sort position *is* the rank; Yahoo never emits the number) + `draft_analysis` (Yahoo ADP: average_pick/round/cost, percent_drafted, preseason variants) | **CAPTURE** — 3 requests/day, top 300 |
| Sleeper | probed `api.sleeper.com/players/nfl/rankings`, `/adp/...`, `/trending/adp` | — | All return `null`/404/`[]`. No public default-ranks endpoint exists as of 2026-08. The players-dump `search_rank` is search relevance, not the draft queue. | **SKIP** — revisit if Sleeper ships a public ranks API |

### ToS posture (conservative, same bar as `collect_odds.py`)
- Both captured endpoints are **public, unauthenticated JSON** backing pages/apps the
  hosts serve to logged-out users (ESPN's own draft room; Yahoo's public
  draft-analysis pages). No login, no key, no paywall, no `robots`-gated HTML scraping.
- Volume: ~4 requests/host/day, 0.5s polite gap, identifying UA
  (`fantasy-alpha-archiver/0.1`). ESPN posture matches the already-accepted ESPN
  core-API futures capture.
- Deliberately NOT captured: Yahoo HTML pages (scraping-unfriendly ToS; the pub-api
  JSON makes it unnecessary), authenticated league endpoints, anything FantasyPros.
- If Yahoo ever closes `pub-api-ro` to anonymous reads, the sanctioned fallback is the
  official Yahoo Fantasy Sports API (free OAuth app registration — needs Jacob to
  create the app key; account creation is out of collector scope).

## 2. First capture (2026-08-11)

```
ok  espn   items=500  bytes=16.1MB   (one kona_player_info call, PPR-rank sorted)
ok  yahoo  items=300  bytes=0.5MB    (3 pages x 100, sort=OR + draft_analysis)
```

Files: `espn_kona_player_info.json` (raw, immutable), `yahoo_players_or_p{0,1,2}.json`
(raw), `host_ranks_parsed.json` (compact join-ready rows per host), `manifest.json`.
Wired into `archive_evidence.py` as the `host_ranks` source (idempotent per day,
append-only, one host failing never kills the run). ESPN raw is ~16MB/day because
`kona_player_info` embeds projections/outlooks — kept whole for immutability; revisit
only if disk matters.

### Delta sanity check (ESPN top-200 vs same-day FFC PPR ADP, 183 name-matched)
- **Host-lower-than-market (fall candidates):** Jordan Love (ESPN 284 vs FFC 150),
  C.J. Stroud (290 vs 162), and every premium kicker (Myers/Fairbairn/Aubrey ~+120).
- **Host-higher-than-market (reach candidates):** Trey McBride (17 vs 35.5),
  Brock Bowers (24 vs 41), Jeremiyah Love (13 vs 26), Travis Hunter (127 vs 163).
- Caveat for the feature: part of the delta is **position-systematic** (ESPN buries
  K/QB2 relative to FFC's mixed-league ADP). The usable signal is the
  **position-normalized residual**, so the feature should enter models as
  `delta_z = z(host_rank − ffc_adp | position)` alongside the raw delta.

## 3. Historical reconstruction (Wayback): NOT worth it

Executable answer: **skip reconstruction; validate on 2026 dailies.**

- CDX density is fine on paper (Yahoo `f1/draftanalysis`: 5–10 capture-months/yr
  2018–2026, ~2–3 in Jul–Sep; ESPN `players/projections`: 8–12/yr 2019–2026) —
  but the pages are **client-rendered shells**. Spot check of the 2024-08-07 Yahoo
  snapshot: 632KB of HTML, **0 player references**. The JSON APIs behind them are
  not archived (ESPN's needs a request header; Yahoo's pub-api isn't crawled).
- Even if extractable, Yahoo's page shows ADP, not the O-Rank queue, and 2–3
  draft-season snapshots/yr is too coarse to time falls.
- The hypothesis can instead be validated **this season**: daily host-rank deltas
  (this collector) × pick-level outcomes, plus FFC's historical ADP splits already
  on disk for the market side.

## 4. How the delta feature enters the system

**DraftBench — host-anchored opponent family** (the review guide's run-contagion
cousin). Current DraftGym opponents draw from ADP with noise. Add a family whose
board is the *host queue*, not the market: opponent `i` values player `p` by
`rank_host(p) + ε`, with a mixing weight `w_anchor` for how much the room leans on
the visible queue vs own priors. Fit `w_anchor` per host from real draft-room pick
distributions when we have them; until then sweep it. Effects the sim then
reproduces mechanically: kickers/QB2s ESPN buries last longer than ADP says
(patience is free); McBride/Bowers-class players ESPN promotes vanish early
(reach or lose them). An agent evaluated against host-anchored rooms gets credit
for exploiting exactly this — the execution edge, separated from valuation skill.

**Draft-room product** — the freshness-stamped call-out: *“He'll fall — ESPN has
him 20 lower than the market”* (and the mirror warning: *“the host has him 15
higher — if you want him, it's now”*). Numbers come from tools per the hard rule:
`host_rank` (this collector) − `ffc_adp` (daily archive), position-normalized,
stamped with both capture dates. Verbal confidence, no percentage theater.

## Next steps
1. Accumulate dailies (already wired into the archiver; zero marginal effort).
2. Name-matching: reuse `norm_name` to build a persistent espn_id/yahoo_id ↔ FFC
   crosswalk in `data/processed/` once (183/200 naive match rate → should be ~100%).
3. Implement the host-anchored opponent family in DraftGym behind a config flag;
   sweep `w_anchor`.
4. Revisit Sleeper quarterly for a public ranks endpoint.
