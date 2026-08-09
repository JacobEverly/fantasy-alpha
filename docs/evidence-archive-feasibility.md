# Historical evidence archive — feasibility findings (2026-08-08)

Empirically verified access paths for dated NFL player-news text 2015–2025 (the "what led up to the breakout" channel from `docs/training-data-design.md`). Full agent report in session transcript; this is the operating summary.

## The headline finds

1. **Sleeper's undocumented GraphQL endpoint serves historical per-player news, free, unauthenticated** — verified live: `POST https://sleeper.com/graphql` with `get_player_news(sport:"nfl", player_id, limit)` returned **933 dated items for Josh Allen back to 2018-05-01**, each with ms timestamp, full blurb, and source attribution (rotoballer 442 / rotowire 334 / fantasy_pros 157). Coverage ≈ 2018→now for all players (ids from the public players dump). **ToS posture: internal research scaffolding only** — it's third-party licensed text obtained sideways; never redistribute; plan to legitimize via a RotoWire/RotoBaller license before commercial launch.
2. **Wayback reconstruction of the Rotoworld lineage is feasible for all 11 target seasons** (measured CDX capture density): headline pages captured multiple times/day most years; 2015–2019 per-player pages have 100k+ capture-days; NBC-era pages dense 2021–2025. Thin spots (2019 site transition, 2021 rebrand) are covered by Sleeper's 2018+ data. Also live and dense: **ProFootballTalk monthly archives** (2010s→now, quotes beat tweets in full) and **ESPN monthly news archives**.
3. **Twitter/X is effectively closed** (academic access dead; enterprise ~$5k/mo) — but the beat-tweet signal is already embedded downstream: blurbs carry "Source: Adam Schefter on Twitter" with timestamps. Skip native Twitter.
4. **No existing dataset does this** (HuggingFace/Kaggle: nothing) — assembling it is a moat, not a chore.

## Licensed paths (for launch)

| Vendor | Verdict |
|---|---|
| **RotoWire commercial API** | The canonical feed (powers ESPN/Yahoo/Sleeper). Live delta-pull news/injuries/depth/lineups; archive licensing is a negotiation (they hold blurbs to the late '90s). Ballpark low-4-figures/yr. **Ask for the historical archive in the same negotiation.** |
| **SportsDataIO Vault** | Only turnkey licensed historical news archive (S3 bulk + `NewsByDate`). Free trial exists — test archive depth before any sales call. ~$500–1k+/mo tier. |
| **FantasyNerds** | $199.95/yr, rolling 7-day news window only — budget live feed, poll daily and archive ourselves. |
| **Sportradar** | Expensive, least fantasy-shaped. Skip. |
| **Reddit dumps** | Exist through 2025-12 (per-subreddit torrents incl. r/fantasyfootball) but Reddit is actively litigating training use. **Eval/labeling only, not trained-in text.** |

## v0 stack (adopted)

**Historical (2015–2025):** (1) structured spine first — nflverse depth charts/injuries/rosters + FFC/MFL ADP (done/free/zero-risk); (2) Sleeper GraphQL backfill 2018–2025 (internal-only flag); (3) Wayback Rotoworld+PFT+ESPN reconstruction for 2015–2018 and gap-fill. Podcast back-catalogs via Whisper = later option (~2,500 hrs, low $100s GPU).

**Live (2026→):** own append-only archiver starting TODAY — nightly: PFT RSS, FFC ADP, Sleeper news, nflverse releases, NBC Rotoworld page. Everything we capture live is timestamped by us and reconstruction-free. RotoWire license at product launch replaces the gray-zone pieces.

**Rule carried over:** blurb text is for internal training/eval; never republished in-product without a license. Every stored item keeps source, URL, capture timestamp.

## Archiver (live capture, running since 2026-08-08)

`scripts/archive_evidence.py` implements the live leg above. One command:

```
.venv/bin/python scripts/archive_evidence.py
```

Snapshots into `data/raw/evidence/<YYYY-MM-DD>/`, append-only and idempotent per day (re-runs only fill gaps — e.g. retrying a player whose fetch 520'd): PFT RSS (raw XML + parsed JSON), FFC ADP (standard/ppr/half-ppr/2qb, 12-team, 2026), Sleeper trending add/drop, Sleeper GraphQL player news for the FFC PPR top-200 (`sleeper_news_internal/` — **internal research only**, rule above applies), the NBC Rotoworld player-news HTML, and a `manifest.json` with per-source status/items/bytes/timestamps. Sleeper ids come from the research repo's `sleeper_players_*.csv`, with a one-time cached players dump (`data/raw/evidence/sleeper_players_cache.json`) filling gaps (kickers, new ids); team DEFs use the team abbrev. First capture (2026-08-08): 30 PFT items, 891 ADP rows, 50 trending entries, 5,483 news items across 200 players, ~7.4MB.

**Run it nightly** (~2.5 min, stdlib only; late evening ET catches the day's news cycle — launchd/cron). One source failing never kills the run; grep the day's `manifest.json` for `"failed"`/`"partial"` and just re-run. Unit tests: `tests/test_archive_evidence.py`.
