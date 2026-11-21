# Reuse map — `fantasy-football-alpha-2026` → product

Audit of the research repo (2026-08-08). This is the engineering spec for lifting the quant layer into `harness/`. Repo: `fantasy-football-alpha-2026/`, Python 3.11+, deps `pdfplumber` + `PyYAML` only. CSV-in/CSV-out, dated filenames, `latest(glob)[-1]` resolution. `config/league.yaml` is the single config (9 scripts read it). `src/fantasy_alpha/` holds 12 small pure-function modules. `scripts/validate_repo.py` (959 lines) is a hard-fail contract checker heavily hardcoded to this one league and draft.

## 1. `src/fantasy_alpha/` module inventory

### `league.py` (29 lines) — league math core — **REUSE AS-IS**
- `required_positions(roster)` :7 — roster dict minus FLEX/BENCH/IR, drops zero-count.
- `snake_picks(teams, slot, rounds)` :16 — overall picks; `offset = slot if round%2 else teams-slot+1` (:27-28). Validates inputs. Cleanest module in the repo.

### `projection.py` (78 lines) — scoring engine — **GENERALIZE**
- `StatLine` frozen dataclass :8-18 — exactly 9 fields (pass yds/tds/int, rush yds/tds, rec/rec yds/tds, fumbles_lost). **No K, DST, bonuses, 2pt, IDP.**
- `point_contributions(line, scoring)` :21 — reads fixed scoring keys (`passing_yard`, `reception`, …singular names); hard KeyError if absent.
- `apply_schedule(line, factors)` :54 — multiplies efficiency/TD channels, deliberately not volume.
- Works for std/half/full-PPR by yaml edit; cannot express per-position scoring, TE premium, yardage bonuses, K/DST.

### `valuation.py` (86 lines) — VORP/replacement — **REUSE (one caveat)**
- `assign_starters(players, teams, required, flex_per_team, flex_eligible)` :23 — fills league-wide starters (:38), then flex (:52); **replacement = next-best unselected at position after starters AND flex consumed** (:58-70); flex-eligible-but-not-required positions inherit next-flex points (:72-74).
- `value_over_replacement` :84 — **signed** VORP, deliberately not zero-clamped (contrast FantasyPros).
- Caveat: raises `ValueError` on thin pools (:40, :54, :69) — product needs graceful degradation. `ProjectedPlayer` used as dict key — identical name+position+points collide silently.

### `turns.py` (36 lines) — turn grouping — **REWRITE (buggy)**
- `group_snake_turns(picks)` :20 — takes pick 1 alone then **blindly pairs the rest** (:28); raises if remainder odd (:27).
- Verified behavior: 10-team slot 2 → correct pairs (19,22)…; **slot 5 → silently pairs picks 9 apart (16,25)**; **even round counts crash**. Every consumer trusts it. Must gate on actual pick adjacency and handle odd/even rounds and end-slots.

### `availability.py` (65 lines) — ADP→pick windows — **REUSE AS-IS**
- `draft_window(adp, picks)` :22 — which of your picks bracket a market price; `recommendation_status` :43 → `fall_only|take_in_this_window|normally_wait`; `fall_severity` :54 — thresholds hardcoded (≤0/≤3/≤10 → normal/modest/meaningful/material) at :61-64, should become config. Explicitly refuses to invent survival probabilities (docstring) — good product posture.

### `avoidance.py` (60 lines) — **REUSE AS-IS**
- `AVOID_CLASSES` (4 labels) :10-15; `reconsider_pick` :26; `market_relation` :49 duplicates the 0/3/10 thresholds — unify with `availability.py`.

### `coaching.py` (64 lines) — prior shrinkage math — **REUSE AS-IS**
`weighted_mean`, `shrunk_weighted_mean(obs, population_prior, prior_weight)` → (shrunk, effective_n), `signed_transition_delta`, `empirical_percentile`, `aggregate_direction(deltas, threshold=0.03)`. Fully generic statistics, zero league coupling. Best direct-lift candidate.

### `schedule.py` (29 lines) — **GENERALIZE (uncalibrated constants)**
- `CHANNEL_PERSISTENCE` :8-15 — hardcoded YoY persistence (0.13–0.28) for 6 channels; comment admits no backtest. `forecast_factor(prior, p) = 1 + p*(prior-1)`. Math fine; constants must be backtested or SOS stays report-only.

### `names.py` (26 lines) — **GENERALIZE**
- 4 hand-entered 2026-specific aliases; NFKD-normalize + slug. Product needs ID-first joins (gsis/sleeper), name-match as fallback only.

### `availability_history.py` (193 lines) — **REUSE AS-IS**
- Snap-count-based games-played history; QB/RB/HB/FB/WR/TE position map :10-17; `build_season_availability` counts REG games with `offense_snaps > 0`, multi-team denominator rule :105-113; `add_rolling_history` adds 3-season rolling fields. Descriptive-only, no risk score.

### `fantasypros.py` (136 lines) — HTML scrapers — **DO NOT SHIP** (see §4)

### `alpha_signals.py` (793 lines) — diagnostics join — **GENERALIZE (split)**
- `ALPHA_SIGNAL_FIELDS` :64-173 = 104-column output contract. `COACHING_METRIC_BY_POSITION` :17-21 covers RB/WR/TE only (QB gets nothing).
- `opportunity_diagnostic` :195 — opportunity ≡ carries+targets (:218). `diagnostic_risk_flags` :249 — labels, never numeric score; hardcodes `games < 17` :266. `availability_diagnostic` :317 — `possible_games = 17*seasons` :323. `team_environment_diagnostic` :339.
- **`fantasypros_league_points(projection, scoring)` :370 — re-scores a third party's component stat line under YOUR scoring instead of trusting their total. Single best reusable idea in the repo.**
- `build_alpha_signal_rows` :415-793 — 12-input mega-join; every row stamped `diagnostic_only_not_applied`, `ranking_eligible=False`. Split into ~6 diagnostic functions; parameterize 17-game constants, position maps, evidence allowlists.

## 2. Scripts inventory

Collectors: FantasyPros ECR/projections/ADP+SOS (regex HTML scraping of paywall previews — do not ship), FFToday ADP (scrapes `tr.smallbody`; **hardcodes `league_size: 12` while the league is 10**; mixes best-ball into redraft), Mike Clay ESPN PDF (pdfplumber, **hardcoded page indices 34-44** :133-136 — the SINGLE projection prior for the whole system), Sleeper players (public JSON, fine), nflverse usage/availability (fine, cached, CC-BY-4.0).

Builders: `build_our_projections.py` (core pipeline: Clay → league scoring → SOS sensitivity → starters → signed VORP → rank; **`FORMULA` string literal :34-36 duplicates yaml and can drift**; `/17*games` hardcoded :132-133), `build_baseline_board.py` (legacy, provider's displayed total), `build_availability_board.py` (ADP→picks via draft_window), `build_turn_rankings.py` (**`roster_guardrail` hardcodes one-QB/one-TE prose :51-56; assumes turns have ≤2 picks :40-48**), `build_coaching_transition_priors.py` (shrinkage over 9 metrics), `build_usage_change_benchmarks.py` (paired-season change distributions — the "is this role change ordinary or extreme?" lookup), `build_alpha_signals.py`, `build_research_queue.py` (thresholds 150/±10/10.0/75/125 hardcoded), `build_player_cards.py`, renderers, `validate_repo.py`.

## 3. Hardcoded league coupling (fix-list for generalization)

- `config/league.yaml`: teams 10, slot 2, half-PPR, QB1/RB2/WR3/TE1/FLEX2/BENCH6, flex=[RB,WR,TE], 15 rounds.
- `validate_repo.py`: exact pick list `[2,19,22,…,142]` :56-58; roster sum == rounds :52-55; TE must == 1 :59-60; SOS must be report_only :61-62; 17-game season :305-307; **named-player assertions** (Bowers avoid class :196, Judkins windows :880-892, Hubbard depth chart :893-904, dated dossier filenames :936). ~40% of the file is draft-specific — port only the ~15 structural invariants (contributions sum to total, signed VORP, ranks ordered, coverage labels present, alpha never ranking-eligible).
- Turn math: `turns.py:26-28` blind pairing; `build_turn_rankings.py` unpacks exactly-2 turns.
- Scoring: formula re-hardcoded 3× (`build_our_projections.py:34-36`, `render_draft_board.py:131-135`, yaml); `collect_mike_clay_projections.py:59` `half_ppr = ppr - 0.5*rec` only valid for half-PPR; collectors pin `scoring=HALF` URLs; `projection.py` has no K/DST/IDP/bonus fields anywhere in the repo.
- Season pins: `season: 2026`, `STATS_SEASONS=(2024,2025)`, `SEASONS=range(2021,2026)`, dated filename pins in `build_vegas_comparison.py:30`, `render_player_dossiers.py:12-13`.

## 4. ToS / fragility rules (carry into product)

- **FantasyPros**: all 4 collectors regex-scrape truncated paywall previews (~10 players/position, 5 ADP rows, 8 SOS teams). Repo's own AGENTS.md:7, README:103, `source_registry.csv`, and parity registry say: licensed API is the correct route. Use `fantasypros.py` as a field spec, not a fetcher.
- **FFToday**: scraping + false `league_size` label + format mixing → replace with first-party/licensed ADP (we now have FFC API history).
- **Mike Clay PDF**: fragile page indices; and `alpha_signals.py:572-575` already detects that ESPN feeds BOTH the Clay prior AND the FantasyPros consensus → `comparison_independence_status=overlapping_espn_component_not_independent`. **"Our projection vs consensus" is not independent today — product needs a second projection prior.**
- Infra fragility: `latest()` is lexicographic across mixed date formats; collectors use exclusive-create `open("x")` (same-second rerun crashes); no retry/backoff anywhere; `data/processed/*` committed to git.
- **Nothing is calibrated** — every row ships `calibration_status=uncalibrated`, `low/high_case_points` deliberately blank; docs forbid presenting qualitative confidence as probability. This restraint is a feature: do not manufacture confidence bands on top.

## 5. Key data contracts (for wrapping as tools)

- **`our_projections_<date>.csv`** (63 cols, 415 rows) — identity/provenance (`as_of, model_version, player_id` gsis>sleeper>slug), 17-col stat line, points block (`neutral_schedule_points`, SOS sensitivity, weeks-1-14 / 15-17 splits), uncertainty flags (all `uncalibrated`/`low`), 9 `*_points` contributions (invariant: sum == `median_points`, enforced validate_repo:412-414), `scoring_config_hash` (sha256[:12] of scoring dict), league value block (`replacement_points`, signed `vorp`, `league_value_rank`, `fantasypros_ecr`, `ecr_minus_league_rank`).
- **`alpha_signals_<date>.csv`** (104 cols) — blocks: our-vs-FP points (incl. `fantasypros_source_minus_league_points` exposing FP scoring drift), opportunity (9), pts/opportunity, team environment (6), availability (16), ECR (4), coaching prior (6), Sleeper injury (5), evidence counts (11), major-injury ledger (8), risk flags, 12 coverage-status fields, 8 source URLs. All rows `ranking_eligible=False`.
- **`availability_board`** (20 cols) — ADP → `last_planned_pick_before_adp` / `first_pick_after` / fall severity / `action_window`, ±8 rank interpretation band.
- **`turn_rankings`** (44 cols), **`avoid_board`** (36 cols; per player-per-pick, 4-value avoid enum, comparator + opportunity-cost block, separated `observed_fact/model_estimate/analyst_hypothesis` + `invalidator`), **`coaching_transition_priors`** (25 cols), **`usage_change_benchmarks`** (15 cols), **`historical_player_usage`** (16 cols, 3013 rows), **`historical_team_usage`** (25 cols, 160 rows), availability histories (28/37 cols), SOS (14 cols, 32×6).
- Universal provenance convention: every table carries `as_of, source_url, format, is_fact/evidence_class, confidence, calibration_status, invalidator, scoring_config_hash`. **Adopt wholesale.** Same for the `*_coverage_status` / `*_eligible` pattern ("missing ≠ zero", "diagnostic ≠ ranking input").

## 6. Verdicts

**Lift as-is:** `league.py`, `coaching.py`, `valuation.py` (degrade instead of raise), `availability.py`, `avoidance.py` (merge thresholds), `availability_history.py`, `fantasypros_league_points()` re-scoring idea, nflverse collectors, provenance schema, coverage-status pattern.

**Generalize:** league.yaml schema (per-position scoring, TE premium, K/DST/IDP, bonuses, IR, auction, 3RR), `projection.py` (extensible scoring), `turns.py` (rewrite), `alpha_signals.py` (split), `schedule.py` (backtest or report-only), `names.py` (ID-first), `build_our_projections.py` (derive formula from config; parameterize /17), `build_turn_rankings.py`, research-queue thresholds, all collectors (idempotent writes, retry, rate limits).

**Do not port:** FantasyPros scrapers, FFToday scraper, `validate_repo.py` (extract ~15 structural invariants only), `render_player_dossiers.py`, `render_draft_board.py` implementation (keep the turn-grouped board concept), season-specific `data/manual/*` rows (keep schemas), committed processed CSVs (use storage + manifest).
