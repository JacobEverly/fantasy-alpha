# FP-Accuracy v0 — FantasyPros draft-accuracy method replication

**Implementation:** `evals/fp_accuracy.py` (v0) · **Tests:** `tests/test_fp_accuracy.py` · **Run:** `.venv/bin/python -m evals.fp_accuracy`
**Generated:** 2026-08-08 · **Scoring preset:** PPR · **Seasons:** 2015–2024 (2025 is the untouched holdout — never scored here)

This is our external anchor against named human experts: the same scoring
machinery FantasyPros uses to rank ~200+ industry experts every year, applied
to our reference contenders (and, later, our model's preseason rankings via
`score_rankings(rankings, season, position)` / `score_overall(...)`).

## Method (as replicated)

Source: [FantasyPros Draft Accuracy Methodology FAQ](https://www.fantasypros.com/about/faq/football-draft-accuracy-methodology/) (fetched and verified 2026-08-08).

1. **Input** — one preseason ranking list per position, frozen before week 1.
2. **Player pool** per position = consensus-top-N ∪ actual-finish-top-N, with
   N = QB 25, RB 50, WR 60, TE 20 (K/DST/IDP tracked by FP but excluded from
   the overall score; skipped entirely in v0).
3. **Slot curve** — every rank converts to points via the trailing 3-year
   average of the points scored by the player who *finished* at that
   positional rank. Both sides go through the curve:
   **Accuracy Gap = |slot_points(expert_rank) − slot_points(actual_finish_rank)|** (lower = better).
4. **Unranked pool players** — in pool via consensus: assigned rank =
   expert's last ranked player + 1. In pool via actual finish only: assigned
   the *worse* of (consensus rank + 1) and (expert's last + 1) — the
   "Raheem Mostert rule", so deep rankers aren't punished.
5. **Out-of-pool penalty** — a player the expert ranked inside the top-N
   range who landed in neither pool half costs
   max(0, expert's gap − average expert's gap).
6. **Weighting** (their 2021+ contest) — each gap is multiplied by 1.0→0.5
   driven by the player's *consensus* rank: 1.0 at/above maxrank, 0.5
   at/below minrank, linear between (QB 18/30, RB 72/96, WR 84/112, TE 18/24).
7. **Overall** = sum of weighted QB+RB+WR+TE position gaps.

## ASSUMPTIONS (v0) — every point where their disclosure is ambiguous or our data differs

- **A1 — ECR proxy.** FantasyPros pools/weights key off their preseason
  Expert Consensus Rankings; historical ECR snapshots are not public. We
  substitute the FFC ADP board (`data/raw/ffc_adp/`, ADP-ascending, ties by
  times-drafted then name). ADP and ECR are highly correlated but not
  identical; pool membership and weights can differ at the margins.
- **A2 — Scoring preset.** FantasyPros grades on **half-PPR**; per project
  decision we score contenders on **PPR** (`--scoring half_ppr` exists but
  FFC half-PPR ADP only starts 2018). This alone makes our absolute numbers
  incomparable to theirs.
- **A3 — Slot-curve window.** Their wording ("rolling 3-year average", "the
  past few years") does not say whether the evaluated season is inside the
  window. We use seasons **S−3..S−1 only** — the leakage-free reading —
  applied to *both* the projected and the actual side (their Cole Beasley
  example shows the actual side is also a slot average, e.g. "WR #28 …
  143.1 pts on average"). Enforced by test.
- **A4 — "Average expert" in the out-of-pool penalty.** Their penalty
  compares against the mean gap across their expert field, which we don't
  have. v0 proxies the average expert with the consensus list itself
  (penalty = max(0, gap_expert − gap_consensus)). Penalties are small for
  sane contenders and exactly 0 for the ADP contender by construction.
- **A5 — Ranks beyond the curve.** A rank deeper than the deepest observed
  finish slot (e.g. a busted pick who scored 0 points, or an off-board
  player) clamps to the deepest slot's value (a near-zero tail). Players in
  the pool with no fantasy points at all take actual rank = (worst observed
  rank + 1) before clamping. FP does not disclose their handling.
- **A6 — Identity.** Players join across sources by `evals.names.norm_name`
  + position (same convention as DraftBench); on a normalized-name collision
  within a position-season the better finish rank wins.
- **A7 — Penalty weighting.** FP describes weighting (step 4) as applying to
  "each score"; we apply the same consensus-rank multiplier to penalties for
  consistency. Not explicitly disclosed.
- **A8 — Weighting era.** The 1.0→0.5 weighting was introduced for their
  2021 contest. We apply it uniformly to all seasons 2015–2024 so scores are
  internally comparable across years (so our 2015–2020 numbers do not match
  the rules FP used in those years).
- **A9 — Unranked "last + 1".** When an expert list omits several pool
  players, each gets last+1 (not last+1, last+2, …). FP's wording ("a rank
  equal to the last player the expert ranked +1") supports this reading.

## Reference contenders, PPR, 2015–2024 (weighted Accuracy Gap — LOWER is better)

| Contender | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | Mean overall | Mean QB | Mean RB | Mean WR | Mean TE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adp_ffc | 13283 | 12210 | 12270 | 9967 | 10189 | 11952 | 11647 | 9466 | 11720 | 11735 | **11444** | 1732 | 4012 | 4507 | 1193 |
| momentum_last_season | 18880 | 17761 | 17016 | 16921 | 16135 | 17336 | 17217 | 15123 | 16670 | 16267 | **16933** | 2662 | 6158 | 6378 | 1735 |
| avg3_weighted | 20204 | 18810 | 17591 | 17669 | 17687 | 18143 | 18854 | 15876 | 18884 | 17673 | **18139** | 2772 | 6653 | 6921 | 1792 |

Contenders:

- **adp_ffc** — FFC ADP-implied positional order (identical to the consensus
  proxy, so its gap isolates pure market error; zero penalties by construction).
- **momentum_last_season** — last season's positional finish order.
- **avg3_weighted** — players ordered by (3·pts[S−1] + 2·pts[S−2] + 1·pts[S−3])/6,
  missing seasons = 0.

Scale calibration: the market (ADP) beats naive momentum by ~5,500 gap points
per season and the stale 3-yr average by ~6,700; the gap between the market
and naive baselines is large relative to year-to-year noise (ADP wins every
single season, every position). A future contender should be read on this
scale: **≤ adp_ffc ≈ market-level; the ~11,400–17,000 band is where "worse
than market, better than naive" lives.** Hindsight-perfect rankings score far
below all three (test-verified), so headroom below ADP exists.

## External anchors — what is publicly verifiable

**FantasyPros does not publish raw Accuracy Gap totals.** Their public pages
expose *ordinal* leaderboards only (expert rank per position/year). So there
is no published number our gap totals can be compared to directly. Verifiable
facts, with sources:

| Year | Draft-accuracy winner | Field size | Numeric score published? | Source (fetched 2026-08-08) |
|---|---|---|---|---|
| 2025 | Seth Miller — Crossroads Fantasy Football (#1 WR expert) | ~212 experts (per docs/benchmark-landscape.md) | No — ranks only | https://www.fantasypros.com/2026/07/2025s-most-accurate-fantasy-football-draft-rankings/ |
| 2024 | Kevin English — Draft Sharks (RB #2, WR #12) | 225 experts | No — ranks only | https://www.fantasypros.com/2025/07/2024s-most-accurate-fantasy-football-draft-rankings/ |
| 2023–2025 multi-year | Jody Smith — Draft Sharks (#1 overall; QB 40 / RB 1 / WR 13 / TE 9 position ranks) | — | No — ranks only | https://www.fantasypros.com/nfl/accuracy/multi-year-draft.php |
| Methodology (pools, slot curve, weights, penalty) | — | — | Formula parameters only (N's, max/min ranks, 1.0/0.5 multipliers) | https://www.fantasypros.com/about/faq/football-draft-accuracy-methodology/ |
| Per-year leaderboards 2015–2025 | — | — | Ordinal ranks only (tables are JS-rendered from an authenticated API) | https://www.fantasypros.com/nfl/accuracy/draft.php |

**Score-scale compatibility — do not fudge equivalence.** To compare our
numbers to a hypothetical published FP gap total we would need, at minimum:
(1) their half-PPR player point totals and finish ranks (we score PPR — A2),
(2) their preseason ECR snapshot for pool membership and weights (we proxy
with FFC ADP — A1), (3) their slot-curve window definition (A3), and (4)
their expert-field mean gaps for the penalty term (A4). None of these are
public, so **our scores anchor contenders *relative to each other* on FP's
scoring geometry; they are not FP-leaderboard numbers.** The legitimate
external comparison available today is *ordinal*: enter the live contest (our
frozen preseason ranks vs. their ~200-expert field, scored by them), or
reconstruct a specific expert's archived preseason rankings and score them
through this replication on identical inputs.

## Reproduce

```
.venv/bin/python -m evals.fp_accuracy                 # 2015–2024, PPR
.venv/bin/python -m evals.fp_accuracy --scoring half_ppr --seasons 2018-2024
.venv/bin/pytest tests/test_fp_accuracy.py -q
```
