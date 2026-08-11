# Play-caller history: coverage + breakout study

Generated 2026-08-11 by `scripts/collect_play_callers.py`. Source: Wikipedia team-season infoboxes (CC BY-SA) + research-repo seed facts. Pro-Football-Reference deliberately not used (ToS).

## Coverage

- (season, team) rows 2008-2025: 576
- play_caller + changed flag coverage 2010+: 100.0%
- coverage 2008+ (2008 lacks an in-table prior season): 100.0%
- seed validation issues: 0

Join key: (season, team) with vegas_features NORMALIZE_TEAM codes (Rams=LA, Chargers=LAC, Raiders=LV, Jaguars=JAX). `team_changed_play_caller` for packet_features = `changed_play_caller` from `data/processed/coaching/play_callers.csv`.

## Breakout rate vs changed play caller (ppr, 2010-2024)

Gate-eligible candidates: 846 (dropped for missing flag/team: 0). Lift = rate_changed / rate_unchanged; season-resampled bootstrap 95% CI, 2000 reps.

| band | n changed | rate changed | n unchanged | rate unchanged | lift | 95% CI |
|---|---|---|---|---|---|---|
| all | 385 | 0.283 | 461 | 0.221 | 1.28 | [1.04, 1.55] |
| near_gate | 229 | 0.293 | 256 | 0.238 | 1.23 | [0.91, 1.71] |
| mid | 150 | 0.267 | 197 | 0.188 | 1.42 | [0.97, 2.07] |
| deep | 6 | 0.333 | 8 | 0.500 | 0.67 | [0.33, 2.33] |

**Verdict:** Changed-play-caller teams show a breakout lift whose bootstrap CI excludes 1.0 - carries post-ADP signal.

## Limitations

- Who *actually* called plays (HC vs OC) is ambiguous from structured sources; the table records the OC (HC when no OC is listed) with `play_caller_ambiguous=1` unless a research-repo seed fact pins it. `changed_play_caller` is the load-bearing flag.
- Wikipedia staff lists reflect end-of-season staff; prose-recorded midseason OC firings are folded back to the opener, but an unrecorded midseason change could mislabel an opener (noise attenuates the lift rather than inflating it).
- Study is descriptive (no controls beyond ADP bands); the `deep` band is tiny. 2025 is the untouched eval holdout and is excluded.
