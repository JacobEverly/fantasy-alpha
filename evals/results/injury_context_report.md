# Injury-context cross-tab — BreakoutBench v0.3 freak-injury split

Built by `evals/build_injury_context.py` (lexicon 2026-08-08.v1) from
`data/processed/labels/injury_context.csv` x `breakouts.csv` (ppr, 2015-2024).

Definitions: bust pool = drafted top-12 positional (ppr ADP). A bust is an
`acute_injury_bust` if season_ending_acute or >=5 games missed in the
ACUTE_TRAUMATIC class (lexical acute terms, or a structural body-part designation
that proved season-ending); a `vanish_bust` if the player went straight to IR with
no report row at all (cause invisible to this source); else `other_injury_bust` if
games < 8 or >=5 games missed to soft-tissue/other designations; else
`performance_bust`.

## Bust class split (pooled 2015-2024, ppr)

| class | n | share |
|---|---|---|
| performance_bust | 35 | 31.2% |
| acute_injury_bust | 24 | 21.4% |
| vanish_bust | 17 | 15.2% |
| other_injury_bust | 36 | 32.1% |
| total busts | 112 | 100% |

Busts with no joinable injury-context row (zero games, no in-season report —
typically preseason injuries invisible to this source): 2 (bucketed as other_injury).

## Breakouts that broke out despite their own season-ending acute injury

0 of 159 labeled breakouts:


## THE NUMBER: bust base rate, all-outcomes vs freak-excluded

Freak-excluded removes acute-injury busts from numerator and pool.
`excl acute` is the confirmed-acute panel (headline, lower bound);
`excl acute+vanish` also drops straight-to-IR unknowns (upper bound).

| season | pool | busts | acute | vanish | rate (all) | excl acute | excl acute+vanish |
|---|---|---|---|---|---|---|---|
| 2015 | 48 | 14 | 3 | 2 | 29.2% | 24.4% | 20.9% |
| 2016 | 48 | 13 | 3 | 1 | 27.1% | 22.2% | 20.5% |
| 2017 | 48 | 11 | 3 | 1 | 22.9% | 17.8% | 15.9% |
| 2018 | 48 | 9 | 3 | 3 | 18.8% | 13.3% | 7.1% |
| 2019 | 48 | 13 | 3 | 1 | 27.1% | 22.2% | 20.5% |
| 2020 | 48 | 12 | 4 | 3 | 25.0% | 18.2% | 12.2% |
| 2021 | 48 | 9 | 1 | 2 | 18.8% | 17.0% | 13.3% |
| 2022 | 48 | 6 | 0 | 2 | 12.5% | 12.5% | 8.7% |
| 2023 | 48 | 13 | 3 | 1 | 27.1% | 22.2% | 20.5% |
| 2024 | 48 | 12 | 1 | 1 | 25.0% | 23.4% | 21.7% |
| **pooled** | 480 | 112 | 24 | 17 | **23.3%** | **19.3%** | **16.2%** |

Pooled shift: 23.3% -> 19.3% (-4.0 pp confirmed-acute; -7.2 pp including vanish busts).

## Face validity: 15 most consequential acute-injury reclassifications

Sorted by overall ADP (earliest picks whose bust label is excused as freak
injury); includes vanish busts, marked as such.

| season | player | pos | ADP | pos rank | finish | games | class | injury sample |
|---|---|---|---|---|---|---|---|---|
| 2020 | Christian McCaffrey | RB | 1.2 | 1 | 54 | 3 | acute_injury | Shoulder |
| 2021 | Christian McCaffrey | RB | 1.2 | 1 | 38 | 7 | vanish | Hamstring |
| 2017 | David Johnson | RB | 1.3 | 1 | 106 | 1 | vanish | straight to IR, no report row |
| 2022 | Jonathan Taylor | RB | 1.3 | 1 | 33 | 11 | vanish | Ankle |
| 2024 | Christian McCaffrey | RB | 1.4 | 1 | 68 | 4 | acute_injury | Calf, Achilles |
| 2020 | Saquon Barkley | RB | 2.5 | 2 | 116 | 2 | vanish | straight to IR, no report row |
| 2015 | LeVeon Bell | RB | 3.1 | 2 | 47 | 6 | vanish | straight to IR, no report row |
| 2015 | Jamaal Charles | RB | 4.5 | 4 | 51 | 5 | vanish | straight to IR, no report row |
| 2020 | Michael Thomas | WR | 5.6 | 1 | 93 | 7 | acute_injury | Ankle |
| 2017 | Odell Beckham Jr. | WR | 6.4 | 3 | 83 | 4 | acute_injury | Finger, Ankle |
| 2023 | Nick Chubb | RB | 12.5 | 5 | 86 | 2 | vanish | straight to IR, no report row |
| 2020 | Joe Mixon | RB | 13.1 | 11 | 49 | 6 | acute_injury | Foot |
| 2016 | Keenan Allen | WR | 13.8 | 8 | 163 | 1 | vanish | straight to IR, no report row |
| 2016 | Rob Gronkowski | TE | 14.2 | 1 | 27 | 8 | acute_injury | Back |
| 2015 | Andrew Luck | QB | 16.9 | 1 | 28 | 7 | acute_injury | Abdomen, Kidney |

## Caveats

- nflverse weekly reports carry body-part text only; IR players drop off the
  report, so attribution carries the last designation forward until a played game,
  and season-ending structural body-part designations are escalated to ACUTE.
- Players who go straight to IR between games leave no report row at all — those
  busts are the `vanish` class (cause unknown, historically mostly ACL-grade).
- Preseason (July/August) injuries never appear; zero-game busts without any
  in-season row cannot be attributed and are counted as other_injury, not acute —
  the confirmed-acute shift is therefore a lower bound.
- `season_ending_acute` in injury_context.csv is computed for every rostered
  player; for fringe players a season-ending designation can also mean a roster
  cut, so interpret the flag jointly with draft capital (as this report does).
