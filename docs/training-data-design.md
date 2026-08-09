# Training data design — the as-of packet and the evidence channel

Drafted 2026-08-08 from Jacob's question: "what dataset do we give the model — can we capture what *led up to* breakouts (beat reporters, credible wording) without overfitting?"

## 1. The organizing principle: point-in-time packets

Every training example is a **packet**: everything knowable about a player as of a dated snapshot, and nothing after it. The production harness in 2026 sees exactly this shape (current stats + current ADP + current reporting), so training on historical packets teaches transferable behavior — how to weigh evidence — rather than trivia. Packet contents:

**Structured channel** (all collected already, or collecting now):
- Usage/production history: weekly stats 2011–2025, snap counts, opportunity shares, YoY change vs `usage_change_benchmarks`
- Player priors: age, experience, draft capital (draft_picks), combine
- Situation: team changes, coaching/coordinator transitions (+ transition priors), **depth chart position and its movement over the offseason** (nflverse depth_charts 2011–2026 — the cheapest structured "camp signal")
- Market: ADP + per-player draft variance (FFC), ADP movement where reconstructable
- Availability: games-played history, injury designations (nflverse injuries)

**Text/evidence channel** (the gap this doc addresses):
- Dated news blurbs, beat reports, camp notes, coach quotes — each item carrying: date, source, author, claim, and an **evidence type** from the research repo's taxonomy (official fact > direct data > named observation > model estimate > consensus > rumor)
- Access paths under feasibility review (see `docs/evidence-archive-feasibility.md` when the scan lands): licensed news APIs (RotoWire et al.), Rotoworld-style archives, Reddit historical dumps, Wayback CDX reconstruction of depth charts/ADP pages, 2026-forward reporter-feed snapshots (registry already exists in the research repo)

**Hard rule:** historical case studies are built from *archived documents with timestamps*, never from a modern LLM's recollection of 2023 — a 2026 model "remembering" what reporters said about Puka is contamination wearing a costume.

## 2. The overfitting answer: three disciplines

The base-rate math forces the design. ~12–20 breakouts/season × ~14 seasons ≈ **200–300 positive examples**. You cannot fit an end-to-end text→breakout classifier on that without memorizing noise ("players described as 'twitchy' in August break out"). So text is never a raw correlational feature. Instead:

1. **Text trains evidence-weighing, not outcome-fitting.** The trainable skill is: classify a claim's evidence type, assess source credibility, decide whether the signal is *already priced into ADP*, and update a projection accordingly. That skill has effectively unlimited training data (every player-week is an exercise) and generalizes. The taxonomy is the regularizer.
2. **Cases build an archetype library; structured data validates it.** The ~250 breakout dossiers ("what led up to it") are qualitative curriculum — they teach the recurring causal shapes: vacated opportunity + role compatibility, scheme change + fit, year-2/3 efficiency in small samples, camp first-team usage before ADP reacts, injury ahead of a starter. Each archetype extracted from cases must then be validated as a **cohort hit-rate on the full structured history** (e.g., "WRs with >18% target share in year 1 + new play-caller: X% breakout rate vs Y% base rate") before the model is allowed to cite it. Text generates hypotheses; tables confirm them.
3. **The ADP filter is the final gate.** A signal that's real but priced in is worth nothing (research repo rule: "discount signals already fully reflected in ADP"). Every validated archetype gets tested *conditional on market price* — alpha is what remains after the market's read.

Additional guards: 2025 stays the untouched holdout; anonymized variants for structure-only evals; preregistered definitions (BreakoutBench v0.2); report base rates with every claim.

## 3. Undrafted players — why they stay in

The v0.2 undrafted-synthesis rows (Puka 2023, Kyren 2023) stay because (a) they're the highest-alpha events and the exact "who's this year's Puka" content users want, and (b) excluding them biases the benchmark toward mild value picks. For *training*, they're not a special case — one continuum of market price vs outcome; the packet just has an ADP of "beyond pool depth."

## 4. Collection priorities (feasibility-weighted)

| Priority | Source | Status |
|---|---|---|
| P0 | nflverse depth charts 2011–2026 | **collecting now** |
| P0 | Everything already landed (stats, snaps, injuries, ADP, combine, draft capital) | done |
| P1 | Dated player-news archive 2015–2025 (licensed API vs archive route) | feasibility agent running |
| P1 | 2026-forward evidence snapshotter (reporter-feed registry from research repo) | build this sprint |
| P2 | Wayback reconstruction: offseason depth charts / ADP movement | pending feasibility |
| P2 | Reddit historical dumps (crowd buzz timing; ToS review for commercial use) | pending feasibility |
| P3 | Historical Vegas win totals / player props; podcast transcripts | later |

ToS discipline carries over from the research repo: licensed or public APIs for anything load-bearing; no scraping prohibited endpoints; every stored item keeps source URL + capture date.
