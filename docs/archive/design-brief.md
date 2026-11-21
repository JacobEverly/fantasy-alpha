# Design Brief — Fantasy Alpha

## Target User
Fantasy football players from casual to sharp, weighted toward the **sharp-curious**: people who already distrust ESPN-grade advice, live in Sleeper and Reddit, listen to analytics podcasts, and want an edge they can verify. They are allergic to hype and respect tools that respect their intelligence — the same skepticism profile as the Boundless base audience, pointed at a different domain. Secondary: casuals arriving via viral sleeper-list content who need the 60-second on-ramp.

## Business Goal
**Set up your league** → ask the first question. One conversion: league configured (7 fields), first chat answer delivered with receipts. Everything on the landing page funnels there; the ledger and benchmark sections exist to remove the trust objection on the way.

## Vibe
Credible · dense · calm · verifiable · quick. "A trading terminal that explains itself" — the confidence of a Bloomberg pane with the polish of a modern dev tool. Numbers are the decoration; timestamps are the jewelry.

## References
- **Linear** (library: linear-homepage) — opacity-based hierarchy on dark, restrained monospace accent for data, severity without coldness.
- **Resend** (library: resend-homepage) — dark minimalism, code/data-first sections, no marketing sugar.
- **Internal Dashboard pattern** (library: dashboard-internal) — the app surfaces: dense tables, status colors, metric cards, skeletons.
- **Animated Comparison Bars pattern** — the benchmark section (us vs. frontier models) is literally this pattern.
- **Ambient Canvas Background pattern** — hero texture of floating dated predictions/ADP deltas; data as atmosphere.

## Anti-References
- **ESPN/sports-media loudness** — no red urgency, no gradient explosions, no lens-flare player photography, no BREAKING chyrons.
- **DFS-tool clutter** — no 40-column tables above the fold, no fifteen filter dropdowns, no "LOCK 🔒" badges.
- **Generic AI-chatbot aesthetics** — no purple/indigo gradients, no sparkle emoji, no "✨ Ask AI" pills, no glassmorphism.
- **Fake precision** — no "94.7% confidence" theater; confidence is verbal and calibrated or it isn't shown.

## Constraints
- Boundless two-layer system; base theme immutables hold (no card borders/shadows, near-black primary buttons, no neon-on-dark, asymmetric splits, ≤10 sections).
- **Dark-first site** using the base theme's dark scale (#0D0D0D ground) — this is the site's base scale, not dark-chrome-on-light.
- Self-hosted fonts; no external asset dependencies (artifact-phase pages must be fully self-contained).
- Every prediction/stat surface carries a timestamp and a data-freshness stamp — this is a product rule that design must make beautiful rather than bury.
- Monetization UI: metered free questions (counter visible, honest), season pass — no trap-pricing dark patterns (competitive scan complaint #2).
