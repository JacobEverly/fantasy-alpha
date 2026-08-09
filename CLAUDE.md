# Fantasy Alpha — product monorepo

AI fantasy-football analyst: chatbot + draft advisor backed by a quant tool layer, with an open model trained via SFT→RL (verifiers + prime-rl). **Read `GAMEPLAN.md` first — it is the operating document** (locked decisions, eval suite, budget, schedule). **For current state + next steps, read the latest `docs/status-*.md`.**

## Layout

- `fantasy-football-alpha-2026/` — Jacob's original research repo (separate git repo, gitignored here). Source of the quant layer: projections, VORP, SOS, coaching priors, availability, alpha signals. Treat as read-mostly upstream; we lift and generalize into `harness/`.
- `harness/` — league-agnostic engine + tool orchestrator serving the product (model-agnostic router).
- `envs/` — verifiers environments (DraftGym, calibration env) for RL and evals.
- `evals/` — DraftBench / CalibBench / AdviceBench + pinned labels. All ship decisions are eval-gated.
- `training/` — prime-rl configs, SFT datagen. 
- `app/` — web app (chat, manual league entry, draft board, public prediction ledger).
- `scripts/` — data collectors (stdlib-only where possible).
- `data/raw/` — immutable dated source snapshots (gitignored); `data/processed/` — derived tables.

## Design system

Web design follows Jacob's website-generator-skills pipeline (capture-inspo → design-brief → site-gen → site-loop → site-refine). **`DESIGN.md` at repo root is the project design system** (extends the Boundless base theme at `~/.claude/design-library/boundless-base.md`); `design-brief.md` holds audience/vibe/anti-references. Key identity: dark-first, Space Grotesk display + Inter body + IBM Plex Mono for data values only, market-green accent (#10B981) never on chrome, gain/loss colors in data only, timestamps + freshness stamps on every stat surface, verbal confidence (no percentage theater), no sports-media urgency styling. Use the site-gen/site-loop skills to build pages from DESIGN.md; refinement skills (polish, critique, harden, etc.) apply after.

## Hard rules

- **No training runs without Jacob's explicit go**, and never on this Mac — rented GPUs only. Budget ceiling $5k; log every run in `docs/budget-ledger.md`.
- **Warn Jacob before any single spend exceeding $50** (2026-08-08 directive). Below that, proceed and log; at or above it, ask first with the estimate.
- Market data (ADP) snapshots are immutable — never overwrite historical captures.
- The 2025 season is the untouched eval holdout: evaluated once per contender at gate time, never during development.
- LLM never invents numbers — projections/values come from tools.
- FantasyPros data only within permitted preview access, never load-bearing for product rankings.
- Python 3.11+, stdlib-preferred collectors; research-repo conventions (dated artifacts, evidence typing) carry over.
