# Competitive landscape — AI fantasy football (researched 2026-08-08, draft season)

Market context: ESPN is now the [official NFL fantasy game](https://espnpressroom.com/press-release/espn-fantasy-football-celebrates-being-named-official-fantasy-game-of-the-nfl-with-new-features-games-campaigns-and-more/) (24M+ players; NFL's own app [sunsetting into ESPN](https://cordcuttersnews.com/the-nfl-is-phasing-out-its-fantasy-app-and-handing-fantasy-football-to-espn-heres-what-players-need-to-know/)). "AI" is table-stakes marketing ([FantasyPros' own AI-tools roundup](https://www.fantasypros.com/2026/07/best-fantasy-football-ai-tools/)). Pattern: **the products with the most real modeling market "AI" the least; the loudest AI branding is usually an LLM wrapper over consensus rankings.**

## AI-native assistants

| Product | Essence | Pricing | Traction / notes |
|---|---|---|---|
| **WalterPicks** | League-synced rec engine (not chat-first) | PRO $64.99/yr, ELITE $169.99/yr | 4.8★/18k ratings, ~650k users claimed; real ML + human analysts |
| **RotoBot AI** | Chat-first agent, heavy betting tilt | $24.99/mo, ~$80–130/yr | 4.7★/1.3k; LLM over live feeds; draft helper "coming soon" |
| **STACKED** | League "front office" + **MCP for Claude/ChatGPT** (20 read-only tools) | unpublished | Precursor-backed; betting-market-derived projections |
| **twif** | Free AI front office; site reshapes around synced roster; draft assistant | Free | Closest free analog to our concept |
| **GameDai** | Chat + personalized live audio | $19 one-time season | Launched 2026-08-07, Sleeper-only |
| **Scoutcast.ai** | Roster-aware daily audio briefs | $5.99/mo / $49.99 season | Audio-first niche |
| **DraftEdge** | "AI Draft Coach," really VORP engine | Kit $34.95; Pro $359/yr | Quant framed as AI |
| **League Loom** | Free MCP connector: ESPN/Sleeper/Fantrax → Claude/ChatGPT | Free | Infrastructure, no model |
| **THE SOLVER / BestBall Edge** | Draft overlay extensions inside Underdog/DK rooms | from $34.99/mo | Sharp best-ball niche; ETR projections |

**Shadow competitor: raw ChatGPT/Gemini/Grok.** Practitioner consensus ([Footballguys](https://forums.footballguys.com/threads/do-any-of-you-use-ai-for-your-leagues.816607/)): stale rosters, clickbait ingestion, ADP groupthink, fails at "who do I pick right now" — exactly the wedge for a grounded engine.

## Incumbents

- **ESPN**: IBM watsonx insights (waiver/trade grades, boom/bust probs, [Auto Control](https://support.espn.com/hc/en-us/articles/8181547483924-How-to-activate-Auto-Control-AI-Team-Management-LM-Leagues-only) abandoned-team management; [14M players / "36B insights"](https://newsroom.ibm.com/2025-09-24-new-ibm-watsonx-ai-powered-insights-help-elevate-espn-fantasy-football-for-2025-fantasy-football-season)). **The Aug 2026 feature release contains zero AI announcements**; no conversational assistant. 
- **Yahoo**: most aggressive monetizer — Fantasy Plus (~$35/yr): [Assistant GM](https://sports.yahoo.com/fantasy/article/introducing-assistant-gm-a-smart-new-feature-exclusive-to-yahoo-fantasy-plus-subscribers-125543697.html) (rules-based lineup automation + alerts), AI draft-kit summaries, live Draft Scout (VOLS). No chatbot.
- **Sleeper**: no assistant; AI mock-draft opponents only; open API makes it every third-party's sync target (~15% share, growing).
- **NFL Pro**: [AWS Bedrock fantasy assistant](https://www.nfl.com/news/nfl-pro-launches-fantasy-ai-assistant-powered-by-aws) (multi-agent, NGS-grounded) but premium-gated and its app surface is sunsetting.
- **CBS/SportsLine**: simulation content, no tooling. **Underdog/DK/FanDuel**: no consumer fantasy assistant found.

## Premium analytics tools

FantasyPros Coach AI = LLM over ECR, gated to HOF tier (~$108/yr). Draft Sharks = most genuine ML ([injury model: 3,500 player-seasons, 1,000+ variables, 80% CI games-missed](https://www.draftsharks.com/injury-predictor/about)), $72–192/yr. 4for4 ($39–99/season) and ETR ($54.99 kit) = accuracy-first, zero AI claims. Fantasy Points "[Ball Knowledge](https://www.fantasypoints.ai/)" = RAG over own content, metered 5–10 free questions/week, full access in $299+ tiers. Stokastic = real sim engine ($9.99–99.99/mo). PFF+ $79.99 early-bird. RotoWire ~$107/yr.

## What to copy (they did the UX research)

1. **Live draft-room sync as the hero feature** (Draft Sharks syncs 10+ hosts; purchase criterion in every roundup). v1 is manual entry; sync is the #1 post-MVP feature.
2. **In-draft overlay extension** (THE SOLVER paradigm) — nobody does it well for ESPN/Yahoo/Sleeper redraft rooms. Open lane.
3. **One league connect → everything reshapes around the roster** + proactive delivery (Tuesday recap / Friday start-sit emails, push alerts).
4. **Tiers + VORP framing + rehearsal loop** (mock trainer, "Rate My Draft" grading) — drives August engagement.
5. **Metered grounded chat as the premium upsell** (5–10 free questions/week → paid) — the emerging monetization standard. Bonus: **MCP/BYO-AI endpoint** for power users (STACKED/League Loom prove demand; serving OUR engine through MCP would be unique).

## Complaint list = feature spec

1. Draft-day sync failures at peak load (FantasyPros has a [dedicated support page](https://support.fantasypros.com/hc/en-us/articles/28876394605339--Waiting-for-draft-sync-error-message-how-to-fix-during-a-live-draft-ESPN-Yahoo-Sleeper-etc) for it) → reliability during the 3-hour window is the whole game.
2. Billing hostility (trial traps, surprise hikes — WalterPicks reviews) → transparent pricing, easy cancel.
3. **Stale/hallucinated AI data** (wrong teams, backups as starters) → grounding + citations + freshness stamps on every answer.
4. **Format-blind consensus advice** → league-math-native answers (our engine is this by construction).
5. Tool sprawl / cluttered UX / per-league caps → one surface.

## Gaps nobody owns (thesis validation)

1. **Engine-first chatbot**: nobody computes true VOR/win-probability from the user's exact league math and shows the work. All "AI" answers re-sort consensus. ← our core plan.
2. **Verifiable grounding + published track record of own advice** (nobody audits their chatbot; 4for4/FP audit rankings only). ← our ledger/benchmark.
3. **Draft-room game theory**: nobody models the other 11 managers (tendencies, run risk, survival probabilities). ← DraftGym's fitted opponents, surfaced in product.
4. **True season-long agent** (proactive waiver/FAAB planning, trade negotiation drafts, act-with-approval). ← the harness, in-season.
5. **BYO-AI rails backed by a proprietary engine** (MCP endpoint serving our quant outputs). ← cheap to add post-MVP.

## Pricing read

Free league hosting is the baseline; paid = edge. One-time draft kits $19–60 (impulse zone), season subs **$35–130 mass band** (AI chat consistently gated to top tiers), sharp tier $170–800 (betting-adjacent). Implication for us: free metered chat → **season pass ~$39–59** → pro tier later; avoid weekly-billing trap patterns (complaint #2).

Caveats: Reddit unreachable (complaints via app-review aggregates + Footballguys); unverified: STACKED pricing, ESPN watsonx 2026 renewal, WalterPicks 650k, share figures (Scoutcast self-published).
