# training/ — SFT data pipeline v0

## Tool-decision supervisor result (2026-09-06)

The matched-state tool-decision experiment is complete. The development-only
calibrated classifier passed the frozen 220-row internal held-out gate, but two
contrastive Qwen3.5-9B LoRA canaries failed their behavioral gates and the
seven-episode DraftGym integration pilot showed no reward lift from forced
lookups. No full adapter and no RL run followed. The next training data must use
measured value-of-information labels from matched counterfactual rollouts, not
additional imitation of the current tool policy. See
`docs/tool-decision-supervisor-experiment.md`.

Key reusable pieces:

```bash
.venv/bin/pytest -q tests/test_tool_decision_dataset.py
.venv/bin/python -m training.tool_decision_supervisor --train  # development only
.venv/bin/python -m evals.tool_decision_scorecard local-heldout
.venv/bin/python -m evals.tool_decision_adversarial score
```

The primary held-out set and candidate manifest are already spent/frozen; do
not retrain or retune them. The adversarial suite is post-hoc diagnostic only.

## T1.2 result (2026-09-05)

T1.2 froze an 829-prompt, conflict-free mix with 70% broad retention and 30%
targeted behavior at the optimizer-loss level. The first representative canary
and its single allowed revision both preserved response syntax but made zero
valid tool calls on unseen tool-opportunity states. The protocol therefore
stopped before full training and RL. See `docs/tinker-sft-t12-experiment-report.md`.

Teacher-trace generation through the harness + the survivor-bias-proof filter
from `docs/training-risk-register.md` risk 3. Output is a **pilot corpus for
Jacob to read** (`data/processed/sft/pilot_v0.jsonl`) — nothing here trains
anything; no training run starts without an explicit go (CLAUDE.md hard rule).

## Teacher model selection (2026-08-08)

Queried the Prime Intellect serverless catalog (`GET
https://api.pinference.ai/api/v1/models`, priced listing). GAMEPLAN §7 pins the
teacher to a **permissive-licensed open model** (sidesteps the distillation
clauses of every proprietary API — OpenAI/Anthropic/Google/xAI/Kimi are out
regardless of price).

Permissive shortlist from the catalog, strongest first:

| model | license lineage | $/Mtok in/out | note |
|---|---|---|---|
| **`qwen/qwen3.5-397b-a17b`** | Qwen / Apache-2.0 | 0.60 / 3.60 | **chosen** — strongest open-weights flagship on the catalog |
| `z-ai/glm-4.7` | GLM / MIT | 0.60 / 2.65 | strong alternate |
| `deepseek/deepseek-v3.2` | DeepSeek / MIT | 0.28 / 0.42 | cheapest capable fallback |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Apache-2.0 | 0.25 / 1.00 | prior-gen Qwen flagship |

Why the 397B: strongest permissive model available; same family as the likely
B2 base model (template/voice consistency); our workload is input-heavy
(~4:1), so the blended rate is ≈ $1.2/Mtok — inside the ~$2/Mtok target even
though the raw output rate ($3.60) is above it. The measured pilot cost was
**$0.55 for 304 traces** (see `docs/budget-ledger.md`), so the rate is a
non-issue at this scale. `z-ai/glm-5` and `deepseek/deepseek-v4-*` were passed
over pending license verification of those newer lines.

## Pipeline

```
.venv/bin/python -m training.sft_datagen ping      # sanity call + cost
.venv/bin/python -m training.sft_datagen sample    # deterministic pilot sample (no API)
.venv/bin/python -m training.sft_datagen generate  # teacher traces (resumable, budget-capped $8)
.venv/bin/python -m training.sft_datagen filter    # -> pilot_v0.jsonl + filter_report.md
```

**Sample** (deterministic, seed `sft_pilot_v0:20260808`): 152 questions =
8 seasons (2016–2023) × (8 BreakoutBench `full_slate` + 4 `bust` + 7 CalibBench
`season_threshold`), ppr, NAMED track (BreakoutBench packets are re-named from
the sequestered `_KEY.json`). **Season discipline (unit-tested):** 2024 is
SFT-val and never in the pilot; 2025 is the sealed gate holdout.

**Per question, the harness (pure code, not the model) assembles:**

1. the candidate packet;
2. historical grounding — cohort base rates computed strictly from seasons
   before the eval season (`evals/harness_breakout.grounding_for` for
   full_slate; analogous ADP-rank cohorts for bust / season_threshold in
   `sft_datagen.py`; leakage asserted in code);
3. up to 3 `EvidenceStore` lookups as of Sept 1 of the season
   (`player_news`/`injury_status`/`depth_chart` — the time gate lives in
   `harness/evidence.py`, never in the prompt).

The teacher (temp 0.7, 2 samples/question, max 700 tok) writes a trace in the
product voice (verbal confidence, cite-or-don't-claim, base-rate anchoring)
ending with one `FINAL PROBABILITY:` line. Budget cap **$8 hard**, tracked
per call from API-reported usage; generation stops before the cap and is
resumable (existing `trace_id`s are skipped).

## The filter (`training/filter.py`, pure + unit-tested)

Implements risk-register rules (a)–(d):

- **(a) process gate:** every number in the trace must appear in the
  tool-provided context (packet/grounding/evidence), up to percent rescaling
  and rounding tolerance; date components and inferable adjacent years count
  as context; prose integers ≤ 10 are exempt. Anything else — invented ages,
  "200+ points in that era", computed percentages — is dropped. This also
  catches teacher-memory leakage (risk 4) when the teacher "knows" a player.
- **(b) grounding gate:** final p must lie in
  `[anchor/3, min(0.95, anchor×5)]` of the code-computed grounding anchor.
  Kills lucky-90% traces regardless of outcome. The risk-register's
  evidence-typed exception for larger deviations is deliberately NOT
  implemented in v0 (drops are logged for review instead).
- **(c) calibration score, never hits:** keep top ~60% by Brier improvement
  vs the anchor, **within (probability-band × outcome) strata**. The first
  pilot run applied a global cut and rule (d) flagged band [0.6, 0.7) —
  kept hit rate 1.00 vs 0.50 pre-filter — because within a band the Brier
  ranking mechanically favors hits. Stratifying makes inclusion
  conditionally independent of the outcome given the stated probability,
  so (d) holds by construction while the cut still ranks anchor discipline.
- **(d) survivorship audit:** per-band kept-vs-pre-filter hit rates with a
  2·SE flag, printed in every `filter_report.md`.

## Outputs (`data/processed/sft/`)

- `raw_traces_v0.jsonl` — all 304 teacher traces with full provenance
  (grounding, anchor, usage, model, temp, sample_idx).
- `pilot_v0.jsonl` — kept traces in chat format
  (`{"messages": [system, user, assistant], "meta": {...}}`).
- `filter_report.md` — kept %, rejection histogram, band-audit table,
  3 kept + 3 rejected example traces verbatim.

## Pilot v0 results (2026-08-08)

304 traces generated (152 questions × 2 samples), $0.55. Kept **162/304
(53%)**; rejections: 106 brier_cutoff, 31 uncited_number, 5 grounding_band.
Band audit clean (no flagged band).

## T1 corpus (anonymized track, 2026-08-09)

The pilot's named-track limitation is addressed at scale by the `t1-*`
commands in `sft_datagen.py`:

```
.venv/bin/python -m training.sft_datagen t1-sample    # deterministic anon sample (no API)
.venv/bin/python -m training.sft_datagen t1-generate  # resumable, budget-capped $9
.venv/bin/python -m training.sft_datagen t1-filter    # -> t1_corpus_v1.jsonl + t1_filter_report.md
```

- **ANONYMIZED generation (risks 3-4):** questions come from the anonymized
  variants (BreakoutBench files are natively anonymized; CalibBench
  `*_anon.json`). The teacher never sees a player name, team, season year, or
  absolute date (asserted at prompt-assembly time). Evidence follows the
  DraftGym masked-mode pattern (`envs/draftgym.py::_masked_dispatch`):
  `player_news` disabled, `injury_status`/`depth_chart` dispatched server-side
  against the sequestered-KEY identity, results re-masked to the anon id with
  teams/gsis ids dropped and dates relativized (`18d_before_as_of`). Errors
  sanitized. Grounding unchanged (code-computed, strictly-prior seasons).
- **Season split (GAMEPLAN amendment 2026-08-09, unit-tested in
  `tests/test_sft_t1.py`):** train = **2008–2022 excluding 2013 and 2018**
  (sealed mid-era diagnostics); 2023–2024 forward val, never SFT data; 2025
  sealed. Presets: standard for 2008–2009 (FFC ADP reaches 2008 in standard
  only), ppr from 2010. Pre-2015 question sets were built with the existing
  breakoutbench/calibbench builders.
- **Families:** full_slate / bust (BreakoutBench) + season_threshold /
  **weekly_h2h** (CalibBench; new for SFT — grounding = A-wins base rates
  over past pairs with a similar season-to-date ppg edge, strictly-prior
  seasons, weekly labels to 1999; h2h as-of = Thursday of week W,
  pre-kickoff). Season-level grounding pools formats (precedent:
  `breakoutbench.family_base_rates`) so 2009–2011 have usable cohorts;
  2008 is weekly_h2h-only (no prior labeled season exists).
- **Filter:** unchanged rules (a)-(d) with the stratified
  (p-band × outcome) cut. Output `t1_corpus_v1.jsonl` = anonymized kept
  traces (`meta.track="anonymized"`) + the named pilot subset re-admitted
  under the amendment (`meta.track="named_pilot"`; pilot rows from 2018 and
  2023 are dropped — see `t1_filter_report.md` for the mix).

## Known limitations / open items

- **Named-track teacher memory (risk 4):** the 162-trace pilot generated from
  NAMED packets per the pilot spec; T1 generation is anonymized (above) and
  the named rows ride along only as a flagged minority subset. The
  counterfactual-perturbation leakage linter is still unbuilt; the T1
  anonymization tests in `tests/test_sft_t1.py` cover the name/date/season
  channels.
- **Evidence licensing:** Sleeper-derived news text
  (`internal_research_only=True`) appears in *user* messages. At SFT time the
  user role must be loss-masked (prime-rl `loss_mask`) so licensed text is
  never trained as completion targets; product republication stays banned.
- Rule (b)'s evidence-typed deviation exception and per-band statistical
  equivalence testing beyond the 2·SE heuristic are v1 items.

## Tinker backend and first T1 result (2026-09-05)

Tinker is now a modular alternative to the Prime/prime-rl path above. It uses
the same frozen JSONL and assistant-only loss masking; no local GPU training is
performed. Install the pinned provider dependencies with `.[tinker]` and pass
the credential only through `TINKER_API_KEY`:

```bash
.venv/bin/python -m training.tinker_sft preflight
.venv/bin/python -m training.tinker_sft smoke              # 24-row paid T1 canary
.venv/bin/python -m training.tinker_sft smoke-existing-t0  # exact legacy pilot_v0 parity run
.venv/bin/python -m training.tinker_sft train
.venv/bin/python -m evals.tinker_sft_scorecard run --arm base
.venv/bin/python -m evals.tinker_sft_scorecard select-checkpoint
.venv/bin/python -m evals.tinker_sft_scorecard run --arm adapter
.venv/bin/python -m evals.tinker_sft_scorecard score
```

Pinned stack: Tinker 0.27.1, tinker-cookbook 0.5.7,
`Qwen/Qwen3.5-9B`, rank-32 LoRA, `qwen3_5_disable_thinking`, last-assistant
loss, batch 32, two epochs, 3e-4→3e-5 linear learning-rate decay, seed
20260808. The first full run reduced development NLL 1.257→0.815 and cost
$3.1568. The frozen evaluation verdict is **revise SFT before RL**: masked
DraftGym had a promising but unstable mean gain, while predictive/calibration
metrics did not show a robust aggregate improvement. See
`docs/tinker-sft-experiment-report.md` and the model card.

## Targeted T1.1 revision (2026-09-05)

T1's diagnosis found a direct supervision mismatch: the 811 traces contained
forecast answers but no model-authored DraftGym or tool actions. The targeted
T1.1 corpus therefore contains 300 development-only examples: 120 conservative
forecast corrections, 120 direct draft choices, and 30 complete tool-call plus
post-result pairs (60 rows). It is materialized and documented under
`training/datasets/`; 2025 remains sealed.

```bash
.venv/bin/python -m training.t11_failure_analysis
.venv/bin/python -m training.t11_dataset validate
.venv/bin/python -m training.tinker_sft t11-preflight
.venv/bin/python -m evals.tinker_sft_t11_scorecard freeze
.venv/bin/python -m training.tinker_sft t11-canary
.venv/bin/python -m training.tinker_sft t11-train
.venv/bin/python -m evals.tinker_sft_t11_scorecard select-checkpoint
.venv/bin/python -m evals.tinker_sft_t11_scorecard run --arm base
.venv/bin/python -m evals.tinker_sft_t11_scorecard run --arm t1
.venv/bin/python -m evals.tinker_sft_t11_scorecard run --arm t11
.venv/bin/python -m evals.tinker_sft_t11_scorecard score
```

The frozen evaluation uses 24 matched masked drafts (2018/2023/2024 × four
slots × two new seeds), plus the same prediction, calibration, structured-output,
and canary surfaces for base, T1, and T1.1. The incremental hard cap is $10;
the preregistered estimate is $4.51.
