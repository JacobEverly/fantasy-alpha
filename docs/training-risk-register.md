# Training Risk Register — SFT + RL fine-tuning

**v1, 2026-08-08. Pre-spend gate document.** Purpose: before any training dollar (GAMEPLAN §7, T0–T3), every common failure mode of SFT + GRPO fine-tuning must either be impossible in our setup or have a named, concrete countermeasure — citing a repo artifact (file + what it enforces) or a specific stack mechanism (config knob / documented behavior, with URL). No vibes.

Stack under review: **verifiers** (envs) + **prime-rl** (SFT/RL) on rented GPUs (GAMEPLAN §1). Phases: **T0** infra smoke · **T1** SFT · **T2** RL pilot · **gate** = 2025 single-shot holdout eval.

Primary references (fetched 2026-08-08):

- prime-rl repo + docs: <https://github.com/PrimeIntellect-ai/prime-rl> — `docs/algorithms.md` (DPPO loss, KL regularizer, importance-ratio clamping, GRPO/MaxRL advantage estimation, length penalty, filters), `docs/training.md` (SFT renderer/loss-masking, `orchestrator.max_off_policy_steps`, eval sources, checkpointing), `docs/configuration.md` (LoRA, `--dry-run` config dumps).
- verifiers repo + docs: <https://github.com/PrimeIntellect-ai/verifiers> — `docs/overview.md` (**v1 tasksets/harnesses/traces is current; v0 `MultiTurnEnv` is deprecated** — see "Plan changes" at bottom).
- GRPO pathologies: DAPO (entropy collapse via clip-higher, length bias, dynamic sampling) <https://arxiv.org/abs/2503.14476>; Dr.GRPO (length/std normalization biases) <https://arxiv.org/abs/2503.20783>; The Entropy Mechanism of RL for LLMs <https://arxiv.org/abs/2505.22617>; GRPO++ engineering tricks <https://cameronrwolfe.substack.com/p/grpo-tricks>.
- vLLM/trainer numerics: vLLM V0→V1 logprob correctness <https://huggingface.co/blog/ServiceNow-AI/correctness-before-corrections>; TRL issue: vLLM logprobs ignore temperature scaling <https://github.com/huggingface/trl/issues/4159>.

## Summary table

| # | Risk | Phase | Countermeasure anchor | Residual |
|---|---|---|---|---|
| 1 | Small-data SFT overfitting | T1 | LoRA + held-out-season val loss (`val.data`) | Low |
| 2 | Catastrophic forgetting | T1/T2 | General-ability canary eval (proposed below) | **Medium — canary not built** |
| 3 | Teacher noise + survivor bias in traces | T1 | Calibration-scored trace filter (proposed rule) | **Medium — filter not implemented** |
| 4 | SFT-data outcome leakage | T1 | As-of packet rule (`docs/training-data-design.md` §1) + leakage linter | Medium — linter not built |
| 5 | Chat-template / loss-masking footguns | T0/T1 | prime-rl renderer-owned tokenization + role masking | Low |
| 6 | Format collapse | T1 | Mixed-format SFT corpus + AdviceBench rubric | Low-medium |
| 7 | DraftGym reward hacking | T2 | Exploit hypothesis list + randomization/holdout counters | Medium |
| 8 | Overfitting to 10 historical seasons | T2 | Season split + anonymization + canaries + env randomization | **Medium-high — central risk** |
| 9 | Entropy / mode collapse | T2 | `kl_tau`, `dppo_mask_*`, entropy monitoring | Low-medium |
| 10 | Sparse terminal reward, 15-pick credit assignment | T2 | `group_size`, paired autopick control, shaped-reward caution | Medium |
| 11 | Async off-policy / vLLM numerics mismatch | T2 | prime-rl token-level IS + KL + `max_off_policy_steps` | Low |
| 12 | Length / format degeneration | T2 | `[orchestrator.algo.length_penalty]` + repetition filter | Low |
| 13 | Era drift 2015→2024 | T2/gate | Per-season eval rows in DraftBench output | Low |
| 14 | Hyperparameter overfitting via iteration (meta-risk) | gate | Preregistered single-shot 2025 gate + attempt cap (proposed protocol) | **Medium — protocol needs pinning** |
| 15 | Eval noise masquerading as progress | all | Bootstrap CIs, paired controls | **Medium — CI artifact missing** |
| 16 | Checkpoint selection bias | T1/T2/gate | Selection on val seasons only; 2025 never in the loop | Low |
| 17 | Irreproducibility | all | `--dry-run` config dumps, seeds, budget ledger | Low-medium |

---

## A. SFT risks

### 1. Small-data overfitting / memorization

**What:** Our SFT corpus is small (~250 breakout dossiers as curriculum; harness-distilled traces in the low thousands at best — `docs/training-data-design.md` §2 gives the base-rate math: 12–20 breakouts/season × ~14 seasons). Full fine-tuning or too many epochs memorizes surface forms and specific player-season outcomes instead of the evidence-weighing skill.

**How it manifests here:** Val loss on held-out seasons rises while train loss falls; the model recites archetype conclusions ("twitchy in August → breakout") without the cohort-validation step; anonymized-variant scores diverge from named-variant scores.

**Countermeasure:**
- **LoRA, not full FT**, at T1 (GAMEPLAN §7: "LoRA on 14–32B"). prime-rl supports it via `[trainer.model.lora]` (rank/alpha/`target_modules`) — <https://github.com/PrimeIntellect-ai/prime-rl> `docs/configuration.md`. Low rank (start 16–32) is itself a capacity regularizer.
- **Held-out-season validation in the trainer loop:** prime-rl SFT supports a validation dataset mirroring the train config (`val.data`, `docs/training.md`). Split SFT traces by season: train on 2015–2022-derived traces, validate on 2023–2024-derived traces. Early-stop on val loss, not train loss.
- **Structural anti-overfit disciplines** from `docs/training-data-design.md` §2: text trains evidence-weighing (unlimited exercises), not outcome-fitting; archetypes must be cohort-validated on structured history before citable; ADP filter as final gate. The corpus design itself removes the memorize-the-answer gradient.

**Residual gap:** Low. Epoch count and LoRA rank still need a pinned starting config (propose: rank 16, α 32, ≤3 epochs, early-stop patience 1 on season-split val loss) written into the T1 config before launch.

**Phase:** T1.

### 2. Catastrophic forgetting / general-capability regression

**What:** LoRA SFT on a narrow domain can still degrade instruction following, general reasoning, and math — capabilities the chatbot needs for everything that isn't a draft pick.

**How it manifests here:** B3 beats B2 on DraftBench/CalibBench but the chat product gets dumber: mangles league-config arithmetic it should delegate, loses multi-turn coherence, breaks JSON tool-call formatting. AdviceBench catches some of this but is domain-internal.

**Countermeasure:**
- **Proposed general-ability canary (needs building, ~half day):** a fixed, seeded 300-item slice — 100 GSM8K, 100 MMLU (mixed), 100 IFEval instruction-following prompts — scored exact-match/constraint-check, run on B2 (base) once to pin the reference, then on every T1/T2 checkpoint. **Regression gate: any category drops >3 points absolute vs B2 → checkpoint rejected regardless of DraftBench gains.** Cheap: ~300 short completions per checkpoint on the existing Prime inference endpoint (same path as `evals/harness_breakout.py`).
- **Stack hook:** prime-rl RL runs periodic evals in-loop via `[[orchestrator.eval.source]]` + `orchestrator.eval.interval` (default 100 steps) — `docs/training.md`. Package the canary as a verifiers env so regression is visible *during* T2, not post-hoc. For T1 (SFT), run it offline per checkpoint.
- LoRA itself bounds forgetting (base weights frozen; adapter can be ablated to recover B2 exactly).

**Residual gap:** **Medium — the canary does not exist yet.** Until it's built and B2-pinned, we cannot see forgetting. Blocker for T1 GO (checklist below).

**Phase:** T1 (build before), T2 (in-loop).

### 3. Teacher-noise distillation and survivor bias in outcome-filtered traces

**What:** Two coupled failures. (a) The SFT teacher is a self-hosted Qwen-class model (GAMEPLAN §1) — its traces contain reasoning errors and hallucinated logic we'd distill wholesale. (b) The subtler one: if we filter traces by *outcome* (keep only traces whose call hit), we select for **lucky overconfidence**. In a domain with 12–20 breakouts/season, a trace that says "90% breakout" and happens to hit is worse pedagogy than one that says "30%, priced at 12%" and misses — but outcome-filtering keeps the first and discards the second. We would train an overconfident model on survivorship-biased reasoning, directly poisoning CalibBench.

**How it manifests here:** B3 posts confident probabilities with terrible Brier/ECE on `evals/calibbench.py`; harness_breakout-style anchoring to base rates disappears; the model learns "assert boldly" because every bold assertion in its training set won.

**Countermeasure — the filter selects for calibrated reasoning quality, never raw outcomes. Proposed concrete rule (to be pinned in the T1 datagen script):**
1. **Process gate (hard):** trace must cite ≥1 tool output for every quantitative claim (regex-checkable against the tool-call log — the harness emits structured calls); no numbers absent from tool outputs (CLAUDE.md hard rule: "LLM never invents numbers"). Fails → drop.
2. **Grounding gate (hard):** any probability in the trace must fall within a plausibility band of the code-computed historical grounding (the `historical_grounding` cohort rates from `evals/harness_breakout.py` stage 1): p within [base_rate/3, min(0.95, base_rate×5)] unless the trace explicitly cites evidence-typed support for a larger deviation. Kills lucky-90% traces *regardless of outcome*.
3. **Calibration score, not hit/miss:** where an outcome exists, score the trace's probability with **Brier improvement vs the cohort base rate** (same scoring as `evals/calibbench.py`). Keep the top ~60% by Brier-vs-base-rate — this retains well-reasoned misses ("30% and it didn't happen" scores fine) and drops lucky hits ("90%" hitting a 15%-base-rate event scores *worse* than the base rate did on the population).
4. **Never condition inclusion on the binary outcome alone.** Audit: post-filter, the kept-trace hit rate must be statistically indistinguishable from the pre-filter hit rate within probability bands (a kept-set hit-rate jump = survivorship leaked back in).

Teacher noise additionally bounded by rule 1–2 (teacher can't inject unsupported numbers) and by B3-vs-B2 eval gating (GAMEPLAN §4: all ship decisions eval-gated).

**Residual gap:** **Medium — rule is specified here but not implemented**; the datagen script must land with rules 1–4 as code + the rule-4 audit printed with every generation run. Blocker for T1 GO.

**Phase:** T1.

### 4. SFT-data leakage (traces referencing outcome knowledge)

**What:** Two channels. (a) Pipeline leakage: a packet accidentally includes post-cutoff data. (b) **Teacher pretraining leakage**: the 2026-era teacher *knows* who broke out in 2015–2024 and writes "traces" that are memory in reasoning's clothing — `docs/training-data-design.md` §1 calls this "contamination wearing a costume."

**How it manifests here:** SFT traces about 2023 that reason toward Puka Nacua with suspicious directness; B3 aces named retrospectives but shows no gain on the anonymized variant — i.e., we spent $300 teaching the model the past.

**Countermeasure:**
- **As-of packet discipline** (`docs/training-data-design.md` §1): every example is "everything knowable as of a dated snapshot, and nothing after it"; historical evidence only from archived timestamped documents. Pattern already enforced in code elsewhere: `evals/harness_breakout.py` computes grounding "STRICTLY LIMITED to seasons before the eval season (hard leakage rule, asserted in code + unit-tested)"; `evals/calibbench.py` packets use strictly-prior weeks/seasons.
- **Teacher-memory counter: generate SFT traces from *anonymized* packets** (names/teams stripped, same pipeline as CalibBench's anonymized variant with sequestered KEY, `evals/README.md`). The teacher can't retrieve memorized outcomes for `Q0417`. Re-name only in the final formatting pass if the product voice needs names — the *reasoning* was produced blind.
- **Memorization canary** (`docs/breakoutbench-design.md` §2): counterfactually perturbed dossiers detect identity-following vs feature-following; run it on B3, not just frontier models.
- **Leakage linter (proposed, not built):** scan generated traces for (i) tokens matching real player names when the packet was anonymized, (ii) any date/week reference past the packet's as-of date, (iii) realized stat lines not present in the packet. Drop + log.

**Residual gap:** Medium — anonymized-generation and the linter are design decisions here, not landed code. The temptation to generate from named packets (better prose) must be explicitly refused.

**Phase:** T1.

### 5. Chat-template / loss-masking / tokenizer footguns

**What:** Classic SFT bugs: loss computed on prompt tokens, template mismatch between training and serving, thinking-mode tokens handled inconsistently, position-dependent templates corrupting masks in multi-turn data.

**What prime-rl handles automatically (cited, `docs/training.md`):**
- Two dataset layouts: `prompt`/`completion` columns — "the trainer masks out the prompt and computes loss only over the completion" — or a `messages` column with **role-based loss masking**, configurable per role via `loss_mask.*` (system/user/assistant/tool).
- "SFT tokenization is renderer-only. The renderers package owns message-to-token conversion and loss attribution end-to-end," specifically so position-dependent templates don't corrupt loss masks. Renderer selected via `[renderer] name` (default `"auto"`), with per-model controls like `enable_thinking = false`.
- Same renderer system is used on the RL side for trainer/inference consistency (`docs/algorithms.md`: "both policies must tokenize identically… handles model selection via the renderer system").

**Our remaining obligations:** (a) serve with the *same* template/thinking setting as trained — we already pin `chat_template_kwargs: {"enable_thinking": false}` at inference in `evals/harness_breakout.py`; the T1 config must match; (b) for multi-turn draft traces with tool messages, set `loss_mask` so tool-output tokens are **masked out** (the model must not learn to generate tool outputs — that's how numbers get invented); (c) T0 smoke test includes decoding one packed batch and eyeballing the mask boundaries.

**Residual gap:** Low, contingent on the T0 mask-inspection step actually being on the T0 checklist (it is, below).

**Phase:** T0/T1.

### 6. Format collapse

**What:** SFT on homogeneous traces (one JSON schema, one dossier shape) collapses the model onto that format; it answers casual chat questions with draft-pick JSON, or can't emit the strict schema the harness parser needs when it matters.

**How it manifests here:** AdviceBench rubric failures (groundedness/actionability graded, GAMEPLAN §4.3); harness parse errors in the app; RL then amplifies whatever format survives (risk 12).

**Countermeasure:** T1 corpus is deliberately mixed by design — GAMEPLAN §7 T1: "DraftGym states + dossier-style analyses + calibration-formatted forecasts," three output shapes; add a slice of plain conversational Q&A in product voice. Post-SFT, AdviceBench + the general canary's IFEval slice (risk 2) test format flexibility. Schema-critical outputs are protected by the harness parser (retry loop), not by hoping the model always complies.

**Residual gap:** Low-medium — the conversational slice isn't in the GAMEPLAN T1 list; add it to the datagen spec (a few hundred examples suffice).

**Phase:** T1.

---

## B. RL / GRPO risks

### 7. Reward hacking of DraftGym

**What:** GRPO optimizes whatever the env pays. DraftGym's reward is realized season points vs an ADP-autopick control (GAMEPLAN §5) — real-outcome-grounded, which blocks *reward-model* hacking, but the simulator around it has exploitable seams. Multi-objective/simulator hacking is the canonical GRPO failure (MO-GRPO, <https://arxiv.org/html/2509.22047v3>; DAPO §reward shaping).

**Concrete exploit hypotheses for OUR env:**

1. **Opponent-model memorization.** Opponents pick `adp + Normal(0, ffc_stdev)` (evals/README.md). A policy can learn the exact σ per player and "snipe" — exploit deterministic quirks of the sampler (e.g., which players reliably fall) rather than learn value. Works in the gym, fails against humans.
2. **Sim-underpriced player farming.** Any join/scoring artifact that systematically underprices a class of players becomes free reward: `draftbench.py` documents that unmatched ADP players "stay on the board and score zero" (~97% name-match) — the *complement* risk is players whose realized points are inflated or whose ADP is stale in old FFC boards. Policy learns "always draft class X" as a data-artifact arbitrage.
3. **Hindsight-lineup exploitation.** Rosters are scored with **hindsight-optimal weekly lineups** (draftbench.py header). This overpays volatile boom/bust benches — the optimizer always starts the boom weeks. Policy learns to hoard variance that a real manager (or the fixed manager policy) couldn't harvest.
4. **Format/board-shape quirks.** v0 has no K/DST and boards thinner than 180 picks with a scarcity guard (draftbench.py header) — policy can learn end-game behaviors that exploit the guard (e.g., deliberately starving a position to trigger it) that don't transfer to real drafts.
5. **Positional-run absence.** v0 opponents have no run contagion (evals/README.md v0 simplifications); policy learns there is never a run, so never reaches — a strategy real drafts punish.

**Counters:**
- **Randomize what can be memorized:** per-episode jitter of opponent σ (scale by U(0.7, 1.4)), random opponent archetype mixes, randomized league shape (teams 8–14, roster counts, scoring preset, slot) — GAMEPLAN §5 already specifies the grid; the RL env must *sample* it per episode, not fix it.
- **Hold out env configurations:** train on a subset of (format × teams × slot) cells; evaluate on held-out cells. A hack that doesn't transfer across cells shows up immediately.
- **Fix hypothesis 3 before T2:** DraftGym reward must use the **fixed manager policy** (GAMEPLAN §5 already says "weekly lineups set by a fixed manager policy") — i.e., do *not* inherit DraftBench v0's hindsight-optimal scorer into the RL reward. DraftBench-as-eval can keep both scorers and report the gap.
- **Paired control as reward normalizer** (GAMEPLAN §5: reward "normalized against the ADP-autopick baseline for that seed") — an exploit available to the policy is often available to the control's board too, netting some artifacts out.
- **Audit top-reward episodes** weekly during T2: read 20 highest-advantage rollouts; reward hacks are visually obvious in draft logs (same weird player every episode).

**Residual gap:** Medium. Hypotheses 1/4/5 need the v1 env work (run contagion, opponent diversity, per-episode randomization) which is specified but unbuilt; hypothesis 2 needs a one-time audit of systematically-mispriced players in the join. No sim is hack-proof — the backstop is that the *gate* is out-of-distribution (2025, anonymized, held-out cells) and the ledger is reality.

**Phase:** T2 (env hardening is a T2-GO precondition).

### 8. Overfitting to 10 historical seasons — the central worry

**What:** DraftGym replays 2015–2024. The policy has thousands of episodes over the *same ~10 answer keys*. Even anonymized, a profile is a near-fingerprint (usage curve + draft capital + ADP ≈ identity); the policy can memorize "this profile hits in this board context" — perfect train reward, zero 2026 transfer.

**How it manifests here:** Train-season reward climbs; 2023–2024 val-season reward flat or falling; memorization canary (perturbed dossiers) fires; 2025 gate disappoints; live ledger embarrasses us publicly.

**Countermeasure stack (defense in depth):**
- **Season-split protocol (pinned here): train = 2015–2022, val = 2023–2024, test = 2025 sealed, live = 2026 ledger.** Enforcement anchors: `evals/draftbench.py` `DEFAULT_SEASONS = tuple(range(2015, 2025))` with the header rule "never add 2025 during development"; `evals/calibbench.py` "refuses it without `--include-holdout` (gate-time only)". **The RL env config must additionally exclude 2023–2024 from rollout seasons** — those are checkpoint-selection val only. This is one line in the DraftGym taskset config and must be asserted in env code the way `harness_breakout.py` asserts its leakage rule.
- **In-loop val on held-out seasons:** prime-rl `[[orchestrator.eval.source]]` (docs/training.md) runs eval envs on an interval — point one at DraftGym-2023/2024 so the train/val reward gap is a live wandb curve, not a post-mortem.
- **Memorization canaries:** perturbed-dossier canary (`docs/breakoutbench-design.md` §2) run on every candidate checkpoint; anonymized DraftBench variant (GAMEPLAN §4.1) as the structure-only check.
- **Env randomization** (risk 7 counters): formats × team counts × slots × opponent noise per episode multiplies distinct board states per season by orders of magnitude, diluting per-season memorization.
- **Group-relative advantage helps structurally:** GRPO baselines within a group on the *same* prompt/seed (prime-rl `docs/algorithms.md`, group-norm advantage), so "knowing season 2019's answer key" is partially netted out within-group — all group members share the season; only *decision quality differences* get credit.

**Residual gap:** **Medium-high even with all counters — this is the risk we spend the most vigilance on.** ~8 train seasons is genuinely few; some memorization is certain. The honest position (GAMEPLAN §4 marketing rule) is that historical results are contamination-suspect and **the 2026 ledger is the only external proof.** Val-season reward and canaries make memorization *visible*; they cannot make it impossible. Anonymized DraftBench variant is still unbuilt (evals/README.md v0 simplification) — required before T2 checkpoint selection.

**Phase:** T2 + gate.

### 9. Entropy / mode collapse

**What:** Policy entropy crashes early in GRPO training; the model commits to a narrow strategy (or single phrasing) and exploration dies — documented as the primary predictable failure of R1-style RL (Entropy Mechanism, <https://arxiv.org/abs/2505.22617>; DAPO's clip-higher motivation, <https://arxiv.org/abs/2503.14476>).

**How it manifests here:** All group rollouts pick identical draft lines → group advantage ≈ 0 everywhere → learning stalls at whatever the first collapsed strategy was (probably "mimic ADP", which scores 0 vs control by construction).

**prime-rl's actual controls (docs/algorithms.md, with defaults):**
- **KL regularizer** `kl_tau` (default `1e-3`): squared log-ratio penalty tethering π to the rollout policy μ; "increase to 1e-2 for strong off-policy drift control."
- **Importance-ratio masking** `dppo_mask_low` / `dppo_mask_high` (default 0.2 symmetric): per-token trust region. Note DAPO's finding that *symmetric* clipping suppresses exploratory (low-probability) tokens — if entropy collapses, the documented fix direction is raising the upper mask (clip-higher).
- **Advantage temperature** `adv_tau` (default 1.0; "reduce to 0.5–0.8 for stabilization").
- **Rollout sampling temperature** under the inference/sampling config — keep ≥1.0 for rollouts (see risk 11 for the logprob caveat).
- **Pre-batch `zero_advantage` filter** drops all-same-reward groups so they don't waste batch (docs/algorithms.md, filtering) — also our *collapse alarm*: rising zero-advantage drop-rate = groups agreeing = entropy dying.

**Run protocol:** log token entropy + zero-advantage drop rate in wandb (`[trainer.wandb]`); alarm thresholds set at T0 from the smoke run; first knob is `kl_tau` up + upper mask up, per docs.

**Residual gap:** Low-medium. Controls exist and are documented; what's missing is our own tuned thresholds — accept one T2 restart as the cost of finding them ($1–1.5k budget includes retries via the GAMEPLAN §7 reserve).

**Phase:** T2.

### 10. Sparse terminal reward + credit assignment over 15-pick episodes

**What:** Reward arrives once per 15-pick episode, and realized-season variance is enormous (DraftBench first run: control-relative std ±271 points for greedy_vorp, ±66 for bestball, evals/README.md). GRPO spreads one noisy scalar over every token of every pick — pick-level credit is weak and slow.

**How it manifests here:** Gradient signal drowned by outcome luck; thousands of episodes needed per reliably-learned behavior; or worse, the policy learns superstitions from noise.

**Countermeasures:**
- **Group size as variance reduction:** prime-rl `group_size` recommended 4–8, "increase for higher variance rewards" (docs/algorithms.md). Ours is a high-variance env: start 16, same seed/board within group so the group baseline nets out season luck shared across the group.
- **Paired control:** reward = points above the ADP-autopick control *in the same seat, same seed* (GAMEPLAN §5; mechanism already proven in `evals/draftbench.py`, whose control "is drafted in the same seat with the same seed"). This is a matched-pairs design — removes board-level and season-level variance components before GRPO ever sees the reward.
- **Cheap episodes:** env steps are table lookups, "thousands of drafts per hour per node" (GAMEPLAN §5) — we can afford the sample count sparse reward demands.
- **Shaped auxiliary rewards — treat as a proxy-hack vector, not a free lunch.** Any dense shaping (e.g., per-pick VORP captured) is a proxy the policy will optimize *instead of* the outcome (classic Goodhart; MO-GRPO documents GRPO's normalization amplifying exactly this). Policy: **T2 runs terminal-reward-only.** If credit assignment proves too slow, the only pre-approved shaping is the win-rate Monte Carlo variant (GAMEPLAN §5 "optional shaped variant") because it is still outcome-grounded, not heuristic-grounded. Heuristic shaping requires a new register entry + eval evidence it doesn't distort picks.
- prime-rl's per-token advantage streams support finer credit later (Hierarchical GRPO exists for multi-role cases, docs/algorithms.md) but that's post-T2 territory.

**Residual gap:** Medium — group_size/episode-count arithmetic is unvalidated until T2's first day; the pilot's explicit purpose is measuring signal-to-noise (GAMEPLAN §8 D6 "env/reward iteration from results").

**Phase:** T2.

### 11. Async off-policy drift / vLLM-trainer numerics mismatch

**What:** In async RL the trainer updates while inference generates with stale weights; separately, vLLM and the trainer compute different logprobs for the *same* weights (kernel/precision differences; vLLM V1 returns pre-temperature-scaling logprobs by default — <https://huggingface.co/blog/ServiceNow-AI/correctness-before-corrections>, <https://github.com/huggingface/trl/issues/4159>). Untreated, "on-policy" training is silently off-policy with biased gradients; MoE models are worst (expert-routing flips make token-level mismatch discontinuous).

**How prime-rl handles it (cited, docs/algorithms.md + docs/training.md):**
- **Token-level importance sampling** against rollout-time logprobs: ratio π/μ per token, clamped at δ — "prevents high-reward tokens sampled under old policy from generating runaway gradients"; highly off-policy tokens soft-mask to zero gradient rather than destabilizing sequences. Rollouts "carry sampling logprobs for importance ratios" by design.
- **KL regularizer** (`kl_tau`) penalizing π↔μ divergence, "maintaining on-policy conditions without explicit KL constraint."
- **Staleness cap:** `orchestrator.max_off_policy_steps` (default 8) — rollouts touched by too many policy versions are discarded; "the primary throughput/on-policyness tradeoff" (docs/training.md).
- Rollout version tracking ("version-salted prefix caches… age off-policy as weights update").

So the framework's core design treats trainer/inference mismatch as *expected* and corrects for it — this is prime-rl's headline feature, not an afterthought.

**Our obligations:** (a) run rollouts at the temperature the correction assumes and verify in T0 that sampling logprobs reflect sampling parameters (the TRL-class bug above lives exactly here); (b) monitor mean IS ratio / clip-fraction in wandb — sustained drift = lower `max_off_policy_steps` or raise `kl_tau`; (c) **if the base model is MoE** (Qwen3.5-MoE fallback, GAMEPLAN §1) treat this risk as elevated: prefer the dense candidate for T2, and if MoE is forced, tighten `max_off_policy_steps` (≤2) and watch clip-fraction like a hawk.

**Residual gap:** Low for dense; medium if MoE is chosen (base-model decision memo, GAMEPLAN §8 D3, must carry this consideration).

**Phase:** T0 (verify logprob sanity), T2.

### 12. Length / format degeneration

**What:** GRPO's classic length pathology — normalization choices reward rambling (Dr.GRPO, <https://arxiv.org/abs/2503.20783>; DAPO overlong-shaping) — plus degenerate outputs (repetition, gibberish) contaminating batches.

**How it manifests here:** The "short rationale channel" (GAMEPLAN §5) balloons into thousand-token essays per pick (slower rollouts, cost) or degenerates into repeated boilerplate; downstream, verbose habits leak into the chat product.

**Countermeasure (all stock prime-rl, docs/algorithms.md):**
- prime-rl's default advantage is group-norm *without* std division ("DR-GRPO variant" — it explicitly adopts Dr.GRPO's fix for the std-normalization bias).
- `[orchestrator.algo.length_penalty]` — linear penalty on output tokens/turns applied before baseline computation, "discourages rambling without hard token limits"; recommended weight 0.1, "0.5+ for severe rambling."
- Pre-batch `repetition` and `gibberish` filters drop degenerate rollouts before they train (filtering section).
- Env-side hard cap on rationale tokens (our DraftGym config) as the blunt backstop.

**Residual gap:** Low.

**Phase:** T2.

### 13. Era drift 2015→2024

**What:** The NFL changed (pass-rate eras, RB devaluation, roster trends). A policy averaged over 2015–2024 may be optimal for a league that no longer exists — and 2026 resembles 2024, not 2017.

**How it manifests here:** Aggregate DraftBench gain driven by 2015–2018 seasons while 2023–2024 is flat — i.e., we learned history, not the present.

**Countermeasure:** Per-season eval rows already exist: `evals/draftbench.py` outputs "one row per agent × season × format × slot × seed" — per-era breakdown is a groupby, and the val seasons (2023–2024) being the *most recent* seasons means checkpoint selection (risk 16) already weights the modern era. Report per-season deltas in every checkpoint eval; a checkpoint whose gains are pre-2020-concentrated fails selection even with a good aggregate. Optionally weight recent-season episodes higher in the T2 rollout mix (prime-rl supports multi-source `ratio` weighting across `[[orchestrator.train.source]]` tables, docs/training.md — one DraftGym source per era band).

**Residual gap:** Low — mechanism exists; the per-season report just has to be part of the standard checkpoint eval printout.

**Phase:** T2/gate.

### 14. Hyperparameter overfitting to dev seasons via iteration — the meta-risk

**What:** Every T2 iteration ("try kl_tau=1e-2… try group 32…") selects on 2015–2024 numbers. Enough iterations and *the process* overfits: the winning recipe is the one that got lucky on dev seasons. Individual-run discipline cannot fix this; only protocol can.

**How it manifests here:** B4 beats B1 on dev seasons after N tweaks, gate on 2025 shows nothing, and — the worst case — we run the 2025 gate *twice*, take the better one, and the holdout is spent.

**Countermeasure — preregistered gate protocol (proposed here; pin before T1):**
1. **2025 is single-shot per contender** — already law: GAMEPLAN §9 "evaluated once per contender at gate time, never during development"; enforced mechanically by `--include-holdout` flags in `evals/draftbench.py` and `evals/calibbench.py` (both refuse 2025 by default).
2. **Attempt cap: 2 gate attempts total for the B4 line** (one for the T2 pilot's best checkpoint, one for T3's), each attempt = one pre-named checkpoint. A third attempt requires new *live* evidence (2026 in-season data), not more dev iteration.
3. **Preregistration artifact:** before running a gate, append to `docs/budget-ledger.md` (exists; every run is logged there per GAMEPLAN §7): checkpoint hash, resolved config (`--dry-run` dump), the predicted DraftBench/CalibBench deltas with CIs, and the pass thresholds — *then* run. Prediction-before-result makes post-hoc rationalization visible.
4. **Checkpoint selection happens on val (2023–2024) before the gate is touched** — the gate confirms, never selects (risk 16).
5. **2026 live ledger is the true test regardless of gate outcome** (GAMEPLAN §4 marketing-honesty rule; `docs/breakoutbench-design.md` §7 freeze-by-Sep-4 protocol) — a gate pass with a bad ledger reverts the model slot to B1.

**Residual gap:** **Medium — the protocol above is proposed, not yet ratified/pinned.** Also the cap is honor-system: nothing in code prevents a third `--include-holdout` run. Cheap hardening: gate scripts append an entry to a `docs/gate-log.md` on every `--include-holdout` invocation. Blocker for T1 GO (ratification, not the hardening).

**Phase:** gate.

---

## C. Process risks

### 15. Eval noise masquerading as progress

**What:** DraftBench deltas have huge variance (±66 to ±271 points std at n=405, evals/README.md). A checkpoint "beating" baseline by 20 points at n=405 can be pure noise; celebrating it selects noise (feeds risk 14).

**How it manifests here:** T2 daily iteration on small eval slices (fewer seeds for speed) → every decision made on statistically-nothing deltas.

**Countermeasure:** Paired-control design (same seat/seed autopick delta, `evals/draftbench.py`) already removes the largest variance components. Decision rule: **no checkpoint comparison without a bootstrap CI over paired per-run deltas, and no "improvement" claim whose 95% CI includes 0.** Seeds are cheap (env is table lookups) — the answer to a wide CI is more seeds, not optimism.

**Residual gap:** **Medium — the CI tooling is referenced by the plan but the artifact is missing: `evals/results/expanded_baseline.md` does not exist in the repo as of this writing** (checked 2026-08-08; `evals/results/` holds gbdt/qwen baselines and answers only). Bootstrap-CI reporting must land in `draftbench.py` (or a small `evals/ci.py`) and the expanded-baseline run must be committed before T1 comparisons mean anything. Blocker for T1 GO.

**Phase:** all.

### 16. Checkpoint selection bias

**What:** Training produces many checkpoints; picking the best-on-eval checkpoint is itself a fit to the selection set. Selecting on the test set destroys it; selecting on noisy val overfits val.

**How it manifests here:** T2 emits a checkpoint per `ckpt.interval`; we pick the 2023–2024 peak; that peak is partly val-season luck.

**Countermeasure:** prime-rl checkpointing is "off by default"; enable `--ckpt` with `ckpt.interval` / `ckpt.keep-last` / `ckpt.keep-interval` (docs/training.md) so a *sparse, predeclared* set of checkpoints exists — selection among ~5 candidates overfits val far less than selection among 500. Selection metric: val-season paired delta with CI (risk 15) + canary pass (risks 2, 8) — a composite is harder to luck into than a single number. 2025 never participates in selection (risk 14 rule 4). The selected checkpoint's val score is *reported as selection-biased* by construction; the gate number is the honest one.

**Residual gap:** Low.

**Phase:** T1/T2/gate.

### 17. Irreproducibility

**What:** "The good run" can't be reproduced: config drifted, seeds unlogged, data snapshot changed, spend untracked.

**How it manifests here:** T3 tries to scale the T2 recipe and gets a different model; nobody can say which of 30 wandb runs was the gate candidate; budget quietly exceeds the $5k ceiling.

**Countermeasure:**
- **Config capture:** prime-rl `--dry-run --output-dir` emits "per-process TOML files reflecting the resolved configuration" (docs/configuration.md) — commit the resolved TOML for every launched run; TOML+CLI layering means the launch command alone is *not* the config.
- **Run logging:** `[trainer.wandb]` for metrics; **every run pre-logged in `docs/budget-ledger.md`** (file exists; GAMEPLAN §7 "Every run gets logged… **No run starts without an explicit go**" — also CLAUDE.md hard rule, which additionally bans local training).
- **Data immutability:** dated snapshots, "never overwrite market history" (GAMEPLAN §6 / CLAUDE.md); evals read from pinned processed CSVs; eval code seeds explicitly (`draftbench.py --seeds`, deterministic question generation in `calibbench.py`).
- **Env versioning:** DraftGym published to the Environments Hub (GAMEPLAN §5) gives the env itself a pinned, pullable version per run.

**Residual gap:** Low-medium — bit-exact RL reproducibility is not achievable (async scheduling is nondeterministic by design; prime-rl's async architecture trades determinism for throughput). We reproduce *configs and conclusions* (re-run → CI-overlapping result), not bitwise checkpoints. Accept and state this.

**Phase:** all.

---

## GO/NO-GO checklists (every box a verifiable artifact)

### T1 (SFT) GO requires

- [ ] **General-ability canary built + B2 reference pinned** — committed script + `evals/results/canary_b2_baseline.*` (risk 2).
- [ ] **Trace filter implemented as code** with rules 1–4 from risk 3, and its survivorship audit printed in datagen output (risk 3).
- [ ] **SFT datagen uses anonymized packets**; leakage-linter pass logged with 0 unexplained hits (risk 4).
- [ ] **Season-split SFT config:** train traces from 2015–2022, `val.data` from 2023–2024; early-stopping criterion written in the config comment (risk 1).
- [ ] **Bootstrap-CI reporting landed in DraftBench** + expanded baseline results file committed (`evals/results/expanded_baseline.md`) (risk 15).
- [ ] **Gate protocol (risk 14, items 1–5) ratified by Jacob** — recorded in this file or GAMEPLAN.
- [ ] **T0 smoke completed**: end-to-end prime-rl SFT on tiny Qwen3.5 (GAMEPLAN §7) including (a) decoded packed batch with visually-verified loss-mask boundaries (risk 5), (b) resolved-config TOML committed (risk 17).
- [ ] **Run pre-logged in `docs/budget-ledger.md` + explicit go from Jacob** (CLAUDE.md hard rule).

### T2 (RL) GO requires all T1 boxes, plus

- [ ] **B3 evaluated:** SFT checkpoint beats/ties B2 on val-season DraftBench+CalibBench with CIs, canary within 3 points of B2 (risks 1, 2, 15).
- [ ] **DraftGym env hardening landed:** per-episode randomization of league shape + opponent noise; held-out env-config cells defined; reward uses fixed-manager-policy lineups, not hindsight-optimal (risk 7).
- [ ] **RL rollout seasons = 2015–2022 only, asserted in env code**; 2023–2024 wired as `[[orchestrator.eval.source]]` val envs (risk 8).
- [ ] **Mispriced-player audit** of the ADP↔outcome join completed (risk 7, hypothesis 2) — findings noted in env README.
- [ ] **Anonymized DraftBench variant built** (currently a v0 gap) — required for checkpoint selection (risk 8).
- [ ] **Memorization canary runnable** on checkpoints (perturbed dossiers, breakoutbench-design §2) (risk 8).
- [ ] **T0-verified logprob sanity** on the chosen base model at the chosen rollout temperature; MoE-vs-dense decision memo addresses risk 11.
- [ ] **Monitoring dashboard:** entropy, zero-advantage drop-rate, mean IS ratio/clip fraction, val-vs-train reward gap, length distribution — all in wandb before step 1 (risks 9, 11, 12).
- [ ] **Terminal-reward-only config confirmed** (no heuristic shaping; length penalty is the only reward modifier) (risk 10).
- [ ] **Checkpoint plan:** `--ckpt` enabled, sparse interval predeclared (risk 16).
- [ ] **Run pre-logged in budget ledger + explicit go** (ceiling check: cumulative spend + this run ≤ $5k).

### Gate (2025) requires

- [ ] Preregistration entry (checkpoint hash, resolved config, predicted deltas + thresholds) appended to `docs/budget-ledger.md` **before** any `--include-holdout` invocation (risk 14).
- [ ] Single run per contender; result published win or lose (breakoutbench-design §9).
- [ ] Ship decision by GAMEPLAN §4 rule: win DraftBench + CalibBench, ≥95% AdviceBench tie — 2026 ledger overrides regardless.

---

## Open gaps, ranked by severity

1. **Anonymized DraftBench variant unbuilt** (risk 8) — the main memorization control for the central risk is a documented v0 simplification, not code. Without it, T2 checkpoint selection can't distinguish memory from skill.
2. **Trace filter + leakage linter unimplemented** (risks 3–4) — specified above, zero lines exist; survivor bias silently poisons T1 otherwise.
3. **Bootstrap-CI artifact missing** (risk 15) — `evals/results/expanded_baseline.md` referenced by plan, absent from repo; until it lands, every "improvement" is unquantified.
4. **General-ability canary unbuilt** (risk 2) — forgetting is invisible until it exists.
5. **verifiers v0→v1 API transition** (see below) — DraftGym design targets a deprecated API.
6. **Gate attempt cap is honor-system** (risk 14) — cheap fix (gate-log append on `--include-holdout`), but currently nothing stops a third attempt.
7. **Hindsight-optimal lineup scoring in the RL reward path** (risk 7 h3) — fine for the *benchmark*, a variance-farming exploit if inherited by the *gym*; must be swapped for the fixed manager policy before T2.

## Plan changes surfaced by stack research

- **verifiers has moved to a v1 API (tasksets / harnesses / traces); v0 `import verifiers as vf` + `MultiTurnEnv` is deprecated and "scheduled for eventual removal"** (<https://github.com/PrimeIntellect-ai/verifiers> `docs/overview.md`). GAMEPLAN §5 says DraftGym is "built on verifiers `MultiTurnEnv`" — build against v1 from the start; a v0 DraftGym would need migration mid-project.
- **prime-rl checkpointing is off by default** (`docs/training.md`) — a T2 run launched with default configs produces *no* checkpoints; `--ckpt` must be explicit in the run template.
- **prime-rl's default advantage already incorporates the Dr.GRPO fixes** (group-norm without std division) and ships length-penalty + repetition/gibberish filters — we do not need custom loss work for risks 9/12; we need monitoring and the documented knob directions.
- **`orchestrator.max_off_policy_steps` (default 8) is the staleness lever** — for our high-variance sparse-reward env, plan to start lower (2–4) and trade throughput for on-policyness; DraftGym rollouts are cheap so the throughput cost is small.
