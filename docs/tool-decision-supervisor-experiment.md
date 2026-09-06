# Tool-decision supervisor experiment

**Completed:** 2026-09-06

**Decision:** redesign the decision policy and outcome labels before shipping;
retain an external/hybrid supervisor as the leading architecture. Do not run
more SFT or begin RL yet.

## What this experiment asked

Can Fantasy Alpha tell the difference between a state where it is safe to act,
a state where one specific evidence lookup is required, and a state where it
should wait because no available tool can resolve the uncertainty?

The implementation uses the actual DraftGym loop. Qwen first proposes a pick
or tool action. The supervisor sees the same observation plus that proposed
action, then approves it, requires a structured lookup, or blocks it. DraftGym
executes the tool through EvidenceStore, injects the result, and calls Qwen on
the updated observation. This candidate-aware design is necessary: an
observation-only supervisor cannot judge whether a particular proposed pick or
tool request is safe and relevant.

## Frozen data and protocol

The primary dataset contains 948 examples in 237 matched groups from 98 whole
DraftGym episodes. Each evidence opportunity contributes a missing-evidence
state, the corresponding resolved post-tool state, an unresolved no-budget
state, and a nearby context-sufficient state. Whole episodes—not individual
rows—are split into 564 training, 164 development, and 220 internal-held-out
examples. The set is balanced by scenario and has no duplicate decision input.

The labels are deliberately outcome-blind. They encode the existing
SurvivalSequencer policy using pre-as-of depth-chart and injury evidence. They
do not claim that making the lookup improves fantasy points. The evaluation
protocol, thresholds, model prompt, candidate artifacts, and seven-episode
integration panel were hashed before held-out labels were opened. Seasons
2013, 2018, 2023, 2024, and 2025 remained untouched.

The primary dataset covers structured depth-chart and injury evidence. A
separate 12-case post-hoc adversarial suite covers the categorical gaps named
in the milestone: missing ADP/ranking evidence, locally solvable roster math,
valid and illegal tool proposals, irrelevant targets, and inconclusive tool
results. It was frozen and scored without refitting. Because it was created
after the primary held-out evaluation, it is a robustness diagnostic—not a
second primary gate.

## Decision-boundary results

| System | Split | Accuracy | Required-tool recall | Unsafe direct action | Unnecessary tools | Immediate-action retention | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Always act | held-out | 50.0% | 0.0% | 100.0% | 0.0% | 100.0% | Fail |
| Always use tool | held-out | 50.0% | 100.0% | 0.0% | 100.0% | 0.0% | Fail |
| Hard-coded rules | held-out | 91.4% | 70.9% | 14.5% | 2.7% | 100.0% | Fail |
| Untouched Qwen3.5-9B | held-out | 58.6% | 98.2% | 1.8% | 70.9% | 1.8% | Fail |
| Learned supervisor | held-out | **99.1%** | **100.0%** | **0.0%** | **1.8%** | **96.4%** | **Pass** |

The calibrated random-forest supervisor has a 0.024 held-out expected
calibration error and 0.0138 multiclass Brier score. Its artifact was fitted on
training plus development only, reloaded successfully, and had the same hash
before and after held-out scoring.

This is strong evidence that a small external component can reproduce this
narrow, structured policy more reliably than the base model or transparent
rules. It is not yet evidence that the policy itself is valuable.

## Contrastive Qwen training

Both canaries used exactly `Qwen/Qwen3.5-9B`, Tinker 0.27.1, cookbook 0.5.7,
the no-thinking renderer, rank-32 LoRA, an 8,192-token limit, and the same
DraftGym action schema used at evaluation.

| Candidate | Training result | Required-tool recall | Correct tool | Unsafe action | Immediate retention | Decision |
|---|---|---:|---:|---:|---:|---|
| Untouched base, development | — | 100.0% | 26.8% | 1.2% | 4.9% | Reject |
| Canary v1 | NLL 0.442 → 0.053 | 0.0% | 0.0% | 50.0% | 100.0% | Reject |
| Canary v2, one allowed revision | NLL 0.518 → 0.048 | 85.4% | 70.7% | 7.3% | 73.2% | Reject |

The base model overcalled tools. The first adapter overcorrected and stopped
calling them. Reweighting tool-required examples in the one permitted revision
improved behavior, but four primary gates still failed. Lower loss and perfect
JSON formatting did not justify a full adapter. Training stopped exactly where
the preregistered protocol required; no SFT candidate saw held-out labels.

## Real DraftGym integration

Seven masked, evidence-enabled historical episodes per arm proved the complete
multi-turn harness works, including forced tools and post-tool continuation.
The sample is intentionally too small for a stable product-quality claim.

| Arm | Completed | Tools | Mean reward vs matched ADP control | Median | Cost |
|---|---:|---:|---:|---:|---:|
| Base Qwen | 7/7 | 0 | **-30.06** | -63.38 | $0.143901 |
| Rules + Qwen | 7/7 | 23 | -83.44 | -148.84 | $0.156259 |
| Learned supervisor + Qwen | 7/7 | 24 | -47.42 | -63.38 | $0.156779 |

Against the same base episodes, the learned hybrid had zero wins, two losses,
and five ties; its mean paired change was -17.36 points. Rules had zero wins,
four losses, and three ties; their mean paired change was -53.38 points. Every
forced lookup executed successfully, but successful retrieval did not reliably
produce a better pick.

This exposes the central limitation: the supervisor learned when the old policy
*wanted evidence*, not when evidence had positive value for the final draft.
The untouched Qwen also behaved differently across surfaces—overcalling tools
in the candidate-review benchmark while making zero tool calls in full draft
episodes—which is evidence of a prompt/trajectory distribution shift.

Tool calls per completed episode were 0.00 for base, 3.29 for rules, and 3.43
for the learned hybrid. Warm local decision latency was approximately 0.002 ms
median for rules and 76.3 ms median for the current learned artifact. That
learned overhead is small beside a remote model request but unnecessarily high
for a tiny classifier and should be optimized before serving. The original
Tinker records retained token/cost ledgers but not per-request timestamps, so
end-to-end provider latency is explicitly unavailable; the experiment was not
rerun solely to manufacture that operational metric.

## Adversarial robustness

On the 12-case post-hoc challenge, the hard-coded supervisor scored 10/12 and
the learned supervisor 11/12. Both failed the zero-unsafe-action gate. The
learned model acted when ADP/ranking evidence was missing and no supported tool
could retrieve it. The rules also treated an inconclusive successful envelope
as resolved. Illegal and irrelevant tool requests were blocked correctly.

These are concrete reasons not to ship the current artifact despite its strong
primary held-out score.

## Spending

Exact incremental price-based spend was **$3.457637520**, below the $5 target
and $10 hard cap. Exact cumulative completed Tinker workload is
**$14.312830878**. The approximate whole-project total is now **$41.07**,
excluding ongoing checkpoint storage. The line-item audit and independent
provider event snapshot are stored with the experiment artifacts.

## Decision and next milestone

**Decision: redesign the data because no current approach is reliable enough
to ship.** The architecture direction is still external/hybrid:

- Keep legality, tool availability, budget enforcement, and fail-closed rules
  outside the main model.
- Use a lightweight learned component for narrow decision-time risk once its
  labels represent value of information rather than policy imitation.
- Let the main model choose and explain the final pick after evidence returns.
- Do not spend on more contrastive SFT or DraftGym RL yet.

The next experiment should label intervention value from matched counterfactual
rollouts: run the same state with immediate action versus the appropriate tool,
measure whether the information changes the action and improves final reward,
then train and evaluate on those outcome-linked labels across a materially
larger paired DraftGym panel.

## Reproduction map

- Dataset builder: `training/tool_decision_dataset.py`
- Frozen primary spec: `training/tool-decision-eval-spec-v1.json`
- Supervisor interface: `harness/tool_supervisor.py`
- Learned trainer: `training/tool_decision_supervisor.py`
- Contrastive recipes: `training/tool_decision_sft.py` and
  `training/tool_decision_sft_v2.py`
- Full interaction driver: `envs/play_llm_supervised.py`
- Frozen evaluator: `evals/tool_decision_scorecard.py`
- Adversarial diagnostic: `evals/tool_decision_adversarial.py`
- Final scorecard: `artifacts/tool-decision-supervisor-v1/frozen-evaluation/scorecard.json`
- Spend audit: `artifacts/tool-decision-supervisor-v1/spend-audit.json`
