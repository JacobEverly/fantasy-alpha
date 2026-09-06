# Outcome-linked evidence supervisor experiment

**Completed:** 2026-09-06 · **Base model:** `Qwen/Qwen3.5-9B` through Tinker ·
**Decision:** **stop_supervision_as_primary_performance_lever**

## Executive result

Stop treating supervision as the primary performance lever; do not begin RL.

On 70 held-out matched decision points, the learned supervisor changed mean
counterfactual reward by **2.64 points** while
selecting tools on **77.1%** of points. In 30 matched full
episodes it changed reward by **-15.75 points** on
average, with 12 wins, 15 losses, and
3 ties versus base Qwen. The paired bootstrap 95% interval is
[-55.70, 24.42].

This is the second half of a deliberate post-training loop. The prior experiment
trained a highly accurate policy-imitation supervisor, but successful tool calls
did not improve full-draft outcomes. This experiment changed the target: predict
whether buying one piece of evidence improves terminal reward, rather than
whether an old policy says evidence is missing.

## Experimental design

- Collected 140 matched decision points from 70 complete masked episodes.
- Replayed the exact same action prefix for control and treatment.
- Control accepted Qwen's proposed pick; treatment performed one relevant
  structured lookup and let the same model reconsider once.
- Continued both branches with the same deterministic policy to isolate the
  local intervention.
- Fit the lightweight classifier and selected its threshold only on 35
  development episodes, grouped by complete episode.
- Froze the artifact, threshold, rules, prompts, 30-episode arm panel, and hashes
  before the 35 held-out episodes were scored.
- Kept seasons 2013, 2018, and 2023–2025 sealed.

## Held-out matched decisions

| Policy | Mean reward change vs act | Tool rate | Harmful interventions selected |
|---|---:|---:|---:|
| Never buy evidence | 0.00 | 0.0% | 0 |
| Always buy evidence | -2.62 | 100.0% | 11 |
| Hand-written rules | -0.37 | 50.0% | 6 |
| Prior imitation supervisor | -4.88 | 30.0% | 5 |
| Outcome-tuned rule | -2.31 | 48.6% | 6 |
| Learned outcome-value supervisor | 2.64 | 77.1% | 6 |

The matched-state result measures local value of information. It is cheaper and
less confounded than replaying a full model trajectory twice, but it cannot prove
that the policy remains valuable after its decisions change later draft states.

## Frozen full-episode comparison

| Arm | Completed | Mean reward | Tool calls | Tinker cost |
|---|---:|---:|---:|---:|
| Base Qwen3.5-9B | 30/30 | -24.06 | 0 | $0.6094 |
| Prior supervisor + Qwen | 30/30 | -29.78 | 77 | $0.6525 |
| Outcome rule + Qwen | 30/30 | -36.71 | 63 | $0.6453 |
| Learned outcome supervisor + Qwen | 30/30 | -39.82 | 412 | $0.8357 |

The 30 episodes are paired by season, draft slot, and random seed. The primary
comparison is the learned outcome supervisor against untouched base Qwen:
12 wins, 15 losses, 3 ties;
median delta **-1.59** and mean delta after removing
the largest gain **-22.50**.

## Preregistered signal gate

| Condition | Result |
|---|---|
| critical improvements exceed regressions | Fail |
| full episode bootstrap ci excludes zero | Fail |
| not driven by one outlier | Fail |
| positive full episode value | Fail |
| positive heldout counterfactual value | Pass |
| predicted value tracks observed benefit | Pass |
| selective tool use | Pass |
| wins more than losses | Fail |

## Interpretation

The experiment tests whether the supervisor knows when evidence is worth buying,
not merely whether it reproduces a hand-written rule. The full decision gate and
every failed condition are recorded in the scorecard. A positive local effect
without a positive full-episode effect is not enough to justify RL.

The learned artifact is intentionally small and external to the main model. This
keeps legality, tool availability, and budget enforcement deterministic while
making the uncertain intervention policy replaceable. It also makes the negative
result from an SFT checkpoint useful: model training is one candidate lever, not
the definition of progress.

One canary exposed a repeated-tool loop because the outcome estimator was trained
on pre-tool states. The affected arm was stopped before a completed record, the
fix was constrained to approving the reconsidered action after one successful
lookup, and the event plus provider-reconciled cost are retained. No artifact,
threshold, feature, or held-out label was changed.

## Limits

- The panel is large enough for a first falsifiable routing result, not a universal
  claim across models, tools, league formats, or future seasons.
- Terminal draft reward is deterministic under the frozen continuation, but is a
  noisy proxy for the causal value of one local pick.
- Helpful interventions are rare, so calibration and policy selection remain
  data-constrained.
- This milestone does not claim a trained mid-trajectory world model or an RL
  policy. RL remains gated on the frozen outcome signal.

Exact experiment spend was **$4.261158**
against a $5 target and $10 hard cap. No held-out labels were used for fitting or
threshold selection.

## Artifacts

- Frozen protocol: `training/value-of-information-spec-v1.json`
- Dataset audit: `artifacts/value-of-information-v1/dataset-audit.json`
- Scorecard: `artifacts/value-of-information-v1/scorecard.json`
- Reviewed pairs: `artifacts/value-of-information-v1/reviewed-pairs.jsonl`
- Spend reconciliation: `artifacts/value-of-information-v1/spend-audit.json`
