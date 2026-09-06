# Fantasy Alpha tool supervisor v1

## Model summary

This is a narrow, external DraftGym supervisor—not a language model. It is a
calibrated 200-tree random forest that reviews a proposed draft pick using only
the current visible observation. It returns `ACT_NOW`, `USE_TOOL`, or
`WAIT_OR_ABSTAIN`, plus confidence, a reason code, and an exact structured tool
call when one is required.

- Artifact: `artifacts/tool-decision-supervisor-v1/learned-supervisor.joblib`
- SHA-256: `cf4d8a496602909635425509c2b81836f14395e26a1f8192ba35153a13c8fef0`
- Runtime dependency: scikit-learn 1.7.2
- Training code: `training/tool_decision_supervisor.py`
- Inference adapter: `harness/tool_supervisor.py::LearnedToolSupervisor`
- Status: research artifact; **not approved for product deployment**

## Training data

The artifact used 728 development-eligible examples (564 training plus 164
development) from Fantasy Alpha tool-decision v1. The underlying 182 matched
groups remain grouped by 77 complete DraftGym episodes. Inputs are anonymized
states from 2015–2017 and 2019–2022. Sealed 2013, 2018, 2023, 2024, and 2025
seasons are absent.

The target labels represent an outcome-blind expert policy around structured
`depth_chart` and `injury_status` lookups. They do not represent observed
fantasy-point lift from the lookup.

## Permitted inputs

The feature allowlist includes draft round and pick, time to the next turn,
roster and visible-board sizes, lookup budget, the proposed candidate's ADP,
ADP uncertainty, prior-season points, position and visible rank, and the status
of a relevant prior tool result. It excludes future tool results, realized
season outcomes, verifier output, final reward, hidden identity, and sealed
data.

## Selection

Model depth, leaf size, calibration, and decision thresholds were selected on
grouped development data only. The final artifact was refit on training plus
development. The held-out evaluator cannot fit or tune the artifact and checks
its hash before and after scoring.

## Results

On the frozen 220-row internal held-out set:

- Accuracy: 99.1%
- Required-tool recall: 100.0%
- Correct-tool rate: 100.0%
- Unsafe direct-action rate: 0.0%
- Unnecessary-tool rate: 1.8%
- Immediate-action retention: 96.4%
- Expected calibration error: 0.0243
- Multiclass Brier score: 0.0138

It passed every preregistered primary classification gate. On a later 12-case
post-hoc edge challenge it scored 11/12, but incorrectly acted when ADP was
missing and no supported tool could retrieve ranking evidence.

When attached to Qwen3.5-9B in seven complete matched DraftGym episodes, it
forced 24 successful tool calls but produced a mean reward of -47.42 versus
-30.06 for unsupervised Qwen. It tied five episodes and lost two. This small
panel is insufficient for a stable effect estimate, but it provides no evidence
that the current intervention policy improves draft outcomes.

Warm single-row inference on the development Mac measured a 76.3 ms median and
82.6 ms p95 across 2,200 calls. This is acceptable for research but excessive
for the model's size; the current nested calibrated-forest implementation should
be simplified or served in batches before production use.

## Intended use

Use this artifact to reproduce the experiment, test the supervisor interface,
and serve as a baseline for outcome-linked value-of-information work. The
external architecture is suitable for enforceable tool legality, budget, and
safety controls.

## Limitations and prohibited claims

- Do not claim that the artifact improves fantasy draft quality.
- Do not use its near-perfect held-out classification as evidence of causal
  product lift; train and held-out rows share the same label-generating policy.
- It has not learned missing-ADP behavior or named free-text search.
- Its post-tool representation is narrow and can confuse an inconclusive result
  with useful evidence unless deterministic guards run first.
- Seven complete episodes are far too few to establish reward improvement.
- Do not use it on 2025 or market it as a validated production safety system.

## Recommended architecture

If future outcome-linked evaluation succeeds, deploy the learned component
inside a hybrid: deterministic guards validate legal actions and tool budgets,
the classifier estimates whether evidence is worthwhile, and the main language
model performs the post-evidence decision and explanation. Until then, redesign
the labels and keep this artifact offline.
