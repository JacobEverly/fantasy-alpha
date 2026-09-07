# Fantasy Alpha

Fantasy Alpha is where I have been testing a practical post-training question:
can a small language model learn to make better decisions over a whole sequence,
rather than just imitate better-looking answers?

I use a fantasy-football draft as the environment because it is compact but
unforgiving. The model makes fifteen related choices, the board changes after
each one, outside information is sometimes useful, and a decision that looks
reasonable on its own can make the rest of the draft harder. Historical seasons
also give me a repeatable outcome instead of a subjective grader.

The first adapter clearly learned its training corpus, but it did not reliably
beat the base model. I then tried external supervisors and, finally, a much
narrower adapter trained only to recognize rare opportunities to override ADP.
Each version looked more promising on its local objective than it did inside a
complete draft. This repository contains the training code, environment,
protocols, and results behind that progression.

## Results so far

All experiments use `Qwen/Qwen3.5-9B` with masked player identities and
point-in-time data. Evaluation settings and stop conditions are written down
before the evaluated runs.

| Experiment | Immediate result | Complete-draft result | Decision |
|---|---:|---:|---|
| Rank-32 LoRA trained through Tinker on 811 traces | Development NLL fell from 1.257 to 0.815 | No reliable improvement over the base model | Revise the training data before RL |
| Tool-decision classifier | 99.1% accuracy on its held-out imitation task | Successful tool calls made drafts worse | Stop using imitation accuracy as the target |
| Outcome-linked supervisor | +2.64 points per isolated held-out decision | −15.75 points per paired draft over 30 episodes | Do not train the current policy with RL |
| Sparse reversible controller | 4 of 30 proposed revisions accepted | −4.92 points per paired draft; 3 wins, 3 losses, 24 ties | Fix clean-state reversion before training |
| High-confidence ADP override, rank-16 LoRA | 100% valid actions, but 0/3 teacher overrides recovered | −29.49 points per draft versus ADP | Stop; retain ADP and the deterministic policy as research references |

The outcome-linked result exposed the central problem. The supervisor found a
weak positive signal for isolated decisions, but its interventions changed the
later draft. All 412 forced lookups succeeded, so the failure was not tool
reliability; it was the difference between judging one decision and judging the
sequence it creates.

This is why the project reports local metrics and full-episode outcomes
separately. A lower loss or a more accurate classifier is useful evidence, but
it is not treated as a product result.

The sparse controller limited itself to one lookup and rejected 26 of 30
revisions. It removed the repeated-tool failure, but not every downstream
effect: two rejected lookups still changed later picks because the extra model
call advanced the seed schedule used for subsequent decisions. The action and
draft state reverted; the sampling path did not.

The final override experiment exposed a different failure. Different draft
seeds had recreated 395 identical visible states, so grouping only by episode
allowed duplicates across local validation folds. On the one-shot 2025 test,
the adapter learned an early-round positional shortcut and the teacher itself
failed to beat ADP. That changed the decision from “improve the adapter” to
“stop until the teacher transports across seasons.”

## Experimental setup

DraftGym replays historical drafts with names hidden from the model. Each run
uses a fixed league format, seat, random seed, opponent policy, evidence date,
and scoring function. The model can make a pick or call a structured evidence
tool. Terminal reward measures realistic roster performance relative to an ADP
control drafted from the same seat and seed.

The main components are:

- `training/`: Tinker and Prime backends, corpus preparation, LoRA runs, and
  frozen experiment specifications.
- `envs/`: the multi-turn DraftGym environment and model runners.
- `harness/`: point-in-time evidence retrieval and external supervisors.
- `evals/`: matched comparisons, scorecards, uncertainty estimates, and
  contamination checks.
- `tests/`: 507 tests covering leakage, determinism, holdout access, replay,
  tool legality, and experiment invariants.

The Tinker integration handles training, checkpoint save and reload, sampling,
and adapter export. The repository contains rank-32 and rank-16 LoRA runs for
`Qwen/Qwen3.5-9B`; the underlying base weights are unchanged. The supervisors
between those runs are small external decision models rather than additional
language-model fine-tunes.

## Evaluation discipline

- Data is restricted to what would have been available at the decision date.
- Player identities are masked in the main draft evaluations.
- Train, development, and evaluation groups are separated by complete question
  or episode. The latest failure analysis shows why exact visible-state
  deduplication must be an additional boundary.
- Checkpoints and thresholds are selected on development data only.
- Frozen specifications record the prompts, model, seeds, code hashes, scoring,
  budget, and decision rule before evaluation.
- Results include paired examples, bootstrap intervals, failed requests, and
  exact request-level spending.
- Sealed seasons are opened only under frozen protocols. The 2025 holdout was
  used once for the completed high-confidence override gate and is now closed
  to further tuning.

The controls are not meant to make historical replay perfectly representative
of a live draft. They make the limitations visible and keep comparisons between
experiments consistent.

## Reading the experiments

For the shortest path through the work:

1. [Project brief](docs/project-brief.md)
2. [High-confidence override experiment](docs/high-confidence-override-experiment.md)
3. [Tinker SFT experiment](docs/tinker-sft-experiment-report.md)
4. [Tool-decision supervisor](docs/tool-decision-supervisor-experiment.md)
5. [Outcome-linked supervisor](docs/value-of-information-experiment.md)
6. [Sparse reversible canary](docs/sparse-reversible-canary.md)
7. [Budget ledger](docs/budget-ledger.md)

The corresponding machine-readable artifacts are under `artifacts/`. Frozen
protocols are under `training/` and include hashes of the code used for each
evaluated run.

## Running locally

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,tinker]'
.venv/bin/pytest -q
```

Paid Tinker operations read the credential from `TINKER_API_KEY`. Keys are not
accepted as command-line arguments and are not written to experiment artifacts.
Most validation and report-generation commands do not require provider access.

## Current direction

The latest test was deliberately narrower than another general drafting agent.
A deterministic teacher found rare historical states where one alternative
cleared both an expected-value and a downside margin. A rank-16 adapter learned
the action format but not the boundary: on the frozen 2025 run it missed all
three teacher overrides and made five false ones. The teacher itself also lost
to ADP in 2025. The project therefore stops this training path rather than
tuning against the holdout. The useful next research question is how to build a
teacher whose value transports across eras before asking a language model to
imitate it.

The historical results establish behavior in this environment only. They do not
establish forecasting skill for a future NFL season or a general result about
language-model supervision.
