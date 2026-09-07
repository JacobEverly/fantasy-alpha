# High-confidence draft overrides

This experiment asks a narrower question than the earlier Fantasy Alpha runs:
can a language model learn when *not* to follow average draft position?

ADP is a strong default. Most apparent improvements disappear when they are
replayed through the rest of a draft, and frequent interventions can make a
team worse even when individual picks look sensible. The useful product is
therefore not a model that redrafts the whole board. It is a conservative
decision layer that keeps ADP unless an alternative clears both an
expected-value test and a downside test.

## Where the labels come from

The teacher is a pair of small gradient-boosted models trained on completed
counterfactual drafts. For each historical decision, the simulator replays the
same draft prefix, tries each of the five best legal candidates, finishes the
draft with the same policy, and scores the resulting roster. Player identities
are hidden. The teacher can see draft position, ADP uncertainty, previous-season
points, roster construction, and the probability that a candidate survives to
the next turn. It cannot see the realized outcome used to create the label.

The teacher only overrides ADP when the same alternative is at least 80 points
better in both its mean model and its downside model. Those margins were chosen
on the 2015–2020 development seasons before this SFT corpus was built.

The resulting corpus contains 1,243 draft states:

- 1,187 `KEEP_ADP` decisions
- 56 `OVERRIDE` decisions
- 170 difficult keep decisions near one of the two margins

The natural override rate is 4.5%. Training up-weights the rare overrides and
the difficult negative examples, but evaluation retains the natural rate.

## Pre-training gate

A local candidate classifier was evaluated with five episode-grouped folds.
No state from a held-out draft episode appeared in that fold's training set.
At the selected diagnostic operating point it recovered 55 of 56 override
states, produced five false overrides among 1,187 keep states, and selected the
exact alternative in 55 cases. State-level average precision was 0.980 against
a 0.045 prevalence.

This is not a product result. The classifier and its operating point were
developed on the same historical development pool. It only establishes that
the visible features contain a learnable version of the teacher's decision
boundary, which is enough to justify one small language-model training run.

## Frozen language-model experiment

The run uses `Qwen/Qwen3.5-9B`, a rank-16 LoRA, one epoch, a 1e-4 learning rate,
and the model's no-thinking chat renderer. The full corpus is approximately
668,000 rendered tokens. Tinker's price-based estimate is $1.13 for training
and development loss evaluation.

The model returns one of two compact actions:

```json
{"decision":"KEEP_ADP"}
```

```json
{"decision":"OVERRIDE","candidate_id":"B016"}
```

The 2025 comparison was frozen before paid training. It uses the same fifteen
seat-and-seed combinations for ADP, the deterministic teacher, untouched Qwen,
and the adapter. Invalid model output falls back to ADP and is counted. The
report separates teacher imitation from completed-roster points; better JSON
formatting alone does not count as a product improvement.

## Results

Pending the one-epoch Tinker run and one-shot 2025 evaluation.

## Historical limits

The 80/80 teacher performed well on an already-opened 2021–2022 panel, but that
panel is provisional rather than fresh held-out evidence. It also failed badly
when transported backward to 2013. That result remains part of the report: the
policy appears sensitive to era and should not be described as a universal
fantasy-football strategy.

## Try the decision card

The repository includes a small local interface to the deterministic teacher:

```bash
.venv/bin/python -m app.decision_card --example
```

It reports the ADP choice, the best alternative, both safety margins, and the
final keep-or-override action. The trained model will be judged against this
same behavior, while the deterministic teacher remains the safe fallback.
