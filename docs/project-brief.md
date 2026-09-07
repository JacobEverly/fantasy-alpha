# Fantasy Alpha: post-training for sequential decisions

Fantasy Alpha is an experiment in teaching a language model to make decisions
whose quality only becomes clear later. A fantasy-football draft is a useful
small environment for this: every choice changes the remaining board, roster
construction matters, and historical seasons provide a concrete terminal
score.

The project began with a broad supervised fine-tune and moved toward narrower
interventions. I built a replay environment, masked player identities, limited
inputs to point-in-time evidence, and compared every treatment with an ADP
control from the same seat and random seed. Training and evaluation run through
a modular Tinker backend, while frozen specifications record the prompts,
splits, decoding settings, costs, and stop rules before a holdout is opened.

## The final experiment

The last run asked whether a 9B model could learn a conservative exception to a
strong baseline. A counterfactual simulator completed the draft after each of
five plausible picks. Two small models estimated expected value and downside,
and produced an override label only when the same alternative cleared both by
80 points. This yielded 1,243 masked states, including 56 overrides.

I trained a rank-16 LoRA on `Qwen/Qwen3.5-9B` for one epoch. The run took 268
seconds, cost $1.13, reduced development NLL from 0.098 to 0.032, and produced a
reloadable, exported adapter. None of those facts established that the policy
worked.

The frozen 2025 evaluation did not. The adapter returned valid structured
actions on every turn, but missed all three teacher-approved overrides and made
five false ones. It finished 29.49 points per draft behind ADP. The teacher was
also 22.36 points behind ADP, meaning that better distillation would not have
rescued the product result.

## What changed after the result

The failure analysis found that 1,243 training rows represented only 848 unique
visible states. Different simulation seeds often recreated the same early-round
prompt, allowing duplicates across episode-grouped validation folds. Fifty-four
of 56 positive rows were in duplicate groups. Positive labels were also heavily
concentrated in rounds one and two and at the edge draft slots. The adapter
learned that shortcut, repeatedly choosing candidate four with the first pick.

I did not retune the prompt or adapter after seeing 2025. The repository records
the failed gates, the duplicate-state audit, paired draft outcomes, exact
$1.2744 experiment cost, and the decision to stop. ADP remains the product
fallback.

## Why I kept the negative result

The useful artifact is not a claim that fine-tuning beats the market. It is a
complete account of how a plausible local metric can fail in a sequential
product, and how to catch that before shipping it. Any next attempt would start
with a teacher that demonstrates forward-season value, split on visible state
identity, collect more independent positive decisions, and gate on completed
draft outcomes rather than imitation alone.

The [full experiment report](high-confidence-override-experiment.md) links the
training configuration, model card, frozen protocol, and machine-readable
results. A small local decision card is included to inspect the deterministic
teacher, but it is explicitly presented as a research tool rather than a live
recommendation engine.
