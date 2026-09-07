# Outcome-trained drafting: first local checkpoint

The first question was not whether Qwen can imitate fantasy-football advice.
It was whether draft-day information contains a learnable decision signal that
improves completed teams.

We formalized the draft as a constrained sequential allocation game and kept
the product reward concrete: season points from weekly starting lineups chosen
without future knowledge. Every comparison uses the same historical board,
seat, opponent policy, and random seed as a legal ADP autopicker.

The first hand-written planner used only legal preseason information: ADP,
historical draft dispersion, roster constraints, positional scarcity, and the
number of picks until the next turn. Across all 270 development drafts it
averaged +6.95 points over ADP, but its later 2021–2022 replication won 26,
lost 29, and tied 35. That is evidence of possible headroom, not a reliable
policy.

We then generated 1,243 masked draft states from 90 complete development
trajectories. For each state, five legal candidates were replayed through the
rest of the draft. The label was the completed roster's realistic season
points. Realized results never appear in the model inputs.

A small gradient-boosted ranker learned from those counterfactual labels. Its
leave-one-season-out development replay won 43 of 75 drafts and lost 32, with
a +14.47 mean and +17.50 ten-percent-trimmed mean versus ADP. This justified a
single frozen later-season evaluation.

That evaluation was mixed:

| policy | drafts | wins | losses | mean vs ADP | trimmed mean | without largest gain | worst draft |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hand-written planner | 30 | 9 | 13 | -7.61 | -6.72 | -13.63 | -178.00 |
| Outcome ranker | 30 | 17 | 13 | +6.62 | +8.42 | -8.31 | -432.64 |

The ranker also lowered turn-level candidate regret from 64.15 points for ADP
to 58.06 and raised the zero-regret choice rate from 27.6% to 29.6%. Those are
useful directional improvements. But nine of its 30 completed drafts fell at
least 100 points below ADP, and the aggregate advantage became negative after
removing one +439.66 outlier.

The precommitted gate therefore fails. We did not spend money evaluating or
training Qwen against this policy. Training an LLM now would make a brittle
teacher easier to imitate; it would not solve the decision problem.

The next credible experiment is a risk-aware ranker trained on the same
development seasons. It should predict both expected candidate regret and
downside risk, defer to ADP when the predicted advantage is small, and be
selected entirely by season-held-out development replay. Because the
2021–2022 outcomes are now opened, they cannot be reused to tune that policy.
A new evaluation requires a separately frozen protocol and one of the reserved
diagnostic seasons, or forward 2026 results.

No paid calls were made for this checkpoint. Exact cumulative completed Tinker
workload remains $19.811052240; approximate whole-project spend remains $46.57,
excluding ongoing checkpoint storage.
