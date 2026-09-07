# Fantasy Alpha draft game

Fantasy Alpha models a snake draft as a finite-horizon stochastic allocation
game. The model does not try to make the most conventional-looking pick. It
tries to build the legal roster expected to score the most points when a
realistic manager sets the lineup each week.

At turn *t*, the state contains the available board, the user's roster, league
scoring and roster constraints, the picks until the user's next turn, public
preseason information, and any evidence already purchased. Player identities
are masked in the primary experimental track.

The legal action set is:

- `TAKE_NOW(player)`: select a legal player now.
- `WAIT(player, alternative)`: pass on a player because a better allocation is
  expected now and the target may survive to the next turn.
- `PIVOT(player)`: select a legal alternative when the preferred path is no
  longer attractive.
- `CHECK_EVIDENCE(query)`: pay the bounded cost of obtaining information that
  could change the next selection.

A player-selection action removes that player, advances the snake draft, and
lets opponent policies make their intervening selections. Opponents are
same-seed noisy ADP agents subject to the same league-wide feasibility rules.
Their selections are uncertain because a player's sampled draft position is
centered on ADP with historical draft dispersion.

The roster must satisfy the league's required QB, RB, WR, TE, FLEX, and bench
capacity. The environment never offers an action that would make completion
impossible. Evidence actions do not alter the board and are separately
budgeted.

The terminal reward is

```text
R(roster) = sum over weeks of points scored by that week's legal starters
```

The weekly manager ranks players using only performance from earlier weeks;
week one uses preseason ADP. This avoids paying the policy for hindsight-perfect
lineup choices. For a matched experiment we report

```text
Delta = R(Fantasy Alpha roster) - R(same-seat legal ADP roster).
```

ADP subtraction is a comparison layer. It does not change the optimization
objective: maximizing `Delta` for a fixed episode is exactly the same as
maximizing the Fantasy Alpha roster's points.

Three references anchor the problem:

1. The legal ADP autopicker selects the lowest-ADP feasible player.
2. The hindsight marginal oracle sees the realized season and greedily selects
   the player with the greatest realized marginal lineup contribution. It is
   deliberately unattainable and only demonstrates that outcomes contain
   exploitable headroom; it never supplies model inputs.
3. The deployable point-in-time planner uses roster constraints, ADP, draft
   dispersion, positional scarcity, and next-turn survival. It sees no
   target-season result.

The oracle is not claimed to be the globally optimal path through every
possible opponent response. It is a reproducible, clairvoyant ceiling
reference. A later rollout planner estimates action value as the mean terminal
reward over point-in-time simulations, with uncertainty and the opportunity
cost of waiting reported separately.
