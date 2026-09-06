# envs/ — RL environments (verifiers)

DraftGym is the draft RL environment from GAMEPLAN §5: one snake-draft seat
against noisy-ADP autopick opponents on real historical FFC boards, rewarded
by what the drafted roster actually scored that NFL season. The simulator,
opponent model, and feasibility/scarcity guards are **imported from
`evals/draftbench.py`** — one simulator, two consumers (benchmark + gym); the
gym adds a step API, an evidence-tool action, and a different (non-hindsight)
reward.

## Files

| File | What |
|---|---|
| `draftgym.py` | Core env. Stdlib + this repo only — zero third-party deps. |
| `verifiers_v1_adapter.py` | verifiers v1 `Taskset`/`Harness`/`Env` wrapper (prime-rl entry point). Import-guarded; core env works without verifiers installed. |
| `play_llm.py` | Serverless baseline driver (Qwen3.5-9B through full episodes) + the shared prompt/action-parsing helpers the adapter reuses. |
| `play_llm_supervised.py` | Candidate-aware interaction driver: model proposal → supervisor approval/tool/block → EvidenceStore → updated observation → model continuation. |

## API

```python
from envs.draftgym import DraftGym

gym = DraftGym(season=2021, preset="ppr", agent_slot=7, seed=3)   # league=LeagueConfig(...) to override 12-team default
obs = gym.reset()                      # Observation (dict) with .render_json()
obs, reward, done, info = gym.step(action)
```

Actions — a dict, one of:

```python
{"pick": "<player_id>"}                                  # draft (optional "rationale": str, recorded for SFT traces)
{"tool": {"name": "player_news", "arguments": {...}}}    # evidence lookup
```

- **Picks** are validated against the same feasibility rules as DraftBench
  (never strand a starting slot, never overflow bench, league-level scarcity
  guard). Illegal picks raise `ValueError` — callers own their fallback
  (`play_llm` retries once, then autopicks).
- **Tool calls** dispatch through a `harness.evidence.EvidenceStore` pinned at
  `as_of = Sept 1 of the season` (published < as_of is enforced inside the
  store). Tools: `player_news`, `search`, `depth_chart`, `injury_status`.
  Results append to `obs["tool_results"]` (cleared at each pick). Capped at
  **5 lookups per pick**; over-cap calls return an error envelope and count in
  `info["cap_hits"]`.
- Opponents auto-play between agent turns with DraftBench's exact autopick
  model (`adp + Normal(0, ffc_stdev)`), same RNG stream — an agent that
  mimics autopick reproduces the control draft pick-for-pick.

Observation (also see `Observation.render_json()`): league shape, pick
number, round, picks-until-next-turn, agent roster, **top-25 available** with
`adp / adp_stdev / position / team / prev_season_points` (previous season
only — leakage guard), feasible positions, lookup budget, tool results,
evidence as-of date.

`season=2025` is refused — untouched gate-time holdout (GAMEPLAN §9).

## Reward — realistic manager, not hindsight (risk register #7)

Terminal-only:

```
reward = realistic_points(agent_roster) − realistic_points(autopick_control_roster)
```

where the control is AutopickADP drafted **in the same seat with the same
seed** (matched-pairs, risk register #10), and `realistic_points` sets each
week-W lineup by **season-to-date points per elapsed week using only weeks
< W** (week 1 = pure ADP order) — a manager who cannot see the future. Booms
are only harvested after they show up in the average; absences decay it.

DraftBench keeps its hindsight-optimal scorer **for benchmarking only**. It
is never the RL reward: hindsight lineups overpay boom/bust variance the
manager can't harvest, which is a reward-hacking vector (risk register #7,
exploit hypothesis 3). `info["agent_points_hindsight"]` is reported as a
diagnostic; tests pin `realistic ≤ hindsight` and `autopick reward == 0`.

## Anonymized mode (memorization counter — risk register #8)

`DraftGym(..., mask_names=True)`: `render_json()` replaces player names with
stable per-episode board ids (`B001`… in ADP order) and picks use those ids;
feature content (adp, stdev, position, prior-season line) is unchanged.
Free-text evidence tools (`player_news`/`search`) are disabled in masked mode
— their text reveals names — while structured lookups (`depth_chart`/
`injury_status`) accept a board id and return rows re-keyed to it, with
errors sanitized so they never echo a real name.

Why: on historical seasons a pretrained model can win episodes by drafting
players it REMEMBERS finishing well — reward inflates via memory, and the
learned policy ("pick who I recall being good") does not transfer to 2026.
Masked boards force strategy to come from features. The training plan will
likely mix masked and named episodes, with the memorization-canary gate
arbitrating; **named-minus-masked reward on paired configs is itself a
memorization measurement** (play_llm reports it).

## How prime-rl consumes it

`envs/verifiers_v1_adapter.py` targets **verifiers v1** (`verifiers.v1`:
tasksets/harnesses/traces — v0 `MultiTurnEnv` is deprecated). Verified
against verifiers **0.1.14** installed in the repo `.venv` (py3.10).

```python
from envs.verifiers_v1_adapter import load_environment

env = load_environment(split="train")   # vf.Env(taskset, harness)
```

- **Taskset**: one row per (season, format, slot, seed) episode;
  `split="train"` → seasons 2015–2022, `split="val"` → 2023–2024
  (checkpoint-selection only — wire it as a prime-rl
  `[[orchestrator.eval.source]]`). The split discipline and the 2025 seal are
  asserted in code (risk register #8).
- **Harness**: a program-form harness — `draftgym_program(task, state)` drives
  the whole multi-pick episode, calling the model through
  `state.get_client(api="chat")` (the v1 interception endpoint), so every
  request/response lands in the framework-owned trajectory/traces that GRPO
  trains on.
- **Reward**: `@vf.reward points_above_autopick` reads the gym's terminal
  reward from state. Raw points scale; prime-rl's group-relative advantage
  (same task row = same board/seed across a group) normalizes within-group.
  Metrics: `fallback_picks`, `tool_calls_used`.

Run config reminders (docs/training-risk-register.md): `group_size` 16 to
start (high-variance reward), terminal-reward-only (no heuristic shaping),
checkpointing ON, per-episode env randomization to be added before T2
(opponent-σ jitter, league-shape sampling — risk 7 counters 1/4/5).

## Baseline driver

```
.venv/bin/python -m envs.play_llm
```

12 named episodes (2018/2021/2024 × slots 1,7 × seeds 0,1; 12-team PPR, 15
rounds) + 2 anonymized duplicates, Qwen/Qwen3.5-9B serverless at temp 0.
Strict-JSON action parsing with one repair retry, then autopick fallback
(counted). Hard budget $1.50; cost appended to `docs/budget-ledger.md`;
results at `evals/results/draftgym_qwen_baseline.json`.

## Tool-decision supervisor integration

`play_llm_supervised.run_episode` preserves the same DraftGym state, parser,
tools, and terminal reward while inserting an optional external supervisor after
each model proposal. The supervisor can approve a legal action, replace a pick
with one exact evidence call, or block and fall back conservatively. Tool results
remain environment-owned and are injected into the next observation; the
supervisor never receives hidden identity or future reward.

The first frozen integration pilot completed seven matched episodes each for
base Qwen, hard-coded rules plus Qwen, and learned supervisor plus Qwen. It
proved the complete lifecycle but did not show reward improvement. See
`docs/tool-decision-supervisor-experiment.md` before expanding or training this
policy.
