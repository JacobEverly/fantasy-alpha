# Fantasy Alpha Qwen3.5-9B Tinker SFT T1.1

Development decision: **revise sft before rl**.

## Model

- Base: `Qwen/Qwen3.5-9B`
- Method: Tinker LoRA SFT, rank 32, no-thinking Qwen renderer
- Dataset: 300 targeted traces, SHA `0ebbb365f1be9fb76804256cbbdf1eccca300a31c01beca0cd1ff446fc6a81b1`
- Selected checkpoint: `tinker://dc343ef3-3c51-56e7-8a56-f66342565fa4:train:0/sampler_weights/fantasy-alpha-t11-qwen35-9b-r32-final-sampler`
- Training steps: 18; development NLL 1.7267 → 0.1113

## Intended use

Development research for masked fantasy draft actions, conservative probability
forecasts, and selective structured depth-chart lookups. It is not a production
claim, gambling model, or evidence of general football expertise.

## Limits

The evaluation seasons were previously opened and are development-only. Historical
backtests remain contamination-suspect. The 2025 gate is still sealed, and no RL
should begin unless the precommitted T1.1 criteria passed. The rule-policy teacher
has only a small, unresolved historical edge over ADP, so imitation can improve
discipline without guaranteeing superior player selection.

In the frozen development evaluation, T1.1 achieved only 52.1% structured answer
coverage, made zero tool calls across 246 opportunities, introduced three illegal
picks, and failed six of eight precommitted decision gates. Its DraftGym mean reward
was 3.04 (median 0.00), versus 12.11 (median 13.72) for the untouched base model.
Twenty-one of 24 T1.1 episodes exactly matched the control reward of zero. The adapter
is therefore a research checkpoint, not a deployable fantasy agent. The evidence says
to revise the SFT mixture and training strength before considering RL.
