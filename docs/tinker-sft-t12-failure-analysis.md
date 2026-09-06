# T1.1 collapse analysis — inputs to T1.2

This analysis uses only existing development artifacts. The 2025 holdout remains sealed.

## What the data proves

- T1 supplied 238,677 supervised answer tokens; T1.1 supplied only 7,762. Despite 300 of 1,111 source rows being targeted, they represented just 3.1% of the unweighted answer-token signal.
- There are 120 same-prompt/different-answer forecast pairs. Keeping both would train contradictory targets.
- T1.1 NLL fell 93.6%, yet structured coverage was 52.1%, 8 prediction families failed, and 3 illegal picks appeared.
- T1.1 made 0 tool calls over 246 opportunities; 21/24 drafts exactly matched the control reward.
- Of 150 draft labels, 78.0% select the visible ADP leader. This is a direct descriptive proxy, not proof that the rule teacher equals the environment's stochastic control policy.

## Evidence-backed T1.2 response

T1.2 removes conflicting old forecast targets, weights by assistant-token loss rather
than row count, lowers the learning rate from 3e-4 to 1e-4, uses one epoch, and selects
checkpoints by format, legality, tool behavior, and task quality before NLL.

## Hypotheses, not established facts

- Two epochs at 3e-4 on the narrow corpus likely over-specialized the adapter.
- Row-count balancing hid the much smaller targeted assistant-token signal.
- Selecting only by NLL favored memorization and ignored retained format behavior.
- Rule-policy imitation can teach discipline but cannot guarantee reward improvement.
