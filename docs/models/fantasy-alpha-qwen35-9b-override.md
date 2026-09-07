# Fantasy Alpha Qwen3.5-9B high-confidence override adapter

## Intended use

This adapter is a narrow decision policy for a masked fantasy-football draft
state. It receives the five best legal candidates and a small set of
point-in-time features. It returns either `KEEP_ADP` or one specific
`OVERRIDE` candidate. It is not a player projection model and does not attempt
to draft a complete team from general football knowledge.

## Base model and training

- Base: `Qwen/Qwen3.5-9B`
- Method: rank-16 LoRA through Tinker
- Renderer: `qwen3_5_disable_thinking`
- Completed schedule: one epoch, batch size 32, learning rate 1e-4, 33 steps
- Corpus: 1,243 masked historical decision states
- Targets: 1,187 keep decisions and 56 override decisions
- Grouping: complete draft episodes remain together in train or development

The teacher is the frozen 80/80 risk-aware ranker. Realized roster outcomes
were used to train that teacher but do not appear in the language model's
input. Positive overrides receive weight 8; 170 near-boundary keep decisions
receive weight 2; other keep decisions receive weight 1.

## Evaluation contract

The untouched base and adapter will run the same fifteen 2025 drafts with the
same seats, opponent seeds, prompt, decoding settings, and legal-action
fallback. Evaluation reports structured-output validity, teacher agreement,
false overrides, intervention rate, completed-roster value relative to ADP,
and exact token-based cost. The final checkpoint is fixed because the run has
one epoch; 2025 is not used for selection.

## Results

- Training runtime: 268 seconds
- Development NLL: 0.0979 to 0.0316
- Reload/export: verified
- 2025 structured-output rate: 100%
- Teacher override recovery: 0 of 3
- False overrides: 5 of 222 teacher-keep states (2.25%)
- Mean completed-roster value versus ADP: −29.49 points over 15 drafts
- Frozen gate: failed
- Exact incremental training and evaluation cost: $1.2744

The adapter learned the response schema but not the intended high-confidence
boundary. It repeatedly chose candidate four at the first pick from slot one,
an apparent shortcut in a corpus where most override labels were concentrated
in early rounds and edge draft slots.

## Limitations

- The teacher's strong 2021–2022 result was observed on an already-opened
  panel and is provisional.
- The teacher failed on a backward 2013 stress test, which suggests meaningful
  era sensitivity.
- Historical replay is not a guarantee of future-season performance.
- This model can imitate a policy boundary without independently understanding
  why the underlying counterfactual simulator assigned value.
- Invalid or illegal output must fall back to ADP.

## Safe fallback

The adapter failed its frozen behavioral and product-safety gates and should
not be used for draft recommendations. ADP is the product fallback. The
deterministic teacher is retained only as the reproducible source of the SFT
labels because it also lost to ADP on the 2025 forward test.
