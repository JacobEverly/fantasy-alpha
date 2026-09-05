# Fantasy Alpha Qwen3.5-9B Tinker SFT T1.2 canary

Status: **research canary rejected; not a full model release**.

- Base: `Qwen/Qwen3.5-9B`
- Method: rank-32 LoRA, 1e-4 learning rate, one epoch
- Frozen mixture SHA: `39f4ccb5790263f2ae37151f31c6cda88a559742814da44ce5f5e0dc904c5d09`
- Canary sampler: `tinker://392d4d66-8f41-5bd7-8c2f-6e2f4d937c6d:train:0/sampler_weights/fantasy-alpha-t12-representative-canary-v2-final-sampler`
- Structured coverage: 100%; valid tool-call rate: 0% on three unseen opportunities
- 2025 was not used; the full 24-draft development evaluation was not opened

The canary demonstrates reloadable Tinker training and retained response syntax,
but it cannot reliably decide when to call the evidence tool. It must not be served,
used for betting, or advanced to RL as though the conditional tool policy passed.
