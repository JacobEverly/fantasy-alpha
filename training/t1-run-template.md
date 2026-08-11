# T1 SFT run template — resolved from the T0 smoke test (2026-08-09)

T0 (pod `5cdcf034a2804e7e859d2f55e4a96d67`, H100 spot) + T0b (pod `76ce0cb1ca0f4e8fb5c3f3cd16dac6d0`, A100 spot)
proved the full path: provision → install → LoRA SFT → checkpoint-at-interval → resume → adapter load +
generation → download → terminate. Every line below is either a fix T0 needed or a config it verified.
Costs in `docs/budget-ledger.md`.

## Pod provisioning (Prime Intellect API)

- `POST https://api.primeintellect.ai/api/v1/pods/` with `team.teamId = cmskuc3x7018j7g10tr01gukl`
  (bills the Plantmangroup team, not the personal account).
- **Fix:** the API error `Missing field: data_center_id` is real — `pod.dataCenterId` is required even
  though docs mark it optional. Take it from the availability response (`dataCenter` field).
- **Fix:** `prepaidForHr` is rejected for datacrunch spot ("Reserved instances are not available") —
  the auto-terminate guardrail must be an **on-pod self-destruct timer** instead: a nohup'd
  `sleep <budget-seconds>; curl -X DELETE .../pods/<id>` armed as the very first action after SSH works.
  This is what bounds the orphaned-pod risk if the driving agent dies (it did once during T0).
- Spot availability churns: `1H100.80S.30V_SPOT` (FIN-01) existed at 02:23 UTC, was gone by 13:58;
  re-query availability and re-resolve cloudId/dataCenterId at launch time, don't hardcode.
- SSH: account had no keys; `POST /api/v1/ssh_keys/` with the public key (key at
  `~/.ssh/fantasy_alpha_t0` on Jacob's Mac, registered as primary). New pods pick up the primary key
  automatically. Pods reachable ~2 min after create (both runs).
- 150 GB disk is comfortable: prime-rl venv 14 GB + Qwen3-4B + 4 weight snapshots @7.7 GB ≈ 60 GB used.

## Install (on pod, ubuntu_22_cuda_12 image)

```bash
git config --global url."https://github.com/".insteadOf "git@github.com:"   # FIX: submodules use SSH URLs, fail on fresh pods
git clone https://github.com/PrimeIntellect-ai/prime-rl.git && cd prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync --all-extras        # ~5 min; installs CPython 3.12 + torch 2.11.0+cu128
```

- **Fix:** if a submodule clone ever aborts, `git submodule sync && git submodule update --init --force ...`
  recovers the partial state (T0 hit this before the URL rewrite).
- No FA3 build needed; `model.attn = "auto"` works on both H100 and A100.

## Dataset

- `python3 -m training.convert_prime_rl <corpus.jsonl> <out_dir>` → `train.jsonl` + `validation.jsonl`
  in one directory; prime-rl's `data.name = "<dir>"` + `splits = ["train"]` loads it via HF
  `load_dataset` auto-detection. `meta` is carried as a JSON string column (harmless).
- **T1 amendment (GAMEPLAN season split):** replace T0's random val split with the season split —
  **train on 2008–2022 excluding 2013 and 2018; 2013+2018 are the SFT validation seasons** (2024 stays
  eval-side, 2025 stays the sealed gate holdout). `convert_prime_rl.py` must split on `meta.season`
  for T1 (noted in its docstring).
- **verifiers adapter note:** `envs/verifiers_v1_adapter.py` asserts the old train/val season split —
  update its split assertions to the 2008–2022-minus-2013/2018 amendment before wiring val envs as
  live eval sources, or the adapter will reject the T1 taskset.

## Trainer config (verified resolved values, `uv run sft @ <cfg.toml>`)

```toml
max_steps = <2-3 epochs>            # T0: 36 steps = 2 epochs of 150 @ bs 8
output_dir = "/root/t0/outputs"

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"   # T1: swap for the chosen 14-32B base
seq_len = 4096

[model.lora]
rank = 16                           # risk-register pin (rank 16, alpha 32)
alpha = 32.0

[data]
type = "sft"                        # REQUIRED discriminator when [val] is present
name = "/root/t0/dataset"
splits = ["train"]
batch_size = 8
micro_batch_size = 1
seq_len = 4096

[val]
interval = 6
eval_on_start = true
[val.data]                          # full mirror of [data] with splits = ["validation"]
type = "sft"
name = "/root/t0/dataset"
splits = ["validation"]
batch_size = 8
micro_batch_size = 1
seq_len = 4096

[optim]
lr = 5e-5                           # LoRA lr; loss fell smoothly, grad norms 3.2 -> 0.6

[ckpt]                              # GAMEPLAN mandate: checkpointing ON (prime-rl defaults it OFF)
interval = 12
[ckpt.weights]
save_adapter_separately = true      # writes weights/step_N/lora_adapters/ in PEFT format

[file_monitor]                      # local metrics.jsonl sink (loss log without W&B)
```

- **Loss masking:** prime-rl's default `LossMaskConfig` is already assistant-only
  (`system/user/tool = false`). Verified in the resolved dump — evidence text in user messages is
  never a completion target (licensing requirement, training/README.md). Always `--dry-run` first and
  check `[data.loss_mask]` in `outputs/configs/sft.toml`.
- **Checkpoint layout:** `checkpoints/step_N/` (full trainer state, resumable) +
  `weights/step_N/` (merged HF model, 7.7 GB for 4B) + `weights/step_N/lora_adapters/`
  (`adapter_config.json` + `adapter_model.safetensors`, ~120 MB — the thing to download).
- **Resume:** `--ckpt.resume-step <N>` (or `-1` = latest) with a bumped `--max-steps` resumes cleanly;
  T0 verified loss continuity (val 1.72 @ step 24 → 1.70 @ step 25 post-resume).
- **Sampling from a checkpoint:** `uv run --with peft python training-side script`; note
  `transformers` in prime-rl's venv rejects `device_map` without accelerate and `torch_dtype` is
  deprecated — use `.from_pretrained(base, dtype=torch.bfloat16).to("cuda")`. peft is NOT in the
  prime-rl venv; add it ad hoc (`uv run --with peft`) or merge the adapter manually
  (`W += B @ A * alpha/r`; adapter keys are `base_model.model.<path>.lora_{A,B}.weight`).

## Monitoring (docs/run-monitoring.md is binding)

- Loss/val curves in `outputs/metrics.jsonl` (file_monitor) and the trainer log.
- T0 reference curve (healthy): train 3.05 → 1.37, val 3.06 → 1.55 over 36 steps, monotonic,
  grad norms 3.2 → 0.65, no NaN, ~5.7 s/step on H100 (11.7 s on A100), peak mem 17.8 GB (4B LoRA @ bs 8×4096 packed).
- Kill rules: first-10-min pathology → kill+fix; val rising 2 consecutive evals → stop, keep best checkpoint.

## Teardown

- `DELETE https://api.primeintellect.ai/api/v1/pods/<id>`, then **verify** `status == TERMINATED` via
  GET and confirm the account-wide active-pod list is empty. Disarm/ignore the self-destruct timer —
  a DELETE on an already-terminated pod is a no-op.

## Lesson from T0 metrics review (2026-08-09)
- Grad-norm spike at step 25 (4.27, ~6x median) was benign (loss unaffected, no recurrence, clipping bounded it) — but T1 should log batch/sample indices whenever grad_norm > 4x running median, so spikes can be traced to specific training examples (a cheap data-quality debugger; a recurring spike at the same batch = inspect those traces for filter escapes).
