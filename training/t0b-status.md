# T0 — COMPLETE (2026-08-09)

All milestones proven, total cost ~$7.98 (both attempts) vs $50 cap:
- Provision → prime-rl install → **LoRA** SFT launch: 9 minutes (first attempt)
- Training: Qwen3-4B-Instruct-2507 on pilot_v0.jsonl — loss 3.05→1.31, val 3.06→1.55, 0 NaNs, grad norms 3.19→0.66 (every docs/run-monitoring.md SFT criterion passed)
- Checkpoints at steps 12/24/36/40 (weights + lora_adapters/ + trainer resume-states)
- Generation from step_40: grounded-reasoning style confirmed (anchors to cohort/base rates, flags data limits)
- Artifacts local: training/checkpoints/t0/{lora_adapters, metrics.jsonl, configs}
- Pod terminated + API-verified both attempts (one via auto-destruct timer, one manual)

Lessons baked into training/t1-run-template.md:
1. Drive pods with short, bounded SSH commands only — long foreground waits stalled the agent twice.
2. `uv` is not on non-interactive SSH PATH → use /root/prime-rl/.venv/bin/python directly.
3. transformers on-pod: no `device_map` without accelerate (`.to("cuda")`), `apply_chat_template(..., return_dict=True)`.
4. Full-weight checkpoint saves ate ~98GB of the 150GB disk by step 40 — T1 must prune old checkpoints or save adapters-only.
5. On-pod self-destruct timer: keep it (saved attempt 1); disarm procedure: pkill -f selfdestruct (verified working).
