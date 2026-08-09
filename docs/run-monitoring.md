# Run monitoring & kill criteria (preregistered)

2026-08-08. Binding for T0–T3. Purpose: detect a failing run at the smallest possible spend. Checkpointing is ON for every run (prime-rl defaults it off — run template pins it), so "kill" always means "stop and keep the best checkpoint," never "lose everything."

## SFT runs (T1-class, ~$300, hours)

| When | Watch | Kill/act rule |
|---|---|---|
| First 10 min (~$5) | Train loss falling smoothly; no NaN/spikes; grad norms stable; tokens/s sane | Any pathology → kill, fix config, restart. A broken template/masking/LR shows here |
| First checkpoint (~30 min) | 20-question smoke: format compliance + Brier vs base model on CalibBench sample | Format broken or Brier worse than base → pause and investigate before continuing |
| Every eval interval | Val loss (held-out 2024 traces) vs train loss | Val rising 2 consecutive evals → stop, take best checkpoint (done, not failed) |

Max money at risk between signals: single-digit dollars.

## RL runs (T2-class, $1–1.5k, 1–2 days)

Dashboard watch list (all logged by prime-rl; val envs = 2023–24 seasons wired as live eval sources):
1. **Train reward** — must move within the first few hundred episodes (rollouts are cheap).
2. **Val-env reward** — the number that matters.
3. **Train−val gap** — widening = memorization/sim-hacking; the primary tell.
4. **Hindsight−realistic scoring gap** per episode — rising = lottery-ticket farming.
5. **Entropy** (collapse) and **KL-to-reference** (explosion) — the classic GRPO pathologies.
6. **Illegal-action/fallback rate** (should fall) and tool-call usage (should not degenerate to 0 or max).
7. **Canary gate** on periodic checkpoints — a FAIL voids the checkpoint's anonymized-track claims.

Budget-fraction gates (preregistered; evaluated at cumulative spend):
- **10% (~$100–150):** train reward above the autopick noise band OR legality/format metrics clearly improving. Else kill.
- **30% (~$400):** val reward improved with CI clear of zero. Else kill.
- **60% (~$900):** val still improving; if plateaued, stop at plateau (success = best checkpoint + knowledge of the ceiling).
- Any time: NaN, entropy collapse, KL runaway, or train−val divergence beyond preset band → kill.

## Who watches

Claude Code monitors the run on a schedule (poll logs/metrics via the pod/API), reports at each gate, and executes the preregistered kills without renegotiating them mid-run. Jacob gets pinged at every gate and on any kill.

## Honest limit

Early signals catch catastrophic failure and gross stagnation for cents on the dollar. They cannot promise the *final* quality of a healthy-looking run — a run can be green throughout and plateau at +ε. That residual risk is why the ladder is sized $50 → $300 → $1–1.5k and why each rung requires the previous one's measured result.
