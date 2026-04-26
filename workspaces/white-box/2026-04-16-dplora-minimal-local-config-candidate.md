# 2026-04-16 DP-LoRA Minimal Local Config Candidate

## Question

What is the smallest honest local candidate config for the `DP-LoRA` successor lane, given the current `DiffAudit` white-box stack?

## Inputs

- `workspaces/white-box/plan.md`
- `workspaces/intake/2026-04-12-smp-lora-gpu-expansion-program.md`
- `workspaces/GPU_TRAINING_HANDOVER.md`
- `scripts/train_smp_lora.py`
- `scripts/evaluate_smp_lora_defense.py`

## Candidate Shape

The current minimal local candidate is:

- `candidate_name = SMP-LoRA under DDPM/CIFAR10 local protocol`
- `role = DP-LoRA-family translation candidate, not paper-faithful DP-LoRA reproduction`
- `attack_panel = GSA-only local white-box evaluator`
- `comparison_board = baseline vs SMP-LoRA candidate vs W-1 strong-v3 reference`

## Frozen Candidate Config

The smallest honest local config candidate is:

- `model_family = DDPM / CIFAR-10`
- `defense_form = LoRA-based defended target model`
- `lambda = 0.1`
- `rank = 4`
- `epochs = 10`

Current supporting local evidence:

- undefended baseline:
  - `AUC = 0.5565217391304348`
- best observed local defended point:
  - `lambda=0.1 / rank=4 / epochs=10 / AUC=0.34375`
- degraded alternatives already known:
  - `100 epochs = 0.3784615384615384`
  - `lambda=0.05 salvage = 0.5770308123249299`

## Why This Is The Minimal Candidate

1. It is the strongest already observed local defended point.
2. It stays on the current local `DDPM/CIFAR10` bridge instead of jumping straight to latent-diffusion paper protocol.
3. It is small enough to support comparability hardening before any new GPU review.

## What Is Explicitly Not Frozen Yet

This note does **not** claim that the following are already settled:

- paper-faithful `DP-LoRA`
- same-protocol comparability to `W-1`
- multi-attack panel
- utility-side `FID/KID` contract
- release-ready GPU packet

## Local Execution Contract

The current minimal candidate should keep using the repo's existing local path:

- training entry:
  - `scripts/train_smp_lora.py`
- local evaluation entry:
  - `scripts/evaluate_smp_lora_defense.py`

The candidate should be interpreted as:

- one frozen local defended config to compare against:
  - local undefended baseline
  - `W-1 strong-v3` reference rung

It should **not** be interpreted as permission to reopen the whole historical sweep.

## Verdict

- `minimal_config_verdict = positive`
- `current_verdict = positive but bounded`
- `frozen_local_candidate = lambda0.1-rank4-epochs10`
- `gpu_release = none`
- `admitted_change = none`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this locks a research candidate contract only; it does not change admitted evidence or higher-layer story.
