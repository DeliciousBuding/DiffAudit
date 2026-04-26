# 2026-04-16 DP-LoRA Protocol Overlap Note

## Question

Can `DP-LoRA` be treated as directly comparable to the current admitted white-box line, or only as a successor candidate with partial overlap?

## Inputs

- `workspaces/intake/2026-04-10-dplora-comparability-intake.md`
- `docs/claude-report-4-9-review.md`
- `docs/paper-reports/markdown/survey/2025-aaai-privacy-preserving-lora-membership-inference-latent-diffusion-models/2025-aaai-privacy-preserving-lora-membership-inference-latent-diffusion-models.md`
- `workspaces/intake/2026-04-12-smp-lora-gpu-expansion-program.md`

## Current Admitted White-Box Reference

The current admitted white-box story is still:

- attack line: `GSA`
- defended comparator: `W-1 = DPDM strong-v3 full-scale`
- local protocol center: `DDPM / CIFAR-10`

That stack is already system-consumable and should not be silently replaced.

## Overlap That Is Real

`DP-LoRA` overlaps with the current white-box defense story on three useful axes:

1. `same problem class`
   - both are privacy-defense candidates for diffusion-model membership inference
2. `same evaluation intent`
   - both are judged by whether attack strength moves toward random / weaker membership signal
3. `local bridge signal exists`
   - this repo already has a local `SMP-LoRA under DDPM/CIFAR10` exploratory bridge shape, plus a frozen comparator campaign concept:
     - `baseline vs SMP-LoRA vs W-1 strong-v3`

## Protocol Mismatch That Still Matters

`DP-LoRA` does **not** currently overlap enough to claim same-protocol comparability:

1. `model family mismatch`
   - current admitted white-box line is `DDPM / CIFAR-10`
   - the `DP-LoRA` paper context is `latent diffusion / Stable Diffusion v1.5 / LoRA adaptation`
2. `defense surface mismatch`
   - `W-1` is a full-model DP-style defended comparator
   - `DP-LoRA` is a parameter-efficient adaptation defense family
3. `attack/evaluation protocol mismatch`
   - the paper reports LoRA/SMP-LoRA against attack-model training on auxiliary/test splits and also reports `AUC-0.5`
   - the current admitted local white-box line is centered on `GSA` and direct comparator-style evaluation
4. `utility coupling mismatch`
   - the paper jointly evaluates privacy and generation quality (`FID/KID`)
   - the current admitted white-box local stack does not yet use that same utility contract

## Practical Interpretation

The honest interpretation is therefore:

- `DP-LoRA` is a `successor defense candidate with partial protocol overlap`
- `DP-LoRA` is **not** yet a direct replacement for `W-1`
- the local `SMP-LoRA under DDPM/CIFAR10` bridge is useful only as a future translation layer, not as proof that the paper protocol is already matched

## What This Unlocks

This note is enough to justify `WB-5.2` as the next bounded step:

- write one minimal local config candidate
- keep `gpu_release = none`
- require that any future release packet name the exact attack panel, split contract, metrics, and output schema

## Verdict

- `protocol_overlap_verdict = partial-overlap only`
- `current_verdict = positive but bounded`
- `execution_release = none`
- `gpu_release = none`
- `admitted_change = none`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this note sharpens candidate truth but does not change admitted white-box claims.
