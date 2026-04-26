# 2026-04-16 Noise-as-a-Probe Protocol / Asset Contract

## Question

Can the current repo support one honest first smoke for `Noise as a Probe`, and if so, on which local target family?

## Inputs Reviewed

- `workspaces/gray-box/2026-04-16-graybox-new-family-shortlist-refresh-verdict.md`
- `workspaces/gray-box/2026-04-06-pia-start.md`
- `docs/paper-reports/markdown/gray-box/2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models/2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models-refined.md`
- current local SD1.5 base snapshot
- current local target-family Recon LoRA checkpoints

## Honest First Target Family

The shortest honest first target family is:

- `StableDiffusion-v1.5 base`
- plus local target-family LoRA:
  - `celeba_partial_target/checkpoint-25000`

Why this target family wins:

1. the required base snapshot already exists locally
2. the target-family LoRA checkpoint already exists locally
3. current repo truth already uses this contract surface in bounded black-box work
4. this is much closer to the paper's latent-diffusion setting than the current `DDPM/CIFAR10` gray-box line

## Boundary

This is still **not** a paper-faithful reproduction:

- paper base: `SD-v1-4`
- local first contract: `SD1.5 + local target-family LoRA`

So the current contract should be read as:

- `latent-diffusion local canary contract`
- not `paper-faithful benchmark contract`

## Minimum Asset / Interface Checklist

The first smoke requires all of the following:

1. `pretrain base`
   - `D:/Code/DiffAudit/Download/shared/weights/stable-diffusion-v1-5`
2. `target model`
   - local target-family LoRA on `celeba_partial_target/checkpoint-25000`
3. `sample pair`
   - at least one member + one non-member on the same target-family image contract
4. `caption / prompt source`
   - either carried from local metadata
   - or generated surrogate captions under a clearly frozen rule
5. `DDIM inversion path`
   - base-model inversion from image + prompt to semantic initial noise
6. `custom-noise generation path`
   - target-model generation starting from injected initial noise
7. `distance scoring`
   - one frozen image-distance metric for the first canary
8. `calibration plan`
   - explicit note that percentile-threshold calibration is not yet part of the first canary unless a small prior non-member set is frozen

## First Bounded Smoke

The first bounded smoke should be:

- `one member + one non-member interface canary`

It should test only:

1. inversion succeeds on the base model
2. custom initial noise can be injected into target-model generation
3. final distance scores can be emitted for both samples

Success condition:

- one auditable record showing the end-to-end three-stage pipeline exists

It should **not** claim:

- attack quality
- threshold quality
- benchmark parity

## GPU Release Gate

`gpu_release = none` until all of the following are frozen:

1. caption/prompt source rule
2. custom-noise generation implementation path
3. first-canary output schema
4. calibration policy for anything larger than the interface canary

## Verdict

- `contract_verdict = positive but bounded`
- `selected_target_family = SD1.5 + celeba_partial_target/checkpoint-25000`
- `first_smoke = member/non-member interface canary`
- `gpu_release = none`
- `admitted_change = none`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this locks a local experimental contract only.
