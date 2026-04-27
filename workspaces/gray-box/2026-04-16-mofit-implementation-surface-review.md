# 2026-04-16 MoFit Implementation-Surface Review

## Question

Given the current repo state, should the first `MoFit` implementation step extend an existing latent-diffusion script, or should it start from a dedicated method-specific scaffold?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-protocol-asset-contract.md`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_structural_memorization_smoke.py`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_blackbox_semantic_aux_probe.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-noise-as-a-probe-implementation-surface-review.md`

## What Already Exists

### 1. Caption bootstrap path exists

Existing scripts already implement:

- metadata caption reuse
- `BLIP` fallback caption generation

This is already enough for the first caption-free initialization layer.

### 2. Text-conditioned latent-diffusion path exists

`run_structural_memorization_smoke.py` already exposes:

- `AutoencoderKL`
- `CLIPTokenizer`
- `CLIPTextModel`
- `UNet2DConditionModel`
- `DDIMScheduler`
- image encode / latent decode / prompt encode

So MoFit does not need a fresh latent-diffusion loading stack.

### 3. Target-side generation path also exists

`run_blackbox_semantic_aux_probe.py` already proves the repo can:

- load `SD1.5 + target-family LoRA`
- run prompt-conditioned generation
- use the same caption bootstrap surface

## What Is Still Missing

### 1. No reusable surrogate optimization loop

There is still no current script that:

- starts from one query image / latent
- iteratively optimizes a model-fitted surrogate under null condition
- records the optimization trace as a reusable artifact

### 2. No reusable fitted-embedding optimization loop

There is also no current script that:

- initializes a condition embedding
- optimizes it directly against the target latent trajectory
- records the fitted embedding plus loss terms

### 3. No MoFit-specific score schema

The repo still lacks one dedicated artifact schema for:

- prompt source
- surrogate trace
- fitted embedding trace
- `L_cond`
- `L_uncond`
- final `L_MoFit`

## Decision

Selected implementation path:

- `dedicated MoFit scaffold`

Why:

1. extending `run_structural_memorization_smoke.py` would overload a method whose core score is `SSIM`-style structure preservation, not surrogate-plus-embedding optimization;
2. extending `run_blackbox_semantic_aux_probe.py` would overload a final-image black-box pipeline that does not own latent optimization loops;
3. `MoFit` has enough unique method surface that a dedicated scaffold is clearer and less likely to pollute existing lanes.

## Next Step

The next bounded step should be:

- `MoFit scaffold selection + minimum output schema`

That step should stay:

- `CPU-only`
- `non-run`
- `gpu_release = none`

## Verdict

- `implementation_surface_verdict = positive but bounded`
- current repo already has the loading and caption substrate needed for a dedicated `MoFit` lane
- current repo still lacks the two core optimization loops and method-specific schema
- `next_step = dedicated MoFit scaffold / schema decision`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this is implementation-surface truth only.
