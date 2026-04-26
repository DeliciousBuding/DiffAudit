# 2026-04-16 Noise-as-a-Probe Implementation Surface Review

## Question

Does the current `Research` codebase already support the first `Noise as a Probe` interface canary, or is there still a missing implementation block?

## Inputs Reviewed

- `workspaces/gray-box/2026-04-16-noise-as-probe-protocol-asset-contract.md`
- `scripts/run_blackbox_semantic_aux_probe.py`
- `scripts/run_structural_memorization_smoke.py`
- `scripts/prepare_clid_local_bridge.py`
- current CLI surface under `src/diffaudit/cli.py`

## What Already Exists

### 1. Target-side latent diffusion loading exists

The repo already has a working local SD1.5 + LoRA loading path:

- base model load
- target-family LoRA attach
- prompt fallback via local metadata or BLIP captioning

### 2. Target-side generation primitives exist

Two useful fragments are already present:

- `run_blackbox_semantic_aux_probe.py`
  - end-to-end SD1.5 target generation with prompt conditioning
- `run_structural_memorization_smoke.py`
  - VAE encode / decode
  - `DDIMScheduler`
  - direct UNet-guided latent stepping

### 3. Distance scoring exists

The repo already contains image-comparison logic such as:

- `SSIM`
- `MAE`
- `MSE / PSNR`

So the final scoring leg is not the main blocker.

## What Is Still Missing

### 1. No explicit DDIM inversion implementation

The current repo does not yet expose a reusable path that:

- takes image + prompt
- performs base-model DDIM inversion
- emits semantic initial noise / latents for later reuse

### 2. No explicit custom-noise target-generation entrypoint

Existing target generation uses:

- normal prompt-based sampling

But the repo does not yet expose a frozen interface for:

- providing externally prepared initial latents/noise
- then running the target model from that exact injected starting point

### 3. No canary-oriented output schema yet

There is not yet a dedicated `Noise as a Probe` canary artifact that records:

- source sample
- prompt source
- inverted latent pointer
- reconstructed output pointer
- distance metrics

## Practical Reading

This means the current implementation surface is:

- `positive but bounded`

Because:

1. the repo already has enough target-side and scoring pieces that the canary is not a fantasy;
2. but the missing inversion + custom-noise interface still prevents an honest end-to-end canary today.

## Verdict

- `implementation_surface_verdict = positive but bounded`
- `ready_now = target loading + generation pieces + distance scoring`
- `missing_block = reusable DDIM inversion + custom-noise target-generation glue`
- `gpu_release = none`
- `next_step = implement a CPU-first Noise-as-a-Probe canary scaffold`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this is implementation-surface truth only.
