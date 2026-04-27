# 2026-04-16 MoFit Protocol / Asset Contract

## Question

Can the current repo support one honest first `MoFit`-style gray-box smoke on a local latent-diffusion target family, or is the branch still too under-specified to become a real CPU-first live lane?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-post-noise-next-family-reselection.md`
- `<DIFFAUDIT_ROOT>/Research/docs/paper-reports/markdown/gray-box/2026-openreview-mofit-caption-free-membership-inference/2026-openreview-mofit-caption-free-membership-inference-refined.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-noise-as-probe-implementation-surface-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-09-graybox-signal-axis-note.md`
- `<DIFFAUDIT_ROOT>/Research/README.md`

## What MoFit Needs

The paper-level flow requires three concrete capabilities:

1. a local text-conditioned latent-diffusion target family;
2. a caption or caption-embedding initialization path without ground-truth captions;
3. an iterative optimization path for:
   - model-fitted surrogate construction
   - model-fitted embedding extraction

## What Already Exists

### 1. Honest local target family exists

The current repo already has a viable first latent-diffusion target family:

- `SD1.5 + celeba_partial_target/checkpoint-25000`

Why this is honest:

- it is the same local target-family surface already used for:
  - `Noise as a Probe`
  - structural-memorization smoke
  - black-box semantic-auxiliary target-side generation

### 2. Target-side latent-diffusion loading exists

The repo already has:

- base `SD1.5` loading
- target-family LoRA attach
- VAE encode/decode
- scheduler-driven latent stepping

So the branch does not start from zero implementation surface.

### 3. Caption bootstrap assumptions exist

The repo already treats local caption bootstrap as a real dependency surface:

- `README.md` explicitly keeps a `BLIP` captioning route in recommended workflows
- multiple existing latent-diffusion notes already rely on local `BLIP`-style caption recovery or prompt fallback

That is enough for a first contract-first MoFit lane.

## What Is Still Missing

### 1. No explicit model-fitted surrogate optimization entrypoint

The repo does not yet expose a reusable loop that:

- starts from query image / latent
- optimizes a surrogate under null condition
- writes reusable intermediate artifacts

### 2. No explicit model-fitted embedding optimization entrypoint

The repo does not yet expose a reusable path that:

- initializes a condition embedding
- optimizes it against the target latent trajectory
- records the fitted embedding and final score terms

### 3. No MoFit-oriented artifact schema yet

A first honest MoFit smoke would need to emit at least:

- prompt source or caption bootstrap source
- surrogate optimization trace
- fitted embedding trace
- `L_cond`
- `L_uncond`
- final `L_MoFit`-style score

## First Honest Local Contract

Selected first contract:

- `target_family = SD1.5 + celeba_partial_target/checkpoint-25000`
- `caption_source = local BLIP bootstrap or cached local caption fallback`
- `threat_model = gray-box caption-free latent diffusion`

This first contract should stay:

- `CPU-only`
- `contract-first`
- `non-run`

## First Bounded Next Step

The first bounded next step should be:

- write a `MoFit` implementation-surface review / scaffold decision

It should answer:

1. where surrogate optimization should plug into the current target-family latent path;
2. whether caption bootstrap should be cached text, BLIP text, or direct text-embedding initialization;
3. what minimum output schema will be required before any smoke is honest.

## Future GPU Release Rule

`gpu_release = none` until the branch explicitly locks:

1. one local target family;
2. one caption bootstrap source;
3. one surrogate optimization entrypoint;
4. one embedding optimization entrypoint;
5. one bounded smoke question and stop condition.

## Verdict

- `contract_verdict = positive but bounded`
- current repo is ready for a real `MoFit` contract-first lane
- current repo is **not** yet ready for an honest first smoke
- `gpu_release = none`
- `next_step = MoFit implementation-surface review / scaffold decision`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this is candidate-generation and contract truth only.
