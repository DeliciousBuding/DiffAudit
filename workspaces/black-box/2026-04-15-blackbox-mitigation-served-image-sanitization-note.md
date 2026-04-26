# 2026-04-15 Black-Box Mitigation Candidate Note: Served-Image Sanitization

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-15 +08:00`
- `task_scope`: `P0-BM-1 / P0-BM-2 / P0-BM-3 / P0-BM-4 / P0-BM-5`
- `selected_candidate`: `served-image-sanitization`
- `gpu_status`: `probe completed`

## Chosen Mitigation Direction

Selected black-box mitigation direction:

- `served-image-sanitization`

Operational definition:

- before images are released to a black-box querier, the service applies a lightweight deterministic post-process
- first probe target:
  - `JPEG recompression`
  - plus optional `downscale -> upscale` sanitization

## Why This Direction Is Realistic

This is a realistic black-box mitigation because:

- hosted image services already compress, resize, or transcode outputs in production;
- the mitigation acts on the only thing a strict black-box attacker necessarily receives:
  - the served final image;
- it does not assume hidden internal access changes or retraining.

So this is closer to:

- deployment-side hardening

than to:

- model retraining
- privileged gray-box defense hooks

## Threat Model

Attacker assumptions for this mitigation line:

- attacker can query the image generation service;
- attacker receives final served images;
- attacker may know or choose prompts / captions depending on the route;
- attacker does **not** get gradients, weights, denoiser activations, or latent trajectories.

In the local repo context, the closest current black-box stack is:

- `Recon` style image-output evidence
- `CLiD clip` local bridge on image/text pairs

This mitigation is therefore targeted at:

- degrading signal recoverable from served images while preserving coarse semantic utility

## Utility Constraints

The mitigation is only interesting if:

1. images remain semantically recognizable;
2. visible quality does not obviously collapse;
3. the post-process is simple enough to plausibly ship in a service layer.

So the first probe should stay mild:

- not a destructive blur baseline
- not a watermark that trivially changes the entire image
- not a retraining-dependent method

## Working Hypothesis

Current black-box evidence lines depend on information present in final served images:

- `Recon` depends on image-level reconstruction-style separation
- `CLiD clip` local bridge consumes image/text pairs after image preprocessing

Hypothesis:

- mild served-image sanitization will preserve coarse semantics but remove or distort some of the fine-grained cues that these black-box scoring paths exploit.

Expected outcome:

- attack metrics should decline more than utility-relevant visual structure does;
- `CLiD clip` is a good first probe target because it already has a runnable local imagefolder-based bridge;
- if the effect is weak on `CLiD`, the same mitigation is unlikely to be a strong black-box hardening story overall.

## Minimal Probe Plan

Probe target:

- current local `CLiD clip` prepared datasets

Probe method:

1. copy the current member/non-member imagefolder datasets;
2. apply deterministic served-image sanitization to both splits;
3. run the same score-finalization logic on the sanitized line;
4. compare against the unsanitized local rung.

First concrete sanitization setting:

- JPEG quality `70`
- resize `512 -> 448 -> 512`

Reason:

- realistic service-side image pipeline perturbation;
- stronger than no-op transcoding;
- still visually plausible.

## Promotion Criterion

Promote this mitigation only if:

1. attack metrics move downward on the sanitized probe;
2. the effect is not trivially explained by catastrophic image destruction;
3. the mitigation can be described as a deployment-side service policy rather than a lab-only hack.

Otherwise classify it as:

- `failed`
- `no-go`
- or `negative but useful`

## Probe Execution

Executed bounded local probe:

1. copied the current prepared `CLiD` member/non-member imagefolder datasets;
2. applied deterministic served-image sanitization to both splits:
   - `JPEG quality = 70`
   - `resize 512 -> 448 -> 512`
3. ran the same local `CLiD clip` scoring bridge on the sanitized datasets;
4. finalized the run into:
   - `workspaces/black-box/runs/clid-served-image-sanitization-probe-20260415-r1/summary.json`

Bounded probe scope:

- `32` member samples
- `32` non-member samples
- same staged `SD1.5` base
- same target-side Recon LoRA

## Probe Result

Sanitized probe summary:

- `AUC = 1.0`
- `ASR = 1.0`
- `TPR@1%FPR = 1.0`

Frozen local CLiD baseline used for comparison:

- `workspaces/black-box/runs/clid-recon-clip-target100-20260415-r1/summary.json`
- baseline metrics:
  - `AUC = 1.0`
  - `ASR = 1.0`
  - `TPR@1%FPR = 1.0`

Observed effect:

- no attack degradation on this bounded probe;
- no evidence that the mitigation changes the ranking story;
- no evidence that the mitigation is competition-usable as a black-box hardening story yet.

## Utility Sanity Check

This should not be misread as “the attack stayed strong only because the images were destroyed.”

Quick member-subset check after sanitization:

- shape preserved for all `32 / 32` samples;
- mean `PSNR = 38.286 dB`;
- minimum `PSNR = 30.82 dB`;
- mean `MAE = 1.879`.

Interpretation:

- the perturbation is real but mild;
- this is not catastrophic image collapse;
- the null result is therefore informative.

## Verdict

Classification:

- `negative but useful`
- current promotion decision: `no-go`

Why:

1. the mitigation did not reduce current local `CLiD` attack metrics;
2. the perturbation remained mild enough that “quality collapse” is not the primary explanation;
3. this is insufficient to justify escalating to a larger black-box mitigation comparator right now.

## Next Best Move

1. keep this mitigation archived as a failed-but-informative deployment-side hardening attempt;
2. move the black-box mitigation queue toward a materially different query-side or serving-side mechanism instead of stronger JPEG-only tuning;
3. keep `Recon / CLiD` baselines frozen and do not overwrite their semantics with this exploratory negative result.
