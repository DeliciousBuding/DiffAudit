# 2026-04-15 Black-Box New Family Feasibility Note

## Status Panel

- `owner`: `research_leader`
- `task_scope`: `P1-BA-1 / P1-BA-2`
- `selected_family`: `semantic-auxiliary-classifier`
- `track`: `black-box`
- `gpu_status`: `not started`

## Decision

For the next black-box family beyond `Recon + CLiD`, choose:

- `semantic-auxiliary-classifier`

Working definition:

- generate a bounded set of target-model outputs from an image-derived prompt;
- score membership with an offline attack classifier built on semantic similarity / embedding features from returned images.

This is inspired by the face-target black-box line, but adapted to the current local CelebA target-family setting.

## Why This Family Wins First

This is the best next black-box family under the current local asset state because:

1. it is more distinct from `Recon + CLiD` than another prompt-perturbation or reconstruction tweak;
2. it matches the current local target-family asset stack:
   - local CelebA target-family image set;
   - local `SD1.5 + target LoRA`;
   - local `BLIP` weights for prompt bootstrapping when needed;
3. it remains a genuine black-box line:
   - the attack only needs final returned images plus offline post-processing;
   - it does not require denoiser internals, latents, or conditional likelihood access.

## Why Not Another Recon-Like Variant First

Reopening another reconstruction-style branch right now is lower value because:

- `Recon` is already a strong frozen black-box headline;
- the immediate opportunity is method-family diversity, not more same-family tuning;
- `CLiD` already covers a condition-discrepancy style line on the local target-family stack.

So the next black-box family should not be “Recon but with another distance.”

## Threat Model

Attacker assumptions:

1. attacker holds a candidate query image;
2. attacker can obtain or synthesize a usable prompt from the image, using cached text or local `BLIP`;
3. attacker can query the target generation service multiple times with bounded budget;
4. attacker only receives final generated images.

Attacker does **not** get:

- weights
- gradients
- latent trajectories
- denoiser outputs
- any intermediate likelihood estimate

## Local Asset Plan

Primary local assets:

- base model:
  - `D:\Code\DiffAudit\Download\shared\weights\stable-diffusion-v1-5`
- prompt bootstrapping:
  - `D:\Code\DiffAudit\Download\shared\weights\blip-image-captioning-large`
- candidate query splits:
  - `D:\Code\DiffAudit\Research\workspaces\black-box\runs\clid-paper-align-asset-prep-20260415-r1\datasets\member`
  - `D:\Code\DiffAudit\Research\workspaces\black-box\runs\clid-paper-align-asset-prep-20260415-r1\datasets\nonmember`
- target-side fine-tuned adapter:
  - `D:\Code\DiffAudit\Research\external\recon-assets\ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models\model-checkpoints\celeba_target\checkpoint-25000`

Optional feature backbones:

- `CLIP ViT-L/14`
- simple pixel/SSIM diagnostics as auxiliary features, not main identity

## Scoring Rule

First scoring proposal:

1. derive a prompt for each query image:
   - use cached metadata caption when present;
   - otherwise use local `BLIP`;
2. query the target model `k` times with the same prompt;
3. compute a small embedding feature set between the query image and the returned image batch:
   - max similarity
   - mean similarity
   - variance across returned samples
   - best-return vs rest gap
4. train a lightweight offline attack classifier on those returned-image features;
5. use the classifier probability as the membership score.

This keeps the line genuinely different from:

- `Recon`: direct reconstruction-distance thresholding
- `CLiD`: conditional discrepancy from internal model behavior

## First Probe Shape

For the later `P1-BA-3` probe, the smallest credible version should be:

- local CelebA target-family stack
- bounded `k` such as `3` or `4` generations per query
- small member/non-member subset
- feature-space scoring with one simple offline classifier

## Promotion Criterion

Promote this line only if:

1. the bounded probe shows stable separation above random;
2. the attack story stays black-box honest;
3. it adds real family diversity beyond `Recon + CLiD`.

Otherwise keep it as:

- `negative but useful`

## Queue Update

- `current task`: `P1-BA-1 / P1-BA-2`
- `next GPU candidate`: `semantic-auxiliary-classifier black-box probe on local CelebA target-family stack`
- `current CPU task after this note`: implement the returned-image feature extractor and bounded query loop for `P1-BA-3`
