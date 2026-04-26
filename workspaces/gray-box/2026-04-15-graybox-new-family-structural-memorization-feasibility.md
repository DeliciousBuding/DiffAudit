# 2026-04-15 Gray-Box New Family Feasibility Note

## Status Panel

- `owner`: `research_leader`
- `task_scope`: `P1-GA-1 / P1-GA-2`
- `selected_family`: `structural memorization`
- `track`: `gray-box`
- `gpu_status`: `not started`

## Decision

For the next gray-box new-family line, choose:

- `structural memorization`

Do not choose, for now:

- `SIDE`
- `MoFit`
- `Noise-as-a-probe`
- `SIMA`

## Why This Family Wins First

This family is the best first `P1-GA` candidate under the current repo state because:

1. it is materially different from the existing gray-box mainline:
   - `PIA` and `SecMI` are noise / posterior-estimation families;
   - `structural memorization` is an image-structure preservation family.
2. it fits the current local asset stack:
   - staged `SD1.5` base exists;
   - local CelebA member / non-member imagefolder assets already exist;
   - local BLIP weights already exist;
   - the current `CLiD` bridge already proves we can drive `encoder / decoder / DDIM-style` paths on the same target-side setup.
3. it is still a membership inference line, not a neighboring extraction-only story.

## Why Not The Others First

### `SIDE`

`SIDE` is valuable, but it is not the right first `P1-GA` move.

Reason:

- the paper is primarily a white-box training-data extraction line;
- its large-model path depends on surrogate-condition construction plus LoRA-based conditional adaptation;
- this is a bigger engineering and threat-model jump than we need for the first new gray-box family verdict.

So `SIDE` should stay in the challenger queue, not be the first low-friction implementation target.

### `MoFit / SIMA / Noise-as-a-probe`

These may still be worthwhile later, but they are currently weaker first choices because:

- we do not yet have a more complete local implementation bridge for them;
- current local paper material and asset readiness are less mature than the structural-memorization line;
- they are more likely to require additional protocol reconstruction before even reaching a clean smoke.

## Mechanism Summary

Core hypothesis from the paper:

- members preserve coarse structure better than non-members during early forward corruption;
- this signal can be measured with a structure-level score instead of a noise-prediction score.

Proposed local approximation for DiffAudit:

1. take an input image `x0`;
2. obtain or approximate a text condition using local `BLIP`;
3. run the target model through a bounded forward-corruption / DDIM-inversion-style path on the same target-side setup already used by the local `CLiD` bridge;
4. decode the corrupted latent back into image space;
5. compute a structure score such as `SSIM(x0, x_t)`;
6. use that score as the membership statistic.

## Local Feasibility

Current local assets that make this feasible:

- base model:
  - `D:\Code\DiffAudit\Download\shared\weights\stable-diffusion-v1-5`
- caption model:
  - `D:\Code\DiffAudit\Download\shared\weights\blip-image-captioning-large`
- prepared local target-family image datasets:
  - `D:\Code\DiffAudit\Research\workspaces\black-box\runs\clid-paper-align-asset-prep-20260415-r1\datasets\member`
  - `D:\Code\DiffAudit\Research\workspaces\black-box\runs\clid-paper-align-asset-prep-20260415-r1\datasets\nonmember`
- current target-side LoRA:
  - `D:\\Code\\DiffAudit\\Download\\black-box\\supplementary\\recon-assets\\ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models\model-checkpoints\celeba_target\checkpoint-25000`

Local implementation leverage:

- we can reuse the staged `SD1.5` loading path already hardened by the `CLiD` bridge;
- we can reuse local dataset export and imagefolder mechanics already exercised by the black-box mitigation probe;
- we only need to add:
  - `BLIP` caption generation or caption reuse;
  - bounded forward-corruption logic;
  - `SSIM` scoring and threshold finalization.

## Main Risks

Main risks before implementation:

1. the exact paper protocol is not fully open-sourced, so the first implementation should be treated as a faithful local approximation, not a paper-faithful claim;
2. the strongest results in the paper were shown on larger-scale text-to-image setups, so a local CelebA target-family rung may understate or distort the signal;
3. `SSIM` may be too brittle unless the corruption step is chosen carefully;
4. BLIP caption quality may add variance, although the paper explicitly argues prompt robustness is one of the advantages of this family.

## Minimum Prototype Plan

CPU-side implementation target for `P1-GA-3`:

1. create a new gray-box workspace for `structural memorization`;
2. reuse the prepared member / non-member imagefolders from the local target-family CelebA stack;
3. implement a bounded scorer that:
   - captions each image with local `BLIP` or uses cached captions;
   - runs one early corruption step or a short bounded schedule;
   - decodes the corrupted image;
   - computes `SSIM(original, corrupted)`;
4. emit `train/test` score files and `summary.json`.

## Suggested Smoke Shape

First smoke:

- `32 / 32` member/non-member subset
- same local target-family CelebA stack
- single early-step setting first
- one bounded GPU run only after the CPU-side script and pathing are already staged

Promotion criterion:

- any stable separation above random with a credible explanation is enough to keep this line alive;
- if the signal is absent or obviously confounded, mark it `negative but useful` early and stop escalation.

## Queue Update

- `current task`: `P1-GA-1 / P1-GA-2`
- `next GPU candidate`: `structural memorization smoke on local CelebA target-family subset`
- `current CPU task after this note`: implement the bounded scorer for `P1-GA-3`

