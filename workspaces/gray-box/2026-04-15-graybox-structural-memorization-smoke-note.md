# 2026-04-15 Gray-Box Structural Memorization Smoke Note

## Status Panel

- `owner`: `research_leader`
- `task_scope`: `P1-GA-3 / P1-GA-4 / P1-GA-5`
- `family`: `structural memorization`
- `run_id`: `structural-memorization-smoke-20260415-r1`
- `gpu_status`: `completed`

## What Was Implemented

Implemented a bounded local faithful-approximation smoke for the structural-memorization family:

1. reused the prepared local CelebA member / non-member imagefolders;
2. loaded staged `SD1.5` plus the same target-side Recon LoRA used by the local black-box bridge;
3. used cached metadata captions, with local `BLIP` fallback available if captions are missing;
4. encoded images to latent space;
5. applied bounded DDIM-style forward corruption with schedule:
   - `0 -> 50 -> 100`
6. decoded the corrupted latent back to image space;
7. used image-space `SSIM` as the membership score.

Implementation artifact:

- `<DIFFAUDIT_ROOT>/Research/scripts/run_structural_memorization_smoke.py`

Run artifact:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/structural-memorization-smoke-20260415-r1/summary.json`

## Smoke Configuration

- dataset:
  - local CelebA target-family stack
- member subset:
  - `16`
- non-member subset:
  - `16`
- total timestep:
  - `100`
- interval:
  - `50`
- guidance scale:
  - `7.5`

## Result

Observed smoke metrics:

- `AUC = 0.375`
- `ASR = 0.53125`
- `TPR@1%FPR = 0.0625`

Score means:

- member mean `SSIM = 0.730527`
- non-member mean `SSIM = 0.750170`

Reconstruction sanity:

- member mean `PSNR = 24.139915`
- non-member mean `PSNR = 24.477987`

## Interpretation

This smoke does not merely show “weak signal.”

It currently shows the wrong signal direction:

- non-members have slightly higher structure-preservation scores than members on this bounded local setup.

That means the current local faithful approximation is not ready for GPU scale-up.

Most likely explanations:

1. the local target-family CelebA setup is not close enough to the paper's large-scale text-to-image pretraining regime;
2. the bounded `0 -> 50 -> 100` schedule is too crude to reproduce the paper's structure-decay behavior;
3. the current approximation may need a more faithful inversion path or a different image-space reconstruction comparison.

## Verdict

Classification:

- `negative but useful`

Promotion decision:

- do **not** promote this line to a mainline GPU comparator yet

Roadmap interpretation:

- `P1-GA-3`: completed
- `P1-GA-4`: completed with negative result
- `P1-GA-5`: completed with `no mainline escalation`

## Next Best Move

Do not spend more GPU on this family immediately.

If this line is reopened later, it should only be with a concrete new hypothesis such as:

1. a more faithful DDIM inversion implementation;
2. a timestep sweep around the early structural peak;
3. a stronger image-structure metric than plain `SSIM`;
4. a setup closer to the paper's pretraining-scale threat model.
