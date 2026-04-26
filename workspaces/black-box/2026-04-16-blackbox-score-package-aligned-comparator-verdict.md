# 2026-04-16 Black-Box Score-Package Aligned Comparator Verdict

## Task

- `BB-6` bounded aligned semantic-aux comparator on the selected `Recon` contract surface

## Question

- After selecting `Recon DDIM public-50 step10` as the first alignment target, does `semantic-auxiliary-classifier` still carry real signal on that exact contract surface, or does it collapse when moved off the local CelebA target-family comparator?

## Executed Evidence

Prepared aligned dataset export:

- `workspaces/black-box/runs/semantic-aux-public50-step10-aligned-prep-20260416-r1/datasets/member`
- `workspaces/black-box/runs/semantic-aux-public50-step10-aligned-prep-20260416-r1/datasets/nonmember`

Aligned bounded comparator:

- `workspaces/black-box/runs/semantic-aux-public50-step10-aligned-20260416-r1/summary.json`
- `workspaces/black-box/runs/semantic-aux-public50-step10-aligned-20260416-r1/outputs/records.json`

Contract surface:

- `derived-public-50 target_member / target_non_member`
- `celeba_partial_target/checkpoint-25000`
- `num_inference_steps = 10`
- `subset = 16 / 16`

## Result

Observed metrics:

- `AUC = 0.859375`
- `ASR = 0.8125`
- `TPR@1%FPR = 0.625`
- `member mean_cos = 0.507871`
- `nonmember mean_cos = 0.392879`
- `gpu_hours = 0.0624`

Comparison against the frozen `Recon` best-single reference:

- current `Recon DDIM public-50 step10` best-single reference:
  - `AUC = 0.866`
- aligned semantic-aux bounded comparator:
  - `AUC = 0.859375`

## Verdict

- `positive but bounded`

The aligned semantic-aux signal survives the contract shift:

1. it does not collapse when moved onto `public-50 step10 / celeba_partial_target`;
2. it remains close to the current `Recon` best-single reference;
3. but this result alone is not yet a score-package win, because the cross-method package itself has not been tested on the same aligned scalar surface.

## Decision

Current decision:

- keep `BB-6` open
- treat the aligned comparator as `execution-positive`
- do not claim package gain yet
- next bounded step is to extract or define the matching `Recon` single-score surface on the same `16 / 16` aligned split, then test one simple package rule

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: if mentioned, say only that semantic-aux remains viable after alignment to the `Recon public-50 step10` contract surface; do not call it a packaged black-box result yet.
