# 2026-04-16 Black-Box Score-Package Verdict

## Task

- `BB-6.2` test one bounded score package against the best single method
- `BB-6.3` record whether the package changes actionability or only calibration

## Question

- On the first aligned `Recon public-50 step10 / celeba_partial_target / 16-16` contract surface, does a trivial cross-method package actually beat the best single method, or is it only a rescaling of the same ordering?

## Executed Evidence

Aligned semantic-aux source:

- `workspaces/black-box/runs/semantic-aux-public50-step10-aligned-20260416-r1/summary.json`
- `workspaces/black-box/runs/semantic-aux-public50-step10-aligned-20260416-r1/outputs/records.json`

Aligned `Recon` source:

- `experiments/recon-runtime-mainline-ddim-public-50-step10/score-artifacts/target_member.pt`
- `experiments/recon-runtime-mainline-ddim-public-50-step10/score-artifacts/target_non_member.pt`

Current scalar surfaces used:

- `Recon = dim0`
- `semantic-aux = mean_cos`

Bounded package rules tested:

1. z-scored sum
2. z-scored max

## Result

Aligned `16 / 16` same-split metrics:

- `Recon dim0`
  - `AUC = 0.820312`
  - `ASR = 0.78125`
- `semantic-aux mean_cos`
  - `AUC = 0.902344`
  - `ASR = 0.84375`
- `z-score sum`
  - `AUC = 0.933594`
  - `ASR = 0.84375`
- `z-score max`
  - `AUC = 0.929688`
  - `ASR = 0.875`

## Verdict

- `positive but bounded`

This package is more than calibration-only:

1. the best `AUC` now comes from a cross-method package (`z-score sum = 0.933594`);
2. the best `ASR` now also comes from a cross-method package (`z-score max = 0.875`);
3. therefore same-protocol black-box packaging can produce real actionability gain on the aligned bounded split.

## Decision

Current decision:

- close `BB-6.2` as `positive`
- close `BB-6.3` as `positive but bounded`
- keep the result at `first aligned bounded package win`
- do not yet promote it above the frozen `Recon` headline package without a larger aligned confirmation rung

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: if mentioned, phrase this as a bounded aligned package win on `public-50 step10`, not as a replacement for the current black-box headline package.
