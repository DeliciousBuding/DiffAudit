# 2026-04-16 PIA 2048 CDI Rung Verdict

## Question

Does the active `PIA 2048 shared-score surface` rung land as a useful extension for the current `CDI` lane, or should it be treated as an over-expensive no-go?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1\scores.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-cdi-feature-collection-surface-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-cdi-internal-canary-verdict.md`

## What Landed

Run root:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1`

Key metrics:

- `AUC = 0.833109`
- `ASR = 0.769043`
- `TPR@1%FPR = 0.050293`
- `member_score_mean = -12.958884`
- `nonmember_score_mean = -32.214287`
- `wall_clock_seconds = 1723.22116`

## Comparison Against 1024 Reference

Reference rung:

- `pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive`
- `AUC = 0.838630`
- `ASR = 0.782715`
- `TPR@1%FPR = 0.048828`
- `wall_clock_seconds = 494.968358`

Interpretation:

1. The attack signal stays alive at `2048 / 2048`; the line does not collapse.
2. Headline quality is slightly below the `1024` rung on `AUC` and `ASR`, but still very close.
3. Low-FPR behavior is also essentially stable.
4. Runtime cost is the real warning:
   - `1723.2s / 495.0s ≈ 3.48x`
   - that is materially heavier than the cleanest naive `2x` expectation.

## Verdict

- `pia_2048_cdi_rung_verdict = positive but cost-heavy`
- the rung is worth keeping as a reusable `PIA` shared-score surface for `CDI` paired follow-up
- but it does **not** justify more open-ended same-family `PIA` scaling by itself
- current honest reading:
  - `signal survives`
  - `cost warning is real`

## Carry-Forward Rule

- keep this rung as a reusable paired-feature surface for `CDI`
- do not immediately scale `PIA` larger again
- the next highest-value GPU move is now:
  - one bounded `SecMI` export / disagreement run on the same `2048` subset using this new `PIA` score surface

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

