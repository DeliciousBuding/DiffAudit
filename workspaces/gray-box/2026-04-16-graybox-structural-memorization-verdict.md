# 2026-04-16 Gray-Box Structural Memorization Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-3 follow-up family review`
- `method_family`: `structural memorization`
- `device`: `gpu-light`
- `decision`: `negative but useful`

## Question

Does the already-implemented local `structural memorization` faithful-approximation smoke justify promotion into the active gray-box family queue, or should it be explicitly frozen as a non-escalating side branch under the current local threat model?

## Executed Evidence

Primary local smoke artifact:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/structural-memorization-smoke-20260415-r1/summary.json`

Local implementation notes reviewed:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-15-graybox-new-family-structural-memorization-feasibility.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-15-graybox-structural-memorization-smoke-note.md`

## Metrics

Observed smoke metrics on the local CelebA target-family approximation:

- `AUC = 0.375`
- `ASR = 0.53125`
- `TPR@1%FPR = 0.0625`

Directionality:

- member mean `SSIM = 0.730527`
- non-member mean `SSIM = 0.750170`

Reconstruction sanity:

- member mean `PSNR = 24.139915`
- non-member mean `PSNR = 24.477987`

Current bounded schedule:

- `0 -> 50 -> 100`

## Verdict

Current verdict:

- `negative but useful`

Reason:

1. the local faithful-approximation smoke is not merely weak; it is currently wrong-direction under the tested setup;
2. non-members preserve slightly higher structure similarity than members on this bounded rung;
3. this means the current `SD1.5 + Recon target LoRA + CelebA target-family` approximation does not justify GPU scale-up or challenger promotion;
4. the result is still useful because it closes one plausible gray-box family branch with a concrete local verdict instead of leaving it in speculative backlog.

## Decision

Current decision:

- do not promote `structural memorization` into the active gray-box challenger queue
- do not spend more GPU on this family under the current local approximation
- if reopened later, require a concrete new hypothesis first:
  - more faithful inversion path
  - early-step sweep around the structural-decay peak
  - stronger structure metric than plain `SSIM`
  - or a setup closer to the paper's larger-scale threat model

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: no sync needed.
- `Research`: this family should now be treated as a frozen side verdict, not as the next gray-box GPU candidate.
