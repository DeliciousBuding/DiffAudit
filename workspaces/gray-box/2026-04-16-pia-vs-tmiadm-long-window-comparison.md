# 2026-04-16 PIA Vs TMIA-DM Long-Window Comparison

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-4.1 / same-split gray-box comparison`
- `methods`: `PIA` vs `TMIA-DM long_window`
- `gpu_status`: `not requested`
- `decision`: `PIA clearly remains ahead`

## Question

After `TMIA-DM long_window` survived one bounded repeat, does it now compete with the current `PIA` gray-box mainline on the same local `CPU-32` split?

## Executed Evidence

Aligned local references:

- `PIA cpu-32 baseline`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-cpu-32/summary.json`
- `TMIA-DM long_window r1`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-protocol-probe-20260416-cpu-32-r1/summary.json`
- `TMIA-DM long_window r2 / seed1`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-protocol-probe-20260416-cpu-32-r2-seed1/summary.json`

Supporting repeat verdict:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-tmiadm-long-window-repeat-verdict.md`

## Metrics

Aligned `cpu-32` comparison:

- `PIA`:
  - `AUC = 0.782227`
  - `ASR = 0.765625`
  - `TPR@1%FPR = 0.09375`
- `TMIA-DM long_window r1`:
  - `AUC = 0.702148`
  - `ASR = 0.703125`
  - `TPR@1%FPR = 0.03125`
- `TMIA-DM long_window r2`:
  - `AUC = 0.663086`
  - `ASR = 0.671875`
  - `TPR@1%FPR = 0.0625`

Observed gaps versus `PIA`:

- `AUC gap`:
  - `PIA - TMIA-DM r1 = 0.080079`
  - `PIA - TMIA-DM r2 = 0.119141`
- `ASR gap`:
  - `PIA - TMIA-DM r1 = 0.0625`
  - `PIA - TMIA-DM r2 = 0.09375`

## Verdict

Current verdict:

- `PIA clearly remains ahead`

Reason:

1. `PIA` beats `TMIA-DM long_window` on every headline metric across both bounded repeats;
2. `TMIA-DM long_window` is real and repeat-positive, but it does not yet challenge `PIA` as the gray-box headline;
3. the current evidence therefore supports `TMIA-DM` as a secondary corroborative branch, not a mainline replacement and not yet a disagreement-exploitation line.

Interpretation:

- this comparison is enough to close the narrow question of whether `TMIA-DM long_window` should immediately overtake `PIA`;
- the answer is `no`;
- future `TMIA-DM` work should focus on bounded refinement, not premature promotion.

## Decision

Current release decision:

- `PIA stays gray-box headline`
- `TMIA-DM long_window stays secondary refinement branch`
- `no GPU release from comparison alone`

Meaning:

1. `GB-4.1` can be treated as completed at the run-level comparison layer;
2. `GB-4.2` and `GB-4.3` should only be reopened if a stronger second family or per-sample disagreement hypothesis appears;
3. the shortest next gray-box task remains a bounded `TMIA-DM long_window` refine/repeat, not a role swap with `PIA`.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording should keep `PIA` as the gray-box mainline and describe `TMIA-DM long_window` as a promising but still secondary corroborative branch.
