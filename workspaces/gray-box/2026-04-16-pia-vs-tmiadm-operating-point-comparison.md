# 2026-04-16 PIA Vs TMIA-DM Operating-Point Comparison

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-4 / gray-box operating-point comparison`
- `methods`: `PIA` vs `TMIA-DM late-window long_window`
- `devices`: `GPU128` and `GPU256`
- `decision`: `headline depends on operating point`

## Question

Now that `TMIA-DM late-window long_window` is repeat-confirmed on GPU, what does it actually beat `PIA` on, and where does `PIA` still remain stronger?

## Executed Evidence

`PIA` references:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-128/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-256/summary.json`

`TMIA-DM late-window` references:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r2-seed1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r2-seed1/summary.json`

## Metrics

### GPU128

- `PIA`:
  - `AUC = 0.817444`
  - `ASR = 0.765625`
  - `TPR@1%FPR = 0.046875`
  - `TPR@0.1%FPR = 0.039062`
- `TMIA-DM r1`:
  - `AUC = 0.825317`
  - `ASR = 0.769531`
  - `TPR@1%FPR = 0.085938`
  - `TPR@0.1%FPR = 0.0625`
- `TMIA-DM r2`:
  - `AUC = 0.836975`
  - `ASR = 0.769531`
  - `TPR@1%FPR = 0.078125`
  - `TPR@0.1%FPR = 0.023438`

### GPU256

- `PIA`:
  - `AUC = 0.841293`
  - `ASR = 0.78125`
  - `TPR@1%FPR = 0.039062`
  - `TPR@0.1%FPR = 0.019531`
- `TMIA-DM r1`:
  - `AUC = 0.839554`
  - `ASR = 0.765625`
  - `TPR@1%FPR = 0.117188`
  - `TPR@0.1%FPR = 0.066406`
- `TMIA-DM r2`:
  - `AUC = 0.837814`
  - `ASR = 0.787109`
  - `TPR@1%FPR = 0.050781`
  - `TPR@0.1%FPR = 0.015625`

## Verdict

Current verdict:

- `headline depends on operating point`

Reason:

1. at `GPU128`, `TMIA-DM late-window` beats `PIA` across `AUC`, `ASR`, and `TPR@1%FPR`;
2. at `GPU256`, `PIA` keeps a slight edge on `AUC`, while `TMIA-DM` is near-parity on `ASR` and often stronger on low-FPR behavior;
3. this means the gray-box story has split into two meaningful operating regimes:
   - `PIA` remains the safer headline if the project wants a single stable global-ranking metric story;
   - `TMIA-DM late-window` becomes especially attractive if the project emphasizes low-FPR operating points or challenger diversity.

Interpretation:

- `PIA` is no longer the only serious gray-box line;
- `TMIA-DM late-window long_window` is a real competitor, but not a universal replacement;
- the project should frame them as two distinct gray-box operating-point stories rather than force an artificial single winner.

## Decision

Current narrative decision:

- `PIA` stays admitted gray-box headline
- `TMIA-DM late-window` becomes the strongest active challenger line
- `materials should explicitly separate headline-metric and low-FPR framing`

Meaning:

1. keep `PIA` as the stable mainline reference for now;
2. treat `TMIA-DM` as the strongest new-family challenger with credible low-FPR advantages;
3. do not collapse the comparison into a single scalar winner claim.

## Next Gate

The next bounded task should be one of:

1. first defense interaction check for `TMIA-DM late-window`;
2. one larger rung only if needed for materials or final mainline arbitration;
3. a structured summary artifact that shows `PIA` and `TMIA-DM` side by side by operating point.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: update gray-box wording to “`PIA` headline + `TMIA-DM late-window` low-FPR challenger,” not “`PIA` only”.
