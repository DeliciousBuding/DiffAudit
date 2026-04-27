# 2026-04-16 TMIA-DM GPU Repeat Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window GPU128 repeat`
- `selected_family`: `TMIA-DM late-window long_window`
- `device`: `cuda:0`
- `decision`: `repeat-confirmed gpu challenger`

## Question

After the first positive `TMIA-DM late-window GPU128` rung, does a matched second GPU run confirm that this is a real challenger line rather than a one-off positive?

## Executed Evidence

GPU runs:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r2-seed1/summary.json`

Current reference:

- `PIA GPU128 baseline`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-128/summary.json`

Prior GPU rung verdict:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-tmiadm-gpu-pilot-verdict.md`

## Metrics

`TMIA-DM late-window GPU128`:

- `r1 / seed0`:
  - `AUC = 0.825317`
  - `ASR = 0.769531`
  - `TPR@1%FPR = 0.085938`
- `r2 / seed1`:
  - `AUC = 0.836975`
  - `ASR = 0.769531`
  - `TPR@1%FPR = 0.078125`

Current aligned `PIA GPU128` reference:

- `AUC = 0.817444`
- `ASR = 0.765625`
- `TPR@1%FPR = 0.046875`

Readout:

- both `TMIA-DM` GPU repeats beat `PIA GPU128` on `AUC`;
- `ASR` stayed above `PIA` on both repeats;
- the useful signal still belongs entirely to `long_window`, not to `short_window` or naive fusion.

## Verdict

Current verdict:

- `repeat-confirmed gpu challenger`

Reason:

1. the GPU result held across two matched runs instead of collapsing after the first positive rung;
2. both repeats stayed competitive with, and slightly ahead of, the current `PIA GPU128` baseline;
3. this is enough to promote `TMIA-DM late-window long_window` from a tentative GPU candidate to a real active challenger line;
4. it is still not enough to replace `PIA` as the overall gray-box headline without a broader ladder or defense-side evidence.

Interpretation:

- `TMIA-DM late-window long_window` is now admitted as a genuine gray-box GPU challenger;
- the main remaining question is scaling and comparative stability, not basic viability;
- the next GPU budget should go to one larger rung or a tighter same-protocol comparison, not to rediscovering whether the line is real.

## Decision

Current release decision:

- `promote to active GPU challenger line`
- `keep PIA as current gray-box headline for now`
- `allow one larger GPU rung`

Meaning:

1. `TMIA-DM late-window` has earned continued GPU budget;
2. materials can now describe it as a repeat-confirmed challenger, not just a candidate;
3. headline replacement still needs more than one rung.

## Next Gate

The next bounded GPU task should be one of:

1. `GPU256` repeat-ladder rung for `TMIA-DM late-window long_window`;
2. a paired `PIA vs TMIA-DM` same-rung comparison note across both `GPU128` and the new rung;
3. a bounded defense interaction check, but only after one more scale rung.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: preserve `conda` env `diffaudit-research` as the admitted CUDA runtime for gray-box GPU work.
- Materials: wording can now say `TMIA-DM late-window long_window` is a repeat-confirmed gray-box GPU challenger, while `PIA` remains the main admitted headline pending broader ladder evidence.
