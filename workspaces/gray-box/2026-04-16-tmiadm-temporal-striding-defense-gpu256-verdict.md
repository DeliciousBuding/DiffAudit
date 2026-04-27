# 2026-04-16 TMIA-DM Temporal-Striding Defense GPU256 Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window temporal-striding defense GPU256`
- `selected_family`: `TMIA-DM late-window long_window`
- `defense`: `temporal-striding(stride=2)`
- `device`: `cuda:0`
- `decision`: `repeat-confirmed scale-positive defense verdict`

## Question

After the `GPU128` positive gate, does `temporal-striding(stride=2)` remain strongly favorable when scaled to the `GPU256` rung, or does the candidate soften back toward the current dropout-defended challenger?

## Executed Evidence

Primary GPU256 rungs:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-temporal-striding-defense-20260416-gpu-256-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-temporal-striding-defense-20260416-gpu-256-r2-seed1/summary.json`

Supporting gates:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-gpu128-verdict.md`

Current GPU256 references:

- undefended `TMIA-DM late-window`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r1/summary.json`
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r2-seed1/summary.json`
- `stochastic-dropout(all_steps)` defended challenger:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-256-r1/summary.json`
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-256-r2-seed1/summary.json`

## Metrics

Undefended `TMIA-DM late-window GPU256`:

- `r1 / seed0`:
  - `AUC = 0.839554`
  - `ASR = 0.765625`
- `r2 / seed1`:
  - `AUC = 0.837814`
  - `ASR = 0.787109`

`stochastic-dropout(all_steps)` on the same GPU256 rung:

- `r1 / seed0`:
  - `AUC = 0.825867`
  - `ASR = 0.746094`
  - `TPR@1%FPR = 0.058594`
- `r2 / seed1`:
  - `AUC = 0.82164`
  - `ASR = 0.765625`
  - `TPR@1%FPR = 0.039062`

`temporal-striding(stride=2)`:

- effective window: `[80, 120]`
- `r1 / seed0`:
  - `AUC = 0.733322`
  - `ASR = 0.6875`
  - `TPR@1%FPR = 0.070312`
  - `TPR@0.1%FPR = 0.042969`
- `r2 / seed1`:
  - `AUC = 0.7173`
  - `ASR = 0.662109`
  - `TPR@1%FPR = 0.019531`
  - `TPR@0.1%FPR = 0.011719`

Readout:

- both scale rungs remain far below the paired undefended challenger;
- both scale rungs also remain substantially below the current dropout-defended challenger on headline `AUC`;
- low-FPR behavior is mixed but still directionally favorable against undefended, and `seed1` is also clearly below the dropout reference;
- the main signal is now stable enough to treat `temporal-striding` as a real defended candidate rather than a speculative ablation.

## Verdict

Current verdict:

- `repeat-confirmed scale-positive defense verdict`

Reason:

1. the candidate survived both `GPU128` and `GPU256` without collapsing;
2. its `AUC` reductions are materially larger than the current `TMIA-DM` dropout defense;
3. the result now has enough scale and repeat structure to justify direct defended operating-point comparison rather than remaining a sidecar note;
4. this still does not automatically replace `PIA + stochastic-dropout(all_steps)` as the admitted defended gray-box headline, because the defended narrative currently centers on the `PIA` mainline rather than the `TMIA-DM` challenger.

## Decision

Current release decision:

- `promote to defended challenger-comparison candidate`
- `do not yet replace defended gray-box headline`
- `next step = defended operating-point comparison + system sync review`

Meaning:

1. the next highest-value task is now comparison and packaging, not another blind defense search;
2. the repo should explicitly compare `TMIA-DM + temporal-striding` against `TMIA-DM + dropout` and defended `PIA`;
3. only after that comparison should any higher-layer wording be reconsidered.

## Handoff Note

- `Platform`: no direct change needed yet.
- `Runtime`: no direct change needed yet.
- Materials: not ready for direct wording change, but the gray-box story now has a stronger `TMIA`-specific defended branch and should be reviewed for summary-layer sync after the operating-point comparison lands.
