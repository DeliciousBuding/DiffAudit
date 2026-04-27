# 2026-04-16 TMIA-DM Temporal-Striding Defense GPU128 Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window temporal-striding defense GPU128`
- `selected_family`: `TMIA-DM late-window long_window`
- `defense`: `temporal-striding(stride=2)`
- `device`: `cuda:0`
- `decision`: `repeat-confirmed gpu128 defense candidate`

## Question

After the bounded `cpu-32` positive signal, does `temporal-striding(stride=2)` still weaken `TMIA-DM late-window` when scaled to the first real `GPU128` rung, or does the apparent defense collapse at a more decision-grade budget?

## Executed Evidence

Primary GPU128 rungs:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-temporal-striding-defense-20260416-gpu-128-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-temporal-striding-defense-20260416-gpu-128-r2-seed1/summary.json`

Supporting CPU gate:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-verdict.md`

Current GPU128 references:

- undefended `TMIA-DM late-window`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r1/summary.json`
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r2-seed1/summary.json`
- `stochastic-dropout(all_steps)` defended challenger:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-128-r1/summary.json`
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-128-r2-seed1/summary.json`

## Metrics

Undefended `TMIA-DM late-window GPU128`:

- `r1 / seed0`:
  - `AUC = 0.825317`
  - `ASR = 0.769531`
  - `TPR@1%FPR = 0.085938`
- `r2 / seed1`:
  - `AUC = 0.836975`
  - `ASR = 0.769531`
  - `TPR@1%FPR = 0.078125`

`stochastic-dropout(all_steps)` on the same GPU128 rung:

- `r1 / seed0`:
  - `AUC = 0.809326`
  - `ASR = 0.75`
  - `TPR@1%FPR = 0.078125`
- `r2 / seed1`:
  - `AUC = 0.819397`
  - `ASR = 0.757812`
  - `TPR@1%FPR = 0.023438`

`temporal-striding(stride=2)`:

- effective window: `[80, 120]`
- `r1 / seed0`:
  - `AUC = 0.727234`
  - `ASR = 0.675781`
  - `TPR@1%FPR = 0.046875`
- `r2 / seed1`:
  - `AUC = 0.711609`
  - `ASR = 0.675781`
  - `TPR@1%FPR = 0.015625`

Readout:

- both GPU repeats remain far below the paired undefended challenger;
- both repeats also beat the current `dropout(all_steps)` defended challenger on headline `AUC`;
- low-FPR behavior improves versus undefended and stays competitive with or better than the existing dropout defense.

## Verdict

Current verdict:

- `repeat-confirmed gpu128 defense candidate`

Reason:

1. the defense remained strongly favorable after moving from `cpu-32` to `GPU128`;
2. the second GPU repeat did not collapse, and if anything pushed `AUC` even lower;
3. this is now the strongest `TMIA-DM`-specific defense result in the repo, clearly outperforming `late_steps_only` and not backfiring like `timestep-jitter`;
4. the evidence is still only at `GPU128`, so it is strong enough for a scale-up gate but not yet for immediate defended-mainline promotion.

## Decision

Current release decision:

- `allow one GPU256 scale rung`
- `keep temporal-striding as challenger-specific defended candidate`
- `do not change admitted defended gray-box headline yet`

Meaning:

1. the next task should be a single `GPU256` temporal-striding rung on the exact same `[80, 120]` contract;
2. if the direction survives there, this candidate becomes strong enough for direct defended operating-point comparison;
3. until then, `PIA + stochastic-dropout(all_steps)` remains the admitted defended gray-box headline by continuity.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: no sync yet; this is now a strong internal gray-box defended candidate, but not yet a promoted public claim.
