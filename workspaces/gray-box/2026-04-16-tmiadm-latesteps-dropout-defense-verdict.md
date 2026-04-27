# 2026-04-16 TMIA-DM Late-Steps Dropout Defense Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window late_steps_only defense`
- `selected_family`: `TMIA-DM late-window long_window`
- `defense`: `stochastic-dropout(late_steps_only, threshold=100)`
- `device`: `cuda:0`
- `decision`: `too weak to replace all_steps`

## Question

Because `TMIA-DM late-window` concentrates its signal on late timesteps, does a matching `late_steps_only` dropout schedule suppress the challenger more efficiently than the current `all_steps` defended mainline?

## Executed Evidence

Primary late-steps defended rung:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-dropout-latesteps-defense-20260416-gpu-128-r1/summary.json`

Current references:

- `TMIA-DM late-window undefended GPU128`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r1/summary.json`
- `TMIA-DM late-window all_steps defended GPU128`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-128-r1/summary.json`

## Metrics

`TMIA-DM late-window GPU128`:

- undefended:
  - `AUC = 0.825317`
  - `ASR = 0.769531`
  - `TPR@1%FPR = 0.085938`
- `late_steps_only(threshold=100)`:
  - `AUC = 0.820984`
  - `ASR = 0.769531`
  - `TPR@1%FPR = 0.085938`
- `all_steps`:
  - `AUC = 0.809326`
  - `ASR = 0.75`
  - `TPR@1%FPR = 0.078125`

## Verdict

Current verdict:

- `too weak to replace all_steps`

Reason:

1. the late-step schedule only produces a tiny `AUC` reduction relative to the undefended rung;
2. it leaves `ASR` unchanged and leaves `TPR@1%FPR` effectively unchanged;
3. the existing `all_steps` schedule is still materially stronger at weakening this challenger;
4. therefore `late_steps_only` should be treated as a narrow ablation, not as the next defended mainline candidate.

## Decision

Current release decision:

- `keep all_steps as defended gray-box mainline`
- `do not promote late_steps_only for TMIA-DM`
- `record as weak challenger-specific defense ablation`

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording can say a late-step-only dropout ablation was tested specifically against `TMIA-DM late-window`, but it was weaker than `all_steps` and does not change the defended headline.
