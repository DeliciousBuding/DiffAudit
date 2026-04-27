# 2026-04-16 TMIA-DM Timestep-Jitter Defense Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window timestep-jitter defense`
- `selected_family`: `TMIA-DM late-window long_window`
- `defense`: `timestep-jitter(radius=10)`
- `device`: `cuda:0`
- `decision`: `negative defense hypothesis`

## Question

Because `TMIA-DM late-window` depends on precise late timestep alignment, does small random timestep jitter weaken the challenger more effectively than the current dropout defenses?

## Executed Evidence

Primary jitter-defense rung:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-timestep-jitter-defense-20260416-gpu-128-r1/summary.json`

Current references:

- `TMIA-DM late-window GPU128 undefended`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r1/summary.json`
- `TMIA-DM late-window GPU128 all_steps defended`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-128-r1/summary.json`

## Metrics

`TMIA-DM late-window GPU128`:

- undefended:
  - `AUC = 0.825317`
  - `ASR = 0.769531`
  - `TPR@1%FPR = 0.085938`
- `timestep-jitter(radius=10)`:
  - `AUC = 0.850098`
  - `ASR = 0.792969`
  - `TPR@1%FPR = 0.109375`
- `stochastic-dropout(all_steps)`:
  - `AUC = 0.809326`
  - `ASR = 0.75`
  - `TPR@1%FPR = 0.078125`

Observed effective timesteps:

- requested window: `[80, 100, 120]`
- jittered window: `[81, 93, 124]`

## Verdict

Current verdict:

- `negative defense hypothesis`

Reason:

1. the tested jitter defense strengthened the challenger instead of weakening it;
2. all headline metrics moved in the wrong direction relative to the undefended rung;
3. the current result is also clearly worse than the existing `all_steps` dropout defense;
4. this means naive late-window timestep jitter should not be pursued as the next defended gray-box candidate.

## Decision

Current release decision:

- `do not promote timestep-jitter as defense`
- `do not spend more GPU budget on this exact jitter hypothesis`
- `keep all_steps dropout as the current stronger defended baseline`

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording can now say a TMIA-specific timestep-jitter defense was tested and rejected because it amplified, rather than suppressed, the challenger.
