# 2026-04-16 TMIA-DM Temporal-Striding Defense Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window temporal-striding defense`
- `selected_family`: `TMIA-DM late-window long_window`
- `defense`: `temporal-striding(stride=2)`
- `device`: `cpu`
- `decision`: `repeat-positive cpu defense hypothesis`

## Question

Because `TMIA-DM late-window` depends on time-resolution consistency, does exposing only a strided subset of late timesteps weaken the challenger more effectively than leaving the full `[80, 100, 120]` window intact?

## Executed Evidence

Primary temporal-striding rungs:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-temporal-striding-defense-20260416-cpu-32-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-temporal-striding-defense-20260416-cpu-32-r2-seed1/summary.json`

Current references:

- `TMIA-DM late-window CPU32 undefended`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-cpu-32-r1/summary.json`
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-cpu-32-r2-seed1/summary.json`
- current defended mainline comparator:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-128-r1/summary.json`

## Metrics

Undefended `TMIA-DM late-window`:

- `r1 / seed0`:
  - `AUC = 0.823242`
  - `ASR = 0.796875`
  - `TPR@1%FPR = 0.28125`
- `r2 / seed1`:
  - `AUC = 0.760742`
  - `ASR = 0.75`
  - `TPR@1%FPR = 0.0625`

`temporal-striding(stride=2)`:

- requested window: `[80, 100, 120]`
- effective window: `[80, 120]`
- `r1 / seed0`:
  - `AUC = 0.697266`
  - `ASR = 0.703125`
  - `TPR@1%FPR = 0.0625`
- `r2 / seed1`:
  - `AUC = 0.696289`
  - `ASR = 0.703125`
  - `TPR@1%FPR = 0.03125`

Readout:

- both bounded repeats moved `AUC` downward by about `0.126` and `0.064` relative to the paired undefended late-window runs;
- `TPR@1%FPR` also dropped in both repeats, especially on `seed0`;
- the resulting CPU signal is now weaker than the current `TMIA-DM` challenger regime and directionally comparable to a meaningful defense, rather than another null perturbation.

## Verdict

Current verdict:

- `repeat-positive cpu defense hypothesis`

Reason:

1. the defense weakened `TMIA-DM late-window` on both same-budget CPU repeats instead of amplifying it;
2. unlike `timestep-jitter(radius=10)`, the new hypothesis did not backfire;
3. unlike `late_steps_only` dropout, this is a materially different interface-side defense family rather than a schedule tweak on the same stochastic defense;
4. the evidence is still only `cpu-32`, so it is strong enough for a minimal GPU gate but not for promotion into defended gray-box narrative yet.

## Decision

Current release decision:

- `allow one minimal GPU128 rung`
- `do not promote as defended gray-box comparator yet`
- `keep stochastic-dropout(all_steps) as current defended mainline`

Meaning:

1. the next step should be one exact-contract `GPU128` rung on `[80, 120]`;
2. do not reopen broad defense search before this gate is resolved;
3. do not change Platform or competition wording until a GPU verdict exists.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: no sync yet; this stays as a gray-box internal candidate until GPU evidence lands.
