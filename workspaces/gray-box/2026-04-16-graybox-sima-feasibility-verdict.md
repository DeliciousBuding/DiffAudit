# 2026-04-16 Gray-Box SimA Feasibility Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-3.3 / GB-3.4`
- `selected_family`: `SimA`
- `gpu_status`: `not requested`
- `decision`: `negative but useful`

## Question

After selecting `SimA` as the next gray-box family, is the current local CIFAR-10 DDPM asset line strong enough to justify promoting it into a real gray-box challenger?

## Executed Evidence

Primary bounded feasibility run:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/sima-cifar10-runtime-feasibility-20260416-cpu-32-r1/summary.json`

Selection note:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-graybox-new-family-sima-selection.md`

Current mainline comparison references:

- `PIA cpu-32 baseline`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-cpu-32/summary.json`
- `PIA defended cpu-32`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-cpu-32/summary.json`

## Metrics

Best bounded `SimA` rung:

- `best_timestep = 120`
- `AUC = 0.542969`
- `ASR = 0.625`
- `TPR@1%FPR = 0.0625`
- `wall-clock = 18.387406s`

Selected timestep scan:

- `t=20`: `AUC = 0.408203`
- `t=40`: `AUC = 0.43457`
- `t=60`: `AUC = 0.466797`
- `t=80`: `AUC = 0.487305`
- `t=100`: `AUC = 0.535156`
- `t=120`: `AUC = 0.542969`

Current local `PIA cpu-32 baseline` reference:

- `AUC = 0.782227`
- `ASR = 0.765625`

## Verdict

Current verdict:

- `negative but useful`

Reason:

1. the bounded local `SimA` path is execution-feasible;
2. it does produce a weak positive separation above random at later timesteps;
3. but the best observed signal (`AUC = 0.542969`) is far below the current local `PIA` baseline and far below challenger quality.

Interpretation:

- `SimA` is a real runnable family on the current local DDPM asset line, so it is no longer just literature-side support;
- however, the current bounded implementation does not justify challenger promotion or GPU release;
- the repo should currently treat `SimA` as a feasible but weak gray-box family on this asset line.

## Decision

Current release decision:

- `no GPU release`
- `no challenger promotion`

Meaning:

1. do not scale this current local `SimA` path to GPU yet;
2. do not replace or dilute the `PIA` headline with this result;
3. keep the run as evidence that the family is locally executable, but not yet strong enough.

## Reopen Rule

Only reopen `SimA` if there is a fresh bounded reason, such as:

1. a more paper-faithful timestep selection or norm-setting hypothesis;
2. a cleaner denoiser-query alignment with the original paper path;
3. evidence that a different asset line should materially improve separation.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording can say `SimA` is now locally execution-feasible, but should also say it is not currently strong enough to join the gray-box mainline or challenger tier.
