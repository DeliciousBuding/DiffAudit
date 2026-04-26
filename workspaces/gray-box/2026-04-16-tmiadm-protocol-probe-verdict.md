# 2026-04-16 TMIA-DM Protocol Probe Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-3 / TMIA-DM protocol probe`
- `selected_family`: `TMIA-DM`
- `gpu_status`: `not requested`
- `decision`: `mixed-positive but not challenger-ready`

## Question

After upgrading `TMIA-DM` to protocol-ready, does the first bounded local CPU probe produce enough signal to justify scaling this family into a real gray-box challenger?

## Executed Evidence

Primary bounded protocol-probe run:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-protocol-probe-20260416-cpu-32-r1\summary.json`

Protocol note:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-tmiadm-protocol-and-asset-note.md`

Current mainline comparison references:

- `PIA cpu-32 baseline`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260408-cpu-32\summary.json`
- `SimA cpu-32 feasibility`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\sima-cifar10-runtime-feasibility-20260416-cpu-32-r1\summary.json`

## Metrics

Bounded `TMIA-DM protocol probe` families:

- `short_window`:
  - `AUC = 0.37793`
  - `ASR = 0.515625`
  - `TPR@1%FPR = 0.0`
- `long_window`:
  - `AUC = 0.702148`
  - `ASR = 0.703125`
  - `TPR@1%FPR = 0.03125`
- `fused`:
  - `AUC = 0.510742`
  - `ASR = 0.546875`
  - `TPR@1%FPR = 0.0`

Current local reference lines:

- `PIA cpu-32 baseline`:
  - `AUC = 0.782227`
  - `ASR = 0.765625`
- `SimA cpu-32 feasibility`:
  - `AUC = 0.542969`
  - `ASR = 0.625`

## Verdict

Current verdict:

- `mixed-positive but not challenger-ready`

Reason:

1. the bounded local `TMIA-DM` path is now execution-real on the current DDPM/CIFAR-10 asset line;
2. the `long_window` family shows a meaningful positive signal and clearly outperforms the current bounded `SimA` probe;
3. the `short_window` family is directionally negative and the naive `fused` combination does not improve over the best family;
4. even the best observed family (`AUC = 0.702148`) still trails the current local `PIA` baseline and is not yet strong enough for challenger promotion.

Interpretation:

- `TMIA-DM` has moved past protocol-only status and now has a locally runnable, evidence-backed positive branch;
- but that positive branch is currently limited to `long_window`, not a robust full-family result;
- the repo should therefore treat `TMIA-DM` as a promising secondary gray-box family, not yet a GPU-released challenger.

## Decision

Current release decision:

- `no GPU release yet`
- `no challenger promotion yet`
- `keep long_window active`
- `prune naive short_window/fused as current headline`

Meaning:

1. do not scale this current protocol probe to GPU yet;
2. keep the `long_window` branch as the shortest bounded follow-up path;
3. do not present `short_window` or naive fusion as evidence that the full family is mature.

## Reopen Rule

The next bounded `TMIA-DM` step should be one of:

1. refine the `long_window` temporal aggregation family on the same CPU asset line;
2. run a small aligned repeat to check whether the `long_window` signal is stable across seeds or timestep subsets;
3. compare `PIA` and `TMIA-DM long_window` on the same split before spending GPU budget.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording can now say `TMIA-DM` has a local positive gray-box protocol-probe branch via `long_window`, but should also say it is not yet a full challenger and has not been GPU-released.
