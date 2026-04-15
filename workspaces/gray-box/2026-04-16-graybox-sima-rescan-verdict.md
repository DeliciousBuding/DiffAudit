# 2026-04-16 Gray-Box SimA Rescan Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-3 follow-up / SimA timestep rescan`
- `device`: `cpu`
- `decision`: `negative but useful`

## Question

- Was the first bounded `SimA` result mainly weak because the scan stopped too early, or does the family remain too weak even after a later-timestep rescan?

## Executed Evidence

Previous feasibility run:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\sima-cifar10-runtime-feasibility-20260416-cpu-32-r1\summary.json`

Current rescan run:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\sima-cifar10-runtime-rescan-20260416-cpu-32-r2\summary.json`

## Metrics

Previous best:

- `best_timestep = 120`
- `AUC = 0.542969`

Rescan window:

- `t=120`: `AUC = 0.542969`
- `t=160`: `AUC = 0.584961`
- `t=200`: `AUC = 0.578125`
- `t=240`: `AUC = 0.542969`

Current best:

- `best_timestep = 160`
- `AUC = 0.584961`
- `ASR = 0.609375`
- `TPR@1%FPR = 0.03125`

## Verdict

- `negative but useful`

Reason:

1. the later-timestep rescan does improve over the first bounded `SimA` result;
2. this means the original weak result was not purely a bad early-window choice;
3. however, even the improved best rung (`AUC = 0.584961`) remains far below current challenger quality and far below local `PIA` strength;
4. therefore the family is still not strong enough for challenger promotion or GPU release on the current asset line.

## Decision

Current decision:

- keep `SimA` as `execution-feasible but weak`
- refine its wording from “weak at `t<=120`” to “improves modestly around `t=160`, still not challenger-grade”
- do not request GPU

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: if mentioned, the honest wording is that `SimA` has a somewhat better later-timestep operating point, but still does not join the gray-box challenger tier.
