# 2026-04-16 TMIA-DM GPU Pilot Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window GPU128 pilot`
- `selected_family`: `TMIA-DM late-window long_window`
- `device`: `cuda:0`
- `decision`: `positive challenger rung`

## Question

After fixing the local `Research` CUDA environment, does the first bounded `TMIA-DM late-window` GPU rung remain strong enough to count as a real gray-box challenger candidate rather than only a CPU-side curiosity?

## Executed Evidence

Environment verification:

- `conda run -n diffaudit-research python scripts/verify_env.py`

Primary GPU pilot:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r1/summary.json`

CPU references:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-cpu-32-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-cpu-32-r2-seed1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-128/summary.json`

Environment fix:

- the correct environment is `conda` env `diffaudit-research`
- the local issue was a wrong editable install target, not lack of CUDA support in the machine

## Metrics

`TMIA-DM late-window GPU128 r1`:

- `AUC = 0.825317`
- `ASR = 0.769531`
- `TPR@1%FPR = 0.085938`
- `TPR@0.1%FPR = 0.0625`

Current aligned `PIA GPU128` reference:

- `AUC = 0.817444`
- `ASR = 0.765625`
- `TPR@1%FPR = 0.046875`

Readout:

- the first admitted GPU rung stayed positive;
- it slightly beat the current `PIA GPU128` baseline on headline metrics;
- the effective signal still sits entirely inside `long_window`; `short_window` and naive `fused` remain non-competitive.

## Verdict

Current verdict:

- `positive challenger rung`

Reason:

1. the local CUDA blocker has been resolved for `Research` by switching to the admitted conda environment and repairing the editable package target;
2. the first `GPU128` pilot completed cleanly on `cuda:0`;
3. the resulting `TMIA-DM late-window long_window` metrics are strong enough to count as a real challenger rung;
4. this is still only one GPU rung, so promotion should wait for at least one repeat or paired comparison pass.

Interpretation:

- `TMIA-DM late-window long_window` is now beyond CPU-only feasibility;
- it has crossed into genuine challenger territory on the current asset line;
- the next rational step is GPU repeat-confirmation, not reopening broad family search.

## Decision

Current release decision:

- `keep GPU line active`
- `allow repeat-confirmation on GPU`
- `do not promote to gray-box headline yet`

Meaning:

1. this branch has earned continued bounded GPU budget;
2. `PIA` remains the admitted gray-box headline until the challenger is repeat-confirmed;
3. materials can now describe `TMIA-DM late-window` as a live GPU challenger line.

## Next Gate

The next task should be:

1. run one matched GPU repeat for `TMIA-DM late-window long_window`;
2. compare repeated GPU stats directly against `PIA GPU128`;
3. only then decide whether headline promotion is justified.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: the `Research` CUDA environment fix is now concrete and should be preserved as the default admitted runtime.
- Materials: wording can now say `TMIA-DM late-window long_window` has landed its first positive GPU challenger rung, but repeat-confirmation is still pending.
