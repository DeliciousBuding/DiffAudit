# 2026-04-16 TMIA-DM GPU256 Repeat Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window GPU256 repeat`
- `selected_family`: `TMIA-DM late-window long_window`
- `device`: `cuda:0`
- `decision`: `repeat-confirmed at GPU256`

## Question

After the first positive `TMIA-DM late-window GPU256` rung, does a matched repeat confirm that the line remains strong at the higher sample count rather than falling back to a noisy one-off result?

## Executed Evidence

GPU256 runs:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r2-seed1\summary.json`

Current references:

- `PIA GPU256 baseline`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260408-gpu-256\summary.json`
- prior `TMIA-DM GPU256` rung note:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-tmiadm-gpu256-rung-verdict.md`

## Metrics

`TMIA-DM late-window GPU256`:

- `r1 / seed0`:
  - `AUC = 0.839554`
  - `ASR = 0.765625`
  - `TPR@1%FPR = 0.117188`
  - `TPR@0.1%FPR = 0.066406`
- `r2 / seed1`:
  - `AUC = 0.837814`
  - `ASR = 0.787109`
  - `TPR@1%FPR = 0.050781`
  - `TPR@0.1%FPR = 0.015625`

Current aligned `PIA GPU256` reference:

- `AUC = 0.841293`
- `ASR = 0.78125`
- `TPR@1%FPR = 0.039062`
- `TPR@0.1%FPR = 0.019531`

## Verdict

Current verdict:

- `repeat-confirmed at GPU256`

Reason:

1. the `TMIA-DM late-window` signal survived the higher rung instead of collapsing after the first `GPU256` run;
2. both repeats stayed tightly clustered on `AUC`, which is enough to treat the line as stable at this scale;
3. compared with `PIA GPU256`, the picture is now metric-dependent rather than one-sided:
   - `PIA` keeps a slight `AUC` edge
   - `TMIA-DM` is competitive on `ASR`
   - `TMIA-DM` is stronger on `TPR@1%FPR` in one repeat and near-parity on the other
4. this is enough to keep the challenger active, but not yet enough to force a headline swap.

Interpretation:

- `TMIA-DM late-window long_window` is now stable across two `GPU128` runs and two `GPU256` runs;
- the remaining project question is comparative positioning, not existence or stability;
- the correct next step is a structured operating-point comparison against `PIA`, not another feasibility recheck.

## Decision

Current release decision:

- `keep TMIA-DM as active gray-box GPU challenger`
- `do not demote after repeat`
- `headline decision still pending`

Meaning:

1. this line has earned continued place in the gray-box narrative;
2. the next non-GPU task should crystallize where it beats `PIA` and where it does not;
3. additional GPU budget should be justified by comparison needs, not by uncertainty about stability.

## Next Gate

The next bounded task should be:

1. write a `PIA vs TMIA-DM` operating-point comparison note over `GPU128` and `GPU256`;
2. only after that decide whether another ladder rung or defense interaction is more valuable;
3. keep `short_window` and naive `fused` closed unless a new hypothesis appears.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording can now say `TMIA-DM late-window long_window` is repeat-confirmed at both `GPU128` and `GPU256`, while final gray-box headline choice still depends on operating-point framing.
