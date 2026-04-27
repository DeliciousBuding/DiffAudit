# 2026-04-16 TMIA-DM GPU256 Rung Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window GPU256 rung`
- `selected_family`: `TMIA-DM late-window long_window`
- `device`: `cuda:0`
- `decision`: `positive scale-up, near-parity with PIA`

## Question

After `TMIA-DM late-window long_window` became a repeat-confirmed `GPU128` challenger, does the first `GPU256` rung keep that line alive at a higher sample count?

## Executed Evidence

Primary rung:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r1/summary.json`

Current references:

- `TMIA-DM GPU128 r1`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r1/summary.json`
- `TMIA-DM GPU128 r2`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r2-seed1/summary.json`
- `PIA GPU256 baseline`:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-256/summary.json`

## Metrics

`TMIA-DM late-window GPU256 r1`:

- `AUC = 0.839554`
- `ASR = 0.765625`
- `TPR@1%FPR = 0.117188`
- `TPR@0.1%FPR = 0.066406`

Current aligned `PIA GPU256` reference:

- `AUC = 0.841293`
- `ASR = 0.78125`
- `TPR@1%FPR = 0.039062`
- `TPR@0.1%FPR = 0.019531`

Readout:

- `TMIA-DM` stayed strong when scaling from `128` to `256`;
- on `AUC` and `ASR` it is near parity but slightly below `PIA GPU256`;
- on low-FPR detection it is materially stronger than `PIA GPU256`.

## Verdict

Current verdict:

- `positive scale-up, near-parity with PIA`

Reason:

1. the `TMIA-DM late-window` signal did not collapse at the next GPU rung;
2. the branch remains fully challenger-worthy at `256`;
3. the comparison against `PIA` is now metric-dependent rather than one-sided:
   - `PIA` still has a slight edge on `AUC/ASR`
   - `TMIA-DM` has a much stronger `TPR@1%FPR`
4. this means the story has matured from “can it compete?” to “which operating point matters?”

Interpretation:

- `TMIA-DM late-window long_window` is now an established gray-box challenger line across more than one GPU rung;
- the project can no longer describe gray-box as a single-family `PIA` story;
- but it is still too early to declare a full headline swap without another repeat or a more explicit operating-point comparison.

## Decision

Current release decision:

- `keep challenger line active`
- `do not demote after scale-up`
- `headline decision still pending`

Meaning:

1. this line has earned continued place in the gray-box narrative;
2. the next question is no longer feasibility, but comparative positioning;
3. a follow-up comparison note should explicitly separate `AUC/ASR` and low-FPR utility.

## Next Gate

The next bounded task should be one of:

1. a `GPU256` repeat for `TMIA-DM late-window`;
2. a gray-box operating-point comparison note between `PIA` and `TMIA-DM`;
3. a first defense interaction check for the `TMIA-DM late-window` line.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: keep the admitted CUDA environment unchanged; this rung depends on it.
- Materials: wording can now say `TMIA-DM late-window long_window` scales to `GPU256` and reaches near-parity with `PIA`, with stronger low-FPR behavior.
