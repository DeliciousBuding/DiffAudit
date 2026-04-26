# 2026-04-16 TMIA-DM Dropout Defense GPU256 Repeat Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window defended GPU256 repeat`
- `selected_family`: `TMIA-DM late-window long_window`
- `defense`: `stochastic-dropout(all_steps)`
- `device`: `cuda:0`
- `decision`: `repeat-confirmed defended scale challenger`

## Question

After the first defended `GPU256` rung stayed alive, does a matched repeat confirm that the defended challenger also remains stable at the higher rung?

## Executed Evidence

Defended GPU256 runs:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-256-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-256-r2-seed1\summary.json`

Current defended reference:

- `PIA GPU256 defended`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256\summary.json`

## Metrics

`TMIA-DM late-window + stochastic-dropout(all_steps)`:

- `r1 / seed0`:
  - `AUC = 0.825867`
  - `ASR = 0.746094`
  - `TPR@1%FPR = 0.117188`
- `r2 / seed1`:
  - `AUC = 0.82164`
  - `ASR = 0.765625`
  - `TPR@1%FPR = 0.042969`

Current defended `PIA GPU256` reference:

- `AUC = 0.82901`
- `ASR = 0.767578`
- `TPR@1%FPR = 0.027344`

## Verdict

Current verdict:

- `repeat-confirmed defended scale challenger`

Reason:

1. the defended `TMIA-DM` line remained strong across two `GPU256` runs instead of falling apart after the first one;
2. both repeats stayed close to defended `PIA` on `AUC/ASR`;
3. both repeats remained at or above defended `PIA` on low-FPR behavior;
4. this is enough to say the defended challenger scales, not just survives.

## Interpretation

- defended `TMIA-DM late-window` is now stable at both `GPU128` and `GPU256`;
- the current dropout defense does not simplify the gray-box field back to one family;
- the real remaining question is narrative framing and whether a new defense can do better, not whether this challenger is real.

## Decision

Current release decision:

- `keep defended TMIA-DM as active scaled challenger`
- `do not claim defended gray-box is PIA-only`
- `allow narrative/material consolidation before more scaling`

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording can now say `TMIA-DM late-window` remains a repeat-confirmed defended challenger at both `GPU128` and `GPU256`.
