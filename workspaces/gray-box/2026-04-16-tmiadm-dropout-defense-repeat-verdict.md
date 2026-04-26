# 2026-04-16 TMIA-DM Dropout Defense Repeat Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window defended repeat`
- `selected_family`: `TMIA-DM late-window long_window`
- `defense`: `stochastic-dropout(all_steps)`
- `device`: `cuda:0`
- `decision`: `repeat-confirmed defended challenger`

## Question

After the first defended `TMIA-DM` rung showed that `stochastic-dropout(all_steps)` weakens but does not remove the challenger, does a matched repeat confirm that this defended challenger is real rather than a one-off?

## Executed Evidence

Defended GPU128 runs:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-128-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-128-r2-seed1\summary.json`

Current defended reference:

- `PIA GPU128 defended`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-128\summary.json`

Prior defended verdict:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-tmiadm-dropout-defense-verdict.md`

## Metrics

`TMIA-DM late-window + stochastic-dropout(all_steps)`:

- `r1 / seed0`:
  - `AUC = 0.809326`
  - `ASR = 0.75`
  - `TPR@1%FPR = 0.078125`
- `r2 / seed1`:
  - `AUC = 0.819397`
  - `ASR = 0.757812`
  - `TPR@1%FPR = 0.0625`

Current aligned `PIA` defended reference:

- `AUC = 0.803955`
- `ASR = 0.757812`
- `TPR@1%FPR = 0.03125`

## Verdict

Current verdict:

- `repeat-confirmed defended challenger`

Reason:

1. the defended `TMIA-DM` line stayed positive across two matched `GPU128` runs;
2. both repeats stayed at or above the current defended `PIA` rung on `AUC`;
3. `TMIA-DM` also remained materially stronger than defended `PIA` on `TPR@1%FPR`;
4. this means the current dropout defense weakens the challenger but does not restore gray-box to a single-family defended story.

Interpretation:

- `TMIA-DM late-window long_window` is now a defended gray-box challenger, not just an undefended one;
- the current `stochastic-dropout(all_steps)` defense does not collapse cross-family threat diversity;
- the next gray-box question should shift from “does the challenger survive defense?” to “how should defended gray-box be framed now?”

## Decision

Current release decision:

- `keep defended TMIA-DM line active`
- `do not claim dropout defense eliminates the challenger`
- `promote defended cross-family comparison into materials`

Meaning:

1. the defended gray-box narrative must now include both `PIA` and `TMIA-DM`;
2. further GPU budget should be driven by comparison or utility questions, not by basic survival checks;
3. the project now has evidence that one defense can weaken both lines without simplifying the field back to one winner.

## Next Gate

The next bounded task should be one of:

1. write a defended `PIA vs TMIA-DM` operating-point comparison note;
2. check whether the same defense interaction persists at `GPU256`;
3. if material pressure is higher than more GPU scale-up, prioritize the comparison note first.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording should now say the current dropout defense weakens both gray-box lines, but `TMIA-DM late-window` remains a repeat-confirmed defended challenger.
