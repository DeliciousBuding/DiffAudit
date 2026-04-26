# 2026-04-16 PIA Vs TMIA-DM Defended Operating-Point Comparison

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-4 / defended gray-box comparison`
- `methods`: `PIA` vs `TMIA-DM late-window long_window`
- `defense`: `stochastic-dropout(all_steps)`
- `decision`: `defended gray-box is still multi-family`

## Question

After repeating the defended `TMIA-DM` rung, how should the project now describe defended gray-box: does `PIA` still dominate cleanly, or does `TMIA-DM` remain a real defended challenger?

## Executed Evidence

Defended references:

- `PIA GPU128 defended`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-128\summary.json`
- `TMIA-DM late-window defended r1`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-128-r1\summary.json`
- `TMIA-DM late-window defended r2`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-128-r2-seed1\summary.json`

Undefended context:

- `PIA GPU128 baseline`
- `TMIA-DM late-window GPU128 baseline`

## Metrics

Defended `GPU128`:

- `PIA`:
  - `AUC = 0.803955`
  - `ASR = 0.757812`
  - `TPR@1%FPR = 0.03125`
- `TMIA-DM r1`:
  - `AUC = 0.809326`
  - `ASR = 0.75`
  - `TPR@1%FPR = 0.078125`
- `TMIA-DM r2`:
  - `AUC = 0.819397`
  - `ASR = 0.757812`
  - `TPR@1%FPR = 0.0625`

Readout:

- on `AUC`, both defended `TMIA-DM` repeats beat defended `PIA`;
- on `ASR`, one repeat ties defended `PIA` and one is slightly below;
- on `TPR@1%FPR`, both defended `TMIA-DM` repeats remain substantially above defended `PIA`.

## Verdict

Current verdict:

- `defended gray-box is still multi-family`

Reason:

1. the current dropout defense weakens both lines, but it does not restore a single dominant family;
2. defended `TMIA-DM` remains fully competitive with defended `PIA` on headline metrics and stronger on low-FPR behavior;
3. the defended gray-box story is therefore no longer “`PIA` plus a weak side note”, but “`PIA` headline with a defended challenger still alive”.

Interpretation:

- the defense narrows the gap but does not collapse the field;
- `TMIA-DM late-window` should now appear in defended gray-box material, not only in attack-side material;
- future defense work should be evaluated against both lines, not only `PIA`.

## Decision

Current narrative decision:

- `PIA` remains the defended headline by continuity
- `TMIA-DM late-window` is the strongest defended challenger
- `dropout defense` should be described as partial mitigation, not family-eliminating mitigation

## Next Gate

The next bounded task should be one of:

1. a defended `GPU256` rung for `TMIA-DM late-window`;
2. a first `TMIA-DM`-specific defense variation if a new bounded hypothesis exists;
3. a material-facing summary table that shows attack-side and defended-side gray-box standings together.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: gray-box defense wording should now say `stochastic-dropout` weakens both `PIA` and `TMIA-DM`, but leaves `TMIA-DM` as a defended challenger rather than eliminating it.
