# 2026-04-16 TMIA-DM Dropout Defense Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window defense interaction`
- `selected_family`: `TMIA-DM late-window long_window`
- `defense`: `stochastic-dropout(all_steps)`
- `device`: `cuda:0`
- `decision`: `weakened but still favorable`

## Question

Once `TMIA-DM late-window` becomes a real GPU challenger, does the current gray-box defended mainline `stochastic-dropout(all_steps)` suppress that challenger enough to neutralize it?

## Executed Evidence

Primary defended rung:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-128-r1\summary.json`

Current references:

- `TMIA-DM GPU128 undefended`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r1\summary.json`
- `PIA GPU128 undefended`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260408-gpu-128\summary.json`
- `PIA GPU128 defended`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-128\summary.json`

## Metrics

`TMIA-DM late-window GPU128`:

- undefended:
  - `AUC = 0.825317`
  - `ASR = 0.769531`
  - `TPR@1%FPR = 0.085938`
- defended with `stochastic-dropout(all_steps)`:
  - `AUC = 0.809326`
  - `ASR = 0.75`
  - `TPR@1%FPR = 0.078125`

Current `PIA GPU128` references:

- undefended:
  - `AUC = 0.817444`
  - `ASR = 0.765625`
  - `TPR@1%FPR = 0.046875`
- defended:
  - `AUC = 0.803955`
  - `ASR = 0.757812`
  - `TPR@1%FPR = 0.03125`

## Verdict

Current verdict:

- `weakened but still favorable`

Reason:

1. the current dropout defense does reduce `TMIA-DM late-window`, so the challenger is not defense-immune;
2. the reduction is modest rather than destructive:
   - `AUC` drops by about `0.016`
   - `ASR` drops by about `0.0195`
3. even after defense, `TMIA-DM` stays above the current defended `PIA` rung on `AUC` and markedly above it on `TPR@1%FPR`;
4. the current evidence therefore says the defended gray-box story is no longer `PIA`-only.

Interpretation:

- `stochastic-dropout(all_steps)` weakens the new challenger, but does not remove it;
- `TMIA-DM late-window long_window` remains operational under the current defended regime;
- the next real question is whether this defended result is repeatable, not whether the line survives at all.

## Decision

Current release decision:

- `keep TMIA-DM defense-interaction line active`
- `do not claim defense neutralizes the challenger`
- `request one defended repeat before stronger narrative promotion`

Meaning:

1. the gray-box defended story now has a genuine cross-family interaction result;
2. materials can say the current dropout defense weakens both lines, but does not collapse `TMIA-DM`;
3. the next bounded GPU task should be a defended repeat, not a fresh unrelated branch.

## Next Gate

The next bounded task should be:

1. repeat the same defended `TMIA-DM GPU128` rung with a second seed;
2. if that holds, write an explicit defended operating-point comparison note against `PIA`;
3. only then decide whether a defended `GPU256` rung is worth the budget.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording can now say the current dropout defense weakens `TMIA-DM late-window`, but does not eliminate it; `TMIA-DM` remains a defended gray-box challenger.
