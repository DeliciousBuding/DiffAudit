# 2026-04-16 PIA Vs TMIA-DM Temporal-Striding Defended Comparison

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-4 / defended gray-box comparison`
- `methods`: `PIA` vs `TMIA-DM late-window long_window`
- `defense`: `TMIA-DM temporal-striding(stride=2)` vs current `stochastic-dropout(all_steps)` references
- `decision`: `TMIA temporal-striding is the strongest defended challenger-specific branch`

## Question

Now that `TMIA-DM temporal-striding(stride=2)` has survived `cpu-32`, `GPU128`, and `GPU256`, how should defended gray-box be described relative to current defended `PIA` and current defended `TMIA-DM + dropout`?

## Executed Evidence

Defended references:

- `PIA GPU256 defended`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256\summary.json`
- `TMIA-DM late-window + stochastic-dropout GPU256 r2`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-256-r2-seed1\summary.json`
- `TMIA-DM late-window + temporal-striding GPU256 r1`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-temporal-striding-defense-20260416-gpu-256-r1\summary.json`
- `TMIA-DM late-window + temporal-striding GPU256 r2`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-temporal-striding-defense-20260416-gpu-256-r2-seed1\summary.json`

Undefended context:

- `PIA GPU256 baseline`
- `TMIA-DM late-window GPU256 baseline`

## Metrics

Defended `GPU256`:

- `PIA + stochastic-dropout(all_steps)`:
  - `AUC = 0.82901`
  - `ASR = 0.767578`
  - `TPR@1%FPR = 0.027344`
- `TMIA-DM + stochastic-dropout(all_steps)`:
  - `AUC = 0.82164`
  - `ASR = 0.765625`
  - `TPR@1%FPR = 0.039062`
- `TMIA-DM + temporal-striding r1`:
  - `AUC = 0.733322`
  - `ASR = 0.6875`
  - `TPR@1%FPR = 0.070312`
- `TMIA-DM + temporal-striding r2`:
  - `AUC = 0.7173`
  - `ASR = 0.662109`
  - `TPR@1%FPR = 0.019531`

Readout:

- on headline `AUC`, both `temporal-striding` repeats are far below defended `PIA` and far below defended `TMIA + dropout`;
- on `ASR`, both `temporal-striding` repeats are also clearly lower than the current defended references;
- low-FPR behavior is mixed across the two `temporal-striding` repeats, but the main decision-grade signal is that the challenger itself is now much more strongly suppressed than under dropout.

## Verdict

Current verdict:

- `TMIA temporal-striding is the strongest defended challenger-specific branch`

Reason:

1. `temporal-striding` is materially stronger than `dropout(all_steps)` on the same `TMIA-DM` attack family;
2. the result survives repeat and scale, so it is no longer an ablation-only note;
3. defended gray-box still should not be rewritten as a single-family story, because the admitted defended headline is still `PIA + stochastic-dropout(all_steps)`;
4. but the defended challenger ordering has changed: `TMIA + temporal-striding` is now the strongest defended `TMIA` branch and should supersede `TMIA + dropout` in higher-layer gray-box summaries.

## Decision

Current narrative decision:

- keep `PIA + stochastic-dropout(all_steps)` as the admitted defended headline by continuity
- replace `TMIA + dropout` with `TMIA + temporal-striding` as the defended gray-box challenger in summary-layer artifacts
- describe `temporal-striding` as a `TMIA`-specific defended candidate, not yet a project-wide replacement defense

## Handoff Note

- `Platform`: no direct product-surface change required yet.
- `Runtime`: no direct runtime contract change required yet.
- Materials: gray-box defended wording should now say the strongest defended challenger is `TMIA-DM + temporal-striding(stride=2)`, while `PIA + stochastic-dropout(all_steps)` remains the admitted headline defended pair.
