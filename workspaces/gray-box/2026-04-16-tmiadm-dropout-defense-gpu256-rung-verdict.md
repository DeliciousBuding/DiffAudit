# 2026-04-16 TMIA-DM Dropout Defense GPU256 Rung Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window defended GPU256 rung`
- `selected_family`: `TMIA-DM late-window long_window`
- `defense`: `stochastic-dropout(all_steps)`
- `device`: `cuda:0`
- `decision`: `positive defended scale-up, low-fpr-favorable`

## Question

After the defended `GPU128` line survived, does `TMIA-DM late-window` stay alive at `GPU256` under the same dropout defense, or does the challenger collapse at the higher rung?

## Executed Evidence

Primary rung:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-dropout-defense-20260416-gpu-256-r1\summary.json`

Current references:

- `PIA GPU256 defended`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256\summary.json`
- `TMIA-DM GPU256 undefended`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r1\summary.json`

## Metrics

`TMIA-DM late-window GPU256`:

- undefended:
  - `AUC = 0.839554`
  - `ASR = 0.765625`
  - `TPR@1%FPR = 0.117188`
- defended:
  - `AUC = 0.825867`
  - `ASR = 0.746094`
  - `TPR@1%FPR = 0.117188`

Current defended `PIA GPU256` reference:

- `AUC = 0.82901`
- `ASR = 0.767578`
- `TPR@1%FPR = 0.027344`

## Verdict

Current verdict:

- `positive defended scale-up, low-fpr-favorable`

Reason:

1. the defended challenger did not collapse when scaled to `GPU256`;
2. compared with defended `PIA`, it is slightly behind on `AUC/ASR`;
3. but it remains much stronger on `TPR@1%FPR`, preserving the low-FPR advantage already seen on the undefended line;
4. this means the defended `TMIA-DM` story is now scale-real, even if it is not yet the safest single headline on every metric.

## Decision

Current release decision:

- `keep defended TMIA-DM scale-up line active`
- `treat low-FPR behavior as the current strongest defended advantage`
- `request one defended GPU256 repeat before stronger narrative promotion`

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording can now say defended `TMIA-DM late-window` scales to `GPU256` and stays low-FPR-favorable, even though defended `PIA` keeps a slight edge on global headline metrics.
