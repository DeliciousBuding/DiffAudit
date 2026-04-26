# 2026-04-16 Gray-Box Second Defense Closure Verdict

## Task

- `GB-1` second gray-box defense

## Question

- Has `TMIA-DM temporal-striding(stride=2)` now cleared the bounded comparison and system-sync gates strongly enough to close `GB-1`, or is another defense-search loop still required first?

## Evidence Base

- `workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-verdict.md`
- `workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-gpu128-verdict.md`
- `workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-gpu256-verdict.md`
- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-defended-operating-point-comparison.md`
- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-temporal-striding-defended-comparison.md`

## Result

Current defended gray-box truth is now explicit:

1. `PIA` remains the defended headline by continuity and global-metric safety;
2. `TMIA-DM temporal-striding(stride=2)` is now the strongest defended challenger inside the `TMIA-DM` branch;
3. `stochastic-dropout(all_steps)` no longer owns the full defended challenger story by default;
4. `SecMI` remains a blocked baseline and is not a gating prerequisite for closing this second-defense round.

## Verdict

- `positive`

`GB-1` can close for the current round. The repo already has a second defended gray-box branch with repeat-confirmed bounded evidence, defended operating-point comparison, and system-layer sync.

## Decision

Current decision:

- close `GB-1` as `positive`
- keep `PIA` as the defended headline
- use `TMIA + temporal-striding(stride=2)` as the defended gray-box challenger reference
- move future gray-box effort to new-family diversity or disagreement work, not another blind defense shortlist

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: gray-box defense wording should prefer `TMIA + temporal-striding(stride=2)` over `TMIA + dropout(all_steps)` as the defended challenger reference.
