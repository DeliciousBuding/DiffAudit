# X-81 Non-Graybox Next-Lane Reselection After X-80 I-A Carry-Forward Audit

## Task

Reselect the next honest non-graybox `CPU-first` lane after:

- the white-box loss-score branch was frozen below immediate same-family continuation,
- `X-79` returned the slot to `I-A`,
- `X-80` cleared one active `I-A` higher-layer residue.

## Verdict

`positive`

The next strongest live lane is **not** another `I-A` micro-audit and **not** a fresh black-box / white-box candidate expansion.

It is:

`X-82 cross-box / system-consumable stale-entry sync after X-81 reselection`

## Why

Current branch truth:

- `active GPU question = none`
- `next_gpu_candidate = none`
- white-box loss-score branch is already frozen as bounded auxiliary evidence
- black-box paper-backed scouting is already closed negative
- `XB-CH-2` still remains `needs-assets`
- gray-box still should yield the slot

But there is still one concrete class of higher-layer drift:

- some active docs still encode older lane state (`X-77` or `X-80`) rather than the current control-plane truth

That means the most honest immediate move is:

1. clear the stale read-path,
2. then reselection again,
3. rather than pretending another candidate surface has already reopened.

## Canonical evidence anchors

- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`

## Next recommended lane

`X-82 cross-box / system-consumable stale-entry sync after X-81 reselection`
