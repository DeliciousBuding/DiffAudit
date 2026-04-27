# X-83 Non-Graybox Next-Lane Reselection After X-82 Stale-Entry Sync

## Task

Reselect the next honest non-graybox `CPU-first` lane after `X-82` cleared the active stale read-path.

## Verdict

`positive`

Once the active stale-entry surfaces are cleared, the strongest next move is no longer another wording-only sync pass and still not an immediate blocked/hold reopen.

The next live lane should therefore move to:

`X-84 non-graybox candidate-surface expansion after X-83 reselection`

## Why

Current repo truth:

- `active GPU question = none`
- `next_gpu_candidate = none`
- black-box paper-backed scouting remains closed negative
- white-box loss-score branch remains frozen as bounded auxiliary evidence
- `XB-CH-2` still remains `needs-assets`
- no new active `I-A` residue is visible after `X-82`

This means the honest next move is to reopen candidate generation itself, rather than:

- forcing another `I-A` micro-audit,
- reopening a blocked branch,
- or repeating the just-finished stale-entry sync pattern.

## Canonical evidence anchors

- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`

## Next recommended lane

`X-84 non-graybox candidate-surface expansion after X-83 reselection`
