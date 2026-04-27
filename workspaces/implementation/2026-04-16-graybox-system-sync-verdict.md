# 2026-04-16 Gray-Box System Sync Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `X-3 / system-consumable sync`
- `focus`: `gray-box standings + unified table`
- `decision`: `synced with challenger-aware gray-box narrative`

## Question

After the recent `TMIA-DM` breakthrough, are the unified comparison artifacts still telling the truth about the gray-box mainline, or are they stuck in an older `PIA-only` worldview?

## Executed Evidence

Primary sync targets:

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-08-unified-attack-defense-table.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/artifacts/unified-attack-defense-table.json`

Gray-box reference anchors:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-tmiadm-gpu-repeat-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-tmiadm-gpu256-repeat-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-tmiadm-dropout-defense-repeat-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-tmiadm-dropout-defense-gpu256-repeat-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-pia-vs-tmiadm-operating-point-comparison.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-pia-vs-tmiadm-defended-operating-point-comparison.md`

## Verdict

Current verdict:

- `synced with challenger-aware gray-box narrative`

Reason:

1. the old unified table still described gray-box as essentially `PIA + provisional G-1`, which is no longer complete;
2. `TMIA-DM late-window` is now repeat-confirmed on both attack-side and defended-side GPU ladders;
3. the system-consumable artifacts now explicitly preserve the correct three-layer reading:
   - `PIA` remains admitted gray-box headline
   - `TMIA-DM late-window` is the strongest active challenger
   - defended gray-box is still multi-family rather than collapsing back to `PIA` only

## Handoff Note

- `Platform`: gray-box surfaces should no longer assume a single admitted family; add room for `headline` and `challenger` phrasing.
- `Runtime`: no schema break was required for this sync, but consumers should expect more than one active gray-box row.
- Materials: the safe wording is now `PIA headline + TMIA late-window challenger`, with defended-side challenger language preserved.
