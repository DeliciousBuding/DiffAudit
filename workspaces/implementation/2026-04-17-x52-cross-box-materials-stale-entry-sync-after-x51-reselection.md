# 2026-04-17 X-52 Cross-Box / Materials Stale-Entry Sync After X-51 Reselection

## Question

Can the repository clear the remaining materials-facing stale entry identified by `X-51`, without changing admitted tables or opening any new GPU release?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/docs/competition-evidence-pack.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/docs/admitted-results-summary.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`

## Sync Findings

One materials-facing evidence pack still lagged behind current gray-box truth:

1. `SecMI` was still described as `blocked baseline`, even though current repo truth is `same-asset independent corroboration line`.
2. `TMIA-DM` was still described as `protocol-and-asset decomposition intake only`, even though the current repo truth is `strongest packaged gray-box challenger`, with its first confidence-gated switching packet already closed as `negative but useful`.
3. The GPU-release reasoning section still grouped `Finding NeMo / SecMI / TMIA-DM` using those stale states.

## Actions Taken

- updated `competition-evidence-pack.md` to preserve:
  - `SecMI = same-asset independent corroboration line`
  - `TMIA-DM = strongest packaged gray-box challenger`
  - `PIA vs TMIA-DM confidence-gated switching = negative but useful`
- left admitted tables unchanged
- left `active GPU question = none`
- left `next_gpu_candidate = none`

## Verdict

- `x52_cross_box_materials_stale_entry_sync_verdict = positive`

More precise reading:

- the issue was real
- it was higher-layer only
- it is now cleared on the active materials-facing evidence pack

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/workspaces/implementation/challenger-queue.md`: update required
- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required because the current live lane advanced again
- `Leader/materials`: safe to reuse the refreshed gray-box challenger wording
- `Platform / Runtime`: no schema change required
