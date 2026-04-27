# 2026-04-17 X-47 Non-Graybox Next-Lane Reselection After X-46 First Bounded Agreement-Board Read

## Question

After `X-46` landed a real but `negative but useful` first four-object agreement-board read for the fresh `I-C` line, what is the highest-value next non-graybox main-slot move?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x46-ic-first-bounded-four-object-agreement-board-read-after-x45-scalar-contract-freeze.md`

## Reselection Review

### 1. Continue same-board `I-C` salvage immediately

Not selected.

Reason:

1. `X-46` already gave the fresh `I-C` line its first honest bounded board read;
2. the result is not blocked and not ambiguous enough to justify immediate same-board salvage;
3. forcing more work on the same board right now would be packet churn rather than honest reselection.

### 2. Return immediately to `I-A` truth-hardening

Not selected as the immediate next move.

Reason:

1. `I-A` is still the strongest carry-forward innovation sidecar;
2. but the repo currently has one more urgent higher-layer mismatch:
   - `mainline-narrative`
   - `comprehensive-progress`
3. both still encode the pre-`X-45/X-46` `I-C` control-plane truth.

So `I-A` remains the likely next substantive lane after sync, but not before sync.

### 3. Run one bounded cross-box / system-consumable stale-entry sync pass

Selected.

Why this is now the strongest move:

1. `X-46` changed the live lane truth:
   - first fresh `I-C` board read exists
   - result is `negative but useful`
   - next lane is reselection
2. higher-layer readers still see the older `X-45` blocker state rather than the newer `X-46` readout;
3. this is exactly the kind of stale entry that has previously deserved main-slot sync before another substantive lane is promoted.

## Selection

- `selected_next_live_lane = X-48 cross-box / system-consumable stale-entry sync after X-47 reselection`

## Verdict

- `x47_non_graybox_next_lane_reselection_after_x46_first_bounded_agreement_board_read = positive`

More precise reading:

1. the fresh `I-C` line has now received an honest first bounded read;
2. that read is sufficient to demote same-board salvage for now;
3. the most honest immediate next move is one bounded stale-entry sync pass before reopening substantive lane choice.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-47 non-graybox next-lane reselection after X-46 first bounded agreement-board read`
- `next_live_cpu_first_lane = X-48 cross-box / system-consumable stale-entry sync after X-47 reselection`
- `carry_forward_cpu_sidecar = I-A higher-layer boundary maintenance`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x47-non-graybox-next-lane-reselection-after-x46-first-bounded-agreement-board-read.md`

## Handoff Decision

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`: update required
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`: update required
- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes the next live lane and identifies higher-layer stale surfaces that should be fixed before the next substantive non-graybox promotion.
