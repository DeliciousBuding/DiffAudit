# 2026-04-17 X-34 Non-Graybox Next-Lane Reselection After X-33 Stale Intake Sync

## Question

After `X-33` cleared the active stale intake/system surfaces, does any currently visible non-graybox branch now honestly deserve the main `CPU-first` slot, or has the executable candidate pool itself gone stale?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x32-next-lane-reselection-after-x31-stale-entry-sync.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x33-crossbox-stale-intake-sync-after-x32-reselection.md`

## Candidate Comparison

### 1. Reopen `XB-CH-2`

Not selected.

Reason:

1. the branch is still `needs-assets`;
2. the paired model/split/shared-metric/bounded-budget reopen contract remains unsatisfied;
3. another immediate pass would only restate the blocker.

### 2. Reopen `GB-CH-2`

Not selected.

Reason:

1. the first bounded packet already closed `negative but useful`;
2. no genuinely new gating variable is visible now;
3. gray-box already yielded the slot and has not earned it back.

### 3. Reopen black-box or white-box hold branches

Not selected.

Reason:

1. `BB-CH-2` still remains `needs-assets`;
2. `WB-CH-2` remains `not-requestable`;
3. `BB-CH-1`, `BB-CH-3`, `WB-CH-1`, and `WB-CH-3` still sit below release with no new boundary shift.

### 4. Re-promote `I-A` or another maintenance sidecar into the main slot

Not selected.

Reason:

1. `I-A` wording is still stable after `X-30`;
2. `X-33` already consumed the remaining cross-box/system stale-sync value;
3. promoting maintenance again would create false momentum rather than a stronger lane.

### 5. Expand the non-graybox candidate surface itself

Selected.

Reason:

1. current visible candidate pool no longer contains one honest ready main-slot lane;
2. the correct next move is to expand candidate generation rather than recycle blocked/hold branches;
3. this keeps the loop honest and matches the repo rule that a stale roadmap must be expanded, not performatively replayed.

## Selection

- `selected_next_live_lane = X-35 non-graybox candidate-surface expansion after X-34 reselection`

## Verdict

- `x34_next_lane_reselection_after_x33_stale_intake_sync = positive`

More precise reading:

1. the reselection succeeded;
2. it did **not** discover a stronger already-existing executable lane;
3. the highest-value next move is now bounded roadmap expansion on the non-graybox side.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-34 non-graybox next-lane reselection after X-33 stale intake sync`
- `next_live_cpu_first_lane = X-35 non-graybox candidate-surface expansion after X-34 reselection`
- `carry_forward_cpu_sidecar = I-A higher-layer boundary maintenance`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x34-next-lane-reselection-after-x33-stale-intake-sync.md`

## Handoff Decision

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`: update required
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`: update required
- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = note-level only`

Reason:

- this step changes lane ordering and admits that the current candidate pool is stale, but it does not change admitted metrics, runtime contracts, or consumer schema.
