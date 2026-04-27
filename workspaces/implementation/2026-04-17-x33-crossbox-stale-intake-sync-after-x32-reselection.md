# 2026-04-17 X-33 Cross-Box / System-Consumable Stale Intake Sync After X-32 Reselection

## Question

After `X-32` promoted stale intake/system sync into the main slot, which active higher-layer or machine-readable surfaces still drift on current `Finding NeMo / Phase E` truth, and can they be aligned without changing admitted contracts?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/docs/reproduction-status.md`
- `<DIFFAUDIT_ROOT>/Research/docs/future-phase-e-intake.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/intake/phase-e-candidates.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x32-next-lane-reselection-after-x31-stale-entry-sync.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-first-truly-bounded-admitted-intervention-review-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-post-first-actual-packet-boundary-review.md`

## Review

### 1. `reproduction-status.md` still encoded old `Phase E` / `Finding NeMo` truth

The active summary still said:

- `Finding NeMo` = current remaining intake-only candidate
- `Finding NeMo` = `zero-GPU hold`
- next step = keep it as intake dossier only

That was behind current repo truth after the executed bounded packet and the post-packet falsifier boundary.

### 2. `future-phase-e-intake.md` still treated `Finding NeMo` as the live intake priority

The active intake document still said:

- fixed ranking = `PIA paper-aligned confirmation` then `Finding NeMo`
- `Finding NeMo` = current intake review priority #1
- `Finding NeMo` = `zero-GPU hold`

Current repo truth is narrower:

- `PIA paper-aligned confirmation` remains document-layer conditional only
- `Finding NeMo` has exited active intake ordering and now lives under executed white-box branch boundary

### 3. `phase-e-candidates.json` still exposed a stale machine-readable intake candidate

The registry still carried one live `intake_review_priority_order` entry:

- `white-box/finding-nemo-local-memorization-fb-mem`
- `current_shape = adapter-complete zero-GPU hold`

That no longer matches current branch posture.

## Applied Sync

1. `reproduction-status.md`
   - rewrote the current priority header
   - upgraded the `finding-nemo` row to `non-admitted actual bounded falsifier + not-requestable`
   - replaced stale `Phase E / zero-GPU hold` bullets with current post-packet boundary truth
2. `future-phase-e-intake.md`
   - removed `Finding NeMo` from active fixed ordering and intake priority order
   - preserved old intake notes only as historical gate context
   - froze current `Phase E` posture to `document-layer conditional only`
3. `phase-e-candidates.json`
   - moved the registry to `document-conditional-only`
   - cleared the active `intake_review_priority_order`

## Verdict

- `x33_crossbox_stale_intake_sync_after_x32_reselection = positive`

More precise reading:

1. the drift was real and still active on higher-layer/system-consumable surfaces;
2. the fix changes current queue truth, not admitted metrics;
3. `Finding NeMo` is no longer encoded as the current `Phase E` intake-only candidate.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-33 cross-box / system-consumable stale intake sync after X-32 reselection`
- `next_live_cpu_first_lane = X-34 non-graybox next-lane reselection after X-33 stale intake sync`
- `carry_forward_cpu_sidecar = I-A higher-layer boundary maintenance`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x33-crossbox-stale-intake-sync-after-x32-reselection.md`

## Handoff Decision

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`: update required
- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`: update required
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/reproduction-status.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/future-phase-e-intake.md`: update required
- `<DIFFAUDIT_ROOT>/Research/workspaces/intake/phase-e-candidates.json`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = note-level only`

Reason:

- this pass sharpens higher-layer and machine-readable queue truth only; it does not change admitted metrics, consumer schema, or runtime requirements.
