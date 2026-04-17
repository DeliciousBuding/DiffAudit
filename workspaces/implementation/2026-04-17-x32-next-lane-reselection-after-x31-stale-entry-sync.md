# 2026-04-17 X-32 Non-Graybox Next-Lane Reselection After X-31 Stale-Entry Sync

## Question

After `X-31` cleared the last visible post-`X-30` control-plane drift, which non-graybox lane now honestly deserves the main `CPU-first` slot?

## Inputs Reviewed

- `D:\Code\DiffAudit\ROADMAP.md`
- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Research\docs\mainline-narrative.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x28-xbch2-shared-surface-contract-freeze-review.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x29-next-lane-reselection-after-x28-shared-surface-freeze-review.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x30-ia-carry-forward-truth-hardening-audit.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x31-stale-entry-sync-after-x30-ia-audit.md`
- `D:\Code\DiffAudit\Research\docs\reproduction-status.md`
- `D:\Code\DiffAudit\Research\docs\future-phase-e-intake.md`
- `D:\Code\DiffAudit\Research\workspaces\intake\phase-e-candidates.json`

## Candidate Comparison

### 1. Reopen blocked or hold execution branches

Not selected.

Reason:

1. `XB-CH-2` is still `needs-assets`, because the paired model/split/shared-metric/bounded-budget reopen contract remains unsatisfied.
2. `GB-CH-2` already produced a bounded `negative but useful` packet and still has no genuinely new gating signal.
3. `BB-CH-2` remains asset-blocked on real query images and endpoint/budget freeze.
4. `WB-CH-2` remains `not-requestable` on the current admitted asset family.

### 2. Promote `I-A` back into the main slot again

Not selected.

Reason:

1. `X-30` already confirmed the current mechanistic / low-FPR / bounded-adaptive `I-A` wording is stable.
2. Re-promoting it immediately would be repeat maintenance, not a stronger lane.

### 3. Promote `cross-box / system-consumable wording maintenance` back into the main slot

Selected.

Reason:

1. It is the only immediately executable non-graybox lane left that does not pretend a blocked/hold branch has reopened.
2. Active intake/system-consumable surfaces still drift on current truth:
   - `Finding NeMo` is still encoded as `Phase E` intake-only `zero-GPU hold` in active higher-layer docs and candidate registry surfaces.
   - `Phase E` still exposes an outdated intake-review ordering even though `Finding NeMo` already moved into an executed bounded packet and post-packet falsifier boundary.
3. Fixing that drift can still change project-level reading for Leader/materials/system-consumable consumers, so it exceeds pure sidecar value.

## Selection

- `selected_next_live_lane = X-33 cross-box / system-consumable stale intake sync after X-32 reselection`

## Verdict

- `x32_next_lane_reselection_after_x31_stale_entry_sync = positive`

More precise reading:

1. no blocked or hold non-graybox branch honestly reopened;
2. `I-A` is stable enough to remain sidecar only;
3. the strongest main-slot choice is maintenance-shaped but still project-level relevant: clear the remaining stale intake/system surfaces.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-32 non-graybox next-lane reselection after X-31 stale-entry sync`
- `next_live_cpu_first_lane = X-33 cross-box / system-consumable stale intake sync after X-32 reselection`
- `carry_forward_cpu_sidecar = I-A higher-layer boundary maintenance`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x32-next-lane-reselection-after-x31-stale-entry-sync.md`

## Handoff Decision

- `D:\Code\DiffAudit\Research\ROADMAP.md`: update required
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`: update required
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`: update required
- `D:\Code\DiffAudit\Research\docs\mainline-narrative.md`: update required
- `D:\Code\DiffAudit\ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = note-level only`

Reason:

- this selection changes lane ordering and higher-layer reading, but it still does not change admitted metrics, runtime contracts, or Platform/Runtime schema.
