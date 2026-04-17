# 2026-04-17 X-49 Non-Graybox Next-Lane Reselection After X-48 Stale-Entry Sync

## Question

After `X-48` cleared the higher-layer stale-entry mismatch, which non-graybox lane is now the strongest honest main-slot choice?

## Inputs Reviewed

- `D:\Code\DiffAudit\ROADMAP.md`
- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Research\docs\mainline-narrative.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x48-cross-box-system-consumable-stale-entry-sync-after-x47-reselection.md`

## Reselection Review

### 1. Reopen fresh `I-C` immediately

Not selected.

Reason:

1. `X-46` already gave the fresh `I-C` line its first honest bounded board read;
2. that read is `negative but useful`, not blocked;
3. no stronger new bounded `I-C` hypothesis has appeared since then.

### 2. Promote `XB-CH-2`

Not selected.

Reason:

1. the transfer / portability branch still remains `needs-assets`;
2. its blocker set is sharper, but not lighter;
3. there is still no honest execution-ready reopen.

### 3. Return to `I-A` truth-hardening

Selected.

Why:

1. `I-A` remains the strongest carry-forward innovation track in current repo truth;
2. the post-`X-46` sync work is now complete, so the main slot can return to substantive work;
3. among currently visible non-graybox candidates, `I-A` is the strongest branch that:
   - is not blocked on assets
   - can still improve project-level wording discipline
   - matters directly for low-FPR and adaptive-attacker honesty

## Selection

- `selected_next_live_lane = X-50 I-A higher-layer boundary maintenance audit after X-49 reselection`

## Verdict

- `x49_non_graybox_next_lane_reselection_after_x48_stale_entry_sync = positive`

More precise reading:

1. current non-graybox control-plane drift is now cleared;
2. no stronger ready branch has reopened above `I-A`;
3. the next honest main-slot move is a bounded return to `I-A` truth-hardening.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-49 non-graybox next-lane reselection after X-48 stale-entry sync`
- `next_live_cpu_first_lane = X-50 I-A higher-layer boundary maintenance audit after X-49 reselection`
- `carry_forward_cpu_sidecar = cross-box / system-consumable wording maintenance`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x49-non-graybox-next-lane-reselection-after-x48-stale-entry-sync.md`

## Handoff Decision

- `D:\Code\DiffAudit\Research\ROADMAP.md`: update required
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`: update required
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`: optional note-level sync
- `D:\Code\DiffAudit\Research\docs\mainline-narrative.md`: optional note-level sync
- `D:\Code\DiffAudit\ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this reselection changes the live main slot again, but it does not yet change any consumer-facing contract by itself.
