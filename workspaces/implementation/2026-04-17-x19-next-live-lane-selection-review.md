# 2026-04-17 X-19 Non-Graybox Next-Live-Lane Selection Review

## Question

After `XB-CH-2` was refreshed and confirmed to still be `needs-assets`, what should the next honest non-graybox `CPU-first` lane become?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`
- `D:\Code\DiffAudit\Research\docs\mainline-narrative.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-crossbox-transfer-portability-blocker-refresh-review.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-post-transfer-blocker-next-lane-reselection-review.md`

## Candidate Comparison

### 1. Reopen `I-A` as the main lane again

Not selected.

Reason:

1. the current `I-A` packet is already landed and remains valid;
2. the highest-value remaining work on that branch is maintenance and wording stability, not a new main-lane verdict;
3. it should stay the carry-forward CPU sidecar rather than reclaim the main slot immediately.

### 2. Reopen blocked transfer or same-family box-local execution

Not selected.

Reason:

1. `XB-CH-2` was just reconfirmed as `needs-assets`;
2. gray-box still has no genuinely new gating signal;
3. white-box same-family rescue remains below release after the first actual `Finding NeMo / I-B` falsifier.

### 3. Execute one higher-layer stale-entry sync pass

Selected.

Reason:

1. the repo control plane is already sharper than some high-layer entry docs;
2. `mainline-narrative.md` still carries an older `2026-04-13` ordering frame and under-exposes the current `Finding NeMo` and `XB-CH-2` truth;
3. `challenger-queue.md` still recommends `XB-CH-2` as if it were the immediate next order even after the refreshed blocker verdict;
4. one bounded cross-box / system-consumable sync pass is executable now, changes project-level reading, and does not waste GPU.

## Selection

- `selected_next_live_lane = X-20 cross-box / system-consumable stale-entry sync after X-19 reselection`

## Verdict

- `x19_next_live_lane_selection_verdict = positive`

More precise reading:

1. the next honest non-graybox move is not another reselection loop and not a blocked-branch reopen;
2. the strongest immediate executable lane is a bounded higher-layer stale-entry sync pass;
3. `next_gpu_candidate` remains `none`.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `current_execution_lane = X-20 cross-box / system-consumable stale-entry sync after X-19 reselection`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x19-next-live-lane-selection-review.md`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `docs/mainline-narrative.md`: update required
- `root_sync = recommended on next Leader pass`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this step changes higher-layer ordering and wording discipline, but it does not change any runtime packet contract or admitted metric table.
