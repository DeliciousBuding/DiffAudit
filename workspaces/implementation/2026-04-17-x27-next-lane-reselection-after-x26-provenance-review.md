# 2026-04-17 X-27 Non-Graybox Next-Lane Reselection After X-26 Provenance Review

## Question

After `X-26` froze the current `PIA` provenance blocker and its consumer reading, what should now become the next honest non-graybox `CPU-first` lane?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x26-pia-provenance-maintenance-main-lane-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-crossbox-transfer-portability-blocker-refresh-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-post-transfer-blocker-next-lane-reselection-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`

## Candidate Comparison

### 1. Reopen `XB-CH-2` as an execution probe

Not selected.

Reason:

1. the branch is still `needs-assets`;
2. no paired model contracts, paired split contracts, or ready shared execution surface appeared during `X-26`;
3. pretending it is now execution-ready would be dishonest.

### 2. Stay on provenance or `I-A` wording again immediately

Not selected.

Reason:

1. `X-26` already froze the current provenance carry-forward boundary;
2. `X-22` already froze the visible `I-A` higher-layer residue;
3. another immediate wording-only pass would be churn, not progress.

### 3. Promote `XB-CH-2` into the next CPU-side blocker/contract review

Selected.

Reason:

1. once provenance maintenance is frozen, `XB-CH-2` becomes the highest-value unresolved non-graybox branch again;
2. it is still blocked for execution, but it is honest to keep pushing its missing shared-surface / paired-contract definition on CPU;
3. this preserves momentum on a project-level cross-box question without inventing a GPU release.

## Selection

- `selected_next_live_lane = X-28 XB-CH-2 shared-surface contract freeze review after X-27 reselection`

## Verdict

- `x27_next_lane_reselection_after_x26_provenance_review = positive`

More precise reading:

1. no honest GPU candidate exists now;
2. no box-local hold branch exposes a stronger executable lane than `XB-CH-2` as a blocker/contract review;
3. the next honest move is to sharpen `XB-CH-2` on CPU rather than reopening execution.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-27 non-graybox next-lane reselection after X-26 provenance review`
- `next_live_cpu_first_lane = X-28 XB-CH-2 shared-surface contract freeze review after X-27 reselection`
- `carry_forward_cpu_sidecar = I-A higher-layer boundary maintenance`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x27-next-lane-reselection-after-x26-provenance-review.md`

## Handoff Decision

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`: update required
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`: update required
- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this step changes lane ordering only; it does not change admitted metrics, runtime contracts, or consumer schema.
