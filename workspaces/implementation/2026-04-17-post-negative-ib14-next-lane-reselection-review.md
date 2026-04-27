# 2026-04-17 Post-Negative-I-B.14 Next-Lane Reselection Review

## Question

After `I-B.15` froze the current `Finding NeMo / I-B` branch as an `actual bounded falsifier`, which class of work should now take the next live non-graybox `CPU-first` slot:

1. reopen `I-A` as the main lane,
2. reopen another box-local execution family,
3. move to cross-box / system-consumable sync?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-post-ia-refresh-next-lane-reselection-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-post-first-actual-packet-boundary-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`

## Candidate Review

### 1. Reopen `I-A` as the main lane

Not selected.

Reason:

1. `I-A` is still the strongest near-term innovation packet
2. but its current remaining work is already the carry-forward CPU sidecar:
   - provenance discipline
   - adaptive / low-FPR wording discipline
   - higher-layer read-order maintenance
3. using the main lane on `I-A` immediately would mostly duplicate maintenance rather than convert the new `I-B` falsifier into system-level truth

### 2. Reopen another box-local execution family

Not selected.

Reason:

1. black-box transfer / portability is still `needs-assets`
2. white-box same-family rerun is now explicitly below release after `I-B.15`
3. gray-box already yielded the current non-graybox slot
4. forcing another box-local reopen now would either:
   - restate an asset blocker
   - or repeat same-family churn

### 3. Cross-box / system-consumable sync

Selected.

Reason:

1. `I-B` just produced a sharper higher-layer boundary than before:
   - not `zero-GPU hold`
   - not defense-positive
   - one real bounded falsifier now exists
2. that narrower reading affects:
   - leader-facing summary wording
   - competition-facing non-admitted wording
   - innovation-funnel truth
3. this is the highest-value way to convert the new negative result into project-level clarity without wasting the next GPU slot

## Selection

- `selected_next_live_lane = X-17 cross-box / system-consumable sync after first actual negative I-B packet`

## Verdict

- `post_negative_ib14_next_lane_reselection_verdict = positive`

More precise reading:

1. the reselection is no longer ambiguous:
   - another box-local reopen is weaker than one sync pass
2. the correct immediate next move is:
   - sync the sharper `I-B` boundary into higher-layer truth
3. GPU remains correctly idle:
   - `next_gpu_candidate = none`

## Next Step

- `next_live_cpu_first_lane = X-17 cross-box / system-consumable sync after first actual negative I-B packet`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `Leader/materials`: wording-only sync suggested
- `Platform/Runtime`: no direct handoff yet
- `competition_material_sync = wording-only sync suggested`
