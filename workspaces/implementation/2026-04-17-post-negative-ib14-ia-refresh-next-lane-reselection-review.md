# 2026-04-17 Post-Negative-I-B.14 I-A-Refresh Next-Lane Reselection Review

## Question

After the `I-A` refresh re-confirmed `PIA + stochastic-dropout` as the strongest near-term innovation track after the first actual negative `I-B` packet, what should become the next live non-graybox `CPU-first` lane?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-ia-refresh-after-negative-ib14-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-16-crossbox-transfer-portability-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-post-first-actual-packet-boundary-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`

## Candidate Comparison

### 1. Cross-box transfer / portability blocker refresh

Selected.

Reason:

1. it remains the highest-value unresolved non-graybox branch in the challenger queue
2. it is still asset-blocked, but that does not make it low-value:
   - it still controls whether the project can ever say anything stronger about cross-box portability
3. after `I-B` narrowed and `I-A` stabilized, this branch is now a cleaner next CPU-first review target than another same-family local reopen

### 2. Reopen gray-box ranking-sensitive hold items

Not selected.

Reason:

1. gray-box already yielded the current non-graybox slot
2. `GB-CH-2` is still a closed bounded packet with no new gating variable
3. reopening it now would mostly restate an old hold

### 3. Reopen white-box local execution again

Not selected.

Reason:

1. `I-B` just produced one actual falsifier and one explicit boundary review
2. white-box same-family rescue remains below release
3. white-box distinct-family reopen is still closed-negative

## Selection

- `selected_next_live_lane = XB-CH-2 transfer / portability blocker refresh review`

## Verdict

- `post_negative_ib14_ia_refresh_next_lane_reselection_verdict = positive`

More precise reading:

1. after the current `I-A` refresh, the next honest non-graybox `CPU-first` lane is now:
   - `XB-CH-2 transfer / portability blocker refresh review`
2. this does **not** authorize a transfer run:
   - the branch is still a blocker review, not an execution release
3. `next_gpu_candidate` should stay:
   - `none`

## Next Step

- `next_live_cpu_first_lane = XB-CH-2 transfer / portability blocker refresh review`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
