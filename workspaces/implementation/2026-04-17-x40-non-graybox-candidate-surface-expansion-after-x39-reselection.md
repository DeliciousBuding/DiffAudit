# 2026-04-17 X-40 Non-Graybox Candidate-Surface Expansion After X-39 Reselection

## Question

After `X-39` established that the visible non-graybox candidate pool is still stale even after the `I-D` yield and stale-surface sync, which new non-graybox candidate surface should be added back into the active pool first?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x39-non-graybox-next-lane-reselection-after-x38-stale-surface-sync.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x35-non-graybox-candidate-surface-expansion-after-x34-reselection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-post-translated-ic-falsifier-next-lane-reselection-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-post-translated-ic-falsifier-system-sync-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-17-blackbox-next-family-candidate-generation-refresh-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-distinct-whitebox-defended-family-import-selection-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x36-id-conditional-future-surface-successor-selection-after-x35-expansion.md`

## Candidate Expansion Review

### 1. Re-expand black-box family generation

Not selected.

Reason:

1. current black-box review still closes `negative but clarifying`;
2. no genuinely new family contract or asset shift has appeared;
3. visible options still collapse into same-family continuation, boundary work, or asset unblock.

### 2. Re-expand white-box defended-family generation

Not selected.

Reason:

1. current white-box import review still closes `none selected`;
2. `DP-LoRA` remains bounded continuation only;
3. `Finding NeMo` remains a falsifier boundary, not an import-ready defended family.

### 3. Re-add `I-D`

Not selected.

Reason:

1. `X-36` already froze the restored `I-D` surface to `no honest bounded successor lane now`;
2. immediate re-add would only replay the just-closed yield.

### 4. Re-add `I-C`, but only as fresh bounded cross-box hypothesis generation

Selected.

Reason:

1. the current `I-C` packet family is indeed frozen as `translated-contract-only + negative falsifier`;
2. but unlike blocked black-box/white-box branches, `I-C` still has a live long-term innovation role if it is reframed as:
   - new bounded cross-box hypothesis generation,
   - not same-pair support hardening,
   - not translated-packet salvage;
3. this is now the strongest honest way to expand the non-graybox pool without pretending an old packet family has reopened.

## Expanded Surface

Add one new explicit non-graybox candidate:

- `IC-CH-1 fresh bounded cross-box hypothesis generation after translated-falsifier freeze`
  - shape: CPU-first review
  - budget: note/review only
  - immediate GPU release: none

## Selection

- `selected_next_live_lane = X-41 I-C fresh bounded cross-box hypothesis generation after X-40 expansion`

## Verdict

- `x40_non_graybox_candidate_surface_expansion_after_x39_reselection = positive`

More precise reading:

1. the candidate-surface expansion succeeded;
2. it did not reopen the old translated `I-C` packet family;
3. it restored `I-C` only as a fresh-hypothesis generation surface.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-40 non-graybox candidate-surface expansion after X-39 reselection`
- `next_live_cpu_first_lane = X-41 I-C fresh bounded cross-box hypothesis generation after X-40 expansion`
- `carry_forward_cpu_sidecar = I-A higher-layer boundary maintenance`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x40-non-graybox-candidate-surface-expansion-after-x39-reselection.md`

## Handoff Decision

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`: update required
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`: update required
- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = note-level only`

Reason:

- this pass expands the active candidate surface and restores one innovation-side review lane, but it still does not change admitted metrics, runtime contracts, or consumer schema.
