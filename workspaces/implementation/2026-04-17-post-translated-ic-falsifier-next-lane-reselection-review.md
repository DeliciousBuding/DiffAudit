# 2026-04-17 Post-Translated-I-C-Falsifier Next-Lane Reselection Review

## Question

After `I-C.14` closed the current translated cross-permission packet as `negative but useful`, which class of work should take the next live `CPU-first` slot:

1. `I-A` truth-hardening / provenance maintenance,
2. a fresh non-graybox box-local reopen,
3. cross-box / system-consumable sync?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-17-blackbox-next-family-candidate-generation-refresh-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-whitebox-post-breadth-next-hypothesis-selection-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-16-crossbox-agreement-analysis-refresh.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-translated-falsifier-review.md`

## Candidate Review

### 1. Reopen `I-A` as the main lane

Not selected.

Reason:

1. `I-A` is already landed as the current strongest near-term innovation packet.
2. Its remaining work is boundary maintenance:
   - provenance carry-forward
   - higher-layer wording discipline
   - low-FPR / adaptive-attacker read-order preservation
3. The roadmap already treats that work as the CPU sidecar, not as the best new main lane.

So reopening `I-A` now would mostly repeat maintenance rather than create a new decision-grade verdict.

### 2. Reopen black-box or white-box locally

Not selected.

Reason:

1. black-box already closed its next-family refresh as `negative but clarifying`:
   - `Recon` is stable,
   - `semantic-aux` is still same-family extension,
   - `variation` is asset-blocked,
   - `CLiD` is boundary tightening rather than a new family.
2. white-box already closed its post-breadth next-hypothesis selection as `negative but clarifying`:
   - `DP-LoRA` is still `no-new-gpu-question`,
   - `Finding NeMo` is still `not-requestable`,
   - `GSA2` is same-family corroboration only.
3. Forcing another box-local reopen right now would mostly restate known holds.

So non-graybox box-local execution is still lower-value than one cross-box sync pass.

### 3. Cross-box / system-consumable sync

Selected.

Reason:

1. `I-C` just produced a sharper boundary than before:
   - translated-contract execution is real,
   - first translated falsifier is negative,
   - current strongest honest wording is now materially narrower.
2. That boundary should be propagated upward before any other lane reopens:
   - root-level research summary
   - cross-box role split
   - higher-layer consumer wording
3. This is the highest-value way to convert the latest negative result into project-level clarity without wasting GPU or re-running same-family probes.

## Selection

- `selected_next_live_lane = X-13 cross-box / system-consumable sync after translated I-C falsifier`

## Verdict

- `post_translated_ic_falsifier_next_lane_reselection_verdict = positive`

More precise reading:

1. the reselection is no longer ambiguous:
   - box-local reopens are currently weaker than one sync pass
2. the correct immediate next move is:
   - sync the sharper `I-C` boundary into cross-box and higher-layer truth
3. GPU remains correctly idle:
   - `next_gpu_candidate = none`
   - no new bounded GPU question is ready

## Next Step

- `next_live_cpu_first_lane = X-13 cross-box / system-consumable sync after translated I-C falsifier`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `Leader/materials`: wording-only sync suggested
- `Platform/Runtime`: no direct handoff; current result sharpens boundary wording but does not change a stable consumer contract
