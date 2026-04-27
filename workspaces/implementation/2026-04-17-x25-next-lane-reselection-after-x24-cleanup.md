# 2026-04-17 X-25 Next-Lane Reselection After X-24 Cleanup

## Question

After `X-24` removed the last visible stale execution-order layer, which non-graybox lane should now occupy the main `CPU-first` slot?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-09-pia-provenance-dossier.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-10-pia-provenance-upstream-identity-note.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-10-pia-provenance-split-protocol-delta.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x24-residual-stale-entry-cleanup-verdict.md`

## Candidate Comparison

### 1. `XB-CH-2` transfer / portability probes

This branch is still valuable, but the current review truth has not changed:

- it remains `needs-assets`
- it still lacks paired model contracts, paired split contracts, and one honest shared-surface hypothesis
- no new fact since `X-18` / `XB-CH-2 refresh` upgrades it into an executable branch

So it remains a real unresolved branch, but not the strongest executable main lane right now.

### 2. `GB-CH-2` ranking-sensitive variable search

This branch already landed one bounded packet and closed `negative but useful`.

Current status is still:

- `hold`
- no genuinely new gating variable
- no honest reopen trigger

So it should stay below release rather than retake the live slot.

### 3. White-box same-family reopen

Current white-box alternatives remain weaker than the current gray-box provenance blocker:

- `Finding NeMo / I-B` is now `actual bounded falsifier`, not a defense-positive reopen
- `Finding NeMo` queue status remains `not-requestable`
- `DP-LoRA / SMP-LoRA` remains `bounded exploration branch + no-new-gpu-question`

So there is still no honest white-box reopen.

### 4. `PIA provenance maintenance`

This is the strongest executable CPU-side lane left after the stale-entry cleanup:

- it is already acknowledged as the carry-forward sidecar
- it still governs the strongest gray-box headline boundary
- it still directly affects higher-layer wording, system-consumable boundary fields, and future release honesty
- unlike the blocked/hold branches above, it can advance immediately without inventing a fake GPU question

## Review

### 1. No blocked/hold branch gained an honest execution release

`XB-CH-2`, `GB-CH-2`, and white-box same-family follow-ups still do not expose a stronger executable non-graybox branch than before.

### 2. The strongest remaining executable lane is the provenance-maintenance lane

The current repo truth already treats `PIA provenance maintenance` as the carry-forward CPU sidecar. After `X-24`, the most honest next move is to promote that sidecar into the main slot rather than re-review a blocker that already stayed blocked.

### 3. GPU posture remains unchanged

There is still no genuinely new bounded hypothesis that survives CPU-first review.

So:

- `active_gpu_question = none`
- `next_gpu_candidate = none`

## Verdict

- `x25_next_lane_reselection_after_x24_cleanup = positive`

More precise reading:

1. the stale-entry cleanup did not create any stronger executable non-graybox reopen;
2. the strongest honest next live lane is now `X-26 PIA provenance maintenance main-lane review after X-25 reselection`;
3. `PIA provenance maintenance` should stop being treated as a passive sidecar only and become the current bounded main CPU-first review lane.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-25 non-graybox next-lane reselection after X-24 cleanup`
- `next_live_cpu_first_lane = X-26 PIA provenance maintenance main-lane review after X-25 reselection`
- `carry_forward_cpu_sidecar = I-A higher-layer boundary maintenance`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x25-next-lane-reselection-after-x24-cleanup.md`

## Handoff Decision

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`: update required
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`: update required
- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this task changed the control-plane lane selection and CPU sidecar assignment;
- it did not change admitted metrics, runtime contracts, or consumer schema.
