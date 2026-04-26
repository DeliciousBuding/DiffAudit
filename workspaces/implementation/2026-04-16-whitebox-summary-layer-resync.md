# 2026-04-16 White-Box Summary-Layer Resync

## Question

After `WB-18` fixed the true current status of the `DP-LoRA` successor lane, does the repository-level summary layer still describe white-box and `SMP-LoRA` truth honestly?

## Inputs

- `workspaces/white-box/2026-04-16-dplora-post-harmonized-lane-status-review.md`
- `workspaces/white-box/2026-04-16-dplora-harmonized-local-board-verdict.md`
- `docs/comprehensive-progress.md`

## Drift Found

The summary layer was stale in three ways:

1. it still described `SMP-LoRA` as if the next live question were:
   - `T06 optimizer/lr frontier`
2. it still implied the exploration line's main unresolved issue was pre-comparator release
3. it did not reflect the new white-box truth that:
   - the harmonized local board is `metric-split`
   - `DP-LoRA` is still alive
   - but the lane currently has `no-new-gpu-question`

## Sync Applied

The summary layer is now updated to say:

- mature mainline remains `PIA + GSA/W-1`
- `SMP-LoRA / DP-LoRA` remains an exploration branch
- the branch now has a bounded harmonized local comparator board
- that board is `metric-split`, not a clean local win
- current `gpu_release = none`

## Verdict

- `summary_layer_resync = positive`
- `whitebox_summary_drift_found = yes`
- `whitebox_summary_now_synced = yes`
- `gpu_release = none`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = suggested`

Reason:

- `docs/comprehensive-progress.md` is directly reused by higher-layer narrative and material preparation, so this wording change should propagate to any derivative competition summary that still says `T06 optimizer/lr frontier`.
