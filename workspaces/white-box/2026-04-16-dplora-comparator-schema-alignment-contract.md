# 2026-04-16 DP-LoRA Comparator Schema-Alignment Contract

## Question

After `WB-8` confirmed that the reconciled `baseline vs frozen SMP-LoRA vs W-1` board is artifact-incomplete, what exact schema must be locked before release review is honest, and does the lane now require one bounded evaluation refresh?

## Inputs

- `workspaces/white-box/2026-04-16-dplora-comparator-release-review.md`
- `workspaces/white-box/2026-04-16-dplora-comparator-contract-reconciliation.md`
- `workspaces/white-box/2026-04-16-dplora-comparator-artifact-board-preflight.md`
- `workspaces/white-box/2026-04-16-dplora-no-go-and-gpu-release-triggers.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-strongv3-3shadow-full.md`
- `outputs/smp-lora-phase2/baseline_nodefense_target-64/evaluation.json`
- `outputs/smp-lora-sweep/sweep_results.json`

## Locked Reading

The current board has two different uses and they must not be conflated:

1. `queue truth`
   - enough to keep the `DP-LoRA` successor lane alive
   - does **not** need full release-review comparability
2. `release review`
   - must compare the three rows on one explicit shared protocol surface
   - cannot rely on mixed metric schemas or mixed evaluation sizes

`WB-8` already settled that the current artifacts only satisfy the first use.

## Required Shared Schema

Any future release-review board must lock the following fields for **all** three rows:

- `row_key`
- `artifact_role`
- `dataset_family`
- `attack_surface`
- `defense_surface`
- `target_eval_size`
- `num_member`
- `num_nonmember`
- `primary_metric = AUC`
- `seed/provenance pointer`
- `artifact path`

Optional family-specific metrics may remain attached:

- `accuracy`
- `ASR`
- `TPR@1%FPR`
- `TPR@0.1%FPR`

But those optional metrics cannot substitute for the locked shared schema above.

## Current Row Audit

### A. Baseline

- already on one local surface
- `num_member = 63`
- `num_nonmember = 63`
- `AUC = 0.5565217391304348`

### B. Frozen SMP-LoRA local candidate

- already on the same local surface as baseline
- `lambda=0.1 / rank=4 / epochs=10`
- `num_member = 63`
- `num_nonmember = 63`
- `AUC = 0.34375`

### C. W-1 strong-v3 full-scale

- not on that local surface
- `target_eval_size = 2000`
- `AUC = 0.488783`
- also carries extra defended metrics:
  - `ASR = 0.4985`
  - `TPR@1%FPR = 0.009`

## Contract Decision

The minimum honest release-review contract is:

- one board
- one shared primary metric
- one shared evaluation surface

Therefore the current board is still **not** release-review-valid.

The cleanest next alignment is:

1. keep `baseline` and frozen `SMP-LoRA` local artifacts unchanged
2. produce **one bounded refresh** for `W-1` on the same local evaluation surface
3. emit at least:
   - `AUC`
   - `target_eval_size`
   - `num_member`
   - `num_nonmember`
   - provenance pointer to the exact `W-1` checkpoint/run

This bounded refresh must be read as:

- `local comparator alignment artifact`

not as:

- a replacement for admitted `W-1 strong-v3 full-scale`
- a new same-protocol bridge closure
- a new white-box headline

## Why This Side Should Move

Refreshing `W-1` onto the local board is cleaner than rewriting the local `baseline / SMP-LoRA` artifacts upward, because:

1. the local successor lane is already frozen on `63 / 63`
2. the current mismatch is caused by the defended reference row
3. one bounded defended refresh is smaller and more honest than pretending the old `2000`-surface result is already comparable

## Verdict

- `schema_alignment_contract = locked`
- `current_board_valid_for_queue_truth = yes`
- `current_board_valid_for_release_review = no`
- `bounded_refresh_required = yes`
- `recommended_refresh_side = W-1`
- `gpu_release = none`
- `next_step = design one bounded W-1 local-surface refresh before any future release review`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this sharpens white-box comparator honesty and future release gating, but it does not change admitted claims.
