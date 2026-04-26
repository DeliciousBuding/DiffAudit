# 2026-04-16 DP-LoRA Comparator Artifact-Board Preflight

## Question

After `WB-7` reconciled the future comparator board to `baseline vs frozen SMP-LoRA vs W-1`, do the current local artifacts already form a usable board for honest release review, or is there still an artifact/schema mismatch that must be fixed first?

## Inputs

- `workspaces/white-box/2026-04-16-dplora-comparator-contract-reconciliation.md`
- `outputs/smp-lora-phase2/baseline_nodefense_target-64/evaluation.json`
- `outputs/smp-lora-sweep/sweep_results.json`
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-strongv3-3shadow-full.md`

## Board Check

### A. Baseline anchor

Present and readable:

- `outputs/smp-lora-phase2/baseline_nodefense_target-64/evaluation.json`
- current values:
  - `AUC = 0.5565217391304348`
  - `accuracy = 0.5263157894736842`
  - `num_member = 63`
  - `num_nonmember = 63`

### B. Frozen SMP-LoRA local candidate

Present and readable:

- `outputs/smp-lora-sweep/sweep_results.json`
- frozen local candidate:
  - `lambda=0.1 / rank=4 / epochs=10`
  - `AUC = 0.34375`
  - `accuracy = 0.39473684210526316`
  - `num_member = 63`
  - `num_nonmember = 63`

### C. W-1 reference

Present and readable:

- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-strongv3-3shadow-full.md`
- current values:
  - `AUC = 0.488783`
  - `ASR = 0.4985`
  - `TPR@1%FPR = 0.009`
  - `target_eval_size = 2000`

## Mismatch

The current board is **not** yet an honest same-protocol comparator board.

Why:

1. `baseline` and frozen `SMP-LoRA` are on the same local `63 / 63` evaluation surface
2. `W-1 strong-v3 full-scale` is reported on a much larger and different evaluation surface:
   - `target_eval_size = 2000`
3. the metrics are also not aligned:
   - baseline / SMP-LoRA currently expose `AUC + accuracy`
   - `W-1` exposes `AUC + ASR + low-FPR metrics`
4. so the board is enough for queue truth, but not enough for a release-review verdict

## Interpretation

This means:

- the lane is still alive
- the reconciled board is conceptually correct
- but the artifact board is still `schema-misaligned`

So the next honest step is not GPU release.

The next honest step is:

- one CPU-side comparator schema / metric alignment note
- or one bounded evaluation refresh that puts the three rows onto an explicitly comparable board

## Verdict

- `artifact_board_preflight = negative but useful`
- `board_exists = partial`
- `board_ready_for_release_review = no`
- `current_blocker = schema-and-eval-surface mismatch`
- `gpu_release = none`
- `next_step = define a comparator schema-alignment contract before any release review`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes white-box queue truth and prevents a false release review, but it does not change admitted white-box claims.
