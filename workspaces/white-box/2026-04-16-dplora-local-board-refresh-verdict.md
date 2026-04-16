# 2026-04-16 DP-LoRA Local Comparator Board Refresh Verdict

## Question

After `WB-10` established that a bounded `W-1` local-surface refresh was executable, what does the first completed same-asset local comparator board now actually say about `baseline vs frozen SMP-LoRA vs W-1`?

## Inputs

- `workspaces/white-box/2026-04-16-dplora-w1-local-surface-refresh-feasibility.md`
- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-local63-20260416/summary.json`
- `outputs/smp-lora-phase2/baseline_nodefense_target-64/evaluation.json`
- `outputs/smp-lora-sweep/sweep_results.json`

## Refreshed Local Board

Shared local asset surface:

- `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-member`
- `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-nonmember`

Rows:

1. `baseline`
   - `AUC = 0.5565217391304348`
   - `num_member = 63`
   - `num_nonmember = 63`
2. frozen `SMP-LoRA` local candidate
   - `lambda=0.1 / rank=4 / epochs=10`
   - `AUC = 0.34375`
   - `num_member = 63`
   - `num_nonmember = 63`
3. `W-1 strong-v3 local63 refresh`
   - `AUC = 0.474175`
   - `ASR = 0.484127`
   - `shadow_train_size = 378`
   - `target_eval_size = 126`
   - `max_samples = 63`

## Ordering on the Shared Primary Metric

On the shared local `AUC` board, lower is better for the defender.

So the local ordering is:

- frozen `SMP-LoRA` local candidate: `0.34375`
- `W-1 strong-v3 local63 refresh`: `0.474175`
- `baseline`: `0.5565217391304348`

## Interpretation

This is the first honest local comparator board that:

- uses the same legacy asset family
- exposes one shared primary metric
- includes all three required rows

What it now shows:

1. the refreshed `W-1` row still improves over baseline on the local board
2. the frozen `SMP-LoRA` local candidate improves further on that same shared primary metric
3. so the successor lane is now more than just conceptual or artifact-ready; it has one completed local-board win over the refreshed `W-1` reference

## Boundaries

This does **not** yet mean:

- admitted `W-1 strong-v3 full-scale` is replaced
- `DP-LoRA` is promoted into an admitted white-box headline
- the full white-box bridge problem is solved

Why not:

- `W-1` still carries defended metrics (`ASR`, low-FPR TPRs) that the local `baseline / SMP-LoRA` rows do not yet expose
- the refreshed board is still a small local surface, not the full admitted `target_eval_size = 2000` surface

## Verdict

- `local_board_refresh_verdict = positive but bounded`
- `same_asset_local_board_exists = yes`
- `shared_primary_metric_board_exists = yes`
- `local_ordering = smp-lora-better-than-w1-better-than-baseline`
- `successor_lane_strength = upgraded`
- `admitted_upgrade = no`
- `gpu_release = none`
- `next_step = refresh the comparator release review around the completed local board before any new GPU request`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this meaningfully upgrades internal white-box queue truth, but it still sits below admitted-claim level.
