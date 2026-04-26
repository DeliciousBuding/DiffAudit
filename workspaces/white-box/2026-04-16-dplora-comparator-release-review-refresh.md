# 2026-04-16 DP-LoRA Comparator Release Review Refresh

## Question

After `WB-11` completed the first honest same-asset local comparator board, how should the old `WB-6` release-review verdict now be refreshed, and does the new local-board win change the current GPU release gate?

## Inputs

- `workspaces/white-box/2026-04-16-dplora-comparator-release-review.md`
- `workspaces/white-box/2026-04-16-dplora-local-board-refresh-verdict.md`
- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-local63-20260416/summary.json`
- `outputs/smp-lora-phase2/baseline_nodefense_target-64/evaluation.json`
- `outputs/smp-lora-sweep/sweep_results.json`
- `workspaces/intake/2026-04-14-baseline-smp-lora-w1-comparator-admission-packet.md`

## What Changed Since `WB-6`

`WB-6` was written when the lane only had:

- a frozen local `baseline`
- a frozen local `SMP-LoRA` candidate
- a conceptual comparator board

It explicitly did **not** yet have:

- one fresh comparator packet executed under a locked board
- one same-asset local result against `W-1`

That gap is now materially narrower.

What now exists:

- `baseline` local row:
  - `AUC = 0.5565217391304348`
- frozen `SMP-LoRA` local candidate:
  - `lambda=0.1 / rank=4 / epochs=10`
  - `AUC = 0.34375`
- refreshed `W-1 strong-v3 local63` row:
  - `AUC = 0.474175`
  - `ASR = 0.484127`
  - `target_eval_size = 126`

## Refreshed Release Read

The successor lane is now stronger than `WB-6` allowed us to say.

Why:

1. the comparator board is no longer only conceptual
2. one honest same-asset local board now exists
3. on the shared local primary metric, frozen `SMP-LoRA` beats refreshed `W-1`, and refreshed `W-1` still beats baseline

So the lane is now:

- beyond `artifact-board planning`
- beyond `execution-feasibility only`
- at `completed bounded local comparator win`

## What Still Does Not Change

The new local board still does **not** justify immediate new GPU release.

Why not:

1. the win is on a small local surface, not the admitted full-scale surface
2. `W-1` still exposes defended metrics (`ASR`, low-FPR TPRs) that local `baseline / SMP-LoRA` rows do not yet match
3. the old admission packet still carries stale assumptions:
   - old `batch14 throughput` framing
   - old comparator rung wording
   - old stop conditions written before the completed local board existed

So the honest reading is:

- `release-review signal improved`
- `gpu-release gate unchanged`

## Refreshed Verdict

`WB-6` should now be read as superseded by a stronger bounded review:

- the successor lane is no longer only `bridge-positive`
- it is now `local-comparator positive`

But it remains:

- `not admitted`
- `not full-scale comparable`
- `not yet a reason to open another GPU question`

## Next Honest Step

The next step is CPU-only:

- refresh the old comparator admission packet and release-review wording
- retire stale `batch14 throughput` comparator framing
- explicitly freeze the new bounded truth:
  - local board exists
  - local ordering favors frozen `SMP-LoRA`
  - no new GPU release follows automatically

Only after that refresh should the lane decide whether a further question even exists.

## Verdict

- `comparator_release_review_refresh = positive but bounded`
- `wb6_status = superseded-by-stronger-local-board-review`
- `local_comparator_win_exists = yes`
- `immediate_gpu_release = none`
- `current_lane_status = strengthened-but-still-pre-release`
- `next_step = refresh the comparator admission packet around the completed local board`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this improves white-box queue truth and release discipline, but it still does not change admitted/system-facing claims.
