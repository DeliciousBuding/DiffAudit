# 2026-04-08 White-Box Follow-Up: DPDM W-1 Target-Only Comparator

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 05:10:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `target-only comparator complete`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Command

```powershell
conda run -n diffaudit-research python -m diffaudit run-dpdm-w1-target-only `
  --workspace D:\Code\DiffAudit\Research\workspaces\white-box\runs\dpdm-w1-target-only-20260408 `
  --checkpoint-path D:\Code\DiffAudit\Research\external\DPDM\runs\dpdm-cifar10-32-eps10-gpu-smoke-v7\checkpoints\final_checkpoint.pth `
  --member-dataset-dir D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow\datasets\target-member `
  --nonmember-dataset-dir D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow\datasets\target-nonmember `
  --dpdm-root D:\Code\DiffAudit\Research\external\DPDM `
  --config-path D:\Code\DiffAudit\Research\external\DPDM\configs\cifar10_32\train_eps_10.0.yaml `
  --device cuda `
  --sigma-points 4 `
  --max-samples 64
```

## B. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-target-only-20260408`

Metrics:

- `AUC = 0.493652`
- `ASR = 0.585938`
- `TPR@1%FPR = 0.015625`
- `TPR@0.1%FPR = 0.0`
- `member_mean_score = 4.803495`
- `nonmember_mean_score = 4.502493`

## C. Interpretation

What this suggests:

- on this first target-only defense-native comparator, the `DPDM` smoke checkpoint does not show the kind of strong separability that the `GSA 1k-3shadow` baseline shows
- directionally, this is consistent with `W-1` being a plausible leakage-reduction direction

What this does not prove:

- a full white-box defended attack result
- a paper-aligned `W-1` defense benchmark
- a shadow-trained defended comparator

This run is explicitly weaker than the `GSA` white-box mainline because it:

- uses only the defended target checkpoint
- does not train a defended shadow attack classifier
- uses a defense-native score rather than the full `GSA` pipeline

## D. Next Step

1. promote this target-only comparator into a defended shadow-trained comparator
2. train or derive a defended shadow checkpoint
3. produce a first same-track white-box attack-defense comparison table
