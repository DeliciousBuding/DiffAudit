# 2026-04-08 White-Box Follow-Up: DPDM W-1 Shadow Comparator

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 06:25:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `defended shadow comparator complete`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Command

```powershell
conda run -n diffaudit-research python -m diffaudit run-dpdm-w1-shadow-comparator `
  --workspace D:\Code\DiffAudit\Research\workspaces\white-box\runs\dpdm-w1-shadow-comparator-20260408 `
  --target-checkpoint-path D:\Code\DiffAudit\Research\external\DPDM\runs\dpdm-cifar10-32-eps10-gpu-smoke-v7\checkpoints\final_checkpoint.pth `
  --shadow-checkpoint-path D:\Code\DiffAudit\Research\external\DPDM\runs\dpdm-cifar10-shadow01-eps10-gpu-smoke-v1\checkpoints\final_checkpoint.pth `
  --target-member-dataset-dir D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow\datasets\target-member `
  --target-nonmember-dataset-dir D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow\datasets\target-nonmember `
  --shadow-member-dataset-dir D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow\datasets\shadow-01-member `
  --shadow-nonmember-dataset-dir D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow\datasets\shadow-01-nonmember `
  --dpdm-root D:\Code\DiffAudit\Research\external\DPDM `
  --config-path D:\Code\DiffAudit\Research\external\DPDM\configs\cifar10_32\train_eps_10.0.yaml `
  --device cuda `
  --sigma-points 4 `
  --max-samples 64
```

## B. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-shadow-comparator-20260408`

Metrics:

- `AUC = 0.506348`
- `ASR = 0.5`
- `TPR@1%FPR = 0.03125`
- `TPR@0.1%FPR = 0.0`
- `shadow_member_mean_score = 0.595372`
- `shadow_nonmember_mean_score = 0.63634`
- `target_member_mean_score = 4.803495`
- `target_nonmember_mean_score = 4.502494`

## C. Interpretation

This is the first defended shadow-trained white-box comparator in the current workspace.

What it suggests:

- with the current `DPDM` smoke checkpoints, the defended comparator stays near random on the target side
- this is directionally consistent with the defense suppressing the separability that the `GSA 1k-3shadow` baseline exposed

What it does not yet prove:

- a paper-aligned `W-1` benchmark
- a final white-box defense claim
- a fair final comparison against multiple defended shadow checkpoints

Current caveats:

- only one defended shadow checkpoint is used
- the `DPDM` checkpoints are smoke-scale and still use `loss.n_noise_samples=1`
- this remains a defense-native comparator, not a direct reuse of the `GSA` `UNet2DModel` path

## D. Next Step

1. promote `DPDM` from smoke-scale to stronger defended shadows
2. add at least one more defended shadow checkpoint
3. assemble a first white-box attack-defense table using `GSA 1k-3shadow` vs `W-1 shadow comparator`
