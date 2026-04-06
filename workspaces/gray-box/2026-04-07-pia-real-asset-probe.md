# 2026-04-07 Gray-Box Follow-Up: PIA Real-Asset Probe Ready

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-07 19:05:00 +08:00`
- `selected_mainline`: `PIA`
- `current_state`: `real-asset probe ready on CPU`
- `gpu_usage`: `not requested`
- `evidence_level`: `real-asset-probe`

## A. Local Asset Placement

This intake is now staged under the gray-box workspace so the path semantics stay explicit:

- dataset root: `Project/workspaces/gray-box/assets/pia/datasets/cifar10`
- model dir: `Project/workspaces/gray-box/assets/pia/checkpoints/cifar10_ddpm`
- source bundle stash: `Project/workspaces/gray-box/assets/pia/sources`
- local-only config: `Project/tmp/configs/pia-cifar10-graybox-assets.local.yaml`

The original source archives remain in `D:/Code/DiffAudit/tmp/data`.

## B. What Was Used

- `cifar-10-python.tar.gz`
  - extracted into the `dataset_root/cifar10` layout that `PIA` expects
- `OneDrive_1_2026-4-7.zip`
  - extracted `DDPM/ckpt_cifar10.pt`
  - reorganized into `model_dir/checkpoint.pt`
  - original zip retained under `workspaces/gray-box/assets/pia/sources`
- existing `member split`
  - `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz`

The extracted checkpoint is a real torch checkpoint with keys:

- `net_model`
- `ema_model`
- `sched`
- `optim`
- `step`
- `x_T`

That matches the current `PIA` loader expectation for `ema_model` / `net_model`.

## C. Commands Run

```powershell
conda run -n diffaudit-research python -m diffaudit probe-pia-assets `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --member-split-root external/PIA/DDPM

conda run -n diffaudit-research python -m diffaudit dry-run-pia `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM

conda run -n diffaudit-research python -m diffaudit runtime-probe-pia `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cpu

conda run -n diffaudit-research python -m diffaudit runtime-preview-pia `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cpu `
  --preview-batch-size 4
```

## D. Result

All four validation layers now return `ready`:

- `probe-pia-assets`
- `dry-run-pia`
- `runtime-probe-pia --device cpu`
- `runtime-preview-pia --device cpu --preview-batch-size 4`

Key runtime-probe facts:

- `weights_key = ema_model`
- `preview_score_shape = [30, 1]`
- `components_loaded = true`
- `model_loaded = true`
- `attacker_instantiated = true`
- `preview_forward = true`

Key runtime-preview facts:

- `member_batch_shape = [4, 3, 32, 32]`
- `nonmember_batch_shape = [4, 3, 32, 32]`
- `member_score_shape = [30, 4]`
- `nonmember_score_shape = [30, 4]`
- `member_score_mean = 10.3009`
- `nonmember_score_mean = 28.5181`

Interpretation:

- the newly staged CIFAR10 dataset root is acceptable to the current `PIA` asset probe
- the newly staged DDPM checkpoint is acceptable to the current `PIA` checkpoint loader
- the current adapter can now load and score real member/non-member CIFAR10 batches on CPU
- this line no longer blocks at the asset-probe layer

## E. What Is Still Missing

This is not yet a benchmark result.

Current remaining gaps:

1. `Project` still has no real `PIA` runtime mainline command beyond `runtime-probe`
2. the current DDPM checkpoint source has not yet been confirmed as paper-aligned provenance
3. no GPU was needed for this phase, so no scheduler request was made
