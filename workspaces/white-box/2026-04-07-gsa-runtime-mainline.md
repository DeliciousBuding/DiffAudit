# 2026-04-07 White-Box Follow-Up: GSA Runtime Mainline On Real Assets

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-07 23:59:00 +08:00`
- `selected_mainline`: `GSA`
- `current_state`: `real-asset closed loop ready on CPU`
- `gpu_usage`: `not requested`
- `evidence_level`: `runtime-mainline`

## A. What Was Added Before Running

- real CIFAR10 buckets under `workspaces/white-box/assets/gsa/datasets`
- real `accelerate` checkpoint roots under `workspaces/white-box/assets/gsa/checkpoints`
- first manifest under `workspaces/white-box/assets/gsa/manifests/cifar10-ddpm-mainline.md`
- Project-side `probe-gsa-assets` and `run-gsa-runtime-mainline` commands

## B. Compatibility Fixes Needed

The current local `GSA` DDPM training script required three compatibility fixes before it could produce reusable checkpoints on this machine:

1. replace deprecated `Accelerator(..., logging_dir=...)` with `project_dir=...`
2. add missing parser field `--prediction_type`
3. stop assuming tracker objects implement `add_images`

Also, the generated `checkpoint-*` directories initially contained `custom_checkpoint_0.pkl` from a registered scheduler object.
That file had to be removed so the upstream gradient extractor could resume without a registered custom object mismatch.

## C. Commands Run

Probe:

```powershell
conda run -n diffaudit-research python -m diffaudit probe-gsa-assets `
  --repo-root workspaces/white-box/external/GSA `
  --assets-root workspaces/white-box/assets/gsa
```

Mainline:

```powershell
conda run -n diffaudit-research python -m diffaudit run-gsa-runtime-mainline `
  --workspace workspaces/white-box/runs/gsa-runtime-mainline-20260407-cpu `
  --repo-root workspaces/white-box/external/GSA `
  --assets-root workspaces/white-box/assets/gsa `
  --resolution 32 `
  --ddpm-num-steps 20 `
  --sampling-frequency 2 `
  --attack-method 1
```

## D. Result

Canonical summary:

- `workspace`: `workspaces/white-box/runs/gsa-runtime-mainline-20260407-cpu`
- `contract_stage`: `target`
- `asset_grade`: `real-asset-closed-loop`
- `provenance_status`: `workspace-verified`

Closed-loop facts:

- all four gradient artifacts were generated
- both `target` and `shadow` sides resumed from real `checkpoint-2`
- the attack classifier stage completed inside the Project-side mainline wrapper

Current local metrics:

- `auc = 0.5`
- `asr = 0.5`
- `tpr@1%fpr = 0.0`
- `tpr@0.1%fpr = 0.0`

## E. Interpretation

What this proves:

- the white-box line is no longer only `toy end-to-end executable`
- the repository now has a real-data, real-checkpoint, command-driven `GSA` closed loop

What this does not prove:

- paper-level attack strength
- stable metrics on realistic data volume
- final benchmark conclusions

The current metrics are weak because this first local closed loop uses extremely small buckets and one-epoch local checkpoints.
So the right claim is:

- `real-asset closed loop ready`

not:

- `paper reproduced`
