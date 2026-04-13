# 2026-04-08 White-Box Follow-Up: GSA 1k-3shadow Asset Prep

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 02:35:00 +08:00`
- `selected_attack`: `GSA`
- `current_state`: `1k-3shadow assets prepared; target training launched with file-backed logs`
- `gpu_usage`: `not required for asset prep`
- `evidence_level`: `asset-ready`

## A. What Was Added

- new dataset root:
  - `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets`
- new source archive copy:
  - `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/sources/cifar-10-python.tar.gz`
- new manifest files:
  - `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/manifests/cifar10-ddpm-1k-3shadow.json`
  - `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/manifests/cifar10-ddpm-1k-3shadow.md`
- new checkpoint roots:
  - `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/checkpoints/target`
  - `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/checkpoints/shadow-01`
  - `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/checkpoints/shadow-02`
  - `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/checkpoints/shadow-03`

## B. Dataset Layout

- `target-member`: `1000`
- `target-nonmember`: `1000`
- `shadow-01-member`: `1000`
- `shadow-01-nonmember`: `1000`
- `shadow-02-member`: `1000`
- `shadow-02-nonmember`: `1000`
- `shadow-03-member`: `1000`
- `shadow-03-nonmember`: `1000`

## C. Probe Result

`probe_gsa_assets` now recognizes the multi-shadow layout and returns a structured `blocked` state instead of throwing:

- datasets: ready
- manifests: ready
- sources: ready
- missing:
  - target `checkpoint-*`
  - shadow `checkpoint-*`

This means the white-box attack side is now blocked only by training outputs, not by asset structure.

## D. Target Training Launch

Target training is now running from the new launch script:

- launcher: `scripts/launch_gsa_training.ps1`
- output dir: `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/checkpoints/target`
- stderr log: `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/checkpoints/target/train.stderr.log`

Observed target-training evidence:

- `Epoch 0 step 1 loss=1.09`
- `Epoch 0 step 10 loss=1.07`
- `Epoch 0 step 20 loss=0.988`
- `Epoch 0 step 29 loss=0.875`
- `checkpoint-1601` has been created under the target checkpoint root
- `checkpoint-3201` has been created under the target checkpoint root
- `checkpoint-4801` has been created under the target checkpoint root
- `checkpoint-6400` has been created under the target checkpoint root
- `Epoch 30 step 992` reached
- `Epoch 96 step 3090` reached while the target process kept running
- final visible training loss in the current log window dropped to the `0.0x` to `0.05x` band
- after epoch training, the run entered the sample-generation stage (`0/1000 -> 992/1000`)

Background automation now in place:

- target trainer PID: `80808`
- shadow watcher PID: `7248`
- once target exits, the watcher launches:
  - `shadow-01-member`
  - `shadow-02-member`
  - `shadow-03-member`
- after shadows finish, the watcher triggers:
  - `python -m diffaudit run-gsa-runtime-mainline --paper-aligned`
- shadow training launcher now defaults `save_images_epochs=100000` so follow-up runs focus on checkpoint production instead of frequent sample-image generation

Current automation status:

- the repaired `shadow-sequence-v3` chain has now spawned `shadow-01-member`
- active shadow trainer PID observed: `89028`

## E. Paper-Aligned Runtime Result

The automatic chain completed the paper-aligned runtime mainline at:

- `workspaces/white-box/runs/gsa-runtime-mainline-20260408-cifar10-1k-3shadow`

Observed runtime metrics:

- `AUC = 0.97514`
- `ASR = 0.919`
- `TPR@1%FPR = 0.55`
- `TPR@0.1%FPR = 0.205`
- `shadow_train_size = 4200`
- `target_eval_size = 2000`

This is the first local white-box result in this workspace that is both:

- multi-shadow
- larger-scale
- paper-aligned enough to stop calling it near-random

## F. W-1 Compatibility Note

The current `DPDM` smoke checkpoint cannot be plugged into the current `GSA` path directly.

Two separate mismatches are now confirmed:

1. checkpoint format mismatch
   - `DPDM`: single-file `final_checkpoint.pth`
   - `GSA`: `accelerate`-style `checkpoint-*` directory
2. model architecture mismatch
   - `DPDM`: `NCSNpp`-based denoiser stack
   - `GSA`: `diffusers.UNet2DModel` DDPM path

So the next white-box defense task is not a trivial file conversion.
It needs a dedicated defense evaluation bridge or a defense-native white-box comparator.

## E. Next Step

1. produce `checkpoint-*` under `target`
2. produce `checkpoint-*` under `shadow-01/02/03`
3. rerun `run-gsa-runtime-mainline --paper-aligned` against this asset root
