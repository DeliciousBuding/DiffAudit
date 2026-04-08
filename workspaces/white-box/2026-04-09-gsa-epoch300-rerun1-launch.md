# 2026-04-09 White-Box Launch: GSA Epoch300 Rerun1

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 +08:00`
- `selected_attack`: `GSA`
- `current_state`: `launched a fresh paper-aligned 1k-3shadow training chain on a new local asset root`
- `gpu_usage`: `single GPU, serial target -> shadow-01 -> shadow-02 -> shadow-03 -> runtime`
- `evidence_level`: `launch-record`

## A. Why This Run Exists

- current admitted white-box attack result is already strong
- current white-box blocker is not "can it run", but whether a stronger rerun can move the paper-aligned attack line beyond the frozen baseline without polluting the admitted asset root
- this launch therefore uses a new local asset root instead of overwriting `gsa-cifar10-1k-3shadow`

## B. Planned Asset Root

- assets root:
  - `workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1`
- runtime workspace:
  - `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1`

## C. Training Parameters

- dataset layout: reuse current `1k-3shadow` CIFAR-10 split
- `resolution = 32`
- `train_batch_size = 32`
- `num_epochs = 300`
- `gradient_accumulation_steps = 1`
- `learning_rate = 1e-4`
- `lr_warmup_steps = 500`
- `mixed_precision = no`
- `save_images_epochs = 100000`
- `save_model_epochs = 50`
- `prediction_type = epsilon`

## D. Expected Logs

Per-split logs should appear under:

- `checkpoints/target/train.stdout.log`
- `checkpoints/target/train.stderr.log`
- `checkpoints/shadow-01/train.stdout.log`
- `checkpoints/shadow-02/train.stdout.log`
- `checkpoints/shadow-03/train.stdout.log`

The final runtime summary should appear at:

- `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`

## E. Acceptance

This run only counts as progress if it produces one of:

1. a new paper-aligned runtime summary with a complete metric set
2. a concrete failure mode with stable logs that explains why the stronger rerun cannot proceed
