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

This run now counts as progress because it has produced a new paper-aligned runtime summary with a complete metric set.

## F. Current Runtime Observation

Latest confirmed state after the latest runtime check:

- launcher PID is still alive:
  - `87052`
- current active stage:
  - runtime summary produced
- current visible progress in `target/train.stderr.log`:
  - reached `Epoch 299`
  - reached at least `step 9600`
- current visible loss band:
  - roughly `0.015` to `0.08`
- current checkpoint root state:
  - `target/` exists under the new rerun root
  - target checkpoints observed:
    - `checkpoint-1601`
    - `checkpoint-3201`
    - `checkpoint-4801`
    - `checkpoint-6401`
    - `checkpoint-8001`
    - `checkpoint-9600`
  - `shadow-01` root now also exposes `checkpoint-9600`
  - `shadow-02` root now also exposes `checkpoint-9600`
  - `shadow-03` root now also exposes `checkpoint-9600`
  - the rerun root initially still failed `probe_gsa_assets` because `manifests/` existed but had no manifest file
  - that blocker has now been removed by adding a rerun manifest:
    - `workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/manifests/cifar10-ddpm-1k-3shadow-epoch300-rerun1.json`
  - after removing this blocker, the runtime mainline was manually relaunched against the rerun asset root
  - the rerun runtime summary is now present at:
    - `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
  - the resulting metrics are:
    - `auc = 0.998192`
    - `asr = 0.9895`
    - `tpr@1%fpr = 0.987`
    - `tpr@0.1%fpr = 0.432`
  - these metrics are stronger than the old `20260408 1k-3shadow` admitted mainline under the same evaluation shape
  - current promotion decision:
    - this rerun should replace the old `20260408 1k-3shadow` summary as the white-box attack main evidence
- current GPU state during the latest check:
  - `memory.used ~= 6411 MiB`
  - `utilization ~= 80%`

So the rerun is no longer blocked on checkpoint production or runtime closure.
The current remaining task is only to sync this stronger rerun summary into the admitted white-box attack mainline and the system read chain.
