# 2026-04-07 White-Box Follow-Up: GSA Asset Intake Judgment

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-07 19:05:00 +08:00`
- `selected_mainline`: `GSA`
- `current_state`: `asset relevance evaluated`
- `gpu_usage`: `not requested`
- `evidence_level`: `asset-intake`

## A. Useful to GSA

### `cifar-10-python.tar.gz`

Useful, but only on the data side.

Reason:

- `GSA` DDPM gradient extraction consumes `imagefolder` data
- the CIFAR10 archive can be converted into better `target-member / target-nonmember / shadow-member / shadow-nonmember` buckets than the current toy synthetic images

This helps the white-box line build more realistic data buckets, but it does **not** solve the checkpoint gap.

## B. Not Directly Useful to GSA

### `DDPM/ckpt_cifar10.pt`

Not directly useful to upstream `GSA`.

Reason:

- current `GSA` DDPM code resumes from `accelerate` checkpoint directories like `checkpoint-*`
- this asset is a single `.pt` file
- the current single-file format is instead directly compatible with `PIA`

### `DDPM/ckpt_tini.pt`

Same judgment as above.

It is likely a `Tiny-ImageNet`-side DDPM checkpoint and fits the gray-box `PIA` style better than the current white-box `GSA` path.

### `gradtts/libritts_grad.pt`

No direct value to the current image `GSA` line.

Reason:

- it is already a gradient artifact, not a white-box DDPM checkpoint
- it belongs to the `GradTTS` / speech side, not the current image DDPM white-box mainline

### `gradtts/ljspeech_grad.pt`

Same as above.

### `cifar-100-python.tar.gz`

No direct value to the current `GSA` line.

Reason:

- current `GSA` repo context in `Project` only has an explicit CIFAR10 preprocessing path
- current `PIA` Project-side integration also only names `CIFAR10` and `TINY-IN`

## C. Why the DDPM Checkpoints Still Do Not Unblock GSA

The current extracted CIFAR10 DDPM checkpoint is a single-file torch checkpoint whose top-level keys are:

- `net_model`
- `ema_model`
- `sched`
- `optim`
- `step`
- `x_T`

This is a good fit for `PIA`.

It is **not** the format the current upstream `GSA` DDPM script resumes from. `GSA` expects a `model_dir/checkpoint-*` directory restorable via `accelerate.load_state(...)`.

So the new checkpoint intake reduces gray-box uncertainty, but it does not directly unblock white-box `GSA`.

## D. Shortest Next White-Box Step

1. Use only the CIFAR10 archive to replace the current toy white-box image buckets with more realistic `imagefolder` splits.
2. Do **not** route `ckpt_cifar10.pt` or `ckpt_tini.pt` directly into upstream `GSA`.
3. Before claiming real white-box progress, first obtain or derive a `GSA`-compatible `checkpoint-*` directory state.
