# GSA Handoff

## Goal

This handoff is for the next worker who needs to move `GSA` from:

- toy end-to-end executable

to:

- non-toy local asset preparation

without overstating evidence quality.

## What Is Already True

- upstream `GSA` gradient extraction works on this machine
- upstream `xgboost` classifier stage works on this machine
- toy end-to-end execution has already been demonstrated under `workspaces/white-box/runs`

## What Is Still Missing

- extracted white-box dataset buckets under this asset root
- `GSA`-compatible training outputs under `checkpoints/`
- any paper-aligned target/shadow checkpoint state

## Real Blocker

The current blocker is not classifier code.

It is checkpoint compatibility:

- `ckpt_cifar10.pt` is a single-file torch checkpoint
- upstream `GSA` DDPM path wants `accelerate` `checkpoint-*` directories

So the next worker must either:

- produce `checkpoint-*` directories by self-training target/shadow models

or:

- obtain already compatible checkpoint directories

## Minimal Next-Step Checklist

- [ ] place raw dataset/source material under `sources/`
- [ ] extract usable data under `datasets/`
- [ ] document exact split ownership in `manifests/`
- [ ] generate or import `checkpoint-*` directories under `checkpoints/`
- [ ] only then run `gen_l2_gradients_DDPM.py --resume_from_checkpoint latest`
- [ ] keep any resulting outputs in `workspaces/white-box/runs`, not here

## Command-Level Path

1. Prepare dataset buckets under `assets/gsa/datasets/`.
2. Train target/shadow models until `assets/gsa/checkpoints/.../checkpoint-*` exists.
3. Run gradient extraction against those `checkpoint-*` directories.
4. Run `test_attack_accuracy.py` on the resulting gradient tensors.

Until step 2 exists, the white-box line should stay in:

- `toy end-to-end executable`

not:

- `paper-aligned`
