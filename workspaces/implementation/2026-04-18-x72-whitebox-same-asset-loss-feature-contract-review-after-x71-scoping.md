# 2026-04-18 X-72 White-Box Same-Asset Loss-Feature Contract Review After X-71 Scoping

## Question

Can the current admitted `DDPM/CIFAR10` white-box asset line honestly support one bounded same-asset `LSA*`-style loss-feature contract?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-18-x71-whitebox-loss-feature-challenger-scoping-review-after-x70-expansion.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/manifests/cifar10-ddpm-1k-3shadow-epoch300-rerun1.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/external/GSA/DDPM/gen_l2_gradients_DDPM.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa.py`
- `<DIFFAUDIT_ROOT>/Research/docs/paper-reports/white-box/2025-popets-white-box-membership-inference-diffusion-models-report.md`
- `<DIFFAUDIT_ROOT>/Research/docs/paper-reports/survey/2025-aaai-privacy-preserving-lora-membership-inference-latent-diffusion-models-report.md`

## Contract Review

### 1. Same-asset identity is already frozen on admitted white-box assets

The current admitted manifest and runtime mainline already freeze the core identity needed for a same-asset review:

- same target/shadow member and non-member dataset buckets
- same target/shadow checkpoint roots
- same `DDPM/CIFAR10` family
- same `ddpm_num_steps = 1000`
- same `sampling_frequency = 10`
- same `prediction_type = epsilon`
- same `attack_method = 1`

So this is not a proposal to import a foreign asset family or change the current admitted white-box surface.

### 2. The current runtime path already computes the relevant loss internally

The current `GSA` DDPM extraction path already traverses the exact attack-side surface a same-asset loss-feature review would need:

- dataloader is already `batch_size = 1`
- the script already computes scalar denoising loss under `attack_method == 1`
- that loss is computed before gradient backpropagation on the same admitted datasets and checkpoints

So the honest repo-grounded reading is:

- the current admitted asset line is sufficient for a bounded `LSA*`-style same-asset contract
- the question is no longer candidate legitimacy
- the question is artifact/export surface

### 3. The current artifact surface is still gradient-only

The current admitted runtime mainline does **not** emit a loss-score payload:

- the runtime summary only records `summary.json`, `attack-output.txt`, and gradient tensor artifacts
- the upstream DDPM extraction script saves only `all_samples_grads`
- the internal closed-loop evaluator consumes gradient tensors and trains the current attack model on gradients, not on exported loss scores

This means the missing piece is not low-FPR metric logic or same-asset comparability. The missing piece is a bounded, artifact-safe per-sample loss-score export surface.

### 4. Honest reading of `WB-CH-4` after contract review

The current strongest honest reading is:

1. `WB-CH-4` remains a real fresh white-box candidate surface.
2. Current admitted `DDPM/CIFAR10` assets do support one bounded same-asset `LSA*`-style contract.
3. That lane is still below execution because current runtime/mainline does not yet export per-sample loss-score artifacts.
4. `LiRA / Strong LiRA` remain above the current bounded host-fit budget and should not be scheduled from this review.

## Verdict

- `x72_whitebox_same_asset_loss_feature_contract_review_verdict = positive but bounded`

More precise reading:

- `contract legitimacy = yes`
- `artifact-safe execution surface = not yet`

Therefore `X-72` closes as:

- current admitted `DDPM/CIFAR10` white-box assets support one bounded same-asset `LSA*`-style loss-feature contract
- but the current runtime/mainline is not yet ready to execute that lane honestly because no loss-score artifacts are exported

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current execution lane = X-73 white-box same-asset loss-score export surface review after X-72 contract review`
- `current CPU sidecar = I-A higher-layer boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/workspaces/implementation/challenger-queue.md`: update required
- `Research/workspaces/white-box/plan.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- root `ROADMAP.md`: update required
- prompt/bootstrap docs: update required
- `Platform/Runtime`: no direct handoff yet
- future handoff trigger: if `X-73` changes exported fields or packet contract, upgrade to note-level system handoff review
- competition/materials sync: note-level only
