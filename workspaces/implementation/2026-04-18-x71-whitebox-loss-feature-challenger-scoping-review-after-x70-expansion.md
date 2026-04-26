# 2026-04-18 X-71 White-Box Loss-Feature Challenger Scoping Review After X-70 Expansion

## Question

Does the restored `WB-CH-4 white-box loss-feature challenger family` contain one honest bounded next lane worth the main CPU-first slot?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x70-non-graybox-candidate-surface-expansion-after-x69-reselection.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\plan.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-distinct-whitebox-defended-family-import-selection-review.md`
- `D:\Code\DiffAudit\Research\docs\paper-reports\white-box\2025-popets-white-box-membership-inference-diffusion-models-report.md`
- `D:\Code\DiffAudit\Research\docs\paper-reports\ocr\white-box\2025-popets-white-box-membership-inference-diffusion-models\pages\page-0016.md`
- `D:\Code\DiffAudit\Research\docs\paper-reports\ocr\white-box\2025-popets-white-box-membership-inference-diffusion-models\pages\page-0017.md`
- `D:\Code\DiffAudit\Research\docs\paper-reports\survey\2025-aaai-privacy-preserving-lora-membership-inference-latent-diffusion-models-report.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\external\GSA\DDPM\gen_l2_gradients_DDPM.py`

## Scoping Review

### 1. The family is real, but not every member is honest to schedule

The restored family contains at least four variants:

1. threshold loss baseline
2. `LSA*`
3. `LiRA`
4. `Strong LiRA`

But the whole family should not be promoted wholesale.

### 2. High-cost LiRA variants are below the current bounded gate

Current repo-grounded reading:

- online `LiRA` requires many shadow models and target-conditioned distribution construction
- `Strong LiRA` is even further above current bounded host-fit budget
- the paper itself treats those variants as much more resource-heavy than the current `GSA` line

So the honest next step is **not**:

- immediate `LiRA` execution
- immediate new GPU release
- a claim that the whole loss-feature family is now execution-ready

### 3. One bounded same-asset lane does survive: `LSA*`-style loss baseline review

What *is* still honest:

- the white-box paper directly compares `GSA` against same-setting loss-feature baselines on `DDPM/CIFAR10`
- the repository already owns the admitted `DDPM/CIFAR10` white-box asset family
- the current `GSA` DDPM extraction path already computes per-sample denoising loss before gradient backpropagation

Therefore the first bounded lane is:

- not a full `LiRA` reproduction
- but a `same-asset LSA* / threshold-loss contract review` on the current admitted white-box surface

### 4. Why this lane is worth the slot

This lane has better project-level value than a generic white-box reread because it can:

1. turn white-box candidate generation back into method diversity rather than family-alias churn;
2. give a cleaner threat-model partner for `SMP-LoRA / DP-LoRA`, whose primary defended target is white-box loss-based MI;
3. stay CPU-first until a real contract is frozen.

## Verdict

- `x71_whitebox_loss_feature_challenger_scoping_verdict = positive but bounded`

The first honest next lane is:

- `X-72 white-box same-asset loss-feature contract review after X-71 scoping`

More precise reading:

1. `WB-CH-4` is a real fresh candidate surface.
2. `LiRA / Strong LiRA` stay below current bounded release.
3. The only honest near-term entry point is a bounded `LSA*`-style same-asset contract review.

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current execution lane = X-72 white-box same-asset loss-feature contract review after X-71 scoping`
- `current CPU sidecar = I-A higher-layer boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/workspaces/implementation/challenger-queue.md`: update required
- `Research/workspaces/white-box/plan.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- root `ROADMAP.md`: update required
- prompt/bootstrap docs: update required
- `Platform/Runtime`: no direct handoff required
- competition/materials sync: note-level only
