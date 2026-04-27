# 2026-04-18 X-102 04 H1 First Actual Retain-Forget Pilot

## Question

After `X-101` landed Step-0 risk aggregation and exported `k=16/32/64` ladders, can `04-H1 risk-targeted SISS / retain-forget mixture` execute one actual bounded pilot on current admitted `DDPM/CIFAR10` assets?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/risk_targeted_unlearning.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_risk_targeted_unlearning.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-prep-full-overlap-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/manifests/cifar10-ddpm-1k-3shadow-epoch300-rerun1.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k32-20260418-r2/summary.json`

## What Landed

### 1. One reusable actual pilot surface now exists

The repo now exposes `run-risk-targeted-unlearning-pilot`.

It performs one bounded training loop on current admitted assets:

1. load the target `DDPM` checkpoint
2. resolve forget members from the exported index file
3. derive the retain set as the remaining target-member images
4. train one hybrid objective:
   - `keep-only`
   - `keep - alpha * forget`
5. save a defended checkpoint plus machine-readable training log

### 2. One first actual pilot run now exists

The first real run used:

- target checkpoint = `checkpoint-9600`
- forget ladder = exported `k32`
- device = `cuda`
- steps = `32`
- batch size = `4`
- `alpha = 0.5`
- `mixture_lambda = 0.5`
- `lr = 1e-5`

Artifacts now exist under:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k32-20260418-r2\`

## Actual Read

### Execution truth

This is a real execution-positive pilot:

- `forget_member_count = 33` files for `32` unique ids
- `retain_member_count = 967` files for `933` unique ids
- branch counts:
  - `keep_only = 17`
  - `keep_minus_forget = 15`
- mean losses:
  - `mean_keep_loss = 0.026602`
  - `mean_forget_loss = 0.022513`
  - `mean_objective = 0.021325`

The defended checkpoint was successfully written as:

- `checkpoint-final/model.safetensors`

### Honest boundary

This is **not** a defense verdict yet.

What is now true:

1. `04-H1` can run a real bounded retain+forget training loop on admitted local assets
2. the exported `k32` forget list is not just bookkeeping; it now drives a real defended checkpoint
3. the repo no longer lacks an actual pilot surface for this family
4. target-member duplicate sample IDs are now explicit research truth and must be handled file-faithfully in later evaluation

What is still missing:

1. forgotten-subset board
2. retained-subset board
3. full-split board
4. defense-aware rerun

So the strongest honest reading is:

- `actual pilot exists`
- not `pilot improves privacy`

## Verdict

- `x102_04_h1_first_actual_pilot_verdict = positive but bounded`

More precise reading:

1. the `04` lane is now execution-real
2. the family-selection question is over
3. the next bottleneck is evaluation, not training-surface existence
4. promotion still depends on attack-side review, especially low-FPR behavior and defense-aware rerun

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = 04-H1 forgotten-subset / full-split attack-side review on defended checkpoint`
- `04 current state = first actual retain+forget pilot executed`
- `H2 adapter = fallback only`

## Canonical Evidence Anchor

Primary anchor:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k32-20260418-r2/summary.json`

Supporting anchors:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-prep-full-overlap-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/cross-box/runs/crossbox-pairboard-gsa-targeted-full-overlap-20260418-r1/summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Platform/Runtime`: no direct handoff yet
