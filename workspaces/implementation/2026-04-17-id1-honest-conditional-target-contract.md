# 2026-04-17 I-D.1 Honest Conditional Target Contract

## Question

Which single conditional-diffusion target contract can the current repo freeze honestly, without overstating existing `DDPM/CIFAR10` results or pretending that all conditional families are already covered?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/reproduction-status.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-noise-as-a-probe-protocol-asset-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-protocol-asset-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-current-contract-hold-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-10-finding-nemo-protocol-reconciliation.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/intake/2026-04-11-dplora-comparability-note.md`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/dit.py`

## Decision

The first honest `I-D` target contract is:

- `target_family = Stable Diffusion v1.5` style `text-conditioned latent diffusion`
- `concrete local contract = stable-diffusion-v1-5 base + celeba_partial_target/checkpoint-25000 LoRA`
- `interpretation = latent-diffusion local canary contract`

This is the right narrow contract because:

1. the base snapshot already exists locally;
2. the target-family LoRA checkpoint already exists locally;
3. the same surface is already reused by bounded latent-diffusion notes such as `Noise as a Probe` and `MoFit`;
4. it is much closer to the intended conditional / commercial family than the current admitted `DDPM/CIFAR10` line.

## Supporting Runtime Surface

The repo also already has one real family-level execution surface on the same broader model family:

- black-box `recon DDIM public` runtime-mainline on `Stable Diffusion v1.5`

But this supporting surface must be read narrowly:

- it supports `Stable Diffusion v1.5` as a real conditional/runtime entry family;
- it does **not** upgrade the whole contract into general conditional-diffusion coverage;
- it remains constrained by `controlled / public-subset / proxy-shadow-member`.

## Frozen Contract Fields

`I-D.1` should freeze only the minimum contract:

- `target_family = text-conditioned latent diffusion`
- `base_model_root = stable-diffusion-v1-5`
- `target_adapter = celeba_partial_target/checkpoint-25000`
- `sample_contract = first bounded packet stays at 1 member + 1 non-member on the same local image contract`
- `prompt_source_rule = metadata text first, BLIP fallback second`
- `execution_surface = DDIM inversion on the base model plus target-model replay from injected latents`
- `cfg_rule = freeze one default guidance setting in I-D.1 and defer any scale variation to I-D.2`

## Why Other Candidates Do Not Win `I-D.1`

- `DDPM/CIFAR10`:
  - current mainline truth explicitly forbids presenting existing unconditional results as conditional-diffusion audit capability.
- `DiT`:
  - current repo truth only proves sample-smoke execution; it is not in a membership-inference protocol.
- `Kandinsky`:
  - there is a real minimum runtime-mainline, but it remains slower and more fragile than the `SD1.5` route and is not the cleanest first contract.
- `Finding NeMo`:
  - the original `SD1.4 / cross-attention value layers` protocol is already frozen as incompatible with current admitted white-box assets.
- `SMP-LoRA / DP-LoRA`:
  - these are defense/comparability branches, not the shortest honest definition of the conditional target family itself.
- `MoFit`:
  - it reuses the right family surface, but its current contract is already on hold as execution-positive yet signal-weak.

## Explicit Anti-Overclaim Boundaries

`I-D.1` does **not** allow any of the following:

- claiming `DDPM/CIFAR10` attack or defense results transfer to conditional diffusion;
- claiming `Stable Diffusion`, `DiT`, and `Kandinsky` are already equally covered;
- claiming `Finding NeMo` is now requestable or benchmark-ready;
- claiming `proxy-shadow-member` public runtime semantics are paper-faithful `target/shadow/member/non-member`;
- claiming any `CFG` or step-count change is monotonic before a bounded probe exists.

## Immediate Next Step

The next honest lane is:

- `I-D.2 bounded CFG-scale probe`

It should stay minimal:

- one family only: the frozen `SD1.5 + celeba_partial_target/checkpoint-25000` contract;
- one scheduler only: `DDIM`;
- one bounded packet only: `1 member + 1 non-member`;
- one tiny guidance-scale comparison only;
- one question only: whether frozen-contract membership readout and low-FPR-facing behavior move in a stable, reviewable direction under a bounded `CFG` change.

## Verdict

- `I-D.1 verdict = positive but bounded`
- `active_gpu_question = none`
- `next_gpu_candidate = I-D.2 SD1.5 CFG micro-probe on the frozen local canary contract`
- `next live CPU-first lane = I-D.2 bounded CFG-scale probe`
- `CPU sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this closes a research-side contract freeze only; it does not yet change exported fields, runner requirements, or competition headline metrics.
