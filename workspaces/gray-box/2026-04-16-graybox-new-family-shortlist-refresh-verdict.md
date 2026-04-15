# 2026-04-16 Gray-Box New-Family Shortlist Refresh Verdict

## Question

Which gray-box family should become the next genuinely new branch after the current `TMIA-DM` packaging round, and what is the first bounded next step?

## Inputs Reviewed

- `Research/ROADMAP.md`
- `workspaces/gray-box/plan.md`
- `workspaces/gray-box/2026-04-06-pia-start.md`
- `workspaces/gray-box/2026-04-16-graybox-new-family-sima-selection.md`
- `workspaces/gray-box/2026-04-16-graybox-sima-feasibility-verdict.md`
- `workspaces/gray-box/2026-04-16-graybox-structural-memorization-verdict.md`
- `workspaces/implementation/2026-04-16-subagent-paper-scout-verdict.md`
- `docs/paper-reports/markdown/gray-box/2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models/2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models-refined.md`
- `docs/paper-reports/markdown/gray-box/2026-openreview-mofit-caption-free-membership-inference/2026-openreview-mofit-caption-free-membership-inference-refined.md`

## Shortlist

Current shortlist:

1. `Noise as a Probe`
2. `MoFit`
3. `SIDe`
4. `SimA reopen`
5. `structural memorization reopen`

## Selection

- `selected_family = Noise as a Probe`

## Why `Noise as a Probe` Wins

1. It is genuinely different from the current gray-box story:
   - current headline families center on trajectory / score / time-noise exposure
   - `Noise as a Probe` opens a new gray-box interface around `controllable initial noise`
2. It still has a bounded engineering decomposition:
   - pretrain inversion
   - target-model generation with custom noise
   - final distance scoring
3. It adds high narrative value:
   - not only "middle-step access leaks"
   - but also "sampling-entry access can leak"
4. Compared with `MoFit`, it avoids the larger surrogate-optimization jump.

## Why The Others Lose For Now

### `MoFit`

- strong idea, but still a larger jump:
  - latent diffusion
  - surrogate optimization
  - caption / embedding machinery
- that is too large a protocol jump for the next bounded gray-box step

### `SIDe`

- interesting bridge paper, but current repo evidence still treats it as more explanatory than execution-near
- it does not currently beat `Noise as a Probe` on shortest credible implementation path

### `SimA reopen`

- already execution-feasible on the current local DDPM line
- but current verdict is still strength-negative
- reopening without a fresh hypothesis would likely just rescan the same weak path

### `structural memorization reopen`

- already frozen as a negative side verdict under the current local threat model

## First Bounded Next Step

The first bounded next step should be:

- `Noise as a Probe protocol / asset contract`

That step should stay:

- `CPU-only`
- `non-run`
- `contract-first`

It should answer:

1. which local target family is honest for the first smoke
2. whether current local assets can support:
   - pretrain inversion
   - custom-noise target generation
   - threshold/calibration split
3. what exact future smoke would count as the first real execution gate

## Future GPU Release Rule

`gpu_release = none` until the contract explicitly locks:

1. one target family
2. one pretrain base
3. one custom-noise generation path
4. one calibration/eval split
5. one bounded smoke question

## Verdict

- `shortlist_refresh_verdict = positive`
- `selected_family = Noise as a Probe`
- `rejected_for_now = MoFit / SIDe / SimA reopen / structural memorization reopen`
- `gpu_release = none`
- `next_step = open a CPU-first Noise-as-a-Probe protocol / asset contract`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this is candidate-generation truth only, not a result upgrade.
