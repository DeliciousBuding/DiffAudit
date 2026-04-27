# 2026-04-16 Post-Noise Next-Family Reselection

## Question

After `Noise as a Probe` has already completed its bounded gray-box branch and current black-box / gray-box / white-box live questions have all narrowed to `no-new-gpu-question` or `zero-GPU hold`, what should become the next live CPU-first research lane?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-graybox-new-family-shortlist-refresh-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/docs/paper-reports/markdown/gray-box/2026-openreview-mofit-caption-free-membership-inference/2026-openreview-mofit-caption-free-membership-inference-refined.md`

## Current State

1. Black-box has no honest immediate reopen:
   - `Recon` is frozen as headline;
   - `semantic-auxiliary-classifier` is strong but lacks a genuinely new feature-family follow-up;
   - `variation` and `CLiD` remain asset/boundary blocked.

2. White-box has no honest immediate breadth reopen:
   - `DP-LoRA / SMP-LoRA` is now a `bounded exploration branch + no-new-gpu-question`;
   - `Finding NeMo` remains `adapter-complete zero-GPU hold`;
   - no second executable defended family currently exists.

3. Gray-box remains the strongest narrative box, but its current live branches are already closed:
   - `TMIA + temporal-striding` is packaged;
   - `Noise as a Probe` is bounded and closed for now.

## Selection Review

### Selected

- `selected_next_live_lane = GB-20 MoFit protocol / asset contract`

Why:

1. `MoFit` is the cleanest remaining genuinely new gray-box mechanism:
   - it opens a caption-free conditional gray-box branch
   - it is not another time/noise-only variant of current `PIA / TMIA / Noise as a Probe`
2. It carries high project-level value:
   - it extends gray-box into the more realistic "image available, true caption unavailable" setting
   - that is a stronger narrative expansion than reopening a weaker same-family branch
3. The engineering jump is now bounded enough for a CPU-first contract task:
   - local `SD1.5 + target-family LoRA` loading already exists
   - VAE / scheduler / latent stepping fragments already exist
   - local caption / BLIP paths are already part of repo assumptions

### Not Selected Now

- `black-box new family reopen`
  - remaining options are more boundary- or asset-blocked than hypothesis-ready
- `Finding NeMo reconsideration review`
  - just closed negatively; sparse registry alone does not justify reopen
- `SimA reopen / structural memorization reopen`
  - both already have negative local verdicts and no fresh hypothesis

## Immediate Task Shape

The next live lane should stay:

- `CPU-only`
- `contract-first`
- `candidate-generation only`

Expected output:

1. lock one honest local target family
2. decide whether current repo can support:
   - surrogate optimization
   - model-fitted embedding extraction
   - score artifact recording
3. define one first bounded smoke and future `gpu_release` conditions

## Verdict

- `selection_verdict = positive`
- `selected_next_live_lane = GB-20 MoFit protocol / asset contract`
- `gpu_release = none`
- `admitted_change = none`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this is next-lane selection truth only.
