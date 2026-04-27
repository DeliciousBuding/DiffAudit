# 2026-04-18 X-70 Non-Graybox Candidate-Surface Expansion After X-69 Reselection

## Question

After `X-69` confirmed that no blocked/hold branch reopened above sidecar maintenance, which fresh non-graybox candidate surface should be restored now?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/README.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-distinct-whitebox-defended-family-import-selection-review.md`
- `<DIFFAUDIT_ROOT>/Research/docs/paper-reports/white-box/2025-popets-white-box-membership-inference-diffusion-models-report.md`
- `<DIFFAUDIT_ROOT>/Research/docs/paper-reports/survey/2025-aaai-privacy-preserving-lora-membership-inference-latent-diffusion-models-report.md`

## Expansion Review

### 1. The previously visible non-graybox pool is still exhausted

- black-box:
  - `BB-CH-4` already closed negative
  - `BB-CH-2` is still `needs-assets`
- cross-box:
  - `XB-CH-2` is still `needs-assets`
- white-box visible candidates:
  - `GSA2` is same-family corroboration only
  - `Finding NeMo` is a non-admitted executed falsifier branch, not current execution-ready breadth
  - `DP-LoRA / SMP-LoRA` is a bounded metric-split branch, not a fresh candidate surface

So a real expansion must come from a not-yet-promoted surface, not from re-reading the currently visible queue rows.

### 2. A fresh white-box paper-backed surface still exists: loss-feature challengers

The current white-box paper stack still contains one family the queue does not yet represent explicitly:

- threshold/loss-feature white-box baselines
- `LSA*`
- `LiRA`
- `Strong LiRA`

This matters because:

1. it is a distinct feature family from the current gradient-centered `GSA/GSA2` line;
2. the main white-box paper already benchmarks this family directly against `GSA`;
3. `SMP-LoRA` is explicitly framed against white-box loss-based MI first, so this family is more relevant to white-box defense reading than another same-family `GSA` continuation.

### 3. This surface is honest to restore as CPU-first review

Restoring this surface does **not** mean releasing a run.

It means:

- promote the family into the candidate pool
- ask whether the current admitted `DDPM/CIFAR10` asset line exposes one bounded same-asset loss-feature lane
- keep GPU closed until that question is reviewed

## Verdict

- `x70_non_graybox_candidate_surface_expansion_verdict = positive`

The fresh candidate surface restored by `X-70` is:

- `WB-CH-4 white-box loss-feature challenger family`

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current execution lane = X-71 white-box loss-feature challenger scoping review after X-70 expansion`
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
