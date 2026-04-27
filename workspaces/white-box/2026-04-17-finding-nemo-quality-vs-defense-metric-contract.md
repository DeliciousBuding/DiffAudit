# 2026-04-17 Finding NeMo Quality-vs-Defense Metric Contract

## Question

Before any future review of the bounded local intervention proposal, what exact metric contract must be satisfied so the line can be judged on defense value, quality cost, and intervention locality together, rather than drifting into one-sided `AUC-only` optimism or unbounded utility damage?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-local-intervention-proposal.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-localization-observable-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-minimum-honest-protocol-bridge.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-08-unified-attack-defense-table.md`
- `<DIFFAUDIT_ROOT>/Research/docs/admitted-results-summary.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-09-pia-gpu512-adaptive-ablation.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-08-whitebox-attack-defense-table.md`

## Core Principle

This future line must never be judged on a single scalar.

The contract must simultaneously cover:

1. `defense effect`
2. `quality / utility preservation`
3. `intervention locality budget`
4. `compute cost`

If any one of these is missing, the proposal is not reviewable.

## Defense Metrics

Any future bounded review of the intervention proposal must report all four attack-side metrics on one fixed admitted white-box evaluation surface:

1. `AUC`
2. `ASR`
3. `TPR@1%FPR`
4. `TPR@0.1%FPR`

And it must compare them against one frozen baseline:

- `GSA 1k-3shadow epoch300 rerun1`

Anti-overclaim rule:

- no positive reading is allowed from `AUC` alone
- no positive reading is allowed if both low-FPR metrics get worse

## Quality / Utility Metrics

Because current admitted assets are `DDPM/CIFAR-10`, the first quality contract must stay modest and honest.

Two quality layers are required:

### A. Mandatory no-sampling quality proxy

Future review must report at least one fixed control-surface drift metric on the non-member/control side:

- `epsilon-prediction drift`
  - measured as relative `L2` drift between baseline and intervened prediction on the fixed control sample packet

Reason:

- this is available on the current local contract without pretending we already have a full generative-quality benchmark

### B. Optional sampling quality proxy

If a future review actually renders samples, it must additionally report at least one image-side proxy, such as:

- `LPIPS` surrogate
- `PSNR` / `MSE`
- or another explicitly frozen image-fidelity proxy

But this sampling proxy is optional only because the current line is still pre-release.

It does **not** replace the mandatory no-sampling drift metric.

## Locality Budget

Any future review must report the exact locality budget, not just “we intervened locally”.

Mandatory locality fields:

1. `selector_count`
2. `timestep_count`
3. `k_channels`
4. `attenuation_alpha`
5. `affected_fraction`

Current frozen default budget:

- `selector_count = 1`
- `timestep_count = 1`
- `k_channels = 8`
- `attenuation_alpha = 0.5`

Anti-overclaim rule:

- if a future review widens beyond this budget, it must be described as a new intervention rung, not the same proposal

## Compute / Cost Fields

Any future review must also report:

1. `device`
2. `wall_clock`
3. `extra_forward_passes`
4. `artifact_count`

Reason:

- locality claims are weaker if they quietly rely on much larger compute than the baseline path

## Pass / Fail Reading Rules

A future bounded review is only allowed to read as `defense-positive` if all of the following hold together:

1. at least one of `AUC` or `ASR` improves
2. at least one low-FPR metric improves
3. the control-surface drift stays within the predeclared review budget
4. the locality budget stays within the declared proposal budget

A future review must read as `negative but useful` if any of the following hold:

- `AUC` improves but both low-FPR metrics worsen
- defense effect appears only after widening locality budget without declaring a new rung
- control-surface drift is large enough that utility loss dominates the claimed privacy gain

## What This Contract Does Not Yet Do

It does not yet:

- authorize a run
- authorize a GPU question
- define final numeric thresholds for release
- define a paper-grade quality benchmark

It only freezes the minimum metric structure required before such a review could even be judged honestly.

## Verdict

- `finding_nemo_quality_vs_defense_metric_contract_verdict = positive but bounded`

More precise reading:

1. `I-B.4` is now satisfied:
   - the intervention line now has an explicit multi-axis review contract
2. this still remains below execution:
   - `gpu_release = none`
   - `next_gpu_candidate = none`
3. the main gain is governance quality:
   - future review cannot hide behind `AUC-only` gains
   - future review cannot hide quality collapse behind vague “local” wording

## Next Step

- `next_live_cpu_first_lane = I-C.1 falsifiable minimal experiment`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `docs/reproduction-status.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `Leader/materials`: no immediate sync; still below release-facing wording
- `Platform/Runtime`: no direct handoff; do not consume this as a released evaluation contract
