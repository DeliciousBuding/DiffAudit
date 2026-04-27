# 2026-04-16 Post-MoFit Gray-Box Next-Family Reselection

## Question

After `MoFit` closed as `current-contract hold`, which bounded live lane is now most worth opening, given that black-box and white-box also currently sit at `gpu_release = none`?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/intake/phase-e-candidates.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/intake/2026-04-16-finding-nemo-reconsideration-gate-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-16-dplora-post-harmonized-lane-status-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-current-contract-hold-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-15-pia-vs-secmi-graybox-comparison.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-15-graybox-ranking-sensitive-disagreement-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/docs/paper-reports/gray-box/2025-cvpr-cdi-copyrighted-data-identification-diffusion-models-report.md`
- `<DIFFAUDIT_ROOT>/Research/docs/paper-reports/gray-box/2024-arxiv-side-extracting-training-data-unconditional-diffusion-models-report.md`

## Selection Review

### 1. Black-box reopen

- `reject`
- black-box truth is already frozen as:
  - `Recon = headline`
  - `semantic-auxiliary-classifier = leading challenger`
  - `CLiD = corroboration / boundary-only`
  - `variation = contract-ready blocked`
- no genuinely new feature-family contract is ready today, so reopening black-box would mostly be maintenance rather than a new verdict

### 2. White-box reopen

- `reject`
- `Finding NeMo` remains `adapter-complete zero-GPU hold`, and its own gate review already says sparse-registry visibility does not by itself create a new hypothesis or budget
- `DP-LoRA / SMP-LoRA` is now a `metric-split bounded exploration branch + no-new-gpu-question`, so reopening it now would just drift back into low-value rerender logic

### 3. Gray-box family reopen on old branches

- `reject`
- `SimA` is already `execution-feasible but weak`
- `structural memorization` is already `negative under the current local faithful approximation`
- `MoFit` is now `execution-positive but signal-weak under the current contract`
- none of these branches currently has a fresh bounded hypothesis strong enough to justify immediate continuation

### 4. `SIDe`

- `reject for now`
- it is still more naturally a training-data-extraction / bridge paper than the shortest honest next DiffAudit audit lane
- it also lacks the same immediate reuse path that current `PIA / SecMI` score artifacts already provide

### 5. `CDI`

- `select`
- current gray-box repo state already has:
  - mature sample-level gray-box signals (`PIA`, `SecMI`)
  - same-split comparison truth
  - per-sample score artifacts on the shared `CIFAR-10` surface
- `CDI` opens a genuinely new gray-box direction:
  - collection-level audit / evidence aggregation
  - stronger audit-facing story than yet another single-sample rung
- compared with `SIDe`, it is a shorter contract-first lane because:
  - it has official code
  - it aligns with DiffAudit's audit framing
  - it can start from current local `CIFAR-10 DDPM` score surfaces instead of requiring a new latent-diffusion or extraction stack

## Verdict

- `selection_verdict = positive`
- `selected_next_live_lane = GB-42 CDI protocol / asset contract`
- `lane_type = gray-box collection-level audit extension`
- `gpu_release = none`
- `next_gpu_candidate = none`

## Carry-Forward Rule

- do not reopen `MoFit` mechanically under the same contract
- keep black-box and white-box in their current hold states
- move immediately to a CPU-first `CDI` contract review that freezes:
  - local target family
  - feature/score source
  - collection schema
  - first canary budget
  - no-go triggers

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

