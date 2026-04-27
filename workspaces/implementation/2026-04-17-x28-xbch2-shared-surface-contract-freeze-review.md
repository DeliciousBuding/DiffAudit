# 2026-04-17 X-28 XB-CH-2 Shared-Surface Contract Freeze Review

## Question

Can the current repo truth now support one honest shared-surface contract for `XB-CH-2` transfer / portability, or should the branch remain blocked with a sharper reopen contract?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-16-crossbox-transfer-portability-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-crossbox-transfer-portability-blocker-refresh-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-16-crossbox-agreement-analysis-refresh.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-16-crossbox-score-calibration-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/artifacts/unified-attack-defense-table.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`

## Candidate Shared Surfaces

### 1. Black-box / gray-box shared surface

Not honest yet.

Reason:

1. black-box headline lives on `SD1.5 + DDIM / public-subset / proxy-shadow-member`
2. gray-box headline lives on local `CIFAR-10 DDPM` membership surfaces
3. current latent-diffusion gray-box candidates (`Noise as a Probe`, `MoFit`) are still bounded challenger / hold surfaces, not a shared promoted contract

### 2. Gray-box / white-box shared surface

Still not honest enough.

Reason:

1. both sit on `CIFAR-10 DDPM`, but their current roles and packet contracts differ materially
2. gray-box uses `PIA / TMIA-DM` score contracts on one admitted local runtime surface
3. white-box uses `GSA / W-1` upper-bound and defended-comparator surfaces with different extraction/evaluation semantics
4. there is still no paired split contract and no paired model board frozen across the two boxes

### 3. Cross-box scalar score surface

Explicitly rejected again.

Reason:

1. `X-4.2` already closed scalar fusion as `negative but useful`
2. box roles are still intentionally layered rather than collapsed
3. a scalar surface would hide, not clarify, the missing portability contract

## Frozen Reopen Contract

`XB-CH-2` may reopen only if one bounded shared-surface packet can be frozen with **all** of the following:

1. one paired model contract
   - same model family or one explicitly justified transfer pair
2. one paired split contract
   - same split semantics across compared surfaces
3. one shared metric hypothesis
   - not just “scores are both large”, but one explicit portability claim on aligned semantics
4. one bounded packet budget
   - small enough for CPU-first verification before any GPU release

Current repo truth satisfies none of these four together.

## Verdict

- `x28_xbch2_shared_surface_contract_freeze_review = positive`

More precise reading:

1. the review successfully freezes the missing shared-surface contract;
2. `XB-CH-2` still remains `needs-assets`;
3. no execution reopen and no GPU release is justified.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-28 XB-CH-2 shared-surface contract freeze review after X-27 reselection`
- `next_live_cpu_first_lane = X-29 non-graybox next-lane reselection after X-28 shared-surface contract freeze review`
- `carry_forward_cpu_sidecar = I-A higher-layer boundary maintenance`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x28-xbch2-shared-surface-contract-freeze-review.md`

## Handoff Decision

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`: update required
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`: update required
- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this pass sharpens blocker requirements and lane ordering only; it does not change admitted metrics, consumer schema, or runtime requirements.
