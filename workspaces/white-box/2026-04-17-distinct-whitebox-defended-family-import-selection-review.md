# 2026-04-17 Distinct White-Box Defended-Family Import / Selection Review

## Question

After black-box and gray-box near-term selection both closed, does white-box now contain any honest import-ready distinct defended family that should become the next live CPU-first lane?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/README.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/README.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-15-whitebox-second-line-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-16-whitebox-defense-breadth-shortlist-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-16-whitebox-bounded-defense-selection-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-16-whitebox-defense-breadth-closure-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-16-whitebox-feature-trajectory-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-16-whitebox-gsa2-bounded-comparator-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-16-dplora-post-harmonized-lane-status-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-whitebox-post-breadth-next-hypothesis-selection-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`

## Review

### 1. White-box still has depth, but not a new import-ready defended family

Current white-box truth is already stable:

- `GSA` remains the admitted white-box attack headline
- `W-1 = DPDM strong-v3 full-scale` remains the defended main rung
- `DP-LoRA` survives only as a bounded metric-split exploration branch
- `GSA2` is a real secondary corroboration line, but same-family

So this review is not about "whether white-box has anything left to do." It is only about whether a distinct defended family is import-ready now.

### 2. Visible candidates still fail the distinct-family selection test

1. `Finding NeMo`
   - still sits at `adapter-complete zero-GPU hold`
   - current admitted `DDPM/CIFAR-10` assets remain structurally incompatible with the original protocol surface
   - it is still observability/mechanism work, not a released defended comparator

2. `Local Mirror`
   - still collapses back into the existing `GSA` family
   - it does not add defended-family diversity at all

3. `DP-LoRA`
   - it remains alive only as a bounded metric-split local branch
   - current lane truth is already `no-new-gpu-question`
   - reopening it would be same-branch continuation, not distinct-family import

4. `GSA2`
   - it is strong and runnable
   - but it is same-family corroboration rather than a distinct defense family

### 3. The honest white-box result is still `none selected`

This round does not expose a new family-import lane. It only makes the current boundary cleaner:

- white-box still has one real defended family on the admitted line
- feature/observability routes remain below release
- same-family corroboration does not count as family import

## Verdict

- `distinct_whitebox_defended_family_import_selection_verdict = negative but clarifying`
- no distinct white-box defended family is import-ready in the current round
- keep:
  - `gpu_release = none`
  - `next_gpu_candidate = none`
- the next live CPU-first lane should now move to:
  - `ranking-sensitive variable search`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/white-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `README.md`: light sync suggested
- `Platform/Runtime`: no direct handoff required
- `Leader/materials`: suggested only; the clean wording remains that white-box currently has one real defended family plus bounded corroboration and observability-side holds
