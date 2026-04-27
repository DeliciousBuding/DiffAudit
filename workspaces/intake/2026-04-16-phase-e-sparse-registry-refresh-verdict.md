# 2026-04-16 Phase E Sparse Registry Refresh Verdict

## Question

After `WB-18`, `GB-18`, and `BB-7` all clarified that the currently explored white-box, gray-box, and black-box side branches no longer contain honest new GPU-worthy questions, what should `phase-e-candidates.json` still contain as an intake-only candidate surface?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/intake/phase-e-candidates.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/intake/2026-04-10-phase-e-intake-ordering-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/intake/2026-04-16-phase-e-registry-refresh-and-dplora-selection-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-post-temporal-striding-graybox-next-question-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-16-post-second-signal-blackbox-next-question-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-16-dplora-post-harmonized-lane-status-review.md`

## Current Truth

1. `DP-LoRA / SMP-LoRA` is no longer an intake-only candidate.
   - It already consumed a full white-box execution chain from `WB-5` through `WB-18`.
   - Its current honest boundary is `metric-split bounded exploration branch + no-new-gpu-question`.
   - That means it no longer belongs on the sparse `Phase E` intake-only candidate surface.

2. `SecMI unblock` is now stale as a registry record.
   - Gray-box no longer treats `SecMI` as an asset-blocked baseline.
   - The current gray-box truth is `SecMI = independent corroboration line`.
   - So the old `gray-box/secmi-unblock` record no longer matches repository truth.

3. `Finding NeMo + local memorization + FB-Mem` remains the only surviving intake-only candidate.
   - It is still explicitly bounded as `adapter-complete zero-GPU hold`.
   - It still requires a separate hypothesis/budget review before any reopen.
   - That keeps it in the registry, but only as a sparse hold item, not as an execution-ready next lane.

## Verdict

- `selection_verdict = negative but stabilizing`
- current `phase-e-candidates.json` should become a `sparse-hold` registry
- current intake-only candidate surface should retain only:
  - `Finding NeMo + local memorization + FB-Mem`
- current intake-only candidate surface should remove:
  - `DP-LoRA`
  - `SecMI unblock`

## Registry Decision

1. Keep `Finding NeMo` as intake review priority `#1`, but mark it explicitly as `hold`.
2. Remove `DP-LoRA` because it has already left intake-only status and is now an executed bounded exploration branch.
3. Remove `SecMI unblock` because the repository truth no longer matches an asset-blocked baseline reopening story.
4. Keep `PIA paper-aligned confirmation` only in `document_layer_conditional`, unchanged.

## Carry-Forward Rule

- `phase-e-candidates.json` should stay sparse until:
  - a genuinely new intake-only candidate appears, or
  - one of the removed branches is re-opened by a fresh hypothesis review rather than by stale registry carry-over
- the next live CPU-first lane should not be inferred mechanically from this sparse registry alone

## Handoff / Sync Decision

- `Research/ROADMAP.md`: update required
- `docs/future-phase-e-intake.md`: follow-up sync suggested
- `docs/reproduction-status.md`: follow-up sync suggested
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
