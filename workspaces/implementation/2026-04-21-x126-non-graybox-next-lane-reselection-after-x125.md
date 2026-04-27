# 2026-04-21 X-126 Non-Graybox Next-Lane Reselection After X-125

## Question

After `X-125` closed the first bounded `PIA + SimA` full-overlap review as `positive but bounded`, what is the highest-value next `CPU-first` lane: another immediate `I-A` truth-hardening pass, a fresh non-graybox reselection, or one more cross-box / system-consumable sync?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/docs/research-autonomous-execution-prompt.md`
- `<DIFFAUDIT_ROOT>/Research/docs/codex-roadmap-execution-prompt.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/docs/future-phase-e-intake.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x125-r25b-pia-sima-support-fusion-bounded-review.md`

## Findings

### 1. The control-plane surfaces are already aligned to post-`X-125` truth

Current live control surfaces already say:

- gray-box should yield the next `CPU-first` slot
- `PIA + SimA` is auxiliary-only
- `active GPU question = none`
- `next_gpu_candidate = none`

So the next issue is not the control board itself.

### 2. Two higher-layer system-consumable docs still encode pre-`X-125` steering

The remaining stale-entry surface is narrower and more specific:

- `docs/mainline-narrative.md`
- `docs/future-phase-e-intake.md`

They still encode the old `02-gray-box` read as:

- first do `SimA`
- then look at `PIA + SimA`
- keep `02` framed as an enabling lane that has not yet landed its bounded support-fusion read

That is now stale, because `X-125` already landed:

- one real `PIA + SimA` full-overlap bounded pairboard
- bounded `AUC / ASR` uplift
- partial `TPR@1%FPR` help
- no stable `TPR@0.1%FPR` lift
- gray-box yield back to non-graybox selection

### 3. Therefore the highest-value immediate move is one bounded stale-entry sync pass

Because the remaining drift is now in system-consumable narrative surfaces rather than in attack execution truth, the strongest next move is:

- one bounded `cross-box / system-consumable stale-entry sync`

not:

- another immediate gray-box extension
- another empty reselection loop
- or an `I-A` wording pass before the stale read-path is cleared

## Verdict

`positive`.

Sharper control truth:

1. gray-box has already yielded honestly after `X-125`
2. the immediate blocker is now stale higher-layer read-path drift
3. the next live lane should therefore be:
   - `X-127 cross-box / system-consumable stale-entry sync after X-126 reselection`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x126-non-graybox-next-lane-reselection-after-x125.md`

## Handoff

- `Research/ROADMAP.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `docs/mainline-narrative.md`: yes
- `docs/future-phase-e-intake.md`: yes
- `Platform/Runtime`: no

Reason:

This is a research-side live-lane ordering and stale-entry decision only. No Runtime or Platform schema change is implied.
