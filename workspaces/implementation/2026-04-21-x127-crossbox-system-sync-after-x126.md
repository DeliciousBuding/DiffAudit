# 2026-04-21 X-127 Cross-Box System-Consumable Sync After X-126

## Question

After `X-126` identified stale higher-layer read-path drift, are the remaining active system-consumable docs now aligned to post-`X-125` truth?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/docs/future-phase-e-intake.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x125-r25b-pia-sima-support-fusion-bounded-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x126-non-graybox-next-lane-reselection-after-x125.md`

## Changes Applied

### 1. `mainline-narrative` no longer encodes pre-`X-125` gray-box steering

Updated `docs/mainline-narrative.md` so that `02-gray-box` now reads as:

- `SimA` packet export landed
- first bounded `PIA + SimA` pairboard landed
- `logistic_2feature` is the best fused candidate
- `AUC / ASR` uplift is real but `TPR@0.1%FPR` lift is not stable
- gray-box is auxiliary and no longer the live slot

### 2. `future-phase-e-intake` no longer tells readers to "first do SimA, then PIA + SimA"

Updated `docs/future-phase-e-intake.md` so that `02-gray-box` now reads as:

- first bounded support-fusion review already exists
- the current outcome is auxiliary-only sidecar retention
- gray-box should yield after the first bounded review

## Verdict

`positive`.

Sharper control truth:

1. the remaining active higher-layer stale-entry layer is now cleared
2. fresh sessions should no longer be steered back into pre-`X-125` gray-box wording
3. the next honest live lane becomes:
   - `X-128 non-graybox next-lane reselection after X-127 stale-entry sync`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x127-crossbox-system-sync-after-x126.md`

## Handoff

- `Research/ROADMAP.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `Platform/Runtime`: no

Reason:

This is a research-side read-path sync only. No admitted metric, schema, or runtime contract changed.
