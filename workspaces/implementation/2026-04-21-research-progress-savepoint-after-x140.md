# 2026-04-21 Research Progress Savepoint After X-140

## Current Control State

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current execution lane = X-141 non-graybox next-lane reselection after X-140 stale-entry sync`
- `GPU-busy CPU sidecar = I-A higher-layer boundary maintenance`

## What Just Landed

### `04-H2` control closure tightened

- `X-138`: selected one minimal `4 / 4` packet-scale follow-up instead of immediate lane-yield
- `X-139`: executed the real `4 / 4` run + review pair
- `X-140`: synced higher-layer entry docs to the sharper `H2` truth

### Current honest `H2` reading

- canonical `probe / prepare / run / review` chain is landed
- first `1 / 1` same-packet transfer-only board was all-zero
- one minimal `4 / 4` follow-up is now also landed
- `4 / 4` target-transfer is non-zero:
  - `AUC = 0.5`
  - `ASR = 0.375`
  - `TPR@1%FPR = 0.5`
  - `TPR@0.1%FPR = 0.5`
- but baseline vs defended deltas remain exactly:
  - `AUC delta = 0.0`
  - `ASR delta = 0.0`
  - `TPR@1%FPR delta = 0.0`
  - `TPR@0.1%FPR delta = 0.0`

So the current control verdict is:

- `04-H2 = minimal contract-complete + bounded 4/4 follow-up negative but useful`
- still not `next_gpu_candidate`
- should yield the next `CPU-first` slot unless a genuinely new bounded hypothesis appears

## Stable Mainline Summary

- admitted attack story remains:
  - black-box `recon`
  - gray-box `PIA + stochastic-dropout`
  - white-box `GSA + W-1`
- `05-cross-box` remains promoted evidence, not the active rerun slot
- gray-box should currently keep yielding the next `CPU-first` slot
- `I-A` remains the strongest near-term innovation layer

## Canonical Anchors Added This Round

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x138-04-h2-bounded-packet-scale-followup-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-review-defense-pilot-4x4-20260421-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x140-crossbox-system-sync-after-x139.md`

## Where To Resume

1. Read `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
2. Read `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
3. Start from `X-141 non-graybox next-lane reselection after X-140 stale-entry sync`

## Storage Boundary Reminder

- upstream/local exploratory code clones -> `Research/external/`
- minimal vendored code actually used by repo -> `Research/third_party/`
- raw downloaded datasets / checkpoints / archives -> `<DIFFAUDIT_ROOT>/Download/`
- lane-normalized admitted assets / manifests -> `Research/workspaces/<lane>/assets/`
- run evidence -> `Research/workspaces/<lane>/runs/`
