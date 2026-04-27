# 2026-04-17 Cross-Permission In-Model Intervention Review

## Question

After `I-C.9` froze the first honest matched pair, can the repository now complete `I-C.10` by:

1. landing an in-model white-box intervention surface on that pair, and
2. reusing the same intervention contract honestly enough to produce a white-gray CPU co-movement canary?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-same-packet-intervention-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-matched-pair-freeze.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/cross-permission-inmodel-packet-canary-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-packet-score-export-matched-pairfreeze-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa_observability.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/pia_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/external/PIA/DDPM/model.py`

## What Landed

### 1. White-box in-model intervention surface now exists

New executable surface:

- `export-gsa-observability-inmodel-packet`

What it now proves:

1. the matched pair can be run once in baseline mode
2. the same matched pair can be rerun with the selected channel mask applied inside the hooked forward path
3. the packet emits both layer-local movement and one downstream `epsilon`-prediction drift readout

Matched-pair canary:

- member:
  - `target-member/00-data_batch_1-00965.png`
- nonmember:
  - `target-nonmember/00-data_batch_1-01278.png`
- artifact:
  - `workspaces/white-box/runs/cross-permission-inmodel-packet-canary-20260417-r1/summary.json`

Observed white-box reading:

- `selected_channel_abs_delta_pre = 0.047201`
- `selected_channel_abs_delta_post = 0.023601`
- `selected_delta_retention_ratio = 0.5`
- `epsilon_prediction_rms_drift_mean = 2.78113e-07`
- `epsilon_prediction_max_abs_drift_mean = 1.549721e-06`

Interpretation:

- the hook is no longer offline tensor masking
- the intervention really happens inside the forward path
- the selected channels move as intended
- downstream output drift exists, but it is extremely small on the current matched pair

### 2. White-gray co-movement is still not honest to claim

The remaining blocker is no longer:

- white-box in-model execution

The remaining blocker is now:

- gray-box architecture compatibility for the same intervention contract

Current white-box selector:

- `mid_block.attentions.0.to_v`

Current `GSA` parameter shape:

- `(512, 512)`

Closest `PIA` structural alias:

- `middleblocks.0.attn.proj_v`

Current `PIA` parameter shape:

- `(256, 256, 1, 1)`

So current repo truth is:

1. `PIA` does not expose the same selector name
2. the closest structural alias is not shape-compatible with the white-box channel contract
3. directly reusing the same `channel_indices` on gray-box would be a fake same-spec bridge

## What This Means

`I-C.10` made real progress, but not full bridge completion.

What is now true:

1. white-box in-model intervention surface is landed
2. matched-pair execution is real on the white-box side
3. the remaining blocker is narrowed to gray-box alias / architecture compatibility

What is still not true:

1. no honest white-gray co-movement canary has executed
2. no same-spec gray-box intervention reuse has been shown
3. no support reading is unlocked
4. no GPU candidate should be restored

## Verdict

- `cross_permission_inmodel_intervention_review_verdict = blocked but useful`

More precise reading:

1. `I-C.10` is execution-positive on the white-box side:
   - in-model intervention is no longer missing
2. `I-C.10` is still bridge-blocked overall:
   - gray-box cannot yet reuse the same selector and channel contract honestly
3. the blocker is now much sharper:
   - selector-alias and architecture compatibility
   - not implementation vagueness
   - not packet identity drift

## Next Step

- `next_live_cpu_first_lane = I-C.11 selector-alias and architecture-compatibility review for gray-box bridge intervention`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `Leader/materials`: no sync needed
- `Platform/Runtime`: no direct handoff; still below bridge-support truth
