# 2026-04-17 Finding NeMo First Bounded Localization/Intervention Packet Selection

## Question

After `I-B.1` through `I-B.4` froze the minimum honest bridge, the first trusted observable, the first bounded intervention proposal, and the quality-vs-defense review contract, what exact first packet should the repository execute next on current admitted white-box assets?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-minimum-honest-protocol-bridge.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-localization-observable-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-local-intervention-proposal.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-quality-vs-defense-metric-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-inmodel-intervention-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-matched-pair-freeze.md`
- `<DIFFAUDIT_ROOT>/Research/tests/test_gsa_observability_adapter.py`

## Candidate Comparison

### 1. Reuse the cross-permission matched pair (`965 / 1278`)

Why it loses as the first `I-B` packet:

- that pair was frozen for `I-C` same-packet bridge semantics, not for `I-B` localization-defense review;
- it drags in gray-box alignment pressure too early;
- it makes the first `I-B` packet harder to interpret because any result can be confused with bridge-specific packet effects.

This pair is valid for `I-C`, but it is too coupled for the first standalone `I-B` execution packet.

### 2. Stay on the native `I-B` control pair (`965 / 467`) with offline masking

Why it loses:

- it preserves the original `I-B` calibration/control semantics;
- but offline tensor masking is now below the best available execution surface;
- the repository already has an honest in-model route, so choosing offline masking would intentionally pick the weaker packet.

### 3. Stay on the native `I-B` control pair (`965 / 467`) with in-model execution

Why it wins:

- it stays inside the original `I-B` contract rather than importing `I-C` bridge semantics;
- it uses the already-landed stronger execution surface:
  - `export-gsa-observability-inmodel-packet`
- it keeps the packet white-box-only and CPU-only;
- it is the narrowest packet that can simultaneously test:
  - targeted local movement,
  - off-mask/control-surface drift,
  - downstream `epsilon`-prediction drift.

## Selected Packet

The first bounded `I-B` execution packet is now frozen as:

- `white-box only`
- `CPU only`
- `native I-B pair`
- `in-model top-k channel attenuation packet`

### Exact frozen shape

1. asset family
   - admitted `GSA epoch300 rerun1` target assets only
2. checkpoint
   - `checkpoint-9600`
3. member sample
   - `target-member/00-data_batch_1-00965.png`
4. control sample
   - `target-nonmember/00-data_batch_1-00467.png`
5. selector
   - `mid_block.attentions.0.to_v`
6. timestep
   - `999`
7. intervention kind
   - `top_abs_delta_k`
8. locality budget
   - `k = 8`
   - `alpha = 0.5`
   - `selector_count = 1`
   - `timestep_count = 1`
9. execution posture
   - in-model hook application
   - no training
   - no GPU
   - no gray-box packet coupling

### Mandatory outputs

The packet must emit:

1. white-box local movement:
   - `selected_channel_abs_delta_pre`
   - `selected_channel_abs_delta_post`
   - `selected_delta_retention_ratio`
2. locality/drift guard:
   - `off_mask_drift`
3. mandatory no-sampling quality proxy:
   - `epsilon_prediction_rms_drift_mean`
   - `epsilon_prediction_max_abs_drift_mean`
4. replayable artifacts:
   - baseline/intervened layer artifacts
   - baseline/intervened prediction artifacts
   - machine-readable `summary.json`
   - machine-readable `records.jsonl`

## Explicit Exclusions

This first packet must **not** include:

1. gray-box `PIA` packet-local readout
2. same-packet bridge semantics from `I-C`
3. split-level `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`
4. neuron naming or mechanism proof language
5. any GPU execution

If those surfaces are added, the task is no longer the first bounded `I-B` packet; it becomes a later rung or an `I-C` packet.

## Verdict

- `finding_nemo_first_bounded_localization_intervention_packet_selection_verdict = positive`

More precise reading:

1. `I-B.5` is now satisfied:
   - the first executable packet is selected
2. the packet is deliberately narrower than `I-C`:
   - it stays white-box only
   - it stays on the native `I-B` control pair
3. the next honest step is implementation/execution, not further packet design churn.

## Next Step

- `next_live_cpu_first_lane = I-B.6 implement first bounded localization/intervention packet`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `workspaces/white-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
