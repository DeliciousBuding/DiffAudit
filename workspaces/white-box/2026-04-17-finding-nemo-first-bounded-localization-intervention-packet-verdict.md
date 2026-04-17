# 2026-04-17 Finding NeMo First Bounded Localization/Intervention Packet Verdict

## Question

After `I-B.5` froze the first executable packet, can that packet actually run on current admitted assets, and if it runs, what exact reading is honest?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-first-bounded-localization-intervention-packet-selection.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-quality-vs-defense-metric-contract.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-cross-permission-inmodel-intervention-review.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\finding-nemo-first-bounded-localization-packet-20260417-r1\summary.json`

## Executed Packet

CPU-only command:

```powershell
conda run -n diffaudit-research python -m diffaudit export-gsa-observability-inmodel-packet `
  --workspace D:\Code\DiffAudit\Research\workspaces\white-box\runs\finding-nemo-first-bounded-localization-packet-20260417-r1 `
  --repo-root D:\Code\DiffAudit\Research\workspaces\white-box\external\GSA `
  --assets-root D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow-epoch300-rerun1 `
  --checkpoint-root D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow-epoch300-rerun1\checkpoints\target `
  --checkpoint-dir D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow-epoch300-rerun1\checkpoints\target\checkpoint-9600 `
  --split target-member `
  --sample-id target-member/00-data_batch_1-00965.png `
  --control-split target-nonmember `
  --control-sample-id target-nonmember/00-data_batch_1-00467.png `
  --layer-selector mid_block.attentions.0.to_v `
  --mask-kind top_abs_delta_k `
  --k 8 `
  --alpha 0.5 `
  --timestep 999 `
  --noise-seed 7 `
  --mask-seed 11 `
  --prediction-type epsilon `
  --device cpu `
  --resolution 32
```

## Run Anchor

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\finding-nemo-first-bounded-localization-packet-20260417-r1\summary.json`

## Result

The packet completed successfully.

Frozen packet identity:

- member:
  - `target-member/00-data_batch_1-00965.png`
- control:
  - `target-nonmember/00-data_batch_1-00467.png`
- checkpoint:
  - `checkpoint-9600`
- selector:
  - `mid_block.attentions.0.to_v`
- mask:
  - `top_abs_delta_k`
  - `k = 8`
  - `alpha = 0.5`

Observed metrics:

- `selected_channel_abs_delta_pre = 0.044396`
- `selected_channel_abs_delta_post = 0.022198`
- `selected_delta_retention_ratio = 0.5`
- `off_mask_drift = 0.0`
- `epsilon_prediction_rms_drift_mean = 2.81695e-07`
- `epsilon_prediction_max_abs_drift_mean = 1.430511e-06`
- `records_written = 2`
- `tensor_artifacts_written = 8`

Interpretation:

1. the packet is execution-ready on current admitted assets;
2. the in-model hook applies the intended local attenuation;
3. the selected local contrast moves exactly in the expected direction;
4. downstream `epsilon`-prediction drift is present but extremely small on this pair.

## Boundary

This packet is still **not** a defense-positive result.

What it proves:

- the first bounded `I-B` packet is executable;
- local intervention semantics are now real rather than note-only;
- the repository can read local movement and a mandatory no-sampling drift proxy in one machine-readable packet.

What it does **not** prove:

- attack-side `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` weakening;
- quality-preserving privacy defense on an evaluation bundle;
- neuron localization;
- second executable white-box defense family;
- any GPU-worthy question.

## Verdict

- `finding_nemo_first_bounded_localization_intervention_packet_verdict = positive but bounded`

More precise reading:

1. `I-B.6` is now satisfied:
   - the first bounded localization/intervention packet executes successfully
2. the correct reading is:
   - `execution-positive / defense-unproven`
3. the next honest question is no longer packet executability:
   - it is how to choose one bounded attack-side evaluation surface for the first quality-vs-defense review.

## Next Step

- `next_live_cpu_first_lane = I-B.7 bounded attack-side evaluation packet selection`
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
