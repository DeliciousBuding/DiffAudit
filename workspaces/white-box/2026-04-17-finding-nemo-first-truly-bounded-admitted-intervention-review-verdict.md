# 2026-04-17 Finding NeMo First Truly Bounded Admitted Intervention Review Verdict

## Question

On the first truly bounded admitted `GSA` target-anchored fixed-mask dual-run packet, does the frozen `Finding NeMo`-style local attenuation mask produce an honest attack-side defense gain under the existing locality budget?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-first-truly-bounded-admitted-packet-launch-review.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\finding-nemo-first-truly-bounded-admitted-intervention-review-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\finding-nemo-first-bounded-localization-packet-20260417-r1\summary.json`

## Executed Packet

- command surface:
  - `run-gsa-runtime-intervention-review`
- admitted family:
  - `GSA epoch300 rerun1`
- device:
  - `cuda`
- packet size:
  - `max_samples = 64`
  - `extraction_max_samples = 64`
- runtime shape:
  - `ddpm_num_steps = 1000`
  - `sampling_frequency = 10`
  - `attack_method = 1`
  - `prediction_type = epsilon`
- frozen mask:
  - selector `mid_block.attentions.0.to_v`
  - timestep `999`
  - `top_abs_delta_k`
  - `k = 8`
  - `alpha = 0.5`
  - channels `[374, 471, 269, 1, 62, 360, 187, 394]`

## Result

### 1. The admitted packet executed cleanly

The bounded real-asset packet is real, not synthetic or stitched:

- baseline gradients ready: `true`
- intervened gradients ready: `true`
- baseline metrics ready: `true`
- intervened metrics ready: `true`
- all eight baseline and eight intervened split bundles wrote `sample_count = 64`

### 2. Attack-side metrics move in the wrong direction

Baseline bounded board:

- `AUC = 0.992065`
- `ASR = 0.960938`
- `TPR@1%FPR = 0.734375`
- `TPR@0.1%FPR = 0.0`

Intervened bounded board:

- `AUC = 0.995605`
- `ASR = 0.976562`
- `TPR@1%FPR = 0.765625`
- `TPR@0.1%FPR = 0.0`

Metric deltas:

- `auc_delta = +0.00354`
- `asr_delta = +0.015624`
- `tpr_at_1pct_fpr_delta = +0.03125`
- `tpr_at_0_1pct_fpr_delta = 0.0`

So on this first honest admitted packet, the fixed mask does **not** suppress the attack-side signal.

### 3. Locality remains clean

The attached locality anchor still looks bounded and stable:

- `selected_delta_retention_ratio = 0.5`
- `off_mask_drift = 0.0`
- `epsilon_prediction_rms_drift_mean = 2.81695e-07`

This matters because the negative readout is not caused by an obviously broken or wildly drifting intervention surface.

## Interpretation

The most honest reading is:

1. the repository now has one real admitted target-anchored fixed-mask intervention-on/off bounded packet;
2. this packet is execution-positive;
3. but it is defense-negative on its own bounded attack-side board.

That is useful because it falsifies the easy overclaim:

- "if the local mask looks clean and attenuates the chosen channels, it should already help the admitted attack-side packet"

Current evidence says no.

## Verdict

- `finding_nemo_first_truly_bounded_admitted_intervention_review_verdict = negative but useful`

More precise reading:

1. `I-B.14` is now satisfied:
   - the first real admitted fixed-mask bounded packet has been executed and reviewed
2. the packet does **not** justify any defense-positive language:
   - attack metrics move slightly upward, not downward
3. the branch should now return to CPU-first review before any same-family GPU rerun:
   - no immediate repeat is honest without a genuinely new bounded hypothesis

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\finding-nemo-first-truly-bounded-admitted-intervention-review-20260417-r1\summary.json`

## Next Step

- `active_gpu_question = none`
- `next_live_cpu_first_lane = I-B.15 post-first-actual-packet boundary / reselection review`
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
