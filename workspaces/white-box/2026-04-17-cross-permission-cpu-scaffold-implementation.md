# 2026-04-17 Cross-Permission CPU Scaffold Implementation

## Question

After `I-C.5` froze the minimum executable surface, did the repository actually implement that scaffold and land real CPU-first artifacts on both the white-box and gray-box sides?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-cross-permission-executable-surface-scaffolding.md`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\gsa_observability.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\pia_adapter.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\cli.py`
- `D:\Code\DiffAudit\Research\tests\test_gsa_observability_adapter.py`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\cross-permission-masked-packet-canary-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-20260417-r1\summary.json`

## What Landed

### 1. White-box scaffold

New scaffold family:

- `export-gsa-observability-masked-packet`

What it now does:

1. loads the admitted canary/control pair
2. derives one mask family from the frozen local contract
3. writes pre-mask and post-mask tensor artifacts
4. emits packet-local fields:
   - `selected_channel_abs_delta_pre`
   - `selected_channel_abs_delta_post`
   - `selected_delta_retention_ratio`
   - `off_mask_drift`

### 2. Gray-box scaffold

New scaffold family:

- `export-pia-packet-scores`

What it now does:

1. loads one fixed member/non-member packet from canonical `PIA` assets
2. exports packet-local per-sample scores
3. emits packet-level fields:
   - `member_score_mean`
   - `nonmember_score_mean`
   - `member_control_score_gap`
4. records the exact packet indices used

## CPU Validation

### White-box canary

Run:

- `workspaces/white-box/runs/cross-permission-masked-packet-canary-20260417-r1/summary.json`

Observed result:

- `status = ready`
- `mask_kind = top_abs_delta_k`
- `k = 8`
- `alpha = 0.5`
- `selected_channel_abs_delta_pre = 0.044396`
- `selected_channel_abs_delta_post = 0.022198`
- `selected_delta_retention_ratio = 0.5`
- `off_mask_drift = 0.0`

Interpretation:

- the white-box scaffold now produces the expected packet-local masked artifact on real admitted assets

### Gray-box canary

Run:

- `workspaces/gray-box/runs/pia-packet-score-export-20260417-r1/summary.json`

Observed result:

- `status = ready`
- `packet_size = 4`
- `member_score_mean = -10.300915`
- `nonmember_score_mean = -28.518116`
- `member_control_score_gap = 18.217201`
- fixed exported packet indices exist on both sides

Interpretation:

- the gray-box scaffold now emits the packet-local score bundle that was missing in `I-C.4`

## What This Does And Does Not Mean

This means:

1. the two missing execution surfaces from `I-C.4` now exist
2. both surfaces have produced real CPU-first artifacts

This does **not** yet mean:

1. the full white-gray bridge packet has been jointly executed end-to-end
2. the current GPU candidate should automatically be restored
3. the `I-C` hypothesis has positive support

The canaries prove execution surface, not support.

## Verdict

- `cross_permission_cpu_scaffold_implementation_verdict = positive but bounded`

More precise reading:

1. `I-C.6` is now satisfied:
   - the minimum scaffold is implemented
   - both sides have real CPU-first artifacts
2. the next honest question is now interpretation and packet assembly:
   - not more theory
   - not immediate GPU release

## Next Step

- `next_live_cpu_first_lane = I-C.7 bounded white-gray CPU bridge canary interpretation and GPU re-release review`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: light sync required
- `Leader/materials`: no sync needed
- `Platform/Runtime`: no direct handoff; this is still scaffold truth, not support truth
